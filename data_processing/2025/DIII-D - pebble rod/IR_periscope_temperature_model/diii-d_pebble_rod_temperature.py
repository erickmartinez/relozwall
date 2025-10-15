import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data_processing.misc_utils.plot_style import load_plot_style
from scipy.interpolate import PPoly, make_smoothing_spline
from typing import Union, Tuple, Callable, Dict
from scipy import ndimage
import matplotlib.ticker as ticker


SHOT = 203783
PATH_TO_DATA = r'./data'
PATH_TO_AVERAGED_TEMPERATURE_DISTRIBUTIONS_H5 = r'./data/dimes_averaged_temperature_distributions_from_laser_tests.h5'
PATH_TO_PPPL_EVAPORATION_MODEL = r'./data/boron_evaporation_model.hdf5'
PATH_TO_LP_DATA = r'../Langmuir Probe/data/dimes_lp'
PEBBLE_ROD_DIAMETER = 0.95 # cm

def model_poly(x, b) -> np.ndarray:
    """
    A polynomial model

    Parameters
    ----------
    x: np.ndarray
        The x data points the polynomial is evaluated at
    b: np.ndarray
        The coefficients of the polynomial

    Returns
    -------

    """
    n = len(b)
    r = np.zeros(len(x))
    for i in range(n):
        r += b[i] * x ** i
    return r

def load_model(path_to_pppl_fit) \
        -> Callable[[Union[float, np.ndarray]], Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Load the fit from

    H.W. Kugel, Y. Hirooka, J. Timberlake et al., Initial boronization of PB-X using ablation of solid boronized probes.
    PPL-2903 (1993)

    Figure 16

    to estimate the evaporation rate at each time

    Parameters
    ----------
    path_to_pppl_fit: str, pathlib.Path

    Returns
    -------
    callable:
        The evaporation model
    """
    path_to_pppl_fit = Path(path_to_pppl_fit)
    with h5py.File(str(path_to_pppl_fit), 'r') as hf:
        # Load the coefficients of the polynomial fit (in log scale) for the boron evaporation rate in
        # (atoms/cm^2/s)
        model_popt = np.array(hf['/model/popt'])

        lb_ppoly_c = np.array(hf['/model/lb_ppoly/c'])
        lb_ppoly_x = np.array(hf['/model/lb_ppoly/x'])
        ub_ppoly_c = np.array(hf['/model/ub_ppoly/c'])
        ub_ppoly_x = np.array(hf['/model/ub_ppoly/x'])

    ppoly_lb = PPoly(lb_ppoly_c, lb_ppoly_x)
    ppoly_ub = PPoly(ub_ppoly_c, ub_ppoly_x)
    def evaporation_rate_model(temperature) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The evaporation rate in atoms/cm^2/s

        Parameters
        ----------
        temperature: np.ndarray
            The temperature in Kelvin

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The evaporation rate in atoms/cm^2/s
            the lower and upper bounds of the evaporation rate
        """
        rate = np.exp(model_poly(temperature, model_popt))
        lb = np.exp(ppoly_lb(temperature))
        ub = np.exp(ppoly_ub(temperature))
        return rate, lb, ub

    return evaporation_rate_model

def remove_spikes_zscore(spectrum, threshold=3, window_size=5):
    """
    Remove spikes using Z-score method with local statistics.

    Parameters:
    -----------
    spectrum : array-like
        Input spectrum/signal
    threshold : float, default=3
        Z-score threshold above which points are considered spikes
    window_size : int, default=5
        Size of the local window for calculating statistics

    Returns:
    --------
    cleaned_spectrum : ndarray
        Spectrum with spikes removed
    spike_mask : ndarray
        Boolean array indicating spike locations
    """
    spectrum = np.array(spectrum)
    cleaned_spectrum = spectrum.copy()

    # Calculate local median and MAD (Median Absolute Deviation)
    local_median = ndimage.median_filter(spectrum, size=window_size)
    mad = ndimage.median_filter(np.abs(spectrum - local_median), size=window_size)

    # Calculate modified Z-score using MAD
    with np.errstate(divide='ignore', invalid='ignore'):
        modified_z_score = 0.6745 * (spectrum - local_median) / mad

    # Identify spikes
    spike_mask = np.abs(modified_z_score) > threshold

    # Replace spikes with local median
    cleaned_spectrum[spike_mask] = local_median[spike_mask]

    return cleaned_spectrum, spike_mask

def load_lp_data(shot, path_to_folder) -> Dict[str, np.ndarray]:
    path_to_folder = Path(path_to_folder)
    with h5py.File( path_to_folder / f'{shot}_LP.h5', 'r') as h5:
        dimes_gp = h5['/LANGMUIR_DIMES']
        t_s = np.array(dimes_gp.get('time')) * 1E-3
        T_eV = np.array(dimes_gp.get('TeV'))
        qpara = np.array(dimes_gp.get('qpara'))
        qperp = np.array(dimes_gp.get('qperp'))
        ne = np.array(dimes_gp.get('ne')) # x1E13 /cm^3
    data = {
        't_s': t_s,
        'T_eV': T_eV,
        'qpara': qpara,
        'qperp': qperp,
    }
    return data

def main(
    shot, path_to_data=PATH_TO_DATA, path_to_averaged_distributions=PATH_TO_AVERAGED_TEMPERATURE_DISTRIBUTIONS_H5,
    path_to_pppl_fit=PATH_TO_PPPL_EVAPORATION_MODEL, pebble_rod_diameter=PEBBLE_ROD_DIAMETER,
        path_to_lp_data=PATH_TO_LP_DATA,
):
    path_to_data = Path(path_to_data)
    path_to_averaged_distributions = Path(path_to_averaged_distributions)
    path_to_pppl_fit = Path(path_to_pppl_fit)
    path_to_lp_data = Path(path_to_lp_data)
    path_to_shot_temperature = path_to_data / f'{shot}_mean_temperature_roi.csv'

    diii_d_df = pd.read_csv(path_to_shot_temperature, comment='#').apply(pd.to_numeric)
    time_s = 1E-3 * diii_d_df['time (ms)'].values
    temperature_k = diii_d_df['T_max (C)'].values + 273.15 #+ diii_d_df['T_max (C)'][0]
    t_min, t_max = np.min(time_s), np.max(time_s)

    lp_data = load_lp_data(shot=shot, path_to_folder=path_to_lp_data)
    time_lp = lp_data['t_s']
    msk_ir_tv = (t_min <= time_lp) & (time_lp <= t_max)
    heat_load = lp_data['qpara'][msk_ir_tv]
    time_lp = time_lp[msk_ir_tv]
    heat_load_despiked, _ = remove_spikes_zscore(spectrum=heat_load, threshold=5, window_size=50)
    spl_heat_load= make_smoothing_spline(x=time_lp, y=heat_load_despiked, lam=0.0025)

    with h5py.File(str(path_to_averaged_distributions), 'r') as f:
        histograms = np.array(f['/histograms'])
        bin_centers = np.array(f['/bin_centers'])
        dimes_mean_temperatures = np.array(f['/DiMES_mean_temperature'])


    def load_histrogram_at_temperature(ir_temperature):
        idx_bin = np.argmin(np.abs(dimes_mean_temperatures - ir_temperature))
        return histograms[idx_bin]



    evaporation_rate_model = load_model(path_to_pppl_fit)
    evaporation_rate = np.full_like(time_s, fill_value=1E-20)
    evaporation_lb = np.full_like(time_s, fill_value=1E-20)
    evaporation_ub = np.full_like(time_s, fill_value=1E-20)
    pebble_rod_area = 0.25 * np.pi * pebble_rod_diameter ** 2  # cm^2

    for i, ti in enumerate(time_s):
        histogram = load_histrogram_at_temperature(temperature_k[i])[1:]
        n_pixels_sum = np.sum(histogram)
        if n_pixels_sum > 0:
            histogram /= n_pixels_sum # <- convert to pdf
        evaporation_rates, lb, ub = evaporation_rate_model(bin_centers[1:]) * histogram
        evaporation_rate[i] = np.sum(evaporation_rates)
        evaporation_lb[i] = np.sum(lb)
        evaporation_ub[i] = np.sum(ub)

    evaporation_rate *= pebble_rod_area
    evaporation_rate *= 1E-16

    evaporation_lb *= pebble_rod_area
    evaporation_ub *= pebble_rod_area
    evaporation_lb *= 1E-16
    evaporation_ub *= 1E-16

    # spl_evaporation_rate = make_smoothing_spline(x=time_s, y=evaporation_rate, lam=0.00005)

    load_plot_style()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, constrained_layout=True, sharex=True, height_ratios=[1.5, 1])
    fig.set_size_inches(4.5, 5.)
    ax1.plot(time_s, evaporation_rate)
    # markers_p, caps_p, bars_p = ax1.errorbar(
    #     time_s, evaporation_rate, yerr=(evaporation_lb, evaporation_ub),
    #     marker='o', ms=9, mew=1.25, mfc='none',  # label=f'{lbl}',
    #     capsize=2.75, elinewidth=1.25, lw=1.5, c='C0', ls='none'
    # )
    #
    # [bar.set_alpha(0.35) for bar in bars_p]
    # [cap.set_alpha(0.35) for cap in caps_p]

    ax1.fill_between(time_s, evaporation_lb, evaporation_ub, alpha=0.25, color='C0', ec='None')
    # ax1.set_yscale('log')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(r'{\sffamily Evaporation rate (x10\textsuperscript{16} atoms/s)}', usetex=True)

    # ax2.plot(time_lp, spl_heat_load(time_lp), label='Heat load')
    ax2.plot(time_lp, heat_load, color='tab:red')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'{\sffamily Heat load (MW/m\textsuperscript{2})}', usetex=True)
    # ax2.set_xlim(1.5, 3)

    ax1.set_title(f'Shot #{shot}')
    ax1.set_xlim(left=1.5, right=t_max)
    ax1.set_ylim(bottom=0, top=2.5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    output_file_tag = f'{shot}_evaporation_rate'
    path_to_figures = Path('./figures/evaporation_rates')
    path_to_figures.mkdir(parents=True, exist_ok=True)

    fig.savefig(f'{path_to_figures}/{output_file_tag}.png', dpi=600)
    plt.show()

if __name__ == '__main__':
    main(
        shot=SHOT, path_to_data=PATH_TO_DATA, path_to_pppl_fit=PATH_TO_PPPL_EVAPORATION_MODEL,
        path_to_averaged_distributions=PATH_TO_AVERAGED_TEMPERATURE_DISTRIBUTIONS_H5, pebble_rod_diameter=PEBBLE_ROD_DIAMETER,
    )



