import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats.distributions import t
from spyder.utils.external.lockfile import unique

from data_processing.misc_utils.plot_style import load_plot_style
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d, make_smoothing_spline
from pathlib import Path
from typing import List, Tuple
from scipy import ndimage

PATH_TO_LASER_EVAPORATION = r'./data/laser_tests/evaporation_rates/evaporation_rates.csv'
PATH_TO_LASER_MASS_LOSS_RATES = r'./data/laser_tests/mass_loss_rates/mass_loss_rate.csv'
PATH_TO_D3D_MASS_LOSS_RATES = r'./data/mass_loss_rate_model/model_results/dmdt_vs_qpara'
PATH_TO_D3D_EVAPORATION_RATES = r'./data/d3d_evaporation_rates'
PATH_TO_LP_DATA = r'./data/dimes_lp'

SHOTS = [203782, 203783, 203784]
T_RANGE = [1.5, 3.5]

def load_lp_data(shot, path_to_folder):
    path_to_folder = Path(path_to_folder)
    with h5py.File( path_to_folder / f'{shot}_LP.h5', 'r') as h5:
        dimes_gp = h5['/LANGMUIR_DIMES']
        t_s = np.array(dimes_gp.get('time')) * 1E-3
        qpara = np.array(dimes_gp.get('qpara'))
    return t_s, qpara

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

def find_rvsout_index_intersections(x, y, y_ref, y_tolerance=1E-6, x_eps=None):
    if x_eps is None:
        x_spacing = np.mean(np.diff(x))
        x_eps = 2 * x_spacing

    idx_peaks = np.argwhere(np.abs(y - y_ref) < y_tolerance)

    if idx_peaks.size == 0:
        return []

    x_peaks = x[idx_peaks]

    # Group consecutive nearby points
    averaged_points = []
    current_group = [x_peaks[0]]

    # print(f'x_eps: {x_eps}')

    # print(f'i: 0, averaged_points {averaged_points}, current group {current_group}')

    for i in range(1, len(x_peaks)):
        if (x_peaks[i] - x_peaks[i - 1]) <= x_eps:
            # print(f'i: {i}, x_peaks[i] {x_peaks[i]}, x_peaks[i-1] {x_peaks[i-1]}, diff {x_peaks[i] - x_peaks[i - 1]}, current group {current_group}')
            current_group.append(x_peaks[i])
        else:
            # Finish current group and start new one
            averaged_points.append(np.mean(current_group))
            current_group = [x_peaks[i]]
            # print(f'i: {i}, averaged_points {averaged_points}, current group {current_group}')
    # Don't forget the last group
    averaged_points.append(np.mean(current_group))
    averaged_points = np.array(averaged_points)

    # print(averaged_points)

    idx_average =[np.argmax(np.abs(x - xi) <= x_eps) for xi in averaged_points]
    x_average = x[idx_average]

    return idx_average, x_average


# A data type to store the times at which the OSP hits DiMES for each shot
osp_dtype = np.dtype([('shot', 'i'), ('t_osp_1', 'd'), ('t_osp_2', 'd')])
# A data type to store qpara at OSP strikes on DiMES for each shot
qpara_dtype = np.dtype([('shot', 'i'), ('qpara', 'd'), ('d_qpara', 'd'),])

def compile_d3d_qpara(
    shots, path_to_lp_data=PATH_TO_LP_DATA, t_range:List[float]=T_RANGE,
) -> Tuple[qpara_dtype, osp_dtype]:

    n = len(shots)

    osp_hit_times = np.empty(0, dtype=osp_dtype)
    qpara_shots = np.empty(0, dtype=qpara_dtype)

    # Plot the data
    fig, axes = plt.subplots(nrows=n, ncols=1, constrained_layout=True, sharex=True)
    fig.set_size_inches(4., 6)

    for i, shot in enumerate(shots):
        print(f'Loading qpara data for shot {shot}')
        # Load qpara from LP data
        t_s, qpara = load_lp_data(shot, path_to_lp_data)
        # Only plot within the desired range
        msk_time = (t_range[0] <= t_s) & (t_s <= t_range[1])
        t_s = t_s[msk_time]
        qpara = qpara[msk_time]
        # Despike and mooth
        qpara_despiked, _ = remove_spikes_zscore(qpara, threshold=1.5, window_size=20)
        spl_qpara = make_smoothing_spline(t_s, qpara_despiked, lam=1E-6)

        # Try to estimate OSP hits on DiMES using rvsout
        # load rvsout
        r_dimes = 1.48 # R at DiMES (m)
        rvsout_df = pd.read_csv(
            fr'./data/dro1/{shot}_voltage_and_rvsout.csv', comment='#'
        ).apply(pd.to_numeric, errors='coerce')
        t_dro1 = rvsout_df['t_ms'].values
        rvsout = rvsout_df['rvsout'].values
        # Just analyze within the range defined in the header
        msk_time = (t_range[0] <= t_dro1*1E-3) & (t_dro1*1E-3 <= t_range[1])
        t_dro1 = t_dro1[msk_time]
        rvsout = rvsout[msk_time]
        # Find the time points at which rsvout intercepts dimes
        _, t_osp_hits = find_rvsout_index_intersections(x=t_dro1, y=rvsout, y_ref=r_dimes, y_tolerance=1E-5)

        # Take the min and the max as the strike points
        osp_range = [t_osp_hits.min() * 1E-3, t_osp_hits.max() * 1E-3]
        osp_hit_times = np.append(osp_hit_times, np.array([(shot, osp_range[0], osp_range[1])], dtype=osp_dtype))

        # Plot qpara for each shot
        axes[i].plot(t_s, qpara, label='Data', alpha=0.25, color='C0')
        # plot smoothened qpara
        axes[i].plot(t_s, spl_qpara(t_s), label=f'Smooth', color='C2')

        axes[i].set_title(f'#{shot}')
        axes[i].set_ylim(bottom=0, top=120)
        axes[i].yaxis.set_major_locator(ticker.MultipleLocator(40))
        axes[i].yaxis.set_minor_locator(ticker.MultipleLocator(10))
        axes[i].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        axes[i].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        if shot != 203784:
            # Shot 203784 has a dwell of 1 s, treat differently
            # Find the peaks of qpara
            idx_peaks, _ = find_peaks(spl_qpara(t_s), prominence=10, distance=500)
            # If there are several peaks restrict to times around 2.1 and 3.1 to get only forward and back sweep strikes
            idx_peaks = np.array([idx_peak for idx_peak in idx_peaks if (2.1 <= t_s[idx_peak]) & (t_s[idx_peak] <= 3.1)])
            # Plot the peaks
            axes[i].plot(t_s[idx_peaks], spl_qpara(t_s[idx_peaks]), ls='none', marker='x', color='r', label='OSP hits DiMES')
            # plot the OSP hits from rvsout
            for t_osp_hit in osp_range:
                axes[i].axvline(x=t_osp_hit, color='tab:red', ls='dashed', lw=1.)

            # Append qpara_shots with the peaks
            for idx_peak in idx_peaks:
                row = np.array([(
                    shot,
                    spl_qpara(t_s[idx_peak]),
                    0.01 * spl_qpara(t_s[idx_peak]) # based on the estimate from shot 203784 which has many points
                )], dtype=qpara_dtype)
                qpara_shots = np.append(qpara_shots, row)
        else:
            # For shot 203784, the OSP dwells for 1 s at DiMES. Average the points
            axes[i].axvspan(xmin=osp_range[0], xmax=osp_range[1], alpha=0.25, color='grey', label='OSP hits DiMES')

            msk_osp_t_dwell = (osp_range[0] <= t_s) & (t_s <= osp_range[1])
            n_t_dwell = np.sum(msk_osp_t_dwell)
            # Get the mean qpara for the dwell time
            qpara_dimes = np.mean(spl_qpara(t_s[msk_osp_t_dwell]))
            # Get the std
            qpara_dimes_std = np.std(spl_qpara(t_s[msk_osp_t_dwell]), ddof=1)
            # Find the standard error
            confidence_level = 0.95
            alpha = 1 - confidence_level
            tval = t.ppf(1 - 0.5 * alpha, n_t_dwell - 1)
            qpara_dimes_se = qpara_dimes_std * tval / np.sqrt(n_t_dwell)

            row = np.array([(
                shot,
                qpara_dimes,
                qpara_dimes_se # report the standard error
            )], dtype=qpara_dtype)
            qpara_shots = np.append(qpara_shots, row)
        axes[i].legend(loc='upper left', fontsize=9, ncol=3)

    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(t_range[0], t_range[1])
    fig.supylabel(r'$Q_{\parallel}$ (MW/m\textsuperscript{2})', usetex=True)
    path_to_figures = Path(r'./figures')
    path_to_figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path_to_figures / 'd3d_compiled_qpara.png'), dpi=600)
    plt.show()
    return qpara_shots, osp_hit_times

evaporation_dtype = np.dtype([
    ('shot', 'd'), ('qpara', 'd'), ('d_qpara', 'd'), ('rate_evaporation', 'd'), ('rate_evaopration_lb', 'd'), ('rate_evaopration_ub', 'd')
])
def compile_evaporation_rates(qpara_shots, osp_hit_times, t_range=[1.5, 3]):
    shots = np.unique(qpara_shots['shot'])
    n = len(shots)
    evaporation_rates = np.zeros(0, dtype=evaporation_dtype)
    # Plot the data
    fig, axes = plt.subplots(nrows=n, ncols=1, constrained_layout=True, sharex=True)
    fig.set_size_inches(4., 6)
    for i, shot in enumerate(shots):
        data = load_evaporation_rates(shot)
        t_s = data['time (s)']
        rate_evaporation = data['evaporation_rate (atoms/s)']
        rate_evaporation_lb = data['evaporation_rate_lb (atoms/s)']
        rate_evaporation_ub = data['evaporation_rate_ub (atoms/s)']

        spl_rate_evaporation = make_smoothing_spline(t_s, rate_evaporation*1E-15, lam=1E-6)
        spl_rate_evaporation_lb = make_smoothing_spline(t_s, rate_evaporation_lb*1E-15, lam=1E-6)
        spl_rate_evaporation_ub = make_smoothing_spline(t_s, rate_evaporation_ub*1E-15, lam=1E-6)
        axes[i].plot(t_s, spl_rate_evaporation(t_s), color='C0')
        osp_hit_shot = osp_hit_times[osp_hit_times['shot'] == shot]
        if shot != 203784:
            # Find the peaks of qpara
            idx_peaks, _ = find_peaks(spl_rate_evaporation(t_s), prominence=0.1, distance=500)
            # If there are several peaks restrict to times around 2.1 and 3.1 to get only forward and back sweep strikes
            idx_peaks = np.array(
                [idx_peak for idx_peak in idx_peaks if (2.1 <= t_s[idx_peak]) & (t_s[idx_peak] <= 3.1)])
            # Plot the peaks
            axes[i].plot(t_s[idx_peaks], spl_rate_evaporation(t_s[idx_peaks]), ls='none', marker='x', color='r',
                         label='OSP hits DiMES')

            qpara_shot = qpara_shots[qpara_shots['shot'] == shot]

            for t_hit in [osp_hit_shot['t_osp_1'].squeeze(), osp_hit_shot['t_osp_2'].squeeze()]:
                axes[i].axvline(t_hit, color='tab:red', ls='dashed', lw=1.)

            # Append evaporation_rates with the peaks
            for j, idx_peak in enumerate(idx_peaks):
                row = np.array([(
                    shot,
                    qpara_shot['qpara'][j],
                    qpara_shot['d_qpara'][j],
                    spl_rate_evaporation(t_s[idx_peak]),
                    spl_rate_evaporation_lb(t_s[idx_peak]),
                    spl_rate_evaporation_ub(t_s[idx_peak]),
                )], dtype=evaporation_dtype)
                evaporation_rates = np.append(evaporation_rates, row)

        else:
            axes[i].axvspan(xmin=osp_hit_shot['t_osp_1'].squeeze(), xmax=osp_hit_shot['t_osp_2'].squeeze(), alpha=0.25, color='grey', label='OSP hits DiMES')
            msk_osp_time = (osp_hit_shot['t_osp_1'].squeeze() <= t_s) & (t_s <= osp_hit_shot['t_osp_2'].squeeze())
            n_osp_time = np.sum(msk_osp_time)
            evaporation_mean = np.mean(spl_rate_evaporation(t_s[msk_osp_time]))
            evaporation_std = np.std(spl_rate_evaporation(t_s[msk_osp_time]), ddof=1)
            confidence_level = 0.95
            alpha = 1 - confidence_level
            t_val = t.ppf(1 - alpha/2, n_osp_time-1)
            evaporation_delta = evaporation_std * t_val #/ np.sqrt(n_osp_time)

            row = np.array([(
                shot,
                qpara_shot['qpara'][0],
                qpara_shot['d_qpara'][0],
                evaporation_mean,
                evaporation_mean - evaporation_delta,
                evaporation_mean + evaporation_delta
            )], dtype=evaporation_dtype)
            evaporation_rates = np.append(evaporation_rates, row)

        axes[i].set_title(f'#{shot}')
        axes[i].legend(loc='upper left', fontsize=9, ncol=3)
        axes[i].set_ylim(bottom=0, top=10)
        axes[i].yaxis.set_major_locator(ticker.MultipleLocator(5))
        axes[i].yaxis.set_minor_locator(ticker.MultipleLocator(1))
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(t_range[0], t_range[1])
    fig.supylabel(r'Evaporation Rate ($\times10^{15}$ atoms/s)', usetex=True)
    path_to_figures = Path(r'./figures')
    path_to_figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path_to_figures / 'd3d_compiled_evaporation_rates.png'), dpi=600)
    plt.show()

    return evaporation_rates


def load_evaporation_rates(shot, path_to_data=PATH_TO_D3D_EVAPORATION_RATES):
    path_to_data = Path(path_to_data)
    path_to_csv = path_to_data / f'{shot}_evaporation_rate.csv'
    df = pd.read_csv(path_to_csv, comment='#').apply(pd.to_numeric, errors='coerce')
    time = df['Time (s)'].values
    evaporation_rate = df['Evaporation rate (atoms/s)'].values  #
    evaporation_rate_lb = df['Evaporation rate lb (atoms/s)'].values
    evaporation_rate_ub = df['Evaporation rate ub (atoms/s)'].values

    data = {
        'time (s)': time,
        'evaporation_rate (atoms/s)': evaporation_rate,
        'evaporation_rate_lb (atoms/s)': evaporation_rate_lb,
        'evaporation_rate_ub (atoms/s)': evaporation_rate_ub,
    }

    return data

dmdt_dtype = np.dtype([('shot', 'd'), ('qpara', 'd'),('d_qpara', 'd'), ('dmdt', 'd'), ('dmdt_error', 'd')])
def compile_mass_loss_rates(qpara_shots):
    shots = np.unique(qpara_shots['shot'])
    n = len(shots)
    dmdt_shots = np.zeros(0, dtype=dmdt_dtype)

    for i, shot in enumerate(shots):
        qpara_ds, dmdt, dmdt_error = load_mass_loss_data(shot)
        qpara_shot = qpara_shots[qpara_shots['shot'] == shot]
        for j in range(len(qpara_shot)):
            # print(f'{shot}, qpara_shot[{j}]: {qpara_shot["qpara"][j]}, qpara_ds[{j}]: {qpara_ds[j]}')
            row = np.array([(shot, qpara_shot["qpara"][j], qpara_shot["d_qpara"][j], dmdt[j], dmdt_error[j])], dtype=dmdt_dtype)
            dmdt_shots = np.append(dmdt_shots, row, axis=0)
    return dmdt_shots


def load_mass_loss_data(shot, path_to_data=PATH_TO_D3D_MASS_LOSS_RATES):
    path_to_data = Path(path_to_data)
    path_to_h5 = path_to_data / f'{shot}_dmdt_vs_qpara.h5'
    with h5py.File(path_to_h5, 'r') as hf:
        qpara = np.array(hf.get(f'{shot}/osp_on_dimes/qpara'))
        dmdt = np.array(hf.get(f'{shot}/osp_on_dimes/dmdt'))
        dmdt_error = np.array(hf.get(f'{shot}/osp_on_dimes/dmdt_delta'))

    return qpara, dmdt, dmdt_error

def main(
    shots:List[int]=SHOTS, path_to_laser_evaporation=PATH_TO_LASER_EVAPORATION,
    path_to_laser_mass_loss_rates=PATH_TO_LASER_MASS_LOSS_RATES,
    path_to_d3d_mass_loss_rates=PATH_TO_D3D_MASS_LOSS_RATES,
    path_to_d3d_evaporation_rates=PATH_TO_D3D_EVAPORATION_RATES
):
    load_plot_style(font='Times New Roman')
    qpara_shots, osp_hit_times = compile_d3d_qpara(shots)
    evporation_rates_diid = compile_evaporation_rates(qpara_shots, osp_hit_times)
    evaporation_rates_laser_df = pd.read_csv(path_to_laser_mass_loss_rates, comment='#').apply(pd.to_numeric, errors='coerce')

    dmdt_shots = compile_mass_loss_rates(qpara_shots)

    # load laser heating mass loss rates
    laser_df = pd.read_csv(path_to_laser_mass_loss_rates, comment='#').apply(pd.to_numeric, errors='coerce')


    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 4)

    markers = ['o', '^', 'v', '<', '>']
    unique_shots = np.unique(qpara_shots['shot'])
    # unique_shots = unique_shots[1:]

    handles = []
    for i, shot in enumerate(unique_shots):
        dmdt_shot = dmdt_shots[dmdt_shots['shot'] == shot]
        label = 'Pebbles/dust (DIII-D)'

        color = 'C0'
        # print(dmdt_shot['shot'])
        if np.any(dmdt_shot['shot'] == 203780) or np.any(dmdt_shot['shot'] == 203781):
            color = 'tab:red'
        eb = ax.errorbar(
            dmdt_shot['qpara'], dmdt_shot['dmdt'] * 1E-20, yerr=dmdt_shot['dmdt_error'] * 1E-20,
            xerr=dmdt_shot['d_qpara']*5,
            marker='o', ls='none', label=label,
            ms=9, mew=1.25, mfc='none',
            capsize=2.75, elinewidth=1.25, lw=1.5, c=color
        )

        if i == 2:
            handles.append(eb)

        shot_txt = f'  {shot}'
        for j, dmdt_shot_j in enumerate(dmdt_shot):
            # print(f'{dmdt_shot_j}')
            ax.text(
                dmdt_shot_j['qpara'], dmdt_shot_j['dmdt']*1E-20,
                shot_txt,
                ha='left', va='bottom',
                color='k',
                fontsize=9,
                fontweight='bold',
                font='Arial'
            )

        markers_p, caps_p, bars_p = eb
        [bar.set_alpha(0.35) for bar in bars_p]
        [cap.set_alpha(0.35) for cap in caps_p]

    eb = ax.errorbar(
        laser_df['heat_load_mean (MW/m2)'].values,
        laser_df['mass_loss_rate (atoms/s)'].values * 1E-20,
        yerr=(laser_df['mass_loss_rate_lb (atoms/s)'].values * 1E-20, laser_df['mass_loss_rate_ub (atoms/s)'].values * 1E-20),
        xerr=1,
        marker='s', ls='none', label='Pebble+dust (laser tests)',
        ms=9, mew=1.25, mfc='none',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='C1'
    )

    handles.append(eb)

    markers_p, caps_p, bars_p = eb
    [bar.set_alpha(0.35) for bar in bars_p]
    [cap.set_alpha(0.35) for cap in caps_p]

    ax.set_xlabel(r'$q$ (MW/m\textsuperscript{2})', usetex=True)
    ax.set_ylabel(r'Total boron emission ($\times10^{20}$ atoms/s)', usetex=True)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 100)
    # ax.set_yscale('log')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))


    ax.legend(loc='upper left', fontsize=10, handles=handles)

    path_to_figures = Path(r'./figures')
    path_to_figures.mkdir(parents=True, exist_ok=True)

    for ext in ['png', 'pdf', 'svg']:
        path_to_figure = path_to_figures / f'boron_emission-laser_vs_diii-d.{ext}'
        fig.savefig(path_to_figure, bbox_inches='tight', dpi=600)

    plt.show()


if __name__ == '__main__':
    main()