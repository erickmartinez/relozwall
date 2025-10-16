import h5py
import numpy as np
import matplotlib.pyplot as plt
from data_processing.misc_utils.plot_style import load_plot_style
import matplotlib.ticker as ticker
from pathlib import Path
import json
from scipy.interpolate import PPoly
from scipy.integrate import simpson
from typing import Tuple, Union
import re
from scipy.stats.distributions import t
import os

PATH_TO_PPPL_FIT = '../../PPPL_boron_evaporation_rates/boron_evaporation_model.hdf5'
# PATH_TO_HISTOGRAMS = './thermography/data/LCT_R5N16-0903_040PCT_2025-09-11_1_histogram.h5'
PATH_TO_HISTOGRAMS = './thermography/data/LCT_R5N16-0905_080PCT_2025-09-11_1_histogram.h5'

PEBBLE_ROD_DIAMETER = 0.95 # cm
"""
The following data is used to determine the size of the pixels
"""
REFERENCE_ROD_DIAMETER = 1.27 # cm
MEASURED_ELLIPSE_RADII = [85.37, 155.85] # minor and major radius in pixels The major radius should be vertical

BORON_MOLAR_MASS = 10.811
BORON_DENSITY = 2.35 # g/cm^3


def load_times_from_json(json_file):
    with open(json_file) as json_file:
        data = json.load(json_file)
        time_s = np.array(data['t (s)'], dtype=float)
    return time_s

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

def load_model(path_to_pppl_fit) -> callable:
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

def boron_atoms_to_grams_of_boron(value, boron_molar_mass=BORON_MOLAR_MASS) -> Union[np.ndarray, float]:
    boron_molar_mass_g = 1.6605390671738466e-24 * boron_molar_mass
    mass_of_boron = value * boron_molar_mass_g
    return mass_of_boron

def main(
    path_to_pppl_fit, path_to_histograms, pebble_rod_diameter, reference_rod_diameter=REFERENCE_ROD_DIAMETER,
    measured_ellipse_radii=MEASURED_ELLIPSE_RADII
):
    # Load the histrogram data with the number of pixels at each temperature
    path_to_histograms = Path(path_to_histograms)
    file_name = path_to_histograms.name
    """
    LCT_R5N15_0602_020PCT_2025-06-03_2_histogram
    """
    pattern = re.compile(r'LCT\_(R\d+N\d+\D\d+)\_(\d+)PCT.*')
    match = pattern.match(file_name)
    if match:
        experiment_id = match.group(1)
        laser_power_setting = int(match.group(2))

    with h5py.File(str(path_to_histograms), "r") as hf:
        time_s = np.array(hf.get('time'))
        histogram_matrix = np.array(hf.get('histogram'))
        bin_centers = np.array(hf.get('bin_centers'))

    path_to_pppl_fit = Path(path_to_pppl_fit)
    evaporation_rate_model = load_model(path_to_pppl_fit)

    evaporation_rate = np.zeros_like(time_s)
    evaporation_lb = np.zeros_like(time_s)
    evaporation_ub = np.zeros_like(time_s)
    pebble_rod_area = 0.25 * np.pi * pebble_rod_diameter ** 2 # cm^2


    for i, ti in enumerate(time_s):
        histogram = histogram_matrix[i, 1:] # < remove data close to 300 K
        n_pixels_sum = np.sum(histogram)
        histogram /= n_pixels_sum # <- convert to pdf
        evaporation_rates, lb, ub = evaporation_rate_model(bin_centers[1:])
        evaporation_rate[i] = np.sum(evaporation_rates* histogram)
        evaporation_lb[i] = np.sum(lb * histogram)
        evaporation_ub[i] = np.sum(ub * histogram)

    evaporation_rate *= pebble_rod_area
    evaporation_rate *= 1E-16

    evaporation_lb *= pebble_rod_area
    evaporation_ub *= pebble_rod_area
    evaporation_lb *= 1E-16
    evaporation_ub *= 1E-16


    total_evaporation = simpson(y=evaporation_rate*1E16, x=time_s)
    total_evaporation_lb = simpson(y=(evaporation_lb)*1E16, x=time_s)
    total_evaporation_ub = simpson(y=(evaporation_ub)*1E16, x=time_s)
    mean_evaporation_rate = np.mean(evaporation_rate)
    std_evaporation_rate = np.std(evaporation_rate, ddof=1)
    mean_evaporation_delta = np.linalg.norm(evaporation_ub - evaporation_rate) / len(time_s)


    confidence_level = 0.91
    alpha = 1 - confidence_level
    t_val = t.ppf(1 - 0.5*alpha, len(time_s)-1)
    evaporation_rate_se = std_evaporation_rate * t_val / np.sqrt(len(time_s))
    evaporation_rate_uncertainty = np.linalg.norm([mean_evaporation_delta, evaporation_rate_se])

    evaporation_rate_max = np.max(evaporation_rate)
    idx_max = np.argmin(np.abs(evaporation_rate - evaporation_rate_max))
    t_max = time_s[idx_max]
    evaporation_rate_max_ub = evaporation_ub[idx_max] - evaporation_rate_max
    evaporation_rate_max_lb = evaporation_rate_max - evaporation_lb[idx_max]

    total_evaporation_g = boron_atoms_to_grams_of_boron(total_evaporation)*1E6
    total_evaporation_lb_g = boron_atoms_to_grams_of_boron(total_evaporation_lb)*1E6
    total_evaporation_ub_g = boron_atoms_to_grams_of_boron(total_evaporation_ub)*1E6

    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3.)
    ax.plot(time_s, evaporation_rate)

    ax.fill_between(time_s, evaporation_lb, evaporation_ub, alpha=0.25, color='C0', ec='None')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'{\sffamily Evaporation rate (x10\textsuperscript{16} atoms/s)}', usetex=True)
    evaporation_rate_txt = (f'Total evaporation: {total_evaporation_g:.3f} '
                            f'[{total_evaporation_lb_g:.3f},{total_evaporation_ub_g:.3f}] µg\n'
                            f'Mean rate: ({mean_evaporation_rate:.1f} ± {evaporation_rate_uncertainty:.0f})'
                            r'x10$^{\mathregular{16}}$ atoms/s}' + '\n'
                            f'Max rate: ({evaporation_rate_max:.1f} [{evaporation_rate_max_lb:.1f}, {evaporation_rate_max_ub:.1f}] '
                            r'x10$^{\mathregular{16}}$ atoms/s}')
    ax.text(
        0.025, 0.95, evaporation_rate_txt,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=10,
        usetex=False
    )

    ax.set_title(f'{experiment_id}')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    output_file_tag = f'evaporation_rate_{experiment_id}'
    path_to_figures = Path('./evaporation_rates/figures')
    path_to_figures.mkdir(parents=True, exist_ok=True)

    fig.savefig(f'{path_to_figures}/{output_file_tag}.png', dpi=600)

    path_to_output_data = Path('./evaporation_rates/data')
    path_to_output_data.mkdir(parents=True, exist_ok=True)

    path_to_output_h5 = path_to_output_data / f'{output_file_tag}.h5'
    with h5py.File(path_to_output_h5, 'w') as h5file:
        time_ds = h5file.create_dataset('time', data=time_s)
        time_ds.attrs['units'] = 's'
        evaporation_rate_ds = h5file.create_dataset('evaporation_rate', data=evaporation_rate*1E14)
        evaporation_rate_ds.attrs['units'] = 'atoms/s'
        evaporation_rate_ds.attrs['pebble_rod_diameter_cm'] = pebble_rod_diameter
        evaporation_rate_ds.attrs['pebble_rod_area_cm2'] = pebble_rod_area
        evaporation_rate_ds.attrs['evaporation_rate_max'] = evaporation_rate_max*1E16
        evaporation_rate_ds.attrs['evaporation_rate_max_lb'] = evaporation_rate_max_lb*1E16
        evaporation_rate_ds.attrs['evaporation_rate_max_ub'] = evaporation_rate_max_ub*1E16
        evaporation_rate_ds.attrs['total_evaporation_g'] = total_evaporation_g*1E-6
        evaporation_rate_ds.attrs['total_evaporation_lb'] = total_evaporation_lb*1E-6
        evaporation_rate_ds.attrs['total_evaporation_ub'] = total_evaporation_ub*1E-6
        evaporation_lb_ds = h5file.create_dataset('evaporation_lb', data=evaporation_lb*1E16)
        evaporation_ub_ds = h5file.create_dataset('evaporation_ub', data=evaporation_ub*1E16)



    plt.show()

if __name__ == '__main__':
    list_files = [fn for fn in os.listdir(str(Path(PATH_TO_HISTOGRAMS).parent)) if fn.endswith('.h5')]
    # main(
    #     path_to_pppl_fit=PATH_TO_PPPL_FIT, path_to_histograms=PATH_TO_HISTOGRAMS, pebble_rod_diameter=PEBBLE_ROD_DIAMETER,
    #     reference_rod_diameter=REFERENCE_ROD_DIAMETER, measured_ellipse_radii=MEASURED_ELLIPSE_RADII
    # )
    for fn in list_files:
        path_to_h5 = Path(PATH_TO_HISTOGRAMS).parent / fn
        main(
            path_to_pppl_fit=PATH_TO_PPPL_FIT, path_to_histograms=path_to_h5,
            pebble_rod_diameter=PEBBLE_ROD_DIAMETER,
            reference_rod_diameter=REFERENCE_ROD_DIAMETER, measured_ellipse_radii=MEASURED_ELLIPSE_RADII
        )






