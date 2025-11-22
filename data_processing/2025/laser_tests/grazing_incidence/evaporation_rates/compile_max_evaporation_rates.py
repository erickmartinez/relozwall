import h5py
import numpy as np
import matplotlib.pyplot as plt
from data_processing.misc_utils.plot_style import load_plot_style
from data_processing.laser_power_mapping import map_laser_power_output
from data_processing.utils import get_experiment_params

import pandas as pd
from pathlib import Path
import os
import re

PATH_TO_EVAPORATION_RATES = r'./data'
PATH_TO_LASER_TESTS = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests'
BEAM_RADIUS = 0.5 * 0.8165  # * 1.5 # 0.707


def get_mean_laser_power(experiment_id, path_to_laser_tests=PATH_TO_LASER_TESTS):
    path_to_matches = find_experiment_files(search_string=experiment_id, search_path=path_to_laser_tests)
    if len(path_to_matches) == 0:
        raise ValueError(f'No laser tests found for experiment {experiment_id}')
    path_to_experiment_file = Path(path_to_matches[0])
    data_df = pd.read_csv(str(path_to_experiment_file), comment='#').apply(pd.to_numeric, errors='coerce')
    power_w = data_df['Laser output peak power (W)'].values
    power_max = np.max(power_w)
    msk_on = power_w >= 0.8 * power_max
    mean_laser_power = power_w[msk_on].mean()
    return mean_laser_power

def find_experiment_files(search_string, search_path):
    """
    Recursively search for CSV files that contain a specific string in their filename.

    Args:
        search_string (str): The string to search for in filenames
        search_path (str): The root directory path to start the search

    Returns:
        list: A list of full paths to matching CSV files
    """
    matching_files = []

    # Walk through the directory tree
    for root, dirs, files in os.walk(search_path):
        for file in files:
            # Check if file ends with .csv and contains the search string
            if file.endswith('.csv') and search_string in file:
                # Get the full path and add to results
                full_path = os.path.join(root, file)
                matching_files.append(full_path)

    return matching_files

def load_evaporation_rate_h5(path_to_h5):
    path_to_h5 = Path(path_to_h5)
    pattern = re.compile(r'.*(R\d+N\d+\D\d+).h5')
    match = pattern.match(str(path_to_h5))
    if match:
        experiment_id = match.group(1)
    else:
        raise ValueError(f'Could not find experiment ID for {path_to_h5.name}')
    with h5py.File(str(path_to_h5), 'r') as h5:
        evaporation_rate_ds = h5['evaporation_rate']
        evaporation_rate_max = float(evaporation_rate_ds.attrs['evaporation_rate_max'])
        evaporation_rate_max_lb = float(evaporation_rate_ds.attrs['evaporation_rate_max_lb'])
        evaporation_rate_max_ub = float(evaporation_rate_ds.attrs['evaporation_rate_max_ub'])
        pebble_rod_area_cm2 = float(evaporation_rate_ds.attrs['pebble_rod_area_cm2'])
        pebble_rod_diameter_cm = float(evaporation_rate_ds.attrs['pebble_rod_diameter_cm'])

    data = {
        'experiment_id': experiment_id,
        'evaporation_rate_max': evaporation_rate_max,
        'evaporation_rate_max_lb': evaporation_rate_max_lb,
        'evaporation_rate_max_ub': evaporation_rate_max_ub,
        'pebble_rod_area_cm2': pebble_rod_area_cm2,
        'pebble_rod_diameter_cm': pebble_rod_diameter_cm
    }
    return data

def gaussian_beam_aperture_factor(beam_radius: float, sample_radius:float) -> float:
    """
    Estimates the factor by which the output power of a laser gaussian beam is reduced when passed through an aperture
    of radius r

    Parameters
    ----------
    beam_radius: float
        The radius of the beam
    sample_radius: float
        The radius of the aperture

    Returns
    -------
    float:
        The estimated factor
    """
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)

def main(path_to_evaporation_rates=PATH_TO_EVAPORATION_RATES, beam_radius=BEAM_RADIUS, path_to_laser_tests=PATH_TO_LASER_TESTS):
    path_to_evaporation_rates = Path(path_to_evaporation_rates)
    files_list = [fn for fn in os.listdir(path_to_evaporation_rates) if fn.endswith('.h5')]
    n_files = len(files_list)
    evaporation_rate = np.zeros(n_files, dtype=np.dtype([
        ('Laser power (W)', 'd'), ('Heat load (MW/m^2)', 'd'), ('Evaporation rate max (atoms/s)', 'd'),
        ('Evaporation rate max lb (atoms/s)', 'd'), ('Evaporation rate max ub (atoms/s)', 'd'),
        ('Pebble rod diameter (cm)', 'd'),
    ]))
    for i, file in enumerate(files_list):
        path_to_file = path_to_evaporation_rates / file
        data = load_evaporation_rate_h5(str(path_to_file))
        laser_power = get_mean_laser_power(data['experiment_id'], str(path_to_laser_tests))
        aperture_factor = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.9*0.5)
        heat_load = laser_power * aperture_factor / data['pebble_rod_area_cm2'] * 1E-2
        evaporation_rate['Laser power (W)'][i] = laser_power
        evaporation_rate['Heat load (MW/m^2)'][i] = heat_load
        evaporation_rate['Evaporation rate max (atoms/s)'][i] = data['evaporation_rate_max']
        evaporation_rate['Evaporation rate max lb (atoms/s)'][i] = data['evaporation_rate_max_lb']
        evaporation_rate['Evaporation rate max ub (atoms/s)'][i] = data['evaporation_rate_max_ub']
        evaporation_rate['Pebble rod diameter (cm)'][i] = data['pebble_rod_diameter_cm']

    results_df = pd.DataFrame(evaporation_rate).sort_values('Heat load (MW/m^2)', ascending=True)
    path_to_output = path_to_evaporation_rates / 'compiled'
    path_to_output.mkdir(parents=True, exist_ok=True)
    path_to_csv = path_to_output / 'evaporation_rates.csv'
    with open(str(path_to_csv), 'w') as csvfile:
        csvfile.write('#'*40 + '\n')
        csvfile.write('# '+f'Laser beam diameter: {beam_radius*2.:.3f} cm\n')
        csvfile.write('#'*40 + '\n')
        results_df.to_csv(csvfile, index=False)

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4, 3)

    markers_p, caps_p, bars_p = ax.errorbar(
        evaporation_rate['Heat load (MW/m^2)'], evaporation_rate['Evaporation rate max (atoms/s)']*1E-16,
        yerr=(evaporation_rate['Evaporation rate max lb (atoms/s)']*1E-16, evaporation_rate['Evaporation rate max ub (atoms/s)']*1E-16),
        marker='o', ms=9, mew=1.25, mfc='none',  # label=f'{lbl}',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='C0', ls='none'
    )

    [bar.set_alpha(0.35) for bar in bars_p]
    [cap.set_alpha(0.35) for cap in caps_p]

    ax.set_xlabel(r'{\sffamily Heat load (MW/m\textsuperscript{2})', usetex=True)
    ax.set_ylabel(r'{\sffamily Evaporation rate (x10\textsuperscript{16} atoms/s)', usetex=True)

    path_to_figures = Path(r'./figures')
    fig.savefig(path_to_figures / 'evaporation_rates_compiled.png', dpi=600)
    plt.show()

if __name__ == '__main__':
    main(path_to_evaporation_rates=PATH_TO_EVAPORATION_RATES, beam_radius=BEAM_RADIUS)

