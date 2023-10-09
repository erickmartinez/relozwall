import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from data_processing.utils import get_experiment_params

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN\20230928'
database_csv = 'Transmission measurements 20230928.csv'

csv_soot_deposition = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests' \
                      r'\surface_temperature\equilibrium_redone\slide_transmission_smausz.csv'

laser_power_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN\laser_output'

samples = [
    {'sample_id': 'GT001688', 'label': 'Graphite rod', 'material': 'Isomolded graphite', 'marker': 'o'},
    {'sample_id': 'GT001688', 'label': 'Graphite rod', 'material': 'Isomolded graphite', 'marker': 'o'},
    {'sample_id': 'GT001688', 'label': 'Graphite rod', 'material': 'Isomolded graphite', 'marker': 'o'},
    {'sample_id': 'GT001688', 'label': 'Graphite rod', 'material': 'Isomolded graphite', 'marker': 'o'},
    {'sample_id': 'GT001688', 'label': 'Graphite rod', 'material': 'Isomolded graphite', 'marker': 'o'},
]

beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707
n_cos, dn_cos = 7.4, 1.9
h_0 = 10.5 * 2.54

graphite_sample_diameter = 0.92


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)


def map_laser_power_settings():
    rdir = os.path.join(base_dir, laser_power_dir)
    file_list = os.listdir(rdir)
    mapping = {}
    for i, f in enumerate(file_list):
        if f.endswith('.csv'):
            params = get_experiment_params(relative_path=rdir, filename=os.path.splitext(f)[0])
            laser_setpoint = int(params['Laser power setpoint']['value'])
            df = pd.read_csv(os.path.join(rdir, f), comment='#').apply(pd.to_numeric)
            laser_power = df['Laser output peak power (W)'].values
            laser_power = laser_power[laser_power > 0.0]
            mapping[laser_setpoint] = laser_power.mean()

    keys = list(mapping.keys())
    keys.sort()
    return {i: mapping[i] for i in keys}


def main():
    df = pd.read_csv(
        os.path.join(base_dir, database_csv),
        # usecols=['Sample ID', 'Laser power setting (%)', 'Film thickness (nm)', 'Deposition rate (nm/s)']
    )
    numeric_columns = ['Laser power setting (%)', 'Film thickness (nm)', 'Deposition rate (nm/s)']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

    sample_id_list = [sid['sample_id'] for sid in samples]
    df = df[df['Sample ID'].isin(sample_id_list)]

    print(df)

    laser_power_mapping = map_laser_power_settings()
    power_settings = np.array([int(k) for k in laser_power_mapping.keys()])

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, axes = plt.subplots(ncols=1, nrows=2, constrained_layout=True)
    fig.set_size_inches(4.0, 5.0)

    deposition_rate = df['Deposition rate (nm/s)'].values
    deposition_rate_lb = df['Deposition rate lb (nm/s)'].values
    deposition_rate_ub = df['Deposition rate ub (nm/s)'].values

    yerr_deposition = (deposition_rate - deposition_rate_lb, deposition_rate_ub - deposition_rate)
    laser_power_setting = df['Laser power setting (%)'].values[0]
    laser_power = laser_power_mapping[laser_power_setting]
    sample_diameter = 0.969
    sample_area = 0.25 * np.pi * (sample_diameter ** 2.0)
    aperture_factor = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.5 * sample_diameter)
    incident_heat_load = aperture_factor * laser_power / sample_area / 100.0

    evaporation_rate_lb = df['Evaporation rate lb (Torr-L/s)'].values * 1E4 / sample_area
    evaporation_rate = df['Evaporation rate (Torr-L/s)'].values * 1E4 / sample_area
    evaporation_rate_ub = df['Evaporation rate ub (Torr-L/s)'].values * 1E4 / sample_area
    yerr_evaporation = (evaporation_rate - evaporation_rate_lb, evaporation_rate_ub - evaporation_rate_lb)
    x = np.array([i+1 for i in range(len(samples))])
    axes[0].errorbar(
        x, deposition_rate, yerr=yerr_deposition,
        marker='o', ms=9, mew=1.25, mfc='none',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='C0'
    )

    axes[1].errorbar(
        x, evaporation_rate, yerr=yerr_evaporation,
        marker='s', ms=9, mew=1.25, mfc='none', 
        capsize=2.75, elinewidth=1.25, lw=1.5, c='C0', label=rf'A = {sample_area:.2f} cm$^{{\mathregular{{2}}}}$'
    )

    axes[0].set_ylabel('nm/s')
    axes[0].set_title(rf'Deposition rate at {incident_heat_load:.0f} MW/m$^{{\mathregular{{2}}}}$')
    
    axes[0].set_ylim(0, 6)
    axes[0].tick_params(which='both', axis='y', labelright=False, right=True, direction='in')
    axes[0].tick_params(which='both', axis='x', direction='out')

    axes[1].set_ylabel(r'Torr-L/s/m$^{\mathregular{2}}$')
    axes[1].set_title('Inferred sublimation')
    
    for ax in axes:
        ax.set_xlim(0, 6)
        ax.set_xlabel('Repetition')

    axes[1].legend(loc='best', frameon=True, fontsize=9)

    fig.savefig(os.path.join(base_dir, 'graphite_sublimation_20230928' + '.svg'), dpi=600)
    fig.savefig(os.path.join(base_dir, 'graphite_sublimation_20230928' + '.png'), dpi=600)
    fig.savefig(os.path.join(base_dir, 'graphite_sublimation_20230928' + '.pdf'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
