import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from data_processing.utils import get_experiment_params
from scipy.stats.distributions import t

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\DPP 2023\figures'
database_csv = 'Transmission measurements - 20231020.csv'

csv_soot_deposition = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests' \
                      r'\surface_temperature\equilibrium_redone\slide_transmission_smausz.csv'

laser_power_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN\laser_output'

samples = [
    {'sample_id': 'GT001688', 'label': 'Graphite rod', 'material': 'Isomolded graphite', 'marker': 'o'},
    # {'sample_id': 'GT001688', 'label': 'Graphite rod', 'material': 'Isomolded graphite', 'marker': 'o'},
    # {'sample_id': 'GT001688', 'label': 'Graphite rod', 'material': 'Isomolded graphite', 'marker': 'o'},
    # {'sample_id': 'GT001688', 'label': 'Graphite rod', 'material': 'Isomolded graphite', 'marker': 'o'},
    # {'sample_id': 'GT001688', 'label': 'Graphite rod', 'material': 'Isomolded graphite', 'marker': 'o'},
]

beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707
n_cos, dn_cos = 7., 2.
h_0 = 10.5 * 2.54

graphite_sample_diameter = 0.92
film_density = 2.2  # g / cm^3
sample_area = 0.25 * np.pi * (graphite_sample_diameter ** 2.)
def nmps2cps(x):
    global film_density, h_0, n_cos, sample_area
    return 2. * np.pi * film_density * (h_0 ** 2.) * x / n_cos / 12.011 * 6.02214076E-3 / sample_area # x1E18


def cps2nmps(x):
    global film_density, h_0, n_cos, sample_area
    return 12.011 * n_cos / (2. * np.pi * film_density * (h_0 ** 2.)) * x * sample_area / 6.02214076E-3

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

    # fig, axes = plt.subplots(ncols=1, nrows=2, constrained_layout=True)
    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(4.0, 2.5)

    secax = ax.secondary_yaxis('right', functions=(nmps2cps, cps2nmps))
    secax.set_ylabel(r'$\mathregular{\times}$10$^{\mathregular{18}}$ C/s/cm$^{\mathregular{2}}$')
    secax.ticklabel_format(axis='y', useMathText=True)

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
    deposition_rate_1 = deposition_rate[0:5]
    deposition_rate_2 = deposition_rate[5::]

    confidence_level = 0.95
    alpha = 1. - confidence_level


    deposition_rate_1_std = np.std(deposition_rate_1, ddof=1)
    n1 = len(deposition_rate_1)
    tval1 = t.ppf(1. - 0.5*alpha, n1-1)
    deposition_rate_1_se = deposition_rate_1_std * tval1 / np.sqrt(n1)

    deposition_rate_2_std = np.std(deposition_rate_2[2::], ddof=1)
    n2 = len(deposition_rate_1[2::])
    tval2 = t.ppf(1. - 0.5 * alpha, n2 - 1)
    deposition_rate_2_se = deposition_rate_2_std * tval2 / np.sqrt(n2)


    stats_first = {
        'mean': np.mean(deposition_rate_1),
        'std': deposition_rate_1_std,
        'n': n1,
        'tval': tval1,
        'standard error': deposition_rate_1_se,
        'mean_cps': nmps2cps(np.mean(deposition_rate_1)),
        'cps_error': nmps2cps(np.mean(deposition_rate_1_se))
    }

    stats_second = {
        'mean': np.mean(deposition_rate_2[2::]),
        'std': deposition_rate_2_std,
        'n': n2,
        'tval': tval2,
        'standard error': deposition_rate_2_se,
        'mean_cps': nmps2cps(np.mean(deposition_rate_2[2::])),
        'cps_error': nmps2cps(np.mean(deposition_rate_2_se))
    }

    yerr_deposition_1 = (deposition_rate[0:5] - deposition_rate_lb[0:5], deposition_rate_ub[0:5] - deposition_rate[0:5])
    yerr_deposition_2 = (deposition_rate[5::] - deposition_rate_lb[5::], deposition_rate_ub[5::] - deposition_rate[5::])
    x = np.array([i+1 for i in range(len(deposition_rate_1))])
    ax.errorbar(
        x, deposition_rate_1, yerr=yerr_deposition_1,
        marker='o', ms=9, mew=1.25, mfc='none',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='C0', label='First set'
    )

    x = np.array([i + 1 for i in range(len(deposition_rate_2))])
    ax.errorbar(
        x, deposition_rate_2, yerr=yerr_deposition_2,
        marker='s', ms=9, mew=1.25, mfc='none',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='C1', label='Second set'
    )

    stats_first_txt = f"$\\bar{{r}} = {stats_first['mean']:.0f}\pm{stats_first['standard error']:.0f}~\\mathrm{{nm/s}}$\n"
    stats_first_txt += f"$({stats_first['mean_cps']:.0f}\pm{stats_first['cps_error']:.0f})\\times 10^{{18}}~\\mathrm{{C/s/cm^2}}$"
    ax.text(
        1, deposition_rate_1[0]+2.5, stats_first_txt,
        # transform=ax.transAxes,
        color='C0', ha='left', va='bottom', fontsize=10
    )

    stats_second_txt = f"$\\bar{{r}} = {stats_second['mean']:.1f}\pm{stats_second['standard error']:.1f}~\\mathrm{{nm/s}}$\n"
    stats_second_txt += f"$({stats_second['mean_cps']:.1f}\pm{stats_second['cps_error']:.1f})\\times 10^{{18}}~\\mathrm{{C/s/cm^2}}$"
    ax.text(
        6, deposition_rate_2[-1] + 4, stats_second_txt,
        # transform=ax.transAxes,
        color='C1', ha='right', va='bottom', fontsize=10
    )

    # axes[1].errorbar(
    #     x, evaporation_rate, yerr=yerr_evaporation,
    #     marker='s', ms=9, mew=1.25, mfc='none',
    #     capsize=2.75, elinewidth=1.25, lw=1.5, c='C0', label=rf'A = {sample_area:.2f} cm$^{{\mathregular{{2}}}}$'
    # )

    ax.set_ylabel('nm/s')
    ax.set_title(rf'Graphite deposition rate at {round(incident_heat_load/10.)*10:.0f} MW/m$^{{\mathregular{{2}}}}$')
    
    # axes[0].set_ylim(0, 6)
    ax.tick_params(which='both', axis='y', labelright=False, right=False, direction='in')
    ax.tick_params(which='both', axis='x', direction='out')

    # axes[1].set_ylabel(r'Torr-L/s/m$^{\mathregular{2}}$')
    # axes[1].set_title('Inferred sublimation')

    ax.set_xlabel('Repetition')
    ax.set_ylim(-1, 21)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.legend(loc='best', frameon=True, fontsize=9)

    secax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    secax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    fig.savefig(os.path.join(base_dir, 'graphite_sublimation' + '.svg'), dpi=600)
    fig.savefig(os.path.join(base_dir, 'graphite_sublimation' + '.png'), dpi=600)
    fig.savefig(os.path.join(base_dir, 'graphite_sublimation' + '.pdf'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
