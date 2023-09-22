import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from data_processing.utils import get_experiment_params

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN\20230723'
database_csv = 'Transmission measurements 20230915.csv'

csv_soot_deposition = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests' \
                      r'\surface_temperature\equilibrium_redone\slide_transmission_smausz.csv'

laser_power_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN\laser_output'

samples = [
    {'sample_id': 'R4N64', 'label': 'GC,  7.5% binder', 'material': 'Glassy carbon', 'marker': 'o', 'size': '850 um', 'fig2': False},
    {'sample_id': 'R4N75', 'label': 'GC, 3.75% binder', 'material': 'Glassy carbon', 'marker': 's', 'size': '850 um', 'fig2': False},
    {'sample_id': 'R4N83', 'label': 'GC, 1.88% binder', 'material': 'Glassy carbon', 'marker': '^', 'size': '850 um', 'fig2': True},
    {'sample_id': 'R4N86', 'label': 'POCO cubes (1.7 mm)', 'material': 'POCO graphite', 'marker': 'v', 'size': '850 um', 'fig2': True},
    {'sample_id': 'R4N88', 'label': 'POCO spheres (1.0 mm)', 'material': 'POCO graphite', 'marker': 'h', 'size': '850 um', 'fig2': True},
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

    print(df.columns)

    laser_power_mapping = map_laser_power_settings()
    power_settings = np.array([int(k) for k in laser_power_mapping.keys()])

    sample_ids = df['Sample ID'].unique()

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)

    fig2, axes2 = plt.subplots(ncols=1, nrows=2, constrained_layout=True)
    fig2.set_size_inches(4.0, 5.5)
    ax2, ax3 = axes2[0], axes2[1]

    markers = ['o', 's', '^', 'v', 'd', '<', '>', 'p', 'h']
    colors = ['C0', 'C1', 'C2', 'C5', 'C4', 'C6', 'C7', 'C8']

    for i, id in enumerate(sample_ids):
        df2 = df[df['Sample ID'] == id]
        df2 = df2.groupby('Laser power setting (%)').agg({
            'Deposition rate (nm/s)': ['mean', 'std'],
            'Deposition rate lb (nm/s)': ['mean', 'std'],
            'Deposition rate ub (nm/s)': ['mean', 'std'],
            'Evaporation rate (Torr-L/s)': ['mean', 'std'],
            'Evaporation rate lb (Torr-L/s)': ['mean', 'std'],
            'Evaporation rate ub (Torr-L/s)': ['mean', 'std'],
        })

        lbl = ""
        marker = samples[i]['marker']
        for s in samples:
            if s['sample_id'] == id:
                lbl = s['label']

        print(id, lbl)

        deposition_rate = df2['Deposition rate (nm/s)']['mean']
        deposition_rate_lb = df2['Deposition rate lb (nm/s)']['mean']
        deposition_rate_ub = df2['Deposition rate ub (nm/s)']['mean']


        yerr_deposition = (deposition_rate - deposition_rate_lb, deposition_rate_ub-deposition_rate)
        laser_power_setting = list(df2.index.values)
        laser_power = np.array([laser_power_mapping[v] for v in laser_power_setting])
        sample_diameter = 1.025
        sample_area = 0.25 * np.pi * sample_diameter ** 2.0
        aperture_factor = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.5 * sample_diameter)
        incident_heat_load = aperture_factor * laser_power / sample_area / 100.0

        evaporation_rate_lb = df2['Evaporation rate lb (Torr-L/s)']['mean'] * 1E4 / sample_area
        evaporation_rate = df2['Evaporation rate (Torr-L/s)']['mean'] * 1E4 / sample_area
        evaporation_rate_ub = df2['Evaporation rate ub (Torr-L/s)']['mean'] * 1E4 / sample_area
        yerr_evaporation = (evaporation_rate-evaporation_rate_lb, evaporation_rate_ub-evaporation_rate_lb)

        ax.errorbar(
            incident_heat_load, deposition_rate,  yerr=yerr_deposition,
            marker=marker, ms=9, mew=1.25, mfc='none', label=f'{lbl}',
            capsize=2.75, elinewidth=1.25, lw=1.5, c=colors[i]
        )

        if samples[i]['fig2']:
            ax2.errorbar(
                incident_heat_load, deposition_rate,  yerr=yerr_deposition,
                marker=marker, ms=9, mew=1.25, mfc='none', label=f'{lbl}',
                capsize=2.75, elinewidth=1.25, lw=1.5, c=colors[i]
            )

            ax3.errorbar(
                incident_heat_load, evaporation_rate,  yerr=yerr_evaporation,
                marker=marker, ms=9, mew=1.25, mfc='none', label=f'{lbl}',
                capsize=2.75, elinewidth=1.25, lw=1.5, c=colors[i]
            )

    ax.set_xlabel('Heat load [MW/m$^{\mathregular{2}}$]')
    ax.set_ylabel('nm/s')
    ax.set_title('Deposition rate')
    ax.set_xlim(5, 40)
    # ax.set_ylim(0, 0.8)
    ax.tick_params(which='both', axis='y', labelright=False, right=True, direction='in')
    ax.tick_params(which='both', axis='x', direction='out')

    for i, axx in enumerate(axes2):
        axx.set_xlabel('Heat load [MW/m$^{\mathregular{2}}$]')
        axx.set_xlim(5, 40)


    ax2.set_ylabel('nm/s')
    ax2.set_title('Deposition rate')

    ax3.set_ylabel(r'Torr-L/s/m$^{\mathregular{2}}$')

    # ax.set_ylim(0, 0.8)
    ax2.tick_params(which='both', axis='y', labelright=False, right=True, direction='in')
    ax2.tick_params(which='both', axis='x', direction='out')

    soot_deposition_df = pd.read_csv(csv_soot_deposition)
    soot_deposition_columns = soot_deposition_df.columns
    soot_deposition_df[soot_deposition_columns[1:]] = soot_deposition_df[soot_deposition_columns[1:]].apply(
        pd.to_numeric)
    soot_deposition_pebble_df = soot_deposition_df[soot_deposition_df['Sample'] != 'GT001688']
    soot_deposition_graphite_df = soot_deposition_df[soot_deposition_df['Sample'] == 'GT001688']
    soot_deposition_pebble_df.sort_values(by=['Laser Power (%)'], ascending=True)
    soot_deposition_graphite_df.sort_values(by=['Laser Power (%)'], ascending=True)

    laser_power_setting_sublimation_graphite = soot_deposition_graphite_df['Laser Power (%)'].values
    laser_power_setting_sublimation_pebble = soot_deposition_pebble_df['Laser Power (%)'].values

    laser_power_graphite = np.array([laser_power_mapping[v] for v in laser_power_setting_sublimation_graphite])
    laser_power_pebble = np.array([laser_power_mapping[v] for v in laser_power_setting_sublimation_pebble])

    film_thickness_pebble = soot_deposition_pebble_df['Thickness (nm)'].values
    film_thickness_pebble_err = film_thickness_pebble * soot_deposition_pebble_df['Error %'].values
    flattop_time_pebble = soot_deposition_pebble_df['Flat top time (s)'].values
    sublimation_rate_pebble = film_thickness_pebble / flattop_time_pebble
    de_by_de_pebble = np.zeros_like(sublimation_rate_pebble, dtype=np.float64)
    msk_pebble = film_thickness_pebble > 0.0
    de_by_de_pebble[msk_pebble] = film_thickness_pebble_err[msk_pebble] / film_thickness_pebble[msk_pebble]
    sublimation_rate_pebble_err = sublimation_rate_pebble * np.sqrt(
        de_by_de_pebble ** 2.0 + 0.1 ** 2.0
    )

    film_thickness_graphite = soot_deposition_graphite_df['Thickness (nm)'].values
    film_thickness_graphite_err = film_thickness_graphite * 0.12  #soot_deposition_graphite_df['Error %'].values
    flattop_time_graphite = soot_deposition_graphite_df['Flat top time (s)'].values
    msk_graphite = (film_thickness_graphite > 0.0) & (flattop_time_graphite > 0.0)
    sublimation_rate_graphite = np.zeros_like(flattop_time_graphite)
    sublimation_rate_graphite[msk_graphite] = film_thickness_graphite[msk_graphite] / flattop_time_graphite[
        msk_graphite]
    de_by_de_graphite = np.zeros_like(sublimation_rate_graphite, dtype=np.float64)
    de_by_de_graphite[msk_graphite] = film_thickness_graphite_err[msk_graphite] / film_thickness_graphite[msk_graphite]
    # sublimation_rate_graphite_err = sublimation_rate_graphite * np.sqrt(
    #     de_by_de_graphite ** 2.0 + 0.1 ** 2.0
    # )
    sublimation_rate_graphite_err  = 0.12 * sublimation_rate_graphite

    sample_area_graphite = 0.25 * np.pi * graphite_sample_diameter ** 2.0
    aperture_factor_graphite = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.5 * graphite_sample_diameter)
    incident_heat_load_graphite = aperture_factor_graphite * laser_power_graphite / sample_area_graphite / 100.0
    incident_heat_load_pebble = aperture_factor_graphite * laser_power_pebble / sample_area_graphite / 100.0

    evaporation_rate_graphite = 21.388E3 * (h_0 ** 2. / n_cos) * sublimation_rate_graphite

    ax.errorbar(
        incident_heat_load_pebble, sublimation_rate_pebble,  #yerr=sublimation_rate_graphite_err,
        marker='^', ms=9, mew=1.25, mfc='none', label='GC, 10.0% binder',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='navy'
    )

    ax.errorbar(
        incident_heat_load_graphite, sublimation_rate_graphite,  # yerr=recession_rate_err,
        marker='D', ms=9, mew=1.25, mfc='none', label='Graphite rod',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='tab:red'
    )

    ax2.errorbar(
        incident_heat_load_pebble, sublimation_rate_pebble,  # yerr=sublimation_rate_graphite_err,
        marker='^', ms=9, mew=1.25, mfc='none', label='GC, 10.0% binder',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='navy'
    )

    ax2.errorbar(
        incident_heat_load_graphite, sublimation_rate_graphite,  # yerr=recession_rate_err,
        marker='D', ms=9, mew=1.25, mfc='none', label='Graphite rod',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='tab:red'
    )

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(10.0))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(5.0))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.legend(loc='best', frameon=True, fontsize=9)

    ax2.legend(loc='best', frameon=True, fontsize=9)
    ax3.ticklabel_format(axis='y', useMathText=True, scilimits=(-3,3))
    ax3.set_title('Sublimation rate')

    fig.savefig(os.path.join(base_dir, 'carbon_deposition_vs_laser_power_binder_content_20230915' + '.svg'), dpi=600)
    fig.savefig(os.path.join(base_dir, 'carbon_deposition_vs_laser_power_binder_content_20230915' + '.png'), dpi=600)
    fig.savefig(os.path.join(base_dir, 'carbon_deposition_vs_laser_power_binder_content_20230915' + '.pdf'), dpi=600)

    fig2.savefig(os.path.join(base_dir, 'carbon_deposition_vs_laser_power_material_20230915' + '.png'), dpi=600)

    plt.show()

if __name__ == '__main__':
    main()