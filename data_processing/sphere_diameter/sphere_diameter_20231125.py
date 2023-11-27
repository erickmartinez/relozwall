import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import platform
from scipy.stats.distributions import t
from data_processing.utils import get_experiment_params
import json

if platform.system() != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'

base_path = r'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/pebble_size'
sublimation_csv = 'sublimation_rates.csv'

firing_csv = r'Documents/ucsd/Postdoc/research/data/firing_tests/merged_db.xlsx'

laser_power_dir = r'Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN\laser_output'

beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)

def pd_norm(x):
    return np.linalg.norm(x)

def map_laser_power_settings():
    global laser_power_dir, drive_path, beam_radius
    if platform.system() != 'Windows':
        laser_power_dir = laser_power_dir.replace('\\', '/')
    rdir = os.path.join(drive_path, laser_power_dir)

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
    global base_path, sublimation_csv, firing_csv, laser_power_dir
    if platform.system() != 'Windows':
        base_path = base_path.replace('\\', '/')
        sublimation_csv = sublimation_csv.replace('\\', '/')
        firing_csv = firing_csv.replace('\\', '/')
        laser_power_dir = laser_power_dir.replace('\\', '/')
    base_path = os.path.join(drive_path, base_path)
    sublimation_csv = os.path.join(drive_path, base_path, sublimation_csv)
    firing_csv = os.path.join(drive_path, firing_csv)

    # load transmission measurements
    transmission_df = pd.read_csv(sublimation_csv)
    transmission_df = transmission_df.loc[:, ~transmission_df.columns.str.contains('Unnamed')]
    transmission_columns = transmission_df.columns
    transmission_df[transmission_columns[2::]] = transmission_df[transmission_columns[2::]].apply(pd.to_numeric)

    row_ids = transmission_df['ROW'].tolist()
    sample_ids = transmission_df['Sample ID'].tolist()

    # load firing database
    firing_df = pd.read_excel(firing_csv, sheet_name='Laser tests')
    firing_df = firing_df[firing_df['ROW'].isin(row_ids)]
    firing_df.rename(columns={'Sample code': 'Sample ID'}, inplace=True)

    transmission_df[['Big spheres', 'Small spheres']] = transmission_df['Pebble material'].str.extract(
        r'\[(\d+)\,(\d+)\]').apply(pd.to_numeric)

    confidence = 0.95
    alpha = 1. - confidence

    bsc = transmission_df['Big spheres'].values

    id2bsc = {}
    for i, sid in enumerate(sample_ids):
        if sid not in id2bsc:
            id2bsc[sid] = bsc[i]

    print(id2bsc)

    print(firing_df)
    merged_df = pd.merge(firing_df, transmission_df, on=['ROW'], how='inner', suffixes=('', '_drop'))
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('Unnamed')]
    merged_df.drop([col for col in merged_df.columns if 'drop' in col], axis=1, inplace=True)

    laser_power_settings = [60, 100]
    laser_power_mapping = map_laser_power_settings()
    lp = np.array([laser_power_mapping[int(x)] for x in merged_df['Laser power setting (%)']])
    sa = 0.25 * np.pi * np.power(merged_df['Sample diameter (cm)'], 2.)
    merged_df['Sample area (cm^2)'] = sa
    merged_df['Laser power [MW/m2]'] = lp * gaussian_beam_aperture_factor(beam_radius,
                                                                          merged_df['Sample diameter (cm)']) / sa / 100.

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

    fig_dr, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig_dr.set_size_inches(4., 5.)

    axes[0].set_yscale('log')

    colors_dr = ['C0', 'C1']
    markers_dr = ['o', 's']

    for i, lps in enumerate(laser_power_settings):
        df_at_ps: pd.DataFrame = merged_df[merged_df['Laser power setting (%)'] == lps]
        df_at_ps = df_at_ps.sort_values(by=['Big spheres']).reset_index()
        yerr = np.array([
            df_at_ps['Evaporation rate (Torr-L/s)'] - df_at_ps['Evaporation rate lb (Torr-L/s)'],
            df_at_ps['Evaporation rate ub (Torr-L/s)'] - df_at_ps['Evaporation rate (Torr-L/s)']
        ])

        lp = df_at_ps['Laser power [MW/m2]'].values[0]

        lbl = f'{5 * round(lp / 5):.0f} MW/m$^{{\\mathregular{{2}}}}$'

        axes[0].errorbar(
            df_at_ps['Big spheres'], 1E4 * df_at_ps['Evaporation rate (Torr-L/s)'] / df_at_ps['Sample area (cm^2)'],
            yerr=1E4 * t.ppf(1 - 0.5 * alpha, 1) * yerr,
            c=colors_dr[i],
            marker=markers_dr[i], lw=1.5, ls='none',
            # mec='k',
            mew=1.25, capsize=3.5, capthick=1.25, ecolor=colors_dr[i], fillstyle='none',
            label=lbl
        )

        df_at_ps2 = firing_df[( firing_df['Sample ID'].isin(sample_ids) ) & (firing_df['Power percent setting (%)'] == lps)].copy()
        df_at_ps2['Big spheres'] = [id2bsc[x] for x in df_at_ps2['Sample ID'].values]
        # print(df_at_ps2[['Sample ID', 'Big spheres', 'Power percent setting (%)']])
        df_at_ps2 = df_at_ps2.sort_values(by=['Big spheres'])
        df_at_ps_g = df_at_ps2.groupby(by=['Sample ID']).agg({
            'Big spheres': ['first'],
            'Recession rate (cm/s)': ['mean', 'std', 'count'],
            'Recession rate error (cm/s)': [pd_norm, 'std']
        })

        df_at_ps_g.fillna(0., inplace=True)
        print(df_at_ps_g[['Big spheres', 'Recession rate (cm/s)', 'Recession rate error (cm/s)']])
        nn = np.stack([df_at_ps_g['Recession rate (cm/s)']['count']-1, np.ones_like(df_at_ps_g['Recession rate (cm/s)']['count'])]).T
        tval = t.ppf(1 - 0.5 *0.32, np.max(nn, axis=1)) # 68 % error


        yerr = np.array([
            (df_at_ps_g['Recession rate error (cm/s)']['pd_norm'] + df_at_ps_g['Recession rate (cm/s)']['std']) * tval / np.sqrt(
                df_at_ps_g['Recession rate (cm/s)']['count'])
        ])

        axes[1].errorbar(
            df_at_ps_g['Big spheres']['first'], df_at_ps_g['Recession rate (cm/s)']['mean'],
            yerr=yerr,
            c=colors_dr[i],
            marker=markers_dr[i], lw=1.5, ls='none',
            # mec='k',
            mew=1.25, capsize=3.5, capthick=1.25, ecolor=colors_dr[i], fillstyle='none'
        )

    axes[0].set_ylabel('Torr-L/s/m$^{\mathregular{2}}$')
    axes[0].set_ylim(1E4, 1E6)
    axes[0].set_title('Sublimation rate')

    axes[1].set_xlabel('Weight % of 850 $\mathregular{\mu}$m spheres')
    axes[1].set_ylabel('cm/s')
    axes[1].set_ylim(0, 1.)
    axes[1].set_title('Recession rate')

    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    axes[1].set_xlim(-5, 105)
    axes[1].text(
        -0.0, -0.3, '220 $\mathregular{\mu}$m', transform=axes[1].transAxes, fontsize=12, fontweight='regular',
        va='bottom', ha='center'
    )
    axes[1].text(
        1.0, -0.3, '850 $\mathregular{\mu}$m', transform=axes[1].transAxes, fontsize=12, fontweight='regular',
        va='bottom', ha='center'
    )

    axes[0].legend(loc='lower right', frameon=True, fontsize=10)

    fig_dr.savefig(os.path.join(base_path, 'pebble_size_scan_20231126.png'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
