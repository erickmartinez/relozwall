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

platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'

base_path = r'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/pebble_size'
sublimation_csv = 'deposition_rates.csv'

firing_csv = r'Documents/ucsd/Postdoc/research/data/firing_tests/merged_db.xlsx'
tracking_csv = r'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/Firing Tests - Mass Loss - Tracking_gsr_20231128_mod.csv'
laser_power_dir = r'Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN\laser_output'

beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707

"""
Parameters to estimate outgassing
"""
cal_p_peak = 0.12  # Torr
cal_og = 3031 # Torr-L/s/m^2
cal_og_err = 0.5


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


def estimate_hydrocarbon_outgassing(peak_pressure):
    global cal_p_peak, cal_og
    return (peak_pressure/cal_p_peak) * cal_og

def normalize_path(the_path):
    global platform_system
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    return the_path

def main():
    global base_path, sublimation_csv, firing_csv, laser_power_dir, tracking_csv
    if platform.system() != 'Windows':
        base_path =  normalize_path(base_path)
        sublimation_csv = normalize_path(sublimation_csv)
        firing_csv = normalize_path(firing_csv)
        laser_power_dir = normalize_path(laser_power_dir)
        tracking_csv = normalize_path(tracking_csv)
    base_path = os.path.join(drive_path, base_path)
    sublimation_csv = os.path.join(drive_path, base_path, sublimation_csv)
    firing_csv = os.path.join(drive_path, firing_csv)
    tracking_csv = os.path.join(drive_path, tracking_csv)

    # load transmission measurements
    transmission_df = pd.read_csv(sublimation_csv)
    transmission_df = transmission_df.loc[:, ~transmission_df.columns.str.contains('Unnamed')]
    transmission_columns = transmission_df.columns
    transmission_df[transmission_columns[2::]] = transmission_df[transmission_columns[2::]].apply(pd.to_numeric)

    tracking_df = pd.read_csv(tracking_csv)
    tracking_df_cols = tracking_df.columns
    tracking_df[tracking_df_cols[1:5]] = tracking_df[tracking_df_cols[1:5]].apply(pd.to_numeric)
    tracking_df[tracking_df_cols[6::]] = tracking_df[tracking_df_cols[6::]].apply(pd.to_numeric)
    tracking_df[['Sample ID', 'ROW', 'Laser power setting (%)']] = tracking_df['Test'].str.extract(
        r'LCT\_(R\d+N\d+)\-?\d*\_ROW(\d+)\_(\d+)PCT\_.*$'
    )

    tracking_df[['ROW', 'Laser power setting (%)']] = tracking_df[['ROW', 'Laser power setting (%)']].apply(pd.to_numeric)
    print(tracking_df[['Sample ID', 'ROW', 'Laser power setting (%)']])
    tracking_df_agg = tracking_df.groupby(by=['ROW']).agg({
        'ROW': ['first'],
        'Max temperature (K)': ['max', 'mean', 'std', 'count'],
        'Graphite sublimation rate (Torr-L/s/m^2)': ['mean', 'std', 'count'],
        'Peak pressure (Torr)': ['first']
    })
    tracking_df_agg.fillna(0.0)
    nn = np.stack([tracking_df_agg['Max temperature (K)']['count'] - 1,
                   np.ones_like(tracking_df_agg['Max temperature (K)']['count'])]).T
    tval = t.ppf(1 - 0.5 * 0.32, np.max(nn, axis=1))  # 68 % error

    max_temp_err = tval * tracking_df_agg['Max temperature (K)']['std'] / np.sqrt(tracking_df_agg['Max temperature (K)']['count'])
    g_sub_err = tval * tracking_df_agg['Graphite sublimation rate (Torr-L/s/m^2)']['std'] / np.sqrt(tracking_df_agg['Graphite sublimation rate (Torr-L/s/m^2)']['count'])
    tracking_summary_df = pd.DataFrame(data={
        'ROW': [x for x in tracking_df_agg['ROW']['first']],
        'Temperature mean (K)': [x for x in tracking_df_agg['Max temperature (K)']['mean']],
        'Temperature std (K)': [x for x in tracking_df_agg['Max temperature (K)']['std']],
        'Temperture count': [x for x in tracking_df_agg['Max temperature (K)']['count']],
        'Peak pressure (Torr)': [x for x in tracking_df_agg['Peak pressure (Torr)']['first']],
        'Max temperature err (K)': [x for x in max_temp_err],
    })
    tracking_summary_df.set_index = tracking_summary_df['ROW']
    print(tracking_summary_df)

    row_ids = transmission_df['ROW'].tolist()
    sample_ids = transmission_df['Sample ID'].tolist()

    # load firing database
    firing_df = pd.read_excel(firing_csv, sheet_name='Laser tests')
    firing_df = firing_df[firing_df['ROW'].isin(row_ids)]
    firing_df.rename(columns={'Sample code': 'Sample ID'}, inplace=True)

    transmission_df[['Small spheres', 'Big spheres']] = transmission_df['Pebble material'].str.extract(
        r'\[(\d+)\,(\d+)\]').apply(pd.to_numeric)

    transmission_df.sort_values(by=['Big spheres', 'Laser power setting (%)'], inplace=True)

    confidence = 0.95
    alpha = 1. - confidence

    bsc = transmission_df['Big spheres'].values

    id2bsc = {}
    for i, sid in enumerate(sample_ids):
        if sid not in id2bsc:
            id2bsc[sid] = bsc[i]

    # print(id2bsc)

    print(firing_df)
    merged_df = pd.merge(firing_df, transmission_df, on=['ROW'], how='inner', suffixes=('', '_drop'))
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('Unnamed')]
    merged_df.drop([col for col in merged_df.columns if 'drop' in col], axis=1, inplace=True)

    # merged_df = pd.merge(merged_df, tracking_summary_df, on=['ROW'], how='inner', suffixes=('', '_drop'))
    # merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('Unnamed')]
    # merged_df.drop([col for col in merged_df.columns if 'drop' in col], axis=1, inplace=True)

    laser_power_settings = [60, 100]
    laser_power_mapping = map_laser_power_settings()
    lp = np.array([laser_power_mapping[int(x)] for x in merged_df['Laser power setting (%)']])
    sa = 0.25 * np.pi * np.power(merged_df['Sample diameter (cm)'], 2.)
    merged_df['Sample area (cm^2)'] = sa
    merged_df['Laser power [MW/m2]'] = lp * gaussian_beam_aperture_factor(beam_radius,
                                                                          0.5*merged_df['Sample diameter (cm)']) / sa / 100.

    merged_df.sort_values(by=['Big spheres', 'Laser power setting (%)'], inplace=True)

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

    fig_dr, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw=dict(hspace=0, height_ratios=[2, 1]), constrained_layout=True)
    fig_dr.set_size_inches(5.5, 6.)

    axes[0].set_yscale('log')

    colors_dr = ['C0', 'C1']
    markers_dr = ['o', 's']
    markers_rr = ['D', '^']

    for i, lps in enumerate(laser_power_settings):
        df_at_ps: pd.DataFrame = merged_df[merged_df['Laser power setting (%)'] == lps]
        df_at_ps = df_at_ps.sort_values(by=['Big spheres']).reset_index()
        # tracking_at_ps = tracking_df[tracking_df['ROW']]
        yerr = np.array([
            np.abs(df_at_ps['Evaporation rate (Torr-L/s)'] - df_at_ps['Evaporation rate lb (Torr-L/s)']),
            np.abs(df_at_ps['Evaporation rate ub (Torr-L/s)'] - df_at_ps['Evaporation rate (Torr-L/s)'])
        ])

        lp = df_at_ps['Laser power [MW/m2]'].values[0]

        lbl = f'{5 * round(lp / 5):.0f} MW/m$^{{\\mathregular{{2}}}}$ (Matrix)'

        axes[0].errorbar(
            df_at_ps['Big spheres'], 1E4 * df_at_ps['Evaporation rate (Torr-L/s)'] / df_at_ps['Sample area (cm^2)'],
            yerr=1E4 * t.ppf(1 - 0.5 * 0.32, 1) * yerr,
            c=colors_dr[i],
            marker=markers_dr[i], lw=1.5, ls='none',
            # mec='k',
            mew=1.25, capsize=3.5, capthick=1.25, ecolor=colors_dr[i], fillstyle='none',
            label=lbl
        )

        # Find the graphite sublimation
        lbl_g = f'{5 * round(lp / 5):.0f} MW/m$^{{\\mathregular{{2}}}}$ (Graphite)'
        tracking_summary_at_ps_df = tracking_summary_df[tracking_summary_df['ROW'].isin(df_at_ps['ROW'].tolist())]
        g_df = pd.merge(tracking_summary_at_ps_df, transmission_df, on='ROW').sort_values(by=['Big spheres'])
        print(g_df[['ROW', 'Big spheres','Graphite sublimation (Torr-L/s/m^2)']])
        axes[0].errorbar(
            g_df['Big spheres'], g_df['Graphite sublimation (Torr-L/s/m^2)'],
            yerr=0.5*g_df['Graphite sublimation (Torr-L/s/m^2)'],
            c=colors_dr[i],
            marker=markers_dr[i], lw=1.5, ls='none',
            # mec='k',
            mew=1.25, capsize=3.5, capthick=1.25, ecolor=colors_dr[i], fillstyle='full',
            label=lbl_g
        )

        # Hydrocarbon outgassing
        lbl_h = f'{5 * round(lp / 5):.0f} MW/m$^{{\\mathregular{{2}}}}$ (CH$_{{\\mathregular{{x}}}}$)'
        hydrocarbon_outgassing = estimate_hydrocarbon_outgassing(g_df['Peak pressure (Torr)'].values)
        axes[0].errorbar(
            g_df['Big spheres'], hydrocarbon_outgassing,
            yerr=hydrocarbon_outgassing*0.5,
            c=colors_dr[i],
            marker=markers_dr[i], lw=1.5, ls='none',
            # mec='k',
            mew=1.25, capsize=3.5, capthick=1.25, ecolor=colors_dr[i], fillstyle='left',
            label=lbl_h
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
            marker=markers_rr[i], lw=1.5, ls='none',
            # mec='k',
            mew=1.25, capsize=3.5, capthick=1.25, ecolor=colors_dr[i], fillstyle='none'
        )

    axes[0].set_ylabel('Torr-L/s/m$^{\mathregular{2}}$')
    axes[0].set_ylim(1, 1E6)
    axes[0].set_title('Outgassing rate')

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

    # axes[0].legend(loc='center left', frameon=True, fontsize=9)
    axes[0].legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0., fontsize=9)

    fig_dr.savefig(os.path.join(base_path, 'pebble_size_scan_20231126.png'), dpi=600)
    fig_dr.savefig(os.path.join(base_path, 'pebble_size_scan_20231126.pdf'), dpi=600)
    fig_dr.savefig(os.path.join(base_path, 'pebble_size_scan_20231126.svg'), dpi=600)
    fig_dr.savefig(os.path.join(base_path, 'pebble_size_scan_20231126.eps'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
