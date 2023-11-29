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

base_path = r'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/baking_time'
sublimation_csv = 'sublimation_rates_baking_time.csv'
tracking_csv = r'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/Firing Tests - Mass Loss - Tracking_gsr_20231128_mod.csv'


firing_csv = r'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/baking_time/recession_rate.csv'

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

def estimate_hydrocarbon_outgassing(peak_pressure):
    global cal_p_peak, cal_og
    return (peak_pressure/cal_p_peak) * cal_og

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
    global base_path, sublimation_csv, firing_csv, laser_power_dir, tracking_csv
    if platform.system() != 'Windows':
        base_path = base_path.replace('\\', '/')
        sublimation_csv = sublimation_csv.replace('\\', '/')
        firing_csv = firing_csv.replace('\\', '/')
        laser_power_dir = laser_power_dir.replace('\\', '/')
    base_path = os.path.join(drive_path, base_path)
    sublimation_csv = os.path.join(drive_path, base_path, sublimation_csv)
    firing_csv = os.path.join(drive_path, firing_csv)
    tracking_csv = os.path.join(drive_path, tracking_csv)


    # load transmission measurements
    transmission_df = pd.read_csv(sublimation_csv)
    transmission_df = transmission_df.loc[:, ~transmission_df.columns.str.contains('Unnamed')]
    transmission_columns = transmission_df.columns
    transmission_df[transmission_columns[2::]] = transmission_df[transmission_columns[2::]].apply(pd.to_numeric)

    row_ids = transmission_df['ROW'].tolist()
    sample_ids = transmission_df['Sample ID'].tolist()

    # print(transmission_df)

    tracking_df = pd.read_csv(tracking_csv)
    tracking_df_cols = tracking_df.columns
    tracking_df[tracking_df_cols[1:5]] = tracking_df[tracking_df_cols[1:5]].apply(pd.to_numeric)
    tracking_df[tracking_df_cols[6::]] = tracking_df[tracking_df_cols[6::]].apply(pd.to_numeric)
    tracking_df[['Sample ID', 'ROW', 'Laser power setting (%)']] = tracking_df['Test'].str.extract(
        r'LCT\_(R\d+N\d+)\-?\d*\_ROW(\d+)\_(\d+)PCT\_.*$'
    )

    tracking_df[['ROW', 'Laser power setting (%)']] = tracking_df[['ROW', 'Laser power setting (%)']].apply(
        pd.to_numeric)
    # print(tracking_df[['Sample ID', 'ROW', 'Laser power setting (%)']])
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

    max_temp_err = tval * tracking_df_agg['Max temperature (K)']['std'] / np.sqrt(
        tracking_df_agg['Max temperature (K)']['count'])
    g_sub_err = tval * tracking_df_agg['Graphite sublimation rate (Torr-L/s/m^2)']['std'] / np.sqrt(
        tracking_df_agg['Graphite sublimation rate (Torr-L/s/m^2)']['count'])
    tracking_summary_df = pd.DataFrame(data={
        'ROW': [x for x in tracking_df_agg['ROW']['first']],
        'Temperature mean (K)': [x for x in tracking_df_agg['Max temperature (K)']['mean']],
        'Temperature std (K)': [x for x in tracking_df_agg['Max temperature (K)']['std']],
        'Temperture count': [x for x in tracking_df_agg['Max temperature (K)']['count']],
        'Peak pressure (Torr)': [x for x in tracking_df_agg['Peak pressure (Torr)']['first']],
        'Max temperature err (K)': [x for x in max_temp_err],
        'Graphite sublimation rate (Torr-L/s/m^2)': [x for x in
                                                     tracking_df_agg['Graphite sublimation rate (Torr-L/s/m^2)'][
                                                         'mean']],
        'Graphite sublimation rate error (Torr-L/s/m^2)': [x for x in g_sub_err]
    })
    tracking_summary_df.set_index = tracking_summary_df['ROW']
    print(tracking_summary_df)

    # load firing database
    firing_df = pd.read_csv(firing_csv)
    firing_df = firing_df[firing_df['ROW'].isin(row_ids)]

    confidence = 0.95
    alpha = 1. - confidence

    # print(firing_df)
    merged_df = pd.merge(firing_df, transmission_df, on=['ROW'], how='left', suffixes=('', '_drop'))
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('Unnamed')]
    merged_df.drop([col for col in merged_df.columns if 'drop' in col], axis=1, inplace=True)

    laser_power_mapping = map_laser_power_settings()
    lp = np.array([laser_power_mapping[int(x)] for x in merged_df['Laser power setting (%)']])
    sa = 0.25 * np.pi * np.power(merged_df['Sample diameter (cm)'], 2.)
    merged_df['Sample area (cm^2)'] = sa
    merged_df['Laser power [MW/m2]'] = lp * gaussian_beam_aperture_factor(beam_radius,
                                                                          merged_df['Sample diameter (cm)']) / sa / 100.


    merged_df.sort_values(by=['Baking time (min)', 'Laser power setting (%)'], inplace=True)
    # print(merged_df)

    merged_df = pd.merge(merged_df, tracking_summary_df, on=['ROW'], how='inner', suffixes=('', '_drop'))
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('Unnamed')]
    merged_df.drop([col for col in merged_df.columns if 'drop' in col], axis=1, inplace=True)
    # merged_df.dropna(inplace=True)
    print(merged_df)

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

    fig_dr, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw=dict(hspace=0, height_ratios=[2, 1]), constrained_layout=True)
    fig_dr.set_size_inches(4., 6.)

    axes[0].set_yscale('log')

    colors_dr = ['C1', 'C1']
    markers_dr = ['s', 's']

    agg_df = merged_df.groupby(by=['Baking time (min)']).agg({
            'Baking time (min)': ['first'],
            'Recession rate (cm/s)': ['mean', 'std', 'count'],
            'Recession rate error (cm/s)': [pd_norm],
            'Evaporation rate (Torr-L/s)': ['mean', 'std'],
            'Evaporation rate lb (Torr-L/s)': ['mean', 'min'],
            'Evaporation rate ub (Torr-L/s)': ['mean', 'max'],
            'Sample area (cm^2)': ['mean'],
            'Peak pressure (Torr)': ['mean', 'std', 'count'],
            'Graphite sublimation rate (Torr-L/s/m^2)': ['mean', 'std', 'count'],
            'Graphite sublimation rate error (Torr-L/s/m^2)': [pd_norm]
        })


    agg_df.fillna(0., inplace=True)
    nn = np.stack([
        agg_df['Recession rate (cm/s)']['count'] - 1,
        np.ones_like(agg_df['Recession rate (cm/s)']['count'])
    ]).T

    tval_s = t.ppf(1 - 0.5 * 0.05, np.max(nn, axis=1))  # 68 % error
    tval_r = t.ppf(1 - 0.5 * 0.32, np.max(nn, axis=1))  # 68 % error

    yerr_s = np.array([
        (agg_df['Evaporation rate (Torr-L/s)']['mean'] - agg_df['Evaporation rate lb (Torr-L/s)']['mean'])/ agg_df['Sample area (cm^2)']['mean'] * 1E4,
        (agg_df['Evaporation rate ub (Torr-L/s)']['mean'] - agg_df['Evaporation rate (Torr-L/s)']['mean'])/ agg_df['Sample area (cm^2)']['mean'] * 1E4
    ]) * tval_r

    yerr_r = np.array([
        (agg_df['Recession rate error (cm/s)']['pd_norm']
         + agg_df['Recession rate (cm/s)']['std']) * tval_r / np.sqrt(
            agg_df['Recession rate (cm/s)']['count']
        )
    ])

    yerr_g = np.array([
        (agg_df['Graphite sublimation rate error (Torr-L/s/m^2)']['pd_norm']
         + agg_df['Graphite sublimation rate (Torr-L/s/m^2)']['std']) * tval_r / np.sqrt(
            agg_df['Graphite sublimation rate (Torr-L/s/m^2)']['count']
        )
    ])

    ch3_outgassing = estimate_hydrocarbon_outgassing(peak_pressure=agg_df['Peak pressure (Torr)']['mean'].values)
    ch3_outgassing_err = 0.5 * ch3_outgassing



    axes[0].errorbar(
        agg_df['Baking time (min)']['first'], 1E4 * agg_df['Evaporation rate (Torr-L/s)']['mean'] / agg_df['Sample area (cm^2)']['mean'],
        yerr=yerr_s,
        c=colors_dr[0],
        marker=markers_dr[0], lw=1.5, ls='none',
        # mec='k',
        mew=1.25, capsize=3.5, capthick=1.25, ecolor=colors_dr[0], fillstyle='none',
        label='Matrix'
    )

    axes[0].errorbar(
        agg_df['Baking time (min)']['first'],
        agg_df['Graphite sublimation rate (Torr-L/s/m^2)']['mean'],
        yerr=yerr_g,
        c=colors_dr[0],
        marker=markers_dr[0], lw=1.5, ls='none',
        # mec='k',
        mew=1.25, capsize=3.5, capthick=1.25, ecolor=colors_dr[0], fillstyle='left',
        label='Graphite'
    )

    axes[0].errorbar(
        agg_df['Baking time (min)']['first'],
        ch3_outgassing,
        yerr=ch3_outgassing_err,
        c=colors_dr[0],
        marker=markers_dr[0], lw=1.5, ls='none',
        # mec='k',
        mew=1.25, capsize=3.5, capthick=1.25, ecolor=colors_dr[0], fillstyle='full',
        label='(CH$_{\\mathrm{3}})$'
    )


    axes[1].errorbar(
        agg_df['Baking time (min)']['first'], agg_df['Recession rate (cm/s)']['mean'],
        yerr=yerr_r,
        c=colors_dr[1],
        marker=markers_dr[1], lw=1.5, ls='none',
        # mec='k',
        mew=1.25, capsize=3.5, capthick=1.25, ecolor=colors_dr[1], fillstyle='none'
    )

    axes[0].set_ylabel('Torr-L/s/m$^{\mathregular{2}}$')
    axes[0].set_ylim(1E0, 1E6)
    axes[0].set_title('Outgassing')


    axes[1].set_xlabel('Baking time [min]')
    axes[1].set_ylabel('cm/s')
    axes[1].set_ylim(0, 1.4)
    axes[1].set_title('Recession rate')

    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

    for ax in axes:
        ax.set_xlim(0, 80)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))


    axes[0].legend(loc='lower right', frameon=True, fontsize=9)
    # axes[0].legend(bbox_to_anchor=(1.05, 1),
    #                loc='upper left', borderaxespad=0., fontsize=9)

    fig_dr.savefig(os.path.join(base_path, 'baking_time_scan_20231126.png'), dpi=600)
    fig_dr.savefig(os.path.join(base_path, 'baking_time_scan_20231126.pdf'), dpi=600)
    fig_dr.savefig(os.path.join(base_path, 'baking_time_scan_20231126.svg'), dpi=600)
    fig_dr.savefig(os.path.join(base_path, 'baking_time_scan_20231126.eps'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
