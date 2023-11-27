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

firing_csv = r'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/baking_time/recession_rate.csv'

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

    print(transmission_df)

    # load firing database
    firing_df = pd.read_csv(firing_csv)
    firing_df = firing_df[firing_df['ROW'].isin(row_ids)]

    confidence = 0.95
    alpha = 1. - confidence

    print(firing_df)
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
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

    fig_dr, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig_dr.set_size_inches(4., 5.)

    axes[0].set_yscale('log')

    colors_dr = ['C0', 'C1']
    markers_dr = ['o', 's']

    agg_df = merged_df.groupby(by=['Baking time (min)']).agg({
            'Baking time (min)': ['first'],
            'Recession rate (cm/s)': ['mean', 'std', 'count'],
            'Recession rate error (cm/s)': [pd_norm],
            'Evaporation rate (Torr-L/s)': ['mean', 'std'],
            'Evaporation rate lb (Torr-L/s)': ['mean', 'min'],
            'Evaporation rate ub (Torr-L/s)': ['mean', 'max'],
            'Sample area (cm^2)': ['mean']
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



    axes[0].errorbar(
        agg_df['Baking time (min)']['first'], 1E4 * agg_df['Evaporation rate (Torr-L/s)']['mean'] / agg_df['Sample area (cm^2)']['mean'],
        yerr=yerr_s,
        c=colors_dr[0],
        marker=markers_dr[0], lw=1.5, ls='none',
        # mec='k',
        mew=1.25, capsize=3.5, capthick=1.25, ecolor=colors_dr[0], fillstyle='none',
        # label=lbl
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
    axes[0].set_ylim(1E4, 1E6)
    axes[0].set_title('Sublimation rate')


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


    # axes[0].legend(loc='lower right', frameon=True, fontsize=10)

    fig_dr.savefig(os.path.join(base_path, 'baking_time_scan_20231126.png'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
