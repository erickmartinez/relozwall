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

base_path = r'Documents/ucsd/Postdoc/research/data/bending_tests'

firing_csv = r'Documents/ucsd/Postdoc/research/data/firing_tests/merged_db.xlsx'
laser_power_dir = r'Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN\laser_output'
bending_xlsx = 'bending_strength_vs_matrix_content.xlsx'

beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707

selected_ids = np.array([
    'R4N03', 'R4N125', 'R4N127', 'R4N131', 'R4N132', 'R4N133', 'R4N134', 'R4N135', 'R4N136', 'R4N137',
    'R4N138', 'R4N139'
])

laser_power_settings = [60, 100]
bending_xlsx = 'bending_strength_vs_matrix_content.xlsx'


def mean_err(x):
    return np.linalg.norm(x) / len(x)


def std_err(x):
    n = len(x)
    return np.std(x, ddof=1) / np.sqrt(n) * t.ppf(1 - 0.5 * 0.05, n - 1)


def normalize_path(the_path):
    global platform_system, drive_path
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    return os.path.join(drive_path, the_path)


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)


def mean_err(x):
    return np.linalg.norm(x) / len(x)


def std_err(x):
    n = len(x)
    return np.std(x, ddof=1) / np.sqrt(n) * t.ppf(1 - 0.5 * 0.05, n - 1)


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


def normalize_path(the_path):
    global platform_system, drive_path
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    return os.path.join(drive_path, the_path)


def main():
    global base_path, firing_csv, laser_power_dir, selected_ids, laser_power_settings
    global bending_xlsx
    if platform.system() != 'Windows':
        base_path = normalize_path(base_path)
        firing_csv = normalize_path(firing_csv)
        laser_power_dir = normalize_path(laser_power_dir)

    bending_xlsx = os.path.join(base_path, bending_xlsx)
    use_cols: list = [
        'ROW', 'Sample code', 'Filler (wt %)', 'Binder (wt %)',
        'Power percent setting (%)', 'Irradiation time (s)', 'Sample diameter (cm)',
        'Recession rate (cm/s)', 'Recession rate error (cm/s)'
    ]
    firing_df = pd.read_excel(firing_csv, sheet_name='Laser tests')
    firing_df = firing_df[firing_df['Sample code'].isin(selected_ids)]
    firing_df = firing_df[firing_df['Power percent setting (%)'].isin(laser_power_settings)]
    firing_df = firing_df[use_cols]
    firing_df[use_cols[2::]] = firing_df[use_cols[2::]].apply(pd.to_numeric)
    firing_df['ROW'] = firing_df['ROW'].apply(pd.to_numeric)
    firing_df.set_index('ROW', inplace=True)
    firing_df['Matrix content (wt %)'] = firing_df['Binder (wt %)'] + firing_df['Filler (wt %)']
    firing_df.drop(columns=['Filler (wt %)', 'Binder (wt %)'])

    laser_power_mapping = map_laser_power_settings()
    lp = np.array([laser_power_mapping[int(x)] for x in firing_df['Power percent setting (%)']])
    sd = firing_df['Sample diameter (cm)'].values
    sa = 0.25 * np.pi * np.power(sd, 2.)
    av = gaussian_beam_aperture_factor(beam_radius, sd)
    firing_df['Laser power [MW/m2]'] = lp * av / sa / 100.

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

    """
    Load bending data
    """
    bending_df = pd.read_excel(os.path.join(base_path, bending_xlsx), sheet_name=0)
    bending_df.drop(columns='Sample ID', inplace=True)
    bending_df = bending_df.apply(pd.to_numeric)
    bending_agg_df = bending_df.groupby(['Matrix wt %']).agg(
        ['mean', std_err, mean_err]
    )

    print(bending_agg_df.columns)

    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 6)

    fig2, ax3 = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig2.set_size_inches(4., 3.)

    cax1 = 'tab:green'
    cax2 = 'tab:red'

    ax1 = axes[0]
    ax2 = ax1.twinx()

    matrix_wt_pct = bending_agg_df.index
    mean_force = bending_agg_df['Fracture force (N)']['mean']
    mean_force_err = bending_agg_df['Fracture force err (N)']['mean_err']
    mean_strength = bending_agg_df['Flexural strength (KPa)']['mean']
    mean_strength_err = bending_agg_df['Flexural strength err (KPa)']['mean_err']

    mean_strength_df = pd.DataFrame(data={
        'Matrix content (wt %)': matrix_wt_pct,
        'Matrix strength mean (KPa)': mean_strength,
        'Matrix strength mean error (KPa)': mean_strength_err
    }).reset_index(drop=True)

    ax1.errorbar(
        matrix_wt_pct, mean_force, yerr=mean_force_err, marker='o', ms=9, mew=1.25, mfc='none',
        capsize=2.75, elinewidth=1.25, lw=1.5, c=cax1
    )

    ax2.errorbar(
        matrix_wt_pct, mean_strength, yerr=mean_strength_err, marker='s', ms=9, mew=1.25, mfc='none',
        capsize=2.75, elinewidth=1.25, lw=1.5, c=cax2
    )

    ax1.set_xlabel('Matrix wt %')
    ax1.set_ylabel('Load (N)', color=cax1)
    ax2.set_ylabel('Bending strength (KPa)', color=cax2)
    ax1.set_title('Breaking load')

    ax1.tick_params(axis='y', labelcolor=cax1)
    ax2.tick_params(axis='y', labelcolor=cax2)

    ax1.set_xlim(2.5, 27.5)

    ax1.set_ylim(0, 4)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    ax2.set_ylim(0, 250)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(25))

    colors = ['C0', 'C1']
    markers = ['o', 's']

    output_df = pd.DataFrame(columns=[
        'Heat load (MW/m^2)', 'Matrix content (wt %)', 'Mean recession rate (cm/s)',
        'Mean recession rate error (cm/s)', 'Standard recession rate error (cm/s)',
        '# points', 'Recession rate min (cm/s)', 'Recession rate max (cm/s)'
    ])

    for i, lps in enumerate(laser_power_settings):
        df = firing_df[firing_df['Power percent setting (%)'] == lps].groupby(['Matrix content (wt %)']).agg(
            {
                'Recession rate (cm/s)': ['mean', 'std', std_err, 'count', 'min', 'max'],
                'Recession rate error (cm/s)': [mean_err],
                'Laser power [MW/m2]': ['mean']
            }
        )

        heat_load = round(df['Laser power [MW/m2]']['mean'].mean() / 5) * 5
        print(df[['Recession rate (cm/s)']])

        new_df = pd.DataFrame(data={
            'Heat load (MW/m^2)': df['Laser power [MW/m2]']['mean'],
            'Matrix content (wt %)': df.index,
            'Mean recession rate (cm/s)': df['Recession rate (cm/s)']['mean'],
            'Mean recession rate error (cm/s)': df['Recession rate error (cm/s)']['mean_err'],
            'Standard recession rate error (cm/s)': df['Recession rate (cm/s)']['std_err'],
            "# points": df['Recession rate (cm/s)']['count'],
            'Recession rate min (cm/s)': df['Recession rate (cm/s)']['min'],
            'Recession rate max (cm/s)': df['Recession rate (cm/s)']['max'],
        }).reset_index(drop=True)

        fb_df = pd.merge(new_df, mean_strength_df, how='inner', on=['Matrix content (wt %)'])
        print(fb_df)

        output_df = pd.concat([new_df, output_df])

        axes[1].errorbar(
            df.index, df['Recession rate (cm/s)']['mean'],
            yerr=df['Recession rate (cm/s)']['std_err'],
            marker=markers[i], ms=9, mew=1.25, mfc='none',
            capsize=2.75, elinewidth=1.25, lw=1.5, c=colors[i],
            label=f'{heat_load:.0f} MW/m$^{{ \\mathregular{{2}} }}$'
        )

        ebc = mpl.colors.to_rgba(colors[i], 0.5)
        ax3.errorbar(
            fb_df['Matrix strength mean (KPa)'], fb_df['Mean recession rate (cm/s)'],
            xerr=fb_df['Matrix strength mean error (KPa)'],
            yerr=fb_df['Standard recession rate error (cm/s)'],
            marker=markers[i], ms=9, mew=1.25, mfc='none',
            capsize=2.75, elinewidth=1.25, lw=1.5, c=colors[i],
            ecolor=ebc,
            label=f'{heat_load:.0f} MW/m$^{{ \\mathregular{{2}} }}$'
        )

    axes[1].set_xlabel('Matrix content (wt %)')
    axes[1].set_ylabel('cm/s')

    ax3.set_xlabel('Bending strength (KPa)')
    ax3.set_ylabel('cm/s')
    ax3.set_title('Recession rate')

    ax3.set_xlim(25, 225)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(25))

    ax3.set_ylim(0., 0.7)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax3.legend(
        loc='upper right', frameon=True
    )

    for ax in axes:
        ax.set_xlim(2.5, 27.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))

    axes[1].set_ylim(0., 0.7)
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    axes[1].set_title('Recession rate')

    axes[1].legend(
        loc='upper right', frameon=True
    )

    fig.savefig(os.path.join(base_path, 'fig_matrix_content.png'), dpi=600)
    fig.savefig(os.path.join(base_path, 'fig_matrix_content.pdf'), dpi=600)

    fig2.savefig(os.path.join(base_path, 'fig_recession_vs_bending_strength.png'), dpi=600)
    fig2.savefig(os.path.join(base_path, 'fig_recession_vs_bending_strength.pdf'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
