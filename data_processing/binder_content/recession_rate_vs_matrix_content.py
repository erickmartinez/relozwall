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

save_path = r'Documents/ucsd/Postdoc/research/data/bending_tests'

beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707

selected_ids = np.array([
    'R4N125', 'R4N127', 'R4N131', 'R4N132', 'R4N133', 'R4N134', 'R4N135', 'R4N136', 'R4N137',
    'R4N138', 'R4N139'
])

laser_power_settings = [60, 100]

"""
OLD DATA
"""
old_data_csv = '../../data/binder_content_scan.csv'


def load_old_data():
    global old_data_csv
    df = pd.read_csv(old_data_csv)
    columns = df.columns
    df[columns[1::]] = df[columns[1::]].apply(pd.to_numeric)
    return df


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)


def mean_err(x):
    return np.linalg.norm(x) / len(x)


def std_err(x):
    return np.std(x, ddof=1) / np.sqrt(len(x)) * t.ppf(1 - 0.5 * 0.05, 10000)


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
    global save_path, bending_xlsx
    if platform.system() != 'Windows':
        base_path = normalize_path(base_path)
        firing_csv = normalize_path(firing_csv)
        laser_power_dir = normalize_path(laser_power_dir)
        save_path = normalize_path(save_path)

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
    Load old data
    """
    old_df = load_old_data()
    old_df['Matrix content (wt %)'] = old_df['Binder content (%)'] + old_df['Filler content (%)']

    old_df = old_df.groupby('Matrix content (wt %)').agg({
        'Erosion Rate (cm/s)': ['mean', 'std', 'count', std_err, 'min', 'max'],
        'Erosion rate error (cm/s)': ['mean', mean_err]
    })

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.25, 3)

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
        })

        output_df = pd.concat([output_df, new_df])

        ax.errorbar(
            df.index, df['Recession rate (cm/s)']['mean'],
            yerr=df['Recession rate error (cm/s)']['mean_err'],
            marker=markers[i], ms=9, mew=1.25, mfc='none',
            capsize=2.75, elinewidth=1.25, lw=1.5, c=colors[i],
            label=f'{heat_load:.0f} MW/m$^{{ \\mathregular{{2}} }}$'
        )

    ax.errorbar(
        old_df.index, old_df['Erosion Rate (cm/s)']['mean'],
        yerr=old_df['Erosion rate error (cm/s)']['mean_err'],
        marker='D', ms=9, mew=1.25, fillstyle='bottom',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='tab:grey',
        label=f'40 MW/m$^{{ \\mathregular{{2}} }}$ (old)'
    )

    ax.set_xlabel('Matrix content (wt %)')
    ax.set_ylabel('Recession rate (cm/s)')

    ax.legend(
        loc='upper right', frameon=True
    )

    fig.savefig(os.path.join(save_path, 'recession_rate_vs_matrix_content.png'), dpi=600)
    output_df.to_csv(os.path.join(save_path, 'recession_rate_vs_matrix_content.csv'), index=False)

    plt.show()


if __name__ == '__main__':
    main()
