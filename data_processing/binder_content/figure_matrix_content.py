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

aquarium_ids = np.array(['R4N147', 'R4N148', 'R4N154'])

laser_power_settings = [60, 100]
bending_xlsx = 'bending_strength_vs_matrix_content.xlsx'
weibull_gc_csv = 'weibull_fit_glassy_carbon.csv'


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



def inlb2t(x):
    return x / 3.


def inlt2b(x):
    return x * 3.


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
        'Density (g/cm^3)',
        'Recession rate (cm/s)', 'Recession rate error (cm/s)',
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

    firing_aquarium_df = pd.read_excel(firing_csv, sheet_name='Laser tests')
    firing_aquarium_df = firing_aquarium_df[firing_aquarium_df['Sample code'].isin(aquarium_ids)]

    firing_aquarium_df = firing_aquarium_df[firing_aquarium_df['Power percent setting (%)'].isin(laser_power_settings)]
    firing_aquarium_df = firing_aquarium_df[use_cols]
    firing_aquarium_df[use_cols[2::]] = firing_aquarium_df[use_cols[2::]].apply(pd.to_numeric)
    firing_aquarium_df['ROW'] = firing_aquarium_df['ROW'].apply(pd.to_numeric)
    firing_aquarium_df.set_index('ROW', inplace=True)
    firing_aquarium_df['Matrix content (wt %)'] = firing_aquarium_df['Binder (wt %)'] + firing_aquarium_df['Filler (wt %)']
    firing_aquarium_df.drop(columns=['Filler (wt %)', 'Binder (wt %)'])

    laser_power_mapping = map_laser_power_settings()
    lp = np.array([laser_power_mapping[int(x)] for x in firing_df['Power percent setting (%)']])
    lp_aquarium = np.array([laser_power_mapping[int(x)] for x in firing_aquarium_df['Power percent setting (%)']])
    sd = firing_df['Sample diameter (cm)'].values
    sa = 0.25 * np.pi * np.power(sd, 2.)
    av = gaussian_beam_aperture_factor(beam_radius, 0.5*sd)
    firing_df['Laser power [MW/m2]'] = lp * av / sa / 100.

    sd_sic = firing_aquarium_df['Sample diameter (cm)'].values
    sa_sic = 0.25 * np.pi * np.power(sd_sic, 2.)
    av_sic = gaussian_beam_aperture_factor(beam_radius, sd_sic)
    firing_aquarium_df['Laser power [MW/m2]'] = lp_aquarium * av_sic / sa_sic / 100.

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

    """
    Load bending data
    """
    bending_df = pd.read_excel(os.path.join(base_path, bending_xlsx), sheet_name=0)
    bending_df = bending_df[bending_df['Q'] == 1]
    bending_df.drop(columns='Sample ID', inplace=True)
    bending_df = bending_df.apply(pd.to_numeric)
    bending_agg_df = bending_df.groupby(['Matrix wt %']).agg(
        ['mean', std_err, mean_err]
    )

    bending_aquarium_df = pd.read_excel(os.path.join(base_path, bending_xlsx), sheet_name='Matrix carbon')
    bending_aquarium_df.drop(columns='Sample ID', inplace=True)
    bending_aquarium_df = bending_aquarium_df.apply(pd.to_numeric)
    bending_aquarium_agg_df = bending_aquarium_df.groupby(['Matrix wt %']).agg(
        ['mean', std_err, mean_err]
    )

    # print(bending_agg_df.columns)

    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 6)

    fig_s_ac, axes_s_ac = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig_s_ac.set_size_inches(4.5, 6)

    fig2, axes3 = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig2.set_size_inches(4.5, 6.5)

    ax3, ax5 = axes3[0], axes3[1]

    cax1 = 'tab:green'
    cax2 = 'tab:red'

    ax1 = axes[0]
    ax2 = ax1.twinx()
    ax2_s_ac = axes_s_ac[0].twinx()
    ax4 = axes3[0].secondary_xaxis('top', functions=(inlt2b, inlb2t))
    ax6 = axes3[1].secondary_xaxis('top', functions=(inlt2b, inlb2t))

    matrix_wt_pct = bending_agg_df.index
    mean_force = bending_agg_df['Fracture force (N)']['mean']
    mean_force_err = bending_agg_df['Fracture force err (N)']['mean_err']
    mean_strength = bending_agg_df['Flexural strength (KPa)']['mean']
    mean_strength_err = bending_agg_df['Flexural strength err (KPa)']['mean_err']

    matrix_wt_pct_aquarium = bending_aquarium_agg_df.index
    mean_force_aquarium = bending_aquarium_agg_df['Fracture force (N)']['mean']
    mean_force_aquarium_err = bending_aquarium_agg_df['Fracture force err (N)']['mean_err']
    mean_strength_aquarium = bending_aquarium_agg_df['Flexural strength (KPa)']['mean']
    mean_strength_aquarium_err = bending_aquarium_agg_df['Flexural strength err (KPa)']['mean_err']

    mean_strength_df = pd.DataFrame(data={
        'Matrix content (wt %)': matrix_wt_pct,
        'Matrix mean breaking load (N)': mean_force,
        'Matrix mean breaking load error (N)': mean_force_err,
        'Matrix strength mean (KPa)': mean_strength,
        'Matrix strength mean error (KPa)': mean_strength_err,
        'Diameter mean (mm)': bending_agg_df['Diameter (mm)']['mean'],
        'Diameter err (mm)': bending_agg_df['Diameter err (mm)']['mean_err']
    }).reset_index(drop=True)

    mean_strength_aquarium_df = pd.DataFrame(data={
        'Matrix content (wt %)': matrix_wt_pct_aquarium,
        'Matrix mean breaking load (N)': mean_force_aquarium,
        'Matrix mean breaking load error (N)': mean_force_aquarium_err,
        'Matrix strength mean (KPa)': mean_strength_aquarium,
        'Matrix strength mean error (KPa)': mean_strength_aquarium_err,
        'Diameter mean (mm)': bending_aquarium_agg_df['Diameter (mm)']['mean'],
        'Diameter err (mm)': bending_aquarium_agg_df['Diameter err (mm)']['mean_err']
    }).reset_index(drop=True)

    """
    Load weibull data for glassy carbon
    Note: This approach does not work
    """
    # weibull_gc_df = pd.read_csv(os.path.join(base_path, 'weibull', weibull_gc_csv)).apply(pd.to_numeric)
    # weibull_gc_df.rename(columns={'Matrix wt %': 'Matrix content (wt %)'}, inplace=True)
    # print(weibull_gc_df)
    # mean_strength_df = pd.merge(left=mean_strength_df, right=weibull_gc_df, on=['Matrix content (wt %)'])

    # ratio, tensile_strength, tensile_strength_error = bending_to_tensile(
    #     bending_strength=mean_strength_df['Matrix strength mean (KPa)'],
    #     shape_parameter=mean_strength_df['Shape param'],
    #     dbs=0.,#mean_strength_df['Matrix strength mean error (KPa)'],
    #     dsp=0.#mean_strength_df['Shape param SE']
    # )

    mean_strength_df['FT_estimate (KPa)'] = mean_strength_df['Matrix strength mean (KPa)'] / 3.
    mean_strength_df['FT_estimate error (KPa)'] = mean_strength_df['Matrix strength mean error (KPa)'] / 3.

    mean_strength_aquarium_df['FT_estimate (KPa)'] = mean_strength_aquarium_df['Matrix strength mean (KPa)'] / 3.
    mean_strength_aquarium_df['FT_estimate error (KPa)'] = mean_strength_aquarium_df['Matrix strength mean error (KPa)'] / 3.
    print('***** MEAN STRENGTH Glassy Carbon *******')
    print(mean_strength_df)

    ax1.errorbar(
        matrix_wt_pct, mean_force, yerr=mean_force_err, marker='o', ms=9, mew=1.25, mfc='none',
        capsize=2.75, elinewidth=1.25, lw=1.5, c=cax1
    )

    axes_s_ac[0].errorbar(
        matrix_wt_pct_aquarium, mean_force_aquarium, yerr=mean_force_aquarium_err, marker='o', ms=9, mew=1.25, mfc='none',
        capsize=2.75, elinewidth=1.25, lw=1.5, c=cax1
    )

    ax2.errorbar(
        matrix_wt_pct, mean_strength, yerr=mean_strength_err, marker='s', ms=9, mew=1.25, mfc='none',
        capsize=2.75, elinewidth=1.25, lw=1.5, c=cax2
    )

    ax2_s_ac.errorbar(
        matrix_wt_pct_aquarium, mean_strength_aquarium, yerr=mean_strength_aquarium_err, marker='s', ms=9, mew=1.25, mfc='none',
        capsize=2.75, elinewidth=1.25, lw=1.5, c=cax2
    )

    ax1.set_xlabel('Matrix wt %')
    ax1.set_ylabel('Load (N)', color=cax1)
    ax2.set_ylabel('$f_{\\mathrm{b}}$ (kPa)', color=cax2)
    ax1.set_title('Glassy carbon')

    axes_s_ac[0].set_title('Activated Carbon')
    axes_s_ac[0].set_xlabel('Matrix wt %')
    axes_s_ac[0].set_ylabel('Load (N)', color=cax1)
    ax2_s_ac.set_ylabel('$f_{\\mathrm{b}}$ (kPa)', color=cax2)

    ax1.tick_params(axis='y', labelcolor=cax1)
    ax2.tick_params(axis='y', labelcolor=cax2)

    axes_s_ac[0].tick_params(axis='y', labelcolor=cax1)
    ax2_s_ac.tick_params(axis='y', labelcolor=cax2)

    ax1.set_xlim(2.5, 27.5)
    axes[0].set_xlim(0, 22.5)
    axes[0].set_ylim(0, 25)
    ax2_s_ac.set_xlim(0, 27.5)

    ax1.set_ylim(0, 4)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    axes_s_ac[0].xaxis.set_major_locator(ticker.MultipleLocator(5))
    axes_s_ac[0].xaxis.set_minor_locator(ticker.MultipleLocator(2.5))

    ax2.set_ylim(0, 250)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(25))

    ax2_s_ac.set_ylim(0, 400)
    ax2_s_ac.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax2_s_ac.yaxis.set_minor_locator(ticker.MultipleLocator(50))

    colors = ['C0', 'C1']
    markers = ['o', 's']

    markers_ac = ['D', '^']

    output_df = pd.DataFrame(columns=[
        'Heat load (MW/m^2)', 'Matrix content (wt %)', 'Mean recession rate (cm/s)',
        'Mean recession rate error (cm/s)', 'Standard recession rate error (cm/s)',
        '# points', 'Recession rate min (cm/s)', 'Recession rate max (cm/s)',
    ])

    output_aquarium_df = pd.DataFrame(columns=[
        'Heat load (MW/m^2)', 'Matrix content (wt %)', 'Mean recession rate (cm/s)',
        'Mean recession rate error (cm/s)', 'Standard recession rate error (cm/s)',
        '# points', 'Recession rate min (cm/s)', 'Recession rate max (cm/s)',
    ])

    for i, lps in enumerate(laser_power_settings):
        df = firing_df[firing_df['Power percent setting (%)'] == lps].groupby(['Matrix content (wt %)']).agg(
            {
                'Recession rate (cm/s)': ['mean', 'std', std_err, 'count', 'min', 'max'],
                'Recession rate error (cm/s)': [mean_err],
                'Laser power [MW/m2]': ['mean'],
                'Density (g/cm^3)': ['mean']
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
            'Density mean (g/cm^3)': df['Density (g/cm^3)']['mean']
        }).reset_index(drop=True)

        fb_df = pd.merge(new_df, mean_strength_df, how='inner', on=['Matrix content (wt %)'])
        print(fb_df)

        output_df = pd.concat([new_df, output_df])

        """
        For Activated carbon
        """
        aquarium_df = firing_aquarium_df[firing_aquarium_df['Power percent setting (%)'] == lps].groupby(
            ['Matrix content (wt %)']).agg(
            {
                'Recession rate (cm/s)': ['mean', 'std', std_err, 'count', 'min', 'max'],
                'Recession rate error (cm/s)': [mean_err],
                'Laser power [MW/m2]': ['mean'],
                'Density (g/cm^3)': ['mean']
            }
        )

        heat_load_aquarium = round(aquarium_df['Laser power [MW/m2]']['mean'].mean() / 5) * 5
        print(aquarium_df[['Recession rate (cm/s)']])

        new_df = pd.DataFrame(data={
            'Heat load (MW/m^2)': aquarium_df['Laser power [MW/m2]']['mean'],
            'Matrix content (wt %)': aquarium_df.index,
            'Mean recession rate (cm/s)': aquarium_df['Recession rate (cm/s)']['mean'],
            'Mean recession rate error (cm/s)': aquarium_df['Recession rate error (cm/s)']['mean_err'],
            'Standard recession rate error (cm/s)': aquarium_df['Recession rate (cm/s)']['std_err'],
            "# points": aquarium_df['Recession rate (cm/s)']['count'],
            'Recession rate min (cm/s)': aquarium_df['Recession rate (cm/s)']['min'],
            'Recession rate max (cm/s)': aquarium_df['Recession rate (cm/s)']['max'],
            'Density mean (g/cm^3)': aquarium_df['Density (g/cm^3)']['mean']
        }).reset_index(drop=True)

        fb_aquarium_df = pd.merge(new_df, mean_strength_aquarium_df, how='inner', on=['Matrix content (wt %)'])
        fb_aquarium_df.sort_values(by=['FT_estimate (KPa)'], inplace=True)

        output_aquarium_df = pd.concat([new_df, output_aquarium_df])

        axes[1].errorbar(
            df.index, df['Recession rate (cm/s)']['mean'],
            yerr=df['Recession rate (cm/s)']['std_err'],
            marker=markers[i], ms=9, mew=1.25, mfc='none',
            capsize=2.75, elinewidth=1.25, lw=1.5, c=colors[i],
            label=f'{heat_load:.0f} MW/m$^{{ \\mathregular{{2}} }}$'
        )

        axes_s_ac[1].errorbar(
            aquarium_df.index, aquarium_df['Recession rate (cm/s)']['mean'],
            yerr=aquarium_df['Recession rate (cm/s)']['std_err']/3,
            marker=markers[i], ms=9, mew=1.25, mfc='none',
            capsize=2.75, elinewidth=1.25, lw=1.5, c=colors[i],
              label=f'{heat_load_aquarium:.0f} MW/m$^{{ \\mathregular{{2}} }}$'
        )

        ebc = mpl.colors.to_rgba(colors[i], 0.5)
        ax3.errorbar(
            fb_df['FT_estimate (KPa)'], fb_df['Mean recession rate (cm/s)'],
            xerr=fb_df['FT_estimate error (KPa)'],
            yerr=fb_df['Standard recession rate error (cm/s)'],
            marker=markers[i], ms=9, mew=1.25, mfc='none',
            capsize=2.75, elinewidth=1.25, lw=1.5, c=colors[i],
            ecolor=ebc,
            label=f'{heat_load:.0f} MW/m$^{{ \\mathregular{{2}} }}$'
        )

        ax5.errorbar(
            fb_aquarium_df['FT_estimate (KPa)'], fb_aquarium_df['Mean recession rate (cm/s)'],
            xerr=fb_aquarium_df['FT_estimate error (KPa)'],
            yerr=fb_aquarium_df['Standard recession rate error (cm/s)']/3,
            marker=markers_ac[i], ms=9, mew=1.25, fillstyle='none',  # mfc='none',
            capsize=2.75, elinewidth=1.25, lw=1.5, c=colors[i],
            ecolor=ebc,
            label=f'{heat_load:.0f} MW/m$^{{ \\mathregular{{2}} }}$'
        )

    axes[1].set_xlabel('Matrix content (wt %)')
    axes[1].set_ylabel('cm/s')

    axes_s_ac[1].set_xlabel('Matrix content (wt %)')
    axes_s_ac[1].set_ylabel('cm/s')

    ax3.set_xlabel('$f_t$ (kPa)')
    ax4.set_xlabel('$f_b$ (kPa)')
    ax3.set_ylabel('Recession rate (cm/s)')
    ax3.set_title('Glassy carbon')
    axes_s_ac[1].set_title('Recession rate')

    ax5.set_title('Activated carbon')
    ax5.set_ylabel('Recession rate (cm/s)')

    """
    Glassy carbon
    """
    ax3.set_xlim(0, 80)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax3.set_ylim(0., 0.7)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    """
    Activated carbon
    """
    ax5.set_xlabel('$f_t$ (kPa)')
    ax6.set_xlabel('$f_b$ (kPa)')

    ax5.set_xlim(0, 80)
    ax5.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax5.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax5.set_ylim(0., 2.)
    ax5.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax5.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    for ax_ac in axes_s_ac:
        ax_ac.set_xlim(19, 26)
        ax_ac.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax_ac.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    axes_s_ac[1].set_ylim(0., 2.0)
    axes_s_ac[1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axes_s_ac[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    ax3.legend(
        loc='upper right', frameon=True
    )

    ax5.legend(
        loc='upper right', frameon=True
    )

    axes_s_ac[1].legend(
        loc='upper right', frameon=True
    )

    for ax in axes:
        ax.set_xlim(2.5, 27.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))

    axes[1].set_ylim(0., 0.8)
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    axes[1].set_title('Recession rate')

    axes[1].legend(
        loc='upper right', frameon=True
    )

    output_df = pd.merge(output_df, mean_strength_df, how='inner', on=['Matrix content (wt %)'])

    output_aquarium_df = pd.merge(output_aquarium_df, mean_strength_aquarium_df, how='inner', on=['Matrix content (wt %)'])

    output_df.to_csv(os.path.join(base_path, 'gc_recession_vs_ft.csv'), index=False)
    output_aquarium_df.to_csv(os.path.join(base_path, 'activated_carbon_recession_vs_ft.csv'), index=False)

    fig.savefig(os.path.join(base_path, 'fig_matrix_content.png'), dpi=600)
    fig.savefig(os.path.join(base_path, 'fig_matrix_content.pdf'), dpi=600)

    fig2.savefig(os.path.join(base_path, 'fig_recession_vs_bending_strength.png'), dpi=600)
    fig2.savefig(os.path.join(base_path, 'fig_recession_vs_bending_strength.pdf'), dpi=600)

    fig_s_ac.savefig(os.path.join(base_path, 'fig_activated_carbon_matrix_content.png'), dpi=600)

    mean_strength_df.to_csv(os.path.join(base_path, 'gc_mean_load_vs_matrix_content.csv'), index=False)
    mean_strength_aquarium_df.to_csv(os.path.join(base_path, 'ac_mean_load_vs_matrix_content.csv'), index=False)

    plt.show()


if __name__ == '__main__':
    main()
