import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
from matplotlib import ticker
from data_processing.utils import get_experiment_params, lighten_color
from scipy.stats.distributions import t

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests'
laser_power_dir = 'MATERIAL_SCAN\laser_output'
merged_db_csv = 'merged_db.csv'
output_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\DPP 2023\figures'

samples = [
    {'sample_id': 'R4N04', 'label': 'GC spheres (10% binder)', 'material': 'Glassy carbon', 'marker': 'o', 'mfc': 'none', 'color': 'navy', 'ax': 0},
    # {'sample_id': 'R4N64', 'label': 'GC spheres (12.5% filler)', 'material': 'Glassy carbon', 'marker': 's', 'mfc': 'none'},
    {'sample_id': 'R4N85', 'label': 'GC spheres (2% binder)', 'material': 'Glassy carbon', 'marker': 's', 'mfc': 'none', 'color': 'royalblue', 'ax': 0},
    {'sample_id': 'R4N23', 'label': 'Mineral graphite', 'material': 'Graphite', 'marker': 'h', 'mfc': 'none', 'color': 'orange', 'ax': 1},
    {'sample_id': 'R4N88', 'label': 'POCO graphite', 'material': 'Graphite', 'marker': 'd', 'mfc': 'none', 'color': 'saddlebrown', 'ax': 1},
    {'sample_id': 'R4N20', 'label': 'SiNx spheres', 'material': 'SiNx', 'marker': '^', 'mfc': 'none', 'color': 'blueviolet','ax': 3},
    {'sample_id': 'R4N21', 'label': 'SiC spheres', 'material': 'SiC', 'marker': 'v', 'mfc': 'none', 'color': 'magenta','ax': 3},
    {'sample_id': 'R4N22', 'label': 'GC spheres (10% binder)', 'material': 'Glassy carbon', 'marker': 'o', 'mfc': 'none', 'color': 'royalblue','ax': 0},
    {'sample_id': 'R4N24', 'label': 'hBN granules', 'material': 'hBN', 'marker': '<', 'mfc': 'none', 'color': 'lightseagreen','ax': 2},
    {'sample_id': 'R4N66', 'label': 'WNiFe', 'material': 'Tungsten', 'marker': '>', 'mfc': 'none', 'color': 'red','ax': 4},
    {'sample_id': 'R4N89', 'label': 'Boron granules', 'material': 'Boron', 'marker': 'D', 'mfc': 'none', 'color': 'green','ax': 2},
    {'sample_id': 'R4N99', 'label': 'Boron granules', 'material': 'Boron', 'marker': 'D', 'mfc': 'none', 'color': 'green','ax': 2},
]

poor_gc_ids = ['R4N04', 'R4N22']

beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707


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


def load_data():
    laser_test_df = pd.read_csv(
        os.path.join(base_dir, merged_db_csv),
        usecols=[
            'Sample code', 'Power percent setting (%)', 'Irradiation time (s)',
            'Recession rate (cm/s)', 'Recession rate error (cm/s)', 'Sample diameter (cm)'  # , 'Ignore'
        ]
    )
    # print(laser_test_df)
    column_names_firing_df = laser_test_df.columns
    laser_test_df[column_names_firing_df[1:]] = laser_test_df[column_names_firing_df[1:]].apply(pd.to_numeric)
    # laser_test_df = laser_test_df[laser_test_df['Irradiation time (s)'] < 10]
    # laser_test_df = laser_test_df[laser_test_df['Ignore'] == 0]
    samples_ids = [v['sample_id'] for v in samples]
    laser_test_df = laser_test_df[laser_test_df['Sample code'].isin(samples_ids)]
    laser_test_df = laser_test_df[laser_test_df['Power percent setting (%)'] > 5]

    laser_test_df['Label'] = ['lbl' for i in range(len(laser_test_df))]
    laser_test_df['Material'] = ['mat' for i in range(len(laser_test_df))]
    laser_test_df['Order'] = [0 for i in range(len(laser_test_df))]
    laser_test_df['Marker'] = ['o' for i in range(len(laser_test_df))]
    laser_test_df['mfc'] = ['none' for i in range(len(laser_test_df))]
    laser_test_df['color'] = ['none' for i in range(len(laser_test_df))]
    laser_test_df['ax'] = ['none' for i in range(len(laser_test_df))]

    for k, s in enumerate(samples):
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'Label'] = s['label']
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'Order'] = k
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'Marker'] = s['marker']
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'mfc'] = s['mfc']
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'color'] = s['color']
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'Material'] = s['material']
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'ax'] = s['ax']

    laser_test_df.sort_values(by=['Order'], inplace=True)
    laser_test_df.reset_index(drop=True, inplace=True)
    return laser_test_df


def main():
    df = load_data()
    df.to_csv(path_or_buf=os.path.join(output_dir, 'dataset.csv'), index=False)
    labels = df['Label'].unique()
    # not_h2_df = df[df['h2'] == False]
    n_labels = len(labels)
    print('Material column exists:', 'Material' in df.columns)

    laser_power_mapping = map_laser_power_settings()
    power_settings = np.array([int(k) for k in laser_power_mapping.keys()])
    norm = mpl.colors.Normalize(vmin=power_settings.min(), vmax=power_settings.max())

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

    fig, axes = plt.subplots(ncols=1, nrows=5, gridspec_kw=dict(hspace=0), sharex=True)
    fig.set_size_inches(4.0, 6.25)
    # fig.subplots_adjust(hspace=0)

    # # inset axes....
    # x1, x2, y1, y2 = 5, 50, 0, 0.12  # subregion of the original image
    # axins = ax.inset_axes(
    #     [0.13, 0.57, 0.45, 0.40],
    #     xlim=(x1, x2), ylim=(y1, y2))#, xticklabels=[], yticklabels=[])

    # ax.indicate_inset_zoom(axins, edgecolor="black")


    def sumsq(x):
        return np.linalg.norm(np.array(x))

    for i, m in enumerate(labels):
        df2 = df[df['Label'] == m]
        df2 = df2.groupby('Power percent setting (%)').agg({
            'Sample diameter (cm)': ['mean', 'std'],
            'Recession rate (cm/s)': ['mean', 'std', 'count', 'min', 'max'],
            'Recession rate error (cm/s)': [sumsq],
            'Marker': ['first'],
            'mfc': ['first'],
            'color': ['first'],
            'Material': ['first'],
            'ax': ['first']
        })

        marker = df2['Marker']['first'].values
        marker = marker[0]
        mfc = df2['mfc']['first'].values[0]
        color = df2['color']['first'].values[0]
        material = df2['Material']['first'].values[0]
        ax_idx = df2['ax']['first'].values[0]
        ax = axes[ax_idx]
        print(f'Label: {m}, material: {material}')

        # df2.fillna(0, inplace=True)
        # df2['Loss Rate Error (cm/s)']['mean'] = df2['Loss Rate Error (cm/s)']['mean'].fillna(0)
        # print(df2['Recession rate error (cm/s)']['mean'])

        laser_power_setting = list(df2.index.values)
        laser_power = np.array([laser_power_mapping[v] for v in laser_power_setting])
        sample_diameter = df2['Sample diameter (cm)']['mean'].values
        sample_area = 0.25 * np.pi * sample_diameter ** 2.0
        aperture_factor = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.5 * sample_diameter)
        # print(f'aperture_factor:', aperture_factor)
        # print(f'laser_power:', laser_power)
        # print(f'sample_diameter', sample_diameter)
        incident_heat_load = aperture_factor * laser_power / sample_area / 100.0
        recession_rate = df2['Recession rate (cm/s)']['mean'].values
        recession_rate_min = df2['Recession rate (cm/s)']['min'].values
        recession_rate_max = df2['Recession rate (cm/s)']['max'].values
        recession_rate_std = df2['Recession rate (cm/s)']['std'].values
        recession_rate_n = df2['Recession rate (cm/s)']['count'].values
        recession_rate_err = df2['Recession rate error (cm/s)']['sumsq'].values

        confidence = 0.67
        alpha = 1. - confidence
        yerr = np.empty((2, len(recession_rate)), dtype=np.float64)
        for j, nn in enumerate(recession_rate_n):
            # if nn > 1:
            #     tval = t.ppf(1 - 0.5*alpha, nn - 1)
            #     err = (recession_rate_err[j] + recession_rate_std[j]) * tval / np.sqrt(nn)
            #     recession_rate_err[j] = err
            if nn > 1:
                yerr[:, j] = (recession_rate[j] - recession_rate_min[j], recession_rate_max[j] - recession_rate[j])
            else:
                yerr[:, j] = recession_rate_err[j]
        # yerr = (np.min(np.stack([recession_rate_err, recession_rate]).T, axis=1), recession_rate_err)

        # if material == 'Boron':
        #     yerr=None

        ax.errorbar(
            incident_heat_load, recession_rate, yerr=yerr,
            marker=marker, ms=9, mew=1.25, label=f'{m}', mfc=mfc,
            capsize=2.75, elinewidth=1.25, lw=1.5, c=color, ecolor=color
        )

        # if i > 1:
        #     axins.errorbar(
        #         incident_heat_load, recession_rate, yerr=yerr,
        #         marker=marker, ms=9, mew=1.0, label=f'{m}', mfc=mfc,
        #         capsize=2.75, elinewidth=1.0, lw=1.25, c=color
        #     )

    # axes[4].set_xlabel(r'Heat load [MW/m$^{\mathregular{2}}$]')
    for ax in axes:
        # ax.set_ylabel('cm/s')
        ax.set_xlim(5, 50)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5.0))
        ax.legend(loc='upper left', frameon=True, fontsize=9)
        ax.tick_params(which='both', axis='y', labelright=False, right=True, direction='out')

    axes[0].set_title('Recession rate [cm/s]')
    axes[-1].set_xlabel(r'Heat load [MW/m$^{\mathregular{2}}$]')

    axes[0].set_ylim(-0.1, 1.0)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    axes[1].set_ylim(-0.02, 0.14)
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.04))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    axes[2].set_ylim(-0.03, 0.21)
    axes[2].yaxis.set_major_locator(ticker.MultipleLocator(0.06))
    axes[2].yaxis.set_minor_locator(ticker.MultipleLocator(0.03))

    axes[3].set_ylim(-0.005, 0.025)
    axes[3].yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    axes[3].yaxis.set_minor_locator(ticker.MultipleLocator(0.005))

    axes[4].set_ylim(0, 0.014)
    axes[4].yaxis.set_major_locator(ticker.MultipleLocator(0.004))
    axes[4].yaxis.set_minor_locator(ticker.MultipleLocator(0.002))
    # axins.yaxis.set_major_locator(ticker.MultipleLocator(0.03))
    # axins.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    # axins.xaxis.set_major_locator(ticker.MultipleLocator(10))
    # axins.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    # axins.tick_params(axis='y', direction='in', labelsize=10, right=True)
    # axins.tick_params(axis='x', direction='in', labelsize=10)
    # axins.tick_params(axis='x', labelbottom=False)


    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8)
    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'recession_rate_vs_laser_power' + '.svg'), dpi=600)
    fig.savefig(os.path.join(output_dir, 'recession_rate_vs_laser_power' + '.png'), dpi=600)
    fig.savefig(os.path.join(output_dir, 'recession_rate_vs_laser_power' + '.pdf'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
