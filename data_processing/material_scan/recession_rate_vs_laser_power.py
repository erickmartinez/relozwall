import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
from matplotlib import ticker
from data_processing.utils import get_experiment_params

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN'
firing_db_csv = 'Firing Tests - Mass Loss - Sheet1.csv'
recipe_db_csv = 'Slurry - tests - Recipe 4.csv'
laser_power_dir = 'laser_output'
samples = [
    {'sample_id': 'R4N04', 'label': 'GC spheres', 'material': 'glassy carbon', 'thermal_conductivity': 0.0651,
     'h2': False, 'CTE': 3.78E-6},
    {'sample_id': 'R4N04-H2', 'label': 'GC spheres (H2)', 'material': 'glassy carbon', 'thermal_conductivity': 0.0651,
     'h2': True, 'CTE': 3.78E-6},
    {'sample_id': 'R4N20', 'label': 'SiNx spheres', 'material': 'SiNx', 'thermal_conductivity': 0.25, 'h2': False,
     'CTE': 3.2E-6},
    {'sample_id': 'R4N21', 'label': 'SiC spheres', 'material': 'SiC', 'thermal_conductivity': 1.2, 'h2': False,
     'CTE': 3.8E-6},
    {'sample_id': 'R4N22', 'label': 'GC spheres', 'material': 'glassy carbon', 'thermal_conductivity': 0.0651,
     'h2': False, 'CTE': 3.78E-6},
    {'sample_id': 'R4N22-H2', 'label': 'GC spheres (H2)', 'material': 'glassy carbon', 'thermal_conductivity': 0.0651,
     'h2': True, 'CTE': 3.78E-6},
    {'sample_id': 'R4N23', 'label': 'Graphite granules', 'material': 'Graphite', 'thermal_conductivity': 2.10465,
     'h2': False, 'CTE': 8.22E-6},
    {'sample_id': 'R4N24', 'label': 'hBN granules', 'material': 'hBN', 'thermal_conductivity': 1.741, 'h2': False,
     'CTE': 3.77E-5},
    {'sample_id': 'R4N46', 'label': 'hBN-coated GC', 'material': 'hBN-coated GC', 'thermal_conductivity': 1.2,
     'h2': False, 'CTE': 3.78E-6}
]

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


def merge_tables():
    firing_df = pd.read_csv(
        os.path.join(base_dir, firing_db_csv),
        usecols=[
            'Sample', 'Power percent setting (%)', 'Irradiation time (s)',
            'Erosion Rate (cm/s)', 'Loss Rate Error (cm/s)', 'Sample Diameter (cm)', 'Ignore'
        ]
    )
    print(firing_df)
    column_names_firing_df = firing_df.columns
    firing_df[column_names_firing_df[1:]] = firing_df[column_names_firing_df[1:]].apply(pd.to_numeric)
    firing_df = firing_df[firing_df['Irradiation time (s)'] < 10]
    firing_df = firing_df[firing_df['Ignore'] == 0]

    recipe_df = pd.DataFrame(data={
        'Sample': [s['sample_id'] for s in samples],
        'Label': [s['label'] for s in samples],
        'Material': [s['material'] for s in samples],
        'h2': [s['h2'] for s in samples],
        'Thermal conductivity (W/cm-K)': [s['thermal_conductivity'] for s in samples],
        'CTE (/K)': [s['CTE'] for s in samples]
    })
    print(recipe_df)
    # recipe_df = pd.read_csv(
    #     os.path.join(base_dir, recipe_db_csv),
    #     usecols=[
    #         'Sample code', 'Type of spheres'
    #     ]
    # )
    # recipe_df.rename(columns={'Sample code': 'Sample'}, inplace=True)
    # recipe_df = recipe_df[recipe_df['Sample'].isin(samples)]
    merged_df = recipe_df.merge(firing_df, how='left', on='Sample')
    return merged_df


if __name__ == '__main__':
    df = merge_tables()
    df.to_csv(os.path.join(base_dir, 'merged_sample_info.csv'), index=False)
    labels = df['Label'].unique()
    not_h2_df = df[df['h2'] == False]
    n_labels = len(labels)

    laser_power_mapping = map_laser_power_settings()
    power_settings = np.array([int(k) for k in laser_power_mapping.keys()])
    norm = mpl.colors.Normalize(vmin=power_settings.min(), vmax=power_settings.max())

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)
    markers = ['o', 's', '^', 'v', 'd', '<', '>', 'p']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    colors = plt.cm.rainbow(np.linspace(0, 1, n_labels))
    cmap2 = plt.cm.jet

    for i, m in enumerate(labels):
        print(f'Label: {m}')
        df2 = df[df['Label'] == m]
        df2 = df2.groupby('Power percent setting (%)').agg({
            'Sample Diameter (cm)': ['mean', 'std'],
            'Erosion Rate (cm/s)': ['mean', 'std'],
            'Loss Rate Error (cm/s)': ['mean', 'max']
        })

        df2.fillna(0, inplace=True)
        # df2['Loss Rate Error (cm/s)']['mean'] = df2['Loss Rate Error (cm/s)']['mean'].fillna(0)
        print(df2['Loss Rate Error (cm/s)']['mean'])

        laser_power_setting = list(df2.index.values)
        laser_power = np.array([laser_power_mapping[v] for v in laser_power_setting])
        sample_diameter = df2['Sample Diameter (cm)']['mean'].values
        sample_area = 0.25 * np.pi * sample_diameter ** 2.0
        aperture_factor = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.5 * sample_diameter)
        incident_heat_load = aperture_factor * laser_power / sample_area / 100.0
        recession_rate = df2['Erosion Rate (cm/s)']['mean'].values
        recession_rate_err = df2['Erosion Rate (cm/s)']['std'].values + df2['Loss Rate Error (cm/s)']['mean'].values
        ax.errorbar(
            incident_heat_load, recession_rate,  # yerr=recession_rate_err,
            marker=markers[i], ms=9, mew=1.25, mfc='none', label=f'{m}',
            capsize=2.75, elinewidth=1.25, lw=1.5, c=colors[i]
        )

    ax.set_xlabel('Heat load (MW/m$^{\mathregular{2}}$)')
    ax.set_ylabel('cm/s')
    ax.set_title('Recession rate')
    ax.set_xlim(5, 50)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.legend(loc='best', frameon=True, fontsize=8)

    fig.savefig(os.path.join(base_dir, 'recession_rate_vs_laser_power' + '.svg'), dpi=600)
    fig.savefig(os.path.join(base_dir, 'recession_rate_vs_laser_power' + '.png'), dpi=600)
    fig.savefig(os.path.join(base_dir, 'recession_rate_vs_laser_power' + '.pdf'), dpi=600)

    """
    Erosion rate vs thermal conductivity
    """
    fig2, ax = plt.subplots(1, 1, constrained_layout=True)
    fig2.set_size_inches(4.0, 3.0)

    markers2 = ['o', '^', 'v', '<', 'd']
    custom_lines = []

    for i, p in enumerate(laser_power_mapping):
        print(f'Laser power setting: {p}')
        df2 = not_h2_df.query(f'`Power percent setting (%)` == {p:.0f}')
        # df2 = not_h2_df[df['Power percent setting (%)'] == p].reset_index(drop=True)
        df2 = df2.groupby('Thermal conductivity (W/cm-K)').agg({
            'Sample Diameter (cm)': ['mean', 'std'],
            'Erosion Rate (cm/s)': ['mean', 'std'],
            'Loss Rate Error (cm/s)': ['mean', 'max']
        })

        print(df2)

        df2.fillna(0, inplace=True)
        # df2['Loss Rate Error (cm/s)']['mean'] = df2['Loss Rate Error (cm/s)']['mean'].fillna(0)
        print(df2['Loss Rate Error (cm/s)']['mean'])

        thermal_conductivity = list(df2.index.values)
        laser_power = laser_power_mapping[p]
        sample_diameter = df2['Sample Diameter (cm)']['mean'].values
        sample_area = 0.25 * np.pi * sample_diameter ** 2.0
        aperture_factor = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.5 * sample_diameter)
        incident_heat_load = np.mean(aperture_factor * laser_power / sample_area / 100.0)
        recession_rate = df2['Erosion Rate (cm/s)']['mean'].values
        recession_rate_err = df2['Erosion Rate (cm/s)']['std'].values + df2['Loss Rate Error (cm/s)']['mean'].values
        lbl = f'Heat load: {incident_heat_load:.0f} MW/m$\\mathregular{{^2}}$'
        nn = len(recession_rate)
        custom_lines.append(mpl.lines.Line2D([0], [0], color=cmap2(norm(p)), lw=2.0, label=lbl))
        ax.plot(thermal_conductivity, recession_rate, lw=1.5, c=cmap2(norm(p)))
        for j, tc, rr in zip(range(nn), thermal_conductivity, recession_rate):
            ax.errorbar(
                tc, rr,  # yerr=recession_rate_err,
                marker=markers2[j], ms=9, mew=1.25, mfc='none',
                # label=lbl,
                capsize=2.75, elinewidth=1.25, lw=1.5, c=cmap2(norm(p))
            )

    ax.set_xscale('log')
    ax.set_xlabel('Thermal conductivity (W/cm-K)')
    ax.set_ylabel('cm/s')
    ax.set_title('Recession rate')
    ax.set_xscale('log')
    ax.set_xlim(0.05, 3)

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.legend(handles=custom_lines, loc='best', frameon=True, fontsize=8)

    fig2.savefig(os.path.join(base_dir, 'recession_rate_vs_thermal_conductivity' + '.svg'), dpi=600)
    fig2.savefig(os.path.join(base_dir, 'recession_rate_vs_thermal_conductivity' + '.png'), dpi=600)
    fig2.savefig(os.path.join(base_dir, 'recession_rate_vs_thermal_conductivity' + '.pdf'), dpi=600)

    """
    Erosion rate vs coefficient of thermal expansion
    """
    fig3, ax = plt.subplots(1, 1, constrained_layout=True)
    fig3.set_size_inches(4.0, 3.0)

    markers2 = ['v', '^', 'o', 'd', '<']
    custom_lines = []

    for i, p in enumerate(laser_power_mapping):
        print(f'Laser power setting: {p}')
        df2 = not_h2_df.query(f'`Power percent setting (%)` == {p:.0f}')
        print(df2[['Material', 'Erosion Rate (cm/s)']])
        # df2 = not_h2_df[df['Power percent setting (%)'] == p].reset_index(drop=True)
        df2 = df2.groupby('CTE (/K)').agg({
            'Sample Diameter (cm)': ['mean', 'std'],
            'Erosion Rate (cm/s)': ['mean', 'std'],
            'Loss Rate Error (cm/s)': ['mean', 'max'],
            'Material': ['first'],
        })

        # print(df2.columns)
        df2.fillna(0, inplace=True)
        print(df2[['Material', 'Erosion Rate (cm/s)']])
        # df2['Loss Rate Error (cm/s)']['mean'] = df2['Loss Rate Error (cm/s)']['mean'].fillna(0)
        print(df2['Loss Rate Error (cm/s)']['mean'])

        cte = list(df2.index.values)
        laser_power = laser_power_mapping[p]
        sample_diameter = df2['Sample Diameter (cm)']['mean'].values
        sample_area = 0.25 * np.pi * sample_diameter ** 2.0
        aperture_factor = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.5 * sample_diameter)
        incident_heat_load = np.mean(aperture_factor * laser_power / sample_area / 100.0)
        recession_rate = df2['Erosion Rate (cm/s)']['mean'].values
        recession_rate_err = df2['Erosion Rate (cm/s)']['std'].values + df2['Loss Rate Error (cm/s)']['mean'].values
        lbl = f'Heat load: {incident_heat_load:.0f} MW/m$\\mathregular{{^2}}$'
        nn = len(recession_rate)
        custom_lines.append(mpl.lines.Line2D([0], [0], color=cmap2(norm(p)), lw=2.0, label=lbl))
        ax.plot(cte, recession_rate, lw=1.5, c=cmap2(norm(p)))
        for j, a, rr in zip(range(nn), cte, recession_rate):
            ax.errorbar(
                a, rr,  # yerr=recession_rate_err,
                marker=markers2[j], ms=9, mew=1.25, mfc='none',
                # label=lbl,
                capsize=2.75, elinewidth=1.25, lw=1.5, c=cmap2(norm(p))
            )

    ax.set_xscale('log')
    ax.set_xlabel('$\\alpha$ (/K)')
    ax.set_ylabel('cm/s')
    ax.set_title('Recession rate')
    ax.set_xscale('log')
    ax.set_xlim(1E-6, 1E-4)

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    # ax.legend(handles=custom_lines, loc='best', frameon=True, fontsize=8)

    fig3.savefig(os.path.join(base_dir, 'recession_rate_vs_cte' + '.svg'), dpi=600)
    fig3.savefig(os.path.join(base_dir, 'recession_rate_vs_cte' + '.png'), dpi=600)
    fig3.savefig(os.path.join(base_dir, 'recession_rate_vs_cte' + '.pdf'), dpi=600)

    plt.show()
