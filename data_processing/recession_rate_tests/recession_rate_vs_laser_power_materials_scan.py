import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from data_processing.utils import get_experiment_params

base_dir = r"C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN\20230723"
merged_db_xslx = r"C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\merged_db.xlsx"
material_propperties_xls = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\material_properties_avearges.xlsx'
laser_power_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN\laser_output'


samples = [
    {'sample_id': 'R4N04', 'label': 'GC spheres (10% filler)', 'material': 'Glassy carbon', 'marker': 'o'},
    # {'sample_id': 'R4N64', 'label': 'GC spheres (12.5% filler)', 'material': 'Glassy carbon', 'marker': 's'},
    {'sample_id': 'R4N66', 'label': 'WNiFe', 'material': 'Tungsten', 'marker': '>'},
    # {'sample_id': 'R4N67', 'label': 'GC spheres (5% filler)', 'material': 'Glassy carbon', 'marker': 's'},
    # {'sample_id': 'R4N69', 'label': 'WNiFe (no filler)', 'material': 'Tungsten', 'marker': '>'},
    {'sample_id': 'R4N72', 'label': 'Cu', 'material': 'Copper', 'marker': 'd'},
    # {'sample_id': 'R4N73', 'label': 'GC spheres, 90% 850um, 10% 180 um (12.5% filler)', 'material': 'Glassy carbon', 'marker': 'o'},
    {'sample_id': 'R4N20', 'label': 'SiNx spheres', 'material': 'SiNx', 'marker': '^'},
    {'sample_id': 'R4N21', 'label': 'SiC spheres', 'material': 'SiC', 'marker': 'v'},
    {'sample_id': 'R4N22', 'label': 'GC spheres (10% filler)', 'material': 'Glassy carbon', 'marker': 'o'},
    {'sample_id': 'R4N23', 'label': 'Mineral graphite', 'material': 'Graphite', 'marker': 'h'},
    {'sample_id': 'R4N24', 'label': 'hBN granules', 'material': 'hBN', 'marker': '<'},
]

beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707

poor_gc_ids = ['R4N04', 'R4N22']



def load_material_properties_df():
    df = pd.read_excel(material_propperties_xls, sheet_name=0, header=0)
    df[['Property value']] = df[['Property value']].apply(pd.to_numeric)
    return df


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
    laser_test_df = pd.read_excel(
        os.path.join(base_dir, merged_db_xslx),
        sheet_name='Laser tests',
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

    empty_property = np.empty(len(laser_test_df))

    laser_test_df['Thermal conductivity (W/cm-K)'] = empty_property
    laser_test_df['Coefficient of thermal expansion (10^{-6}/K)'] = empty_property
    laser_test_df['Specific heat (J/g/K)'] = empty_property
    laser_test_df['Label'] = ['lbl' for i in range(len(laser_test_df))]
    laser_test_df['Material'] = ['mat' for i in range(len(laser_test_df))]
    laser_test_df['Order'] = [0 for i in range(len(laser_test_df))]
    laser_test_df['Marker'] = ['o' for i in range(len(laser_test_df))]

    materials_df = load_material_properties_df()
    for i, s in enumerate(samples):
        properties_df = materials_df[materials_df['Material'] == s['material']]
        cte = properties_df.loc[properties_df['Property'] == 'Coefficient of thermal expansion', 'Property value'].mean()
        tc = properties_df.loc[properties_df['Property'] == 'Thermal conductivity', 'Property value'].mean()
        cp = properties_df.loc[properties_df['Property'] == 'Specific heat', 'Property value'].mean()
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'Thermal conductivity (W/cm-K)'] = tc
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'Coefficient of thermal expansion (10^{-6}/K)'] = cte
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'Specific heat (J/g/K)'] = cp
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'Label'] = s['label']
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'Order'] = i
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'Marker'] = s['marker']
        laser_test_df.loc[laser_test_df['Sample code'] == s['sample_id'], 'Material'] = s['material']

    laser_test_df.sort_values(by=['Order'], inplace=True)
    laser_test_df.reset_index(drop=True, inplace=True)
    return laser_test_df


if __name__ == '__main__':
    df = load_data()
    df.to_csv(path_or_buf=os.path.join(base_dir, 'dataset_all_materials.csv'), index=False)
    labels = df['Label'].unique()
    # not_h2_df = df[df['h2'] == False]
    n_labels = len(labels)
    print('Material column exists:', 'Material' in df.columns)

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
            'Sample diameter (cm)': ['mean', 'std'],
            'Recession rate (cm/s)': ['mean', 'std'],
            'Recession rate error (cm/s)': ['mean', 'max'],
            'Marker': ['first']
        })

        marker = df2['Marker']['first'].values
        marker = marker[0]

        df2.fillna(0, inplace=True)
        # df2['Loss Rate Error (cm/s)']['mean'] = df2['Loss Rate Error (cm/s)']['mean'].fillna(0)
        # print(df2['Recession rate error (cm/s)']['mean'])

        laser_power_setting = list(df2.index.values)
        laser_power = np.array([laser_power_mapping[v] for v in laser_power_setting])
        sample_diameter = df2['Sample diameter (cm)']['mean'].values
        sample_area = 0.25 * np.pi * sample_diameter ** 2.0
        aperture_factor = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.5 * sample_diameter)
        incident_heat_load = aperture_factor * laser_power / sample_area / 100.0
        recession_rate = df2['Recession rate (cm/s)']['mean'].values
        recession_rate_err = df2['Recession rate (cm/s)']['std'].values + df2['Recession rate error (cm/s)']['mean'].values
        ax.errorbar(
            incident_heat_load, recession_rate,  # yerr=recession_rate_err,
            marker=marker, ms=9, mew=1.25, mfc='none', label=f'{m}',
            capsize=2.75, elinewidth=1.25, lw=1.5, c=colors[i]
        )

    ax.set_xlabel('Heat load [MW/m$^{\mathregular{2}}$]')
    ax.set_ylabel('cm/s')
    ax.set_title('Recession rate (high heat loads)')
    ax.set_xlim(5, 50)
    ax.set_ylim(0, 0.2)
    ax.tick_params(which='both', axis='y', labelright=False, right=True, direction='in')
    ax.tick_params(which='both', axis='x', direction='out')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    ax.legend(loc='best', frameon=True, fontsize=8)

    fig.savefig(os.path.join(base_dir, 'recession_rate_vs_laser_power_all_materials' + '.svg'), dpi=600)
    fig.savefig(os.path.join(base_dir, 'recession_rate_vs_laser_power_all_materials' + '.png'), dpi=600)
    fig.savefig(os.path.join(base_dir, 'recession_rate_vs_laser_power_all_materials' + '.pdf'), dpi=600)


    plt.show()