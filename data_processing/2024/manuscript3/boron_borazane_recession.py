import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import os
import json

laser_power_csv = r'../../laser_power_mapping/laser_power_mapping.csv'
beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}')


def mean_err(x):
    return np.linalg.norm(x) / len(x)


def map_laser_power_settings():
    global laser_power_csv
    df: pd.DataFrame = pd.read_csv(laser_power_csv).apply(pd.to_numeric)
    mapping = {}
    for i, row in df.iterrows():
        mapping[row['Laser power setting (%)']] = row['Laser power (W)']

    keys = list(mapping.keys())
    keys.sort()
    return {i: mapping[i] for i in keys}


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)


def main():
    data_df: pd.DataFrame = pd.read_excel(
        io='data/laser_tests.xlsx', sheet_name=0,
        usecols=[
            'ROW', 'Power percent setting (%)', 'Irradiation time (s)', 'Recession rate (cm/s)',
            'Recession rate error (cm/s)', 'Sample diameter (cm)'
        ]
    ).apply(pd.to_numeric)
    laser_mapping = map_laser_power_settings()
    data_df['Laser power (W)'] = [laser_mapping[int(lp)] for lp in data_df['Power percent setting (%)'].values]
    diameter = 1.  # data_df['Sample diameter (cm)'].values
    av = gaussian_beam_aperture_factor(beam_radius, 0.5 * diameter)
    area = 0.25 * np.pi * diameter ** 2.
    data_df['Heat load (MW/m^2)'] = np.round(data_df['Laser power (W)'].values * av / area / 100. / 5) * 5.
    agg_df = data_df.groupby(by=['Heat load (MW/m^2)']).agg(
        ['mean', mean_err]
    )

    agg_df.sort_values(by=['Heat load (MW/m^2)']).reset_index(inplace=True, drop=True)

    carbon_df = pd.read_csv(
        filepath_or_buffer='data/recession_vs_heat_load_30KPa.csv'
    ).apply(pd.to_numeric)

    heat_load = agg_df.index
    nu = agg_df['Recession rate (cm/s)']['mean'].values
    nu_err = agg_df['Recession rate error (cm/s)']['mean_err'].values

    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3.)

    ax.set_xlabel(r'$q$ {\sffamily (MW/m\textsuperscript{2})}', usetex=True)
    ax.set_ylabel(r'$\nu$ {\sffamily (cm/s)}', usetex=True)

    ax.errorbar(
        heat_load*0.95, nu, yerr=nu_err, marker='o', ms=9, mew=1.25, mfc='none',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='C1', ls='none',
        label='Laser heating'
    )

    # ax.errorbar(
    #     carbon_df['Heat load (MW/m2)']*0.9,
    #     carbon_df['Recession rate (cm/s)'],
    #     yerr=(carbon_df['Recession rate (cm/s)'] * 0.5),
    #     marker='s', color='C0',
    #     ms=9, mew=1.25, mfc='none', ls='none',
    #     capsize=2.75, elinewidth=1.25, lw=1.5,
    #     label=r'Glassy carbon ($F_{\mathrm{b}}$ = 1.5 N)'
    # )

    ax.legend(
        loc='upper left', frameon=True
    )

    ax.set_xlim(-0.5, 40)
    ax.set_ylim(1E-4, 0.5)
    ax.set_title('Surface recession')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))


    # ax.set_yscale('log')

    fig.savefig('figures/boron_pebble_rod_recession.png', dpi=600)
    fig.savefig('figures/boron_pebble_rod_recession.pdf', dpi=600)
    fig.savefig('figures/boron_pebble_rod_recession.svg', dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
