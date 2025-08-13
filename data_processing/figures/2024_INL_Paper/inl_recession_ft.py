import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import platform
import os
import json
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'

data_path = '/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/simulations'

simulated_db = [
    {'lbl': 'Simulation $f_{\mathrm{t}}$ = 60 KPa', 'file': 'Ben_Fig10_dt_sim_ft_60.csv', 'marker': 's', 'ft': 60},
    {'lbl': 'Simulation $f_{\mathrm{t}}$ = 70 KPa', 'file': 'Ben_Fig10_dt_sim_ft_70.csv', 'marker': 's', 'ft': 70},
    {'lbl': 'Simulation $f_{\mathrm{t}}$ = 80 KPa', 'file': 'Ben_Fig10_dt_sim_ft_80.csv', 'marker': 's', 'ft': 80},
    {'lbl': 'Simulation $f_{\mathrm{t}}$ = 70 KPa (not $T$ limit)', 'file': 'Ben_Fig10_dt_sim_ft_70_no_T_limit.csv',
     'marker': 'x', 'ft': 70},
]

experimental_csv = 'recession_vs_heat_load_30KPa.csv'

recession_db = [
    {'material': 'Glassy carbon', 'file': 'gc_recession_vs_ft.csv'},
    {'material': 'Activated carbon', 'file': 'activated_carbon_recession_vs_ft.csv'}
]


def normalize_path(the_path):
    global platform_system, drive_path
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    return os.path.join(drive_path, the_path)


def load_plot_style():
    with open('../../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def main():
    global data_path, simulated_db, experimental_csv
    base_path = normalize_path(base_path)
    load_plot_style()

    norm = mpl.colors.Normalize(vmin=2, vmax=80)
    cmap = mpl.cm.get_cmap('jet')

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.)
    colors_sim = ['C0', 'C1', 'C2', 'C3']

    ax.set_yscale('log')
    ax.set_ylim(5E-5, 2)
    ax.set_xlabel(r'Heat load (MW/m$^{\mathregular{2}}$)')
    ax.set_ylabel('Recession rate (cm/s)')

    ax.set_xlim(0, 55)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    for i, r in enumerate(simulated_db):
        csv = os.path.join(base_path, r['file'])
        lbl = r['lbl']
        marker = r['marker']
        ft = r['ft']
        sim_df = pd.read_csv(csv, comment='#').apply(pd.to_numeric)
        heat_load = sim_df['Heat load (MW/m2)'].values
        recession_rate = sim_df['Recession rate (cm/s)'].values

        ax.plot(heat_load, recession_rate, c=cmap(norm(ft)), marker=marker, fillstyle='full', ls='--', mew=1.5)


    r3n41_42_df = pd.read_csv(os.path.join(base_path, experimental_csv)).apply(pd.to_numeric)
    ax.errorbar(
        r3n41_42_df['Heat load (MW/m2)'],
        r3n41_42_df['Recession rate (cm/s)'],
        # yerr=r3n41_42_df['Recession rate error (cm/s)'],
        marker='o', color=cmap(norm(30)),
        ms=9, mew=1.25, mfc='none', ls='none',
        capsize=2.75, elinewidth=1.25, lw=1.5,
    )

    markers_ft = ['o', '^']
    for i, r in enumerate(recession_db):
        csv = os.path.join(base_path, r['file'])
        ft_df = pd.read_csv(csv).apply(pd.to_numeric)
        print(ft_df)
        for j, row in ft_df.iterrows():
            ax.errorbar(
                round(row['Heat load (MW/m^2)']/5)*5, row['Mean recession rate (cm/s)'],
                # yerr=row['Standard recession rate error (cm/s)']/3.,
                c=cmap(norm(row['FT_estimate (KPa)'])),
                ms=9, mew=1.25, mfc='none', ls='none',
                capsize=2.75, elinewidth=1.25, lw=1.5,
                marker=markers_ft[i]
            )

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, orientation='vertical', label=r'$f_{\mathrm{t}}$ (KPa)')
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    cbar.ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    legend_elements = [Line2D([0], [0], color='k', mfc='k', lw=1.5, label='Simulation', marker='s'),
                       Line2D([0], [0], marker='o', color='k', label=r'Glassy carbon',
                              markerfacecolor='none', mew=1.25, markersize=10, ls='none'),
                       Line2D([0], [0], marker='^', color='k', label=r'Activated carbon',
                              markerfacecolor='none', mew=1.25, markersize=10, ls='none')
                       ]

    ax.legend(handles=legend_elements, loc='lower right', frameon=True)
    fig.savefig(os.path.join(base_path, 'figure_recession_heat_load_ft.png'), dpi=600)
    fig.savefig(os.path.join(base_path, 'figure_recession_heat_load_ft.pdf'), dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
