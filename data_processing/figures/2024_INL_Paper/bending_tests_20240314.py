import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import platform
import os
import json


platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'

base_path = '/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/bending_tests'

data_dir = 'R4N137'

csv = '3PBT_R4N137 - 031_2024-01-08_1.csv'



simulated_db = [
    {'lbl': 'Simulation ($f_{\mathrm{t}}$ = 60 KPa)', 'file': 'Ben_Fig10_dt_sim_ft_60.csv', 'marker': 's', 'ft': 60},
    {'lbl': 'Simulation ($f_{\mathrm{t}}$ = 70 KPa)', 'file': 'Ben_Fig10_dt_sim_ft_70.csv', 'marker': 's', 'ft': 70},
    {'lbl': 'Simulation ($f_{\mathrm{t}}$ = 80 KPa)', 'file': 'Ben_Fig10_dt_sim_ft_80.csv', 'marker': 's', 'ft': 80},
    {'lbl': 'Simulation ($f_{\mathrm{t}}$ = 70 KPa, not T limit)', 'file': 'Ben_Fig10_dt_sim_ft_70_no_T_limit.csv',
     'marker': 'x', 'ft': 70},
]

experimental_csv = 'recession_vs_heat_load_30KPa.csv'

recession_db = [
    {'material': 'GC', 'file': 'gc_recession_vs_ft.csv'},
    {'material': 'AC', 'file': 'activated_carbon_recession_vs_ft.csv'}
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
    global base_path, data_dir, csv
    base_path = normalize_path(base_path)
    data_dir = os.path.join(base_path, data_dir)
    load_plot_style()

    bending_df = pd.read_csv(os.path.join(data_dir, csv), comment='#').apply(pd.to_numeric)

    bending_df = bending_df[bending_df['Displacement (mm)']>=0]
    bending_df = bending_df.sort_values(by=['Time (s)', 'Displacement (mm)'])
    displacement = bending_df['Displacement (mm)'].values
    displacement_err = bending_df['Displacement err (mm)'].values
    force = bending_df['Force (N)'].values

    max_force = force.max()
    idx_max = np.argmin(np.abs(force - max_force))
    displacement = displacement[0:idx_max+1]
    force = force[0:idx_max+1]


    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.)

    ax.errorbar(
        displacement, force, yerr=0.5,
        marker='o', color='C0',
        ms=9, mew=1.25, mfc='none', ls='-',
        capsize=2.75, elinewidth=1.25, lw=1.5,
        label=r'R4N1371'
    )

    ax.set_xlabel('d (mm)')
    ax.set_ylabel('F (N)')
    plt.show()

if __name__ == '__main__':
    main()
