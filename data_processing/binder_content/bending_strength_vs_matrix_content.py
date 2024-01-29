import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import platform
from scipy.stats.distributions import t

data_path = r'Documents/ucsd/Postdoc/research/data/bending_tests'
data_xlsx = 'bending_strength_vs_matrix_content.xlsx'

platform_system = platform.system()
if platform_system == 'Windows':
    drive_path = r'C:\Users\erick\OneDrive'
else:
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'

def normalize_path(the_path):
    global platform_system, drive_path
    the_path = os.path.join(drive_path, the_path)
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    return the_path

def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

def std_err(x):
    return np.linalg.norm(x) /len(x)


def main():
    global data_path, data_xlsx
    data_path = normalize_path(data_path)
    data_df = pd.read_excel(os.path.join(data_path, data_xlsx), sheet_name=0)
    data_df.drop(columns='Sample ID', inplace=True)
    data_df = data_df.apply(pd.to_numeric)
    data_agg_df = data_df.groupby(['Matrix wt %']).agg(
        ['mean', std_err]
    )

    print(data_agg_df.columns)

    load_plt_style()

    fig, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.)

    cax1 = 'C0'
    cax2 = 'C1'

    ax2 = ax1.twinx()
    matrix_wt_pct = data_agg_df.index
    mean_force = data_agg_df['Fracture force (N)']['mean']
    mean_force_err = data_agg_df['Fracture force err (N)']['std_err']
    mean_strength = data_agg_df['Flexural strength (KPa)']['mean']
    mean_strength_err = data_agg_df['Flexural strength err (KPa)']['std_err']
    ax1.errorbar(
        matrix_wt_pct, mean_force, yerr=mean_force_err, marker='o', ms=9, mew=1.25, mfc='none',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='C0'
    )

    ax2.errorbar(
        matrix_wt_pct, mean_strength, yerr=mean_strength_err, marker='s', ms=9, mew=1.25, mfc='none',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='C1'
    )

    ax1.set_xlabel('Matrix wt %')
    ax1.set_ylabel('Load (N)', color=cax1)
    ax2.set_ylabel('Bending strength (KPa)', color=cax2)

    ax1.tick_params(axis='y', labelcolor=cax1)
    ax2.tick_params(axis='y', labelcolor=cax2)

    ax1.set_xlim(2.5, 27.5)

    ax1.set_ylim(0, 4)
    ax2.set_ylim(0, 250)

    fig.savefig(os.path.join(data_path, 'bending_strength_vs_matrix_content.png'), dpi=600)

    plt.show()



if __name__ == '__main__':
    main()
