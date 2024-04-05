import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
import os

experimental_files = [
    {'matrix wt %':  5, 'csv': '3PBT_R4N137 - 033_2024-01-08_1_processed.csv'},
    {'matrix wt %': 20, 'csv': '3PBT_R4N139 - 040_2024-01-16_2_processed.csv'},
    {'matrix wt %': 25, 'csv': '3PBT_R4N136 - 030_2024-01-08_1_processed.csv'},
]

def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def main():
    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.)

    ax.set_xlabel('Deformation (mm)')
    ax.set_ylabel('Load (N)')

    colors = ['C0', 'C2', 'C1', 'C3', 'C4', 'C5']
    markers = ['o', 's', '^', 'v', 'D', '<', '>']

    for i, ef in enumerate(experimental_files):
        mc, csv = ef['matrix wt %'], ef['csv']
        bending_df = pd.read_csv(os.path.join('data', csv)).apply(pd.to_numeric)
        deformation = bending_df['Deformation (mm)'].values
        force_4cm = bending_df['Force 4cm (N)'].values
        force_4cm_err = bending_df['Force 4cm err (N)'].values

        # print(bending_df[['Deformation (mm)', 'Force 4cm (N)']])
        # find the number of points where force > 0
        n = len(force_4cm[force_4cm==0])
        idx_start = n - 2
        deformation = deformation[idx_start::]
        force_4cm = force_4cm[idx_start::]
        force_4cm_err = force_4cm_err[idx_start::]
        deformation -= deformation.min()
        lbl = f'{mc:>2d} % matrix'
        ebc = mpl.colors.to_rgba(colors[i], 0.5)
        ax.errorbar(
            deformation, force_4cm, yerr=0.5, xerr=0.1,
            marker=markers[i], mfc='none', capsize=2.5,
            ms=9, mew=1., ecolor=ebc,
            elinewidth=1.0, lw=1.75, c=colors[i],
            label=lbl
        )
    ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.set_xlim(-0.1, 0.8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.set_ylim(-1, 6)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    plt.show()



if __name__ == '__main__':
    main()


