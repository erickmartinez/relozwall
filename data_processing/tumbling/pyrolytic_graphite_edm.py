import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json


base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\tumbling\pyrolytic_graphite'
data_xls = 'particle_sizes.xlsx'

def load_data():
    df = pd.read_excel(os.path.join(base_dir, data_xls), sheet_name='averages').apply(pd.to_numeric)
    return df

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

def main():
    df = load_data()
    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)

    ax.errorbar(
        df['Tumbling time (days)'], df['Mean particle size (mm)'],
        yerr=2.*df['Particle size std'].values,
        ls='none',
        color='C0', marker='o', ms=8, fillstyle='none', label='Outgassed pebbles',
        capsize=2.5, mew=1.25, elinewidth=1.25,
    )

    ax.set_xlabel('Tumbiling time [days]')
    ax.set_ylabel('Particle size [mm]')
    ax.set_title('Pyrolytic graphite')

    fig.savefig(os.path.join(base_dir, 'tumbling_progress.png'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()