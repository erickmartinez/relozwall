import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data_processing.misc_utils.plot_style import load_plot_style
import matplotlib.ticker as ticker

EXCEL_FILE = r'./R5N16_ejected_pebble_sizes.xlsx'

def load_data(path_to_excel):
    df = pd.read_excel(path_to_excel, sheet_name='All images', usecols=['Equivalent diameter (um)'])
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def plot_histogram(df, ax: plt.Axes = None):
    if ax is None:
        load_plot_style(font='Times New Roman')
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        fig.set_size_inches(5.0, 3)

    diameters = df['Equivalent diameter (um)'].values
    counts, bin_edges = np.histogram(diameters, bins=60)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    print(f'Total measurements: {len(diameters)}')

    ax.bar(bin_centers, counts, align='center', width=np.diff(bin_edges)*0.9, color='C0', edgecolor='white', linewidth=1)
    ax.set_xlabel('Particle size (µm)')
    ax.set_ylabel('Counts')
    ax.set_title('Ejected particles R5N16')
    ax.set_xlim(left=-50, right=1000)

def main(path_to_excel):
    df = load_data(path_to_excel)
    load_plot_style(font='Times New Roman')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3)
    plot_histogram(df, ax=ax)
    path_to_figures = Path('./figures')
    path_to_figures.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(path_to_figures / f'ejected_particle_sizes_R5N16.{extension}', dpi=600)

    plt.show()


if __name__ == '__main__':
    main(path_to_excel=EXCEL_FILE)