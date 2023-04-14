import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
from matplotlib import ticker
from data_processing.utils import lighten_color

csv_file = '../../data/friction_measurements.csv'

samples = {
    'R3N56': r'$d_{p}$ = 850 $\mathregular{\mu}$m, Epoxy, $A_{v}$ = 2.9 mm$\mathregular{^2}$',
    'R4N33': r'$d_{p}$ = 850 $\mathregular{\mu}$m, Amine, $A_{v}$ = 2.9 mm$\mathregular{^2}$',
    'R4N37': r'$d_{p}$ = 220 $\mathregular{\mu}$m, Amine, $A_{v}$ = 2.9 mm$\mathregular{^2}$',
    'R4N39': r'$d_{p}$ = 850 $\mathregular{\mu}$m, Amine, $A_{v}$ = 5.9 mm$\mathregular{^2}$'
}


def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


if __name__ == '__main__':
    df = pd.read_csv(csv_file)
    columns = df.columns
    df[columns[2:-1]] = df[columns[2:-1]].apply(pd.to_numeric)
    df['Date'] = df['Date'].apply(pd.to_datetime)
    df = df[df['Sample'].isin(list(samples.keys()))]

    force = df['Average Friction Force (N)'].values
    force_err = force * 0.09  # temp_df['Friction Force Std (N)'].values
    area = df['Contact Area (cm2)'].values
    area_err = df['Contact Area Error (cm2)'].values
    force_n = force / area
    force_n_err = force_n * np.linalg.norm([area_err / area, force_err / force])
    # labels = [samples[s] for s in df['Sample'].values]
    labels = [f'{sk} ' for sk in df['Sample'].values]
    print(force_n)
    print(force_n_err)

    load_plt_style()
    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)
    mew = 1.5
    ls = '-'
    width = 1.0
    error_kw = dict(lw=1.0, capsize=3.5, capthick=1.0, ecolor='tab:red')
    xpos = np.arange(len(labels))
    ax.barh(
        xpos, force_n,
        xerr=force_n_err,
        align='center',
        error_kw=error_kw,
        color='C0',
        lw=0.5, ec='k', #lighten_color('C0', 1.5),
        ls=ls,
    )

    ax.set_yticks(xpos, labels, color='k', rotation=0)
    ax.set_xlabel(r'N/cm$\mathregular{^2}$')
    ax.set_title('Friction force')

    ax.set_xlim(0, 0.1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    plt.show()
    fig.savefig('../../data/Q8/friction_measurements.png', dpi=600)
    fig.savefig('../../data/Q8/friction_measurements.svg', dpi=600)