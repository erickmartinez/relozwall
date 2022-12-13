import json
import os
import sys

import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('../')

data_csv = '../../data/sphere_diameter_scan.csv'

def load_data():
    df = pd.read_csv(data_csv)
    columns = df.columns
    df[columns[1::]] = df[columns[1::]].apply(pd.to_numeric)
    return df

def load_plt_style():
    with open('../../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

def rss(series: pd.Series):
    return np.sqrt(series.dot(series)) / len(series)


if __name__ == '__main__':
    df = load_data()
    df = df.groupby('Sample ID').agg({
        '850 um %': ['mean'],
        '185 um %': ['mean'],
        'Density (g/cm3)': ['mean'],
        'Density error (g/cm3)': ['mean', rss],
        'Erosion Rate (cm/s)': ['mean', 'min', 'max'],
        'Erosion rate error (cm/s)': ['mean', rss]
    })
    print(df)
    df.sort_values(by=[('185 um %', 'mean')], inplace=True)

    load_plt_style()
    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.0)

    ax.errorbar(
        df['850 um %'], df['Erosion Rate (cm/s)']['mean'],
        yerr=df['Erosion rate error (cm/s)']['rss'],
        c='C0',
        marker='^', lw=1.5,
        # mec='k',
        mew=1.25, capsize=3.5, capthick=1.25, ecolor='C0', fillstyle='none'
    )

    # ax[1].errorbar(
    #     df['185 um %'], df['Density (g/cm3)']['mean'],
    #     yerr=df['Density error (g/cm3)']['rss'],
    #     c='C1',
    #     marker='^', lw=1.5,
    #     # mec='k',
    #     mew=1.25, capsize=3.5, capthick=1.25, ecolor='C1', fillstyle='none'
    # )

    ax.set_xlabel('Weight % of 850 $\mathregular{\mu}$m spheres')
    ax.set_ylabel('Erosion rate (cm/s)')
    # ax.set_xlim(0,100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))

    ax.set_ylim(0.05, 0.25)
    ax.set_xlim(-5, 105)
    ax.text(
        -0.0, -0.2, '185 $\mathregular{\mu}$m', transform=ax.transAxes, fontsize=12, fontweight='regular',
        va='bottom', ha='center'
    )
    ax.text(
        1.0, -0.2, '850 $\mathregular{\mu}$m', transform=ax.transAxes, fontsize=12, fontweight='regular',
        va='bottom', ha='center'
    )
    #
    # ax[1].set_xlabel('% of 185 $\mathregular{\mu}$m spheres')
    # ax[1].set_ylabel('Density (g/cm$^{\mathregular{3}}$)')

    fig.savefig(os.path.join('../../data', 'erosion_vs_sphere_diameter.png'), dpi=600)
    plt.show()
