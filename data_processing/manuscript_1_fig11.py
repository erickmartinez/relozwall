import json
import os
import sys

import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('../')

sphere_data_csv = '../data/sphere_diameter_scan.csv'
binder_data_csv = '../data/binder_content_scan.csv'

def load_data():
    df1 = pd.read_csv(sphere_data_csv)
    df2 = pd.read_csv(binder_data_csv)
    columns1 = df1.columns
    columns2 = df2.columns
    df1[columns1[1::]] = df1[columns1[1::]].apply(pd.to_numeric)
    df2[columns2[1::]] = df2[columns2[1::]].apply(pd.to_numeric)
    return df1, df2

def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

def rss(series: pd.Series):
    return np.sqrt(series.dot(series)) / len(series)


if __name__ == '__main__':
    sphere_df, binder_df = load_data()
    sphere_df = sphere_df.groupby('Sample ID').agg({
        '850 um %': ['mean'],
        '185 um %': ['mean'],
        'Density (g/cm3)': ['mean'],
        'Density error (g/cm3)': ['mean', rss],
        'Erosion Rate (cm/s)': ['mean', 'min', 'max'],
        'Erosion rate error (cm/s)': ['mean', rss]
    })

    binder_df = binder_df.groupby('Sample ID').agg({
        'Binder content (%)': ['mean'],
        '850 um %': ['mean'],
        '185 um %': ['mean'],
        'Density (g/cm3)': ['mean'],
        'Density error (g/cm3)': ['mean', rss],
        'Erosion Rate (cm/s)': ['mean', 'min', 'max'],
        'Erosion rate error (cm/s)': ['mean', rss]
    })
    binder_df.sort_values(by=[('Binder content (%)', 'mean')], inplace=True)

    sphere_df.sort_values(by=[('185 um %', 'mean')], inplace=True)

    load_plt_style()
    fig, ax = plt.subplots(ncols=1, nrows=2, constrained_layout=True)
    fig.set_size_inches(4.0, 4.75)

    ax[0].errorbar(
        sphere_df['850 um %'], sphere_df['Erosion Rate (cm/s)']['mean'],
        yerr=sphere_df['Erosion rate error (cm/s)']['rss'],
        c='C0',
        marker='o', lw=1.5,
        # mec='k',
        mew=1.25, capsize=3.5, capthick=1.25, ecolor='C0', fillstyle='none'
    )

    ax[0].set_title('Recession rate')

    ax[1].errorbar(
        binder_df['Binder content (%)'], binder_df['Erosion Rate (cm/s)']['mean'],
        yerr=binder_df['Erosion rate error (cm/s)']['rss'],
        c='C1',
        marker='s', lw=1.5,
        # mec='k',
        mew=1.25, capsize=3.5, capthick=1.25, ecolor='C1', fillstyle='none'
    )

    ax[1].set_title('Recession rate')

    ax[0].set_xlabel('Wt % of 850 $\mathregular{\mu}$m spheres')
    ax[0].set_ylabel('cm/s')
    # ax.set_xlim(0,100)
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.025))

    ax[0].set_ylim(0.05, 0.25)
    ax[0].set_xlim(-5, 105)
    ax[0].text(
        -0.0, -0.25, '185 $\mathregular{\mu}$m', transform=ax[0].transAxes, fontsize=11, fontweight='regular',
        va='bottom', ha='center'
    )
    ax[0].text(
        1.0, -0.25, '850 $\mathregular{\mu}$m', transform=ax[0].transAxes, fontsize=11, fontweight='regular',
        va='bottom', ha='center'
    )

    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax[1].set_xlim(0, 21)
    ax[1].set_ylim(0.025, 0.225)

    ax[1].set_xlabel('Binder content wt %')
    ax[1].set_ylabel('cm/s')

    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))

    for i, axi in enumerate(ax.flatten()):
        panel_label = chr(ord('`') + i + 1)
        axi.text(
            -0.15, 1.15, f'({panel_label})', transform=axi.transAxes, fontsize=13, fontweight='bold',
            va='top', ha='right'
        )

    fig.savefig(os.path.join('../data', 'erosion_vs_sphere_diameter_and_binder_content.png'), dpi=600)
    fig.savefig(os.path.join('../data', 'erosion_vs_sphere_diameter_and_binder_content.pdf'), dpi=600)
    fig.savefig(os.path.join('../data', 'erosion_vs_sphere_diameter_and_binder_content.eps'), dpi=600)
    plt.show()
