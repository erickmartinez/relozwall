import matplotlib as mpl
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import utils


data_csv = '../data/sphere_diameter_scan.csv'

def load_data():
    df = pd.read_csv(data_csv)
    columns = df.columns
    df[columns[1::]] = df[columns[1::]].apply(pd.to_numeric)
    return df

def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


if __name__ == '__main__':
    df = load_data()
    df.sort_values(by=['185 um %'], inplace=True)

    load_plt_style()
    fig, ax = plt.subplots(ncols=1, nrows=2, constrained_layout=True)
    fig.set_size_inches(4.0, 4.5)

    ax[0].errorbar(
        df['185 um %'], df['Erosion Rate (cm/s)'],
        yerr=df['Erosion rate error (cm/s)'],
        c='C0',
        marker='^', lw=1.5,
        # mec='k',
        mew=1.25, capsize=3.5, capthick=1.25, ecolor='C0', fillstyle='none'
    )

    ax[1].errorbar(
        df['185 um %'], df['Density (g/cm3)'],
        yerr=df['Density error (g/cm3)'],
        c='C1',
        marker='^', lw=1.5,
        # mec='k',
        mew=1.25, capsize=3.5, capthick=1.25, ecolor='C1', fillstyle='none'
    )

    ax[0].set_xlabel('% of 185 $\mathregular{\mu}$m spheres')
    ax[0].set_ylabel('Erosion rate (cm/s)')

    ax[1].set_xlabel('% of 185 $\mathregular{\mu}$m spheres')
    ax[1].set_ylabel('Density (g/cm$^{\mathregular{3}}$)')

    fig.savefig(os.path.join('../data', 'erosion_vs_sphere_diameter.png'), dpi=600)
    plt.show()
