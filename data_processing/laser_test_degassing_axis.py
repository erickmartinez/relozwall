import h5py
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
from matplotlib import ticker
import matplotlib as mpl
import json
import re
import shutil
import platform

input_file = '../data/degassing_database_gc_graphite.csv'
k_B = 1.380649E-23  # J / K
temperature = 20.0  # K
chamber_volume = 2717.91  # cm^3
mean_dt = 1.4 # s


def number_density(chamber_pressure, chamber_temperature):
    """
    kb = 1.380649e-23  # J/k = N * m / K
    pressure = 133.322 * pressure   # N / m^2
    n = pressure / (kb * (temperature + 273.15))  # (N / m^2) / (N * m ) = (#/m^3)
    n = n * 1E-6
    """
    p_k = 133.322 * chamber_pressure / 1.380649e-17
    return p_k / (chamber_temperature + 273.15)


if __name__ == '__main__':
    outgassing_df: pd.DataFrame = pd.read_csv(input_file)
    # outgassing_df = outgassing_df[(outgassing_df['Laser Power Setpoint (%)'] == 100) & (outgassing_df['Sample ID'] != 'R3N40')]
    column_names = outgassing_df.columns
    outgassing_df[column_names[2:]] = outgassing_df[column_names[2:]].apply(pd.to_numeric)
    mean_dt = outgassing_df
    outgassing_df['dP (mTorr)'] = outgassing_df['Peak Pressure (mTorr)'] - outgassing_df['Base Pressure (mTorr)'].min()
    n = len(outgassing_df)
    outgassed_particles = np.empty(n, dtype=np.float64)
    outgassing_df['Gas Concentration x10^15 (1/cm^3)'] = 1E-15*number_density(outgassing_df['dP (mTorr)']*1E-3, temperature)
    print(number_density(outgassing_df['dP (mTorr)']*1E-3, temperature))
    print(outgassing_df[['Sample', 'dP (mTorr)', 'Erosion Rate (cm/s)']])

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1)  # , constrained_layout=True)
    fig.set_size_inches(4.5, 3.5)

    colors = ['C0', 'C1']

    ax.errorbar(
        outgassing_df['Gas Concentration x10^15 (1/cm^3)'].values,
        outgassing_df['Erosion Rate (cm/s)'].values,
        yerr= outgassing_df['Erosion Rate Error (cm/s)'].values,
        capsize=2.5, mew=1.25, marker='o', ms=8, elinewidth=1.25,
        ls='none', c=colors[0], fillstyle='none'
    )


    ax.set_xlabel('Gas Content $\\times\\mathregular{10^{15}}$ (1/cm$\\mathregular{^3}$)')
    ax.set_ylabel('Erosion Rate (cm/s)', color=colors[0])
    # ax.set_xlim(left=0.0, right=1.0)
    # ax.set_ylim(bottom=0.0, top=2.0)
    ax.tick_params(axis='y', labelcolor=colors[0])

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    ax2 = ax.twinx()

    ax.set_zorder(1)
    ax2.set_zorder(0)
    ax.patch.set_visible(False)

    ax2.errorbar(
        outgassing_df['Gas Concentration x10^15 (1/cm^3)'].values,
        outgassing_df['Particle velocity mode (cm/s)'],
        yerr=outgassing_df['Particle velocity std (cm/s)'],
        capsize=2.5, mew=1.25, marker='s', ms=8, elinewidth=1.25,
        ls='none', c=colors[1], fillstyle='none'
    )

    ax2.set_ylabel('Particle velocity (cm/s)', color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])

    fig.tight_layout()

    basename = os.path.splitext(os.path.basename(input_file))[0]
    path = os.path.dirname(input_file)
    fig.savefig(os.path.join(path, basename + '_erosion_rates.png'), dpi=600)
    plt.show()
