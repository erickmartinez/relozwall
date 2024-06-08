import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from data_processing.utils import latex_float, latex_float_with_error

"""
Optical and Electrical Properties of Boron
Nobuyoshi Morita and Akira Yamamoto 1975 Jpn. J. Appl. Phys. 14 825
doi: 10.1143/JJAP.14.825
"""


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def main():
    n_df: pd.DataFrame = pd.read_csv(
        'Morita-Yakamoto1975_refractive_index_rhombohedral_boron.csv', comment='#'
    )
    k_df: pd.DataFrame = pd.read_csv(
        'Morita-Yakamoto1975_exctinction_coefficient_rhombohedral_boron.csv', comment='#'
    )
    n_df = n_df.apply(pd.to_numeric)
    k_df = k_df.apply(pd.to_numeric)
    wl_n_nm = n_df['Wavelength (um)'].values * 1000.
    wl_k_nm = n_df['Wavelength (um)'].values * 1000.
    refractive_index = n_df['n'].values
    extinction_coefficient = k_df['k'].values

    reflectance = (1.0003 - refractive_index) / (1.0003 + refractive_index)
    reflectance *= reflectance

    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.5)

    ax.plot(wl_n_nm, refractive_index, marker='o', ls='none', c='C0', label='n', mfc='none')
    ax.plot(wl_k_nm, extinction_coefficient, marker='s', ls='none', c='C1', label='k', mfc='none')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('n, k')

    ax.set_xlim(300, 1000.)
    ax.set_ylim(0, 3.5)

    ax.legend(
        loc='upper right', frameon=True
    )

    ax.set_title('Boron (Morita & Yakamoto 1975)')

    fig_r, ax_r = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig_r.set_size_inches(4.0, 3.5)

    ax_r.plot(wl_n_nm, reflectance*100., marker='o', ls='none', c='C0', label='n', mfc='none')

    ax_r.set_xlabel('Wavelength (nm)')
    ax_r.set_ylabel('R')

    ax_r.set_xlim(300, 1000.)
    ax_r.set_ylim(0, 100.)

    plt.show()


if __name__ == '__main__':
    main()
