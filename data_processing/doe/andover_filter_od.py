import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib import ticker
from utils import get_experiment_params

transmission_csv = '../../data/100FL07.csv'
output_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera'

target_wl = 1070.0

def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


if __name__ == '__main__':
    filter_params = get_experiment_params(
        os.path.dirname(transmission_csv), os.path.splitext(os.path.basename(transmission_csv))[0]
    )
    transmission_df = pd.read_csv(transmission_csv, comment='#').apply(pd.to_numeric)
    transmission_df['OD'] = -np.log10(transmission_df['Tx. (%)'] / 100.0)
    wavelength_nm = transmission_df['Wavelength (nm)'].values
    optical_density = transmission_df['OD'].values
    f = interp1d(wavelength_nm, optical_density)
    od_at_target_wl = f(target_wl)
    part_no = filter_params['Part No.']['value']

    load_plt_style()

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)

    ax.plot(wavelength_nm, optical_density)
    ax.plot([target_wl], [od_at_target_wl], 'or', ms=8, mew=1.25, mfc='None', )

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Optical density')
    ax.set_xlim(850, 1100)
    ax.set_ylim(-0.25, 3.5)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(50.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(25.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))


    ax.set_title(f'Andover {part_no}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.annotate(
        f"{od_at_target_wl:.1f}\n({target_wl:.0f} nm)",
        xy=(target_wl, od_at_target_wl), xycoords='data',
        xytext=(0.5, 0.7), textcoords=ax.transAxes,
        ha='center', va='top',
        bbox=props, fontsize=9,
        arrowprops=dict(
            arrowstyle="->", color="0.5",
            shrinkA=10, shrinkB=5,
            patchA=None, patchB=None,
            connectionstyle='angle,angleA=-90,angleB=180,rad=5'
        )
    )

    fig.savefig(
        os.path.join(output_dir, os.path.splitext(os.path.basename(transmission_csv))[0] + '_optical_density.png'),
        dpi=300
    )

    plt.show()
