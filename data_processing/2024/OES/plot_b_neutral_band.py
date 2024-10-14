import os.path

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import json
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf
from experiments.calibrate_mx200 import save_path
from scipy.interpolate import interp1d

path_to_asc1 = './data/Echelle_data/echelle_20240815/MechelleSpect_003.asc'
path_to_asc2 = './data/Echelle_data/echelle_20240827/MechelleSpect_018.asc' #12
f_pattern = re.compile(r'(.*)?\:\s+(.*)')

# Python implementation to
# read last N lines of a file
# through Exponential search

# Function to read
# last N lines of the file
def LastNlines(fname, N):
    # assert statement check
    # a condition
    assert N >= 0

    # declaring variable
    # to implement
    # exponential search
    pos = N + 1

    # list to store
    # last N lines
    lines = []

    # opening file using with() method
    # so that file get closed
    # after completing work
    with open(fname) as f:

        # loop which runs
        # until size of list
        # becomes equal to N
        while len(lines) <= N:

            # try block
            try:
                # moving cursor from
                # left side to
                # pos line from end
                f.seek(-pos, 2)

            # exception block
            # to handle any run
            # time error
            except IOError:
                f.seek(0)
                break

            # finally block
            # to add lines
            # to list after
            # each iteration
            finally:
                lines = list(f)

            # increasing value
            # of variable
            # exponentially
            pos *= 2

    # returning the
    # whole list
    # which stores last
    # N lines
    return lines[-N:]

def load_asc_file(path_to_file):
    df = pd.read_csv(
        path_to_file,  sep=r'\s+', engine='python',
        usecols=[0, 1],
        names=['wl (nm)', 'counts']
    ).apply(pd.to_numeric, errors='coerce').dropna()
    # read the last 29 lines of the file
    footer = LastNlines(path_to_file, 29)

    params = {}
    for line in footer:
        matches = f_pattern.match(line)
        if not matches is None:
            params[matches.group(1)] = matches.group(2)
    return df, params

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    plt.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}')


def transmission_window(wl):
    return 12.783 + 0.13065 * wl - 8.467e-5 * wl **2.

def main():
    global path_to_asc1, path_to_asc2
    labsphere_df = pd.read_csv(
        './data/PALabsphere_2014.txt', sep=' ', comment='#',
        usecols=[0], names=['Radiance (W/cm2/ster/nm)']
    ).apply(pd.to_numeric)
    radiance = labsphere_df['Radiance (W/cm2/ster/nm)']
    n = len(radiance)
    wl = 350. + np.arange(n) * 10.


    radiance_interp = interp1d(x=wl, y=radiance)

    br_df, params_br = load_asc_file(path_to_asc1)
    bp_df, params_bp = load_asc_file(path_to_asc2)

    # gain = float(params_bp['Gain level'])
    # pre_amplifier_gain = float(params_bp['Pre-Amplifier Gain'])
    # exposure_time = float(params_bp['Exposure Time (secs)'])
    # data_type = params_bp['Data Type']
    # calibration_temperature = float(params_bp['Calibration Temperature'])

    wl_br= br_df['wl (nm)'].values
    counts_br = br_df['counts'].values
    n_br = len(counts_br)

    wl_bp = bp_df['wl (nm)'].values
    counts_bp = bp_df['counts'].values
    n_bp = len(counts_bp)

    msk_br = (818. <= wl_br) & (wl_br <=824)
    msk_bp = (818. <= wl_bp) & (wl_bp <= 824)

    # y_pred = radiance_at_temperature(5000, x_pred, 1E5)
    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.)
    ax.plot(wl_br[msk_br], counts_br[msk_br], c='C0', label='Boron rod')
    ax.plot(wl_bp[msk_bp], counts_bp[msk_bp], c='C1', label='Boron pebble rod')
    ax.set_xlabel(r'$\lambda$ {\sffamily (nm)}', usetex=True)
    ax.set_ylabel('Counts')

    # params_txt = r'Gain:' + '\t' + rf'{gain:.0f}' + '\n'
    # params_txt += r'Preamp gain:' + '\t' + rf'{pre_amplifier_gain:.0f}' + '\n'
    # params_txt += r'Exposure time:' + '\t' + rf'{exposure_time:.3f} (s)' + '\n'
    # params_txt += r'Cal temp:' + '\t' + rf'{calibration_temperature:.0f} K'

    ax.ticklabel_format(axis='y', useMathText=True)
    mf = ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(mf)

    # ax.text(
    #     0.01, 0.98, params_txt,
    #     transform=ax.transAxes,
    #     ha='left', va='top',
    #     fontsize=10,
    #     usetex=True
    # )


    fig.savefig('./figures/b_neutral_spectra.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
