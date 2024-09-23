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

path_to_asc = './data/Echelle_data/echelle_20240910/MechelleSpect_001.asc'
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


def radiance_at_temperature(temperature: float, wavelength_nm: float, A=1.) -> float:
    hc = 6.62607015 * 2.99792458  # x 1E -34 (J s) x 1E8 (m/s) = 1E-26 (J m)
    hc2 = hc * 2.99792458  # x 1E -34 (J s) x 1E16 (m/s)^2 = 1E-18 (J m^2 s^{-1})
    factor = 2. * 1E14 * hc2 * np.power(wavelength_nm, -5.0)  # W / cm^2 / nm
    arg = 1E6 * hc / wavelength_nm / 1.380649 / temperature  # 26 (J m) / 1E-9 m / 1E-23 J/K
    return A * factor / (np.exp(arg) - 1.)

def model_bb(wavelength_nm: np.ndarray, b):
    temperature, factor = b[0], b[1]
    return factor * radiance_at_temperature(temperature=temperature, wavelength_nm=wavelength_nm)


def res_bb(b, x, y):
    return model_bb(wavelength_nm=x, b=b) - y


def main():
    global path_to_asc
    df, params = load_asc_file(path_to_asc)
    gain = float(params['Gain level'])
    pre_amplifier_gain = float(params['Pre-Amplifier Gain'])
    exposure_time = float(params['Exposure Time (secs)'])
    data_type = params['Data Type']
    calibration_temperature = float(params['Calibration Temperature'])

    wl = df['wl (nm)'].values
    counts = df['counts'].values
    n = len(counts)


    # y_pred = radiance_at_temperature(5000, x_pred, 1E5)
    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.)
    ax.plot(wl, counts)
    ax.set_xlabel(r'$\lambda$ {\sffamily (nm)}', usetex=True)
    ax.set_ylabel('Counts')

    params_txt = r'Gain:' + '\t' + rf'{gain:.0f}' + '\n'
    params_txt += r'Preamp gain:' + '\t' + rf'{pre_amplifier_gain:.0f}' + '\n'
    params_txt += r'Exposure time:' + '\t' + rf'{exposure_time:.3f} (s)' + '\n'
    params_txt += r'Cal temp:' + '\t' + rf'{calibration_temperature:.0f} K'

    ax.ticklabel_format(axis='y', useMathText=True)
    mf = ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(mf)

    ax.text(
        0.01, 0.98, params_txt,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=10,
        usetex=True
    )

    file_tag = os.path.splitext(os.path.basename(path_to_asc))[0]
    parent_dir = os.path.basename(os.path.dirname(path_to_asc))

    save_dir = os.path.join('./figures/Echelle_plots', parent_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    fig.savefig(os.path.join(save_dir, file_tag + '_plot.png'), dpi=600)
    plt.show()



if __name__ == '__main__':
    main()