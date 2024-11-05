import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import os
import json
from scipy.optimize import least_squares, OptimizeResult
from scipy.interpolate import interp1d
import data_processing.confidence as cf
from matplotlib.lines import Line2D
import matplotlib as mpl


spectrum_csv = r'./data/brightness_data_fitspy/echelle_20240815/MechelleSpect_007.csv'

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    plt.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}')


fit_points = []

def onpick1(event):
    if isinstance(event.artist, Line2D):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        clicked_points = np.column_stack([xdata[ind], ydata[ind]])
        # print('onpick1 line:', np.column_stack([xdata[ind], ydata[ind]]))
        print('onpick1 line:', clicked_points.mean(axis=0))


def main():
    global spectrum_csv, onpick1
    spectrum_df = pd.read_csv(spectrum_csv).apply(pd.to_numeric)
    wl = spectrum_df['Wavelength (nm)'].values
    brightness = spectrum_df['Brightness (photons/cm^2/s/nm)'].values
    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(6.5, 5.)
    ax.set_xlabel(r"$\lambda$ {\sffamily (nm)}", usetex=True)
    ax.set_ylabel(r"B (photons/cm$^{\mathregular{2}}$/s/nm)")

    ax.set_ylim(bottom=-1E11, top=4E12)

    line = ax.plot(
        wl, brightness, ms=6, color='C0', mfc='none', picker=True, pickradius=1, ls='none', marker='o'
    )

    fig.canvas.mpl_connect('pick_event', onpick1)

    plt.show()

if __name__ == '__main__':
    main()


