import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json

cte = 1E-4 # /°C
T0 = 22.0 # °C
diameters = [0.91, 1.5]

def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['xtick.top'] = True

def t2dt(val):
    return val - T0

def convert_ax_t2dt(ax_t:plt.axis):
    x1, x2 = ax_t.get_xlim()

def main():
    load_plt_style()
    d0 = np.array(diameters)
    temperature = np.linspace(T0, 280.0)
    dT = t2dt(temperature)
    dL = cte * np.outer(dT, d0)
    d = d0 + dL

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)

    ax.plot(temperature, d[:,0])

    ax2 = ax.twiny()

