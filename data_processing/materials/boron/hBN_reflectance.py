import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import os
import platform
import json

data_path = r'Documents/ucsd/Research/Literature/Boron/Boron nitride'
csv = 'Jin-Xiang 2009 hBN reflectance.csv'


platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'


def normalize_path(the_path):
    global platform_system, drive_path
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    return os.path.join(drive_path, the_path)


def main():
    global data_path
    base_path = normalize_path(base_path)
    r_df = pd.read_csv(os.path.join(base_path, csv))
    wl = r_df['Wavelength (nm)'].values
    reflectance = r_df['Reflectance (%)'].values

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

    f = interp1d(x=wl, y=reflectance)
    dx = 0.2
    n = (800 - 200) / dx + 1
    x_interp = 200. + dx * np.arange(0, n)
    y_interp = f(x_interp)

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('R (%)')

    ax.set_xlim(200., 800.)
    ax.set_ylim(0, 30.)

    r_800 = f(800)
    txt = fr'R($\lambda$=800 nm) = {r_800:.0f} %'

    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.plot(x_interp, y_interp)

    ax.text(
        0.95, 0.95, txt, transform=ax.transAxes, ha='right', va='top'
    )

    plt.show()



if __name__ == '__main__':
    main()