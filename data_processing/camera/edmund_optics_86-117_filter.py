import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os, sys
sys.path.append('../')

from matplotlib import ticker, patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import json
from scipy import interpolate

data_file = './1025nm_50mm_OD4_SP_filter_86-117_curv_84733.csv'
laser_wavelength = 1070.0  # um

def main():
    optical_density_df = pd.read_csv(os.path.join(data_file), comment='#').apply(pd.to_numeric)
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)
    wavelength = optical_density_df['Wavelength (nm)'].values
    od = optical_density_df['O.D.'].values
    f = interpolate.interp1d(wavelength, od, bounds_error=False)
    wavelength_interp = np.linspace(465.0, 1365.0, 500)
    od_interp = f(wavelength_interp)
    fig, ax = plt.subplots(constrained_layout=True)
    fig.set_size_inches(4.75, 3.75)
    ax.set_xlabel('Wavelength (m,)')
    ax.set_ylabel('O.D.')
    ax.set_xlim(wavelength_interp.min(), wavelength_interp.max())
    ax.set_ylim(0, 10)

    ax.plot(
        wavelength_interp, od_interp, ls='-', label=f'Edmund Optics 86-117',
        c='C0',
    )

    ax.axvline(
        x=laser_wavelength, ls=':', lw=1.25, color='tab:gray'
    )

    connectionstyle = "angle3,angleA=0,angleB=90"
    bbox = dict(boxstyle="round", fc="wheat", alpha=1.0)
    arrowprops = dict(
        arrowstyle="->", color="k",
        shrinkA=5, shrinkB=5,
        patchA=None, patchB=None,
        connectionstyle=connectionstyle
    )

    od_at_laser_wl = f(laser_wavelength)
    offset = 10
    x1 = laser_wavelength
    y1 = od_at_laser_wl
    txt = f"Laser WL: {laser_wavelength:.3f} nm\nR = {od_at_laser_wl:.1f}"
    ax.annotate(
        txt,
        xy=(x1, y1), xycoords='data',  # 'figure pixels', #data',
        xytext=(-15 * offset, -2 * offset), textcoords='offset points',  # 'data',
        arrowprops=arrowprops,
        bbox=bbox, fontsize=9,
        ha='left'
    )

    ax.set_title('Edmund Optics 86-117 Shortpass filter')
    filetag = os.path.basename(data_file)

    fig.savefig(filetag + '.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()