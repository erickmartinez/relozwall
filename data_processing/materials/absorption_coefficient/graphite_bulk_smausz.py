import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from data_processing.utils import latex_float, latex_float_with_error

probe_wl_nm = 650.


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def main():
    global probe_wl_nm
    alpha_err = 0.3  # ~30 % according to plot
    alpha_df = pd.read_csv('Smausz2017_alpha_ellipsometry.csv', comment='#').apply(pd.to_numeric)
    wl = alpha_df['Wavelength (nm)'].values
    alpha = alpha_df['alpha (1/um)'].values #* 1E4
    f = interp1d(x=wl, y=alpha)
    alpha_probe = f(probe_wl_nm)
    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$\alpha_{\mathregular{ellip}}$ ($\times$ 10$^{\mathregular{4}}$ cm$^{\mathregular{-1}}$)')

    ax.fill_between(wl, alpha*(1.-alpha_err), alpha*(1+alpha_err), color='C0', alpha=0.3, ec='none')
    ax.plot(wl, alpha, color='C0')
    ax.plot([probe_wl_nm], [alpha_probe], marker='o', color='tab:red')

    offset = 50
    connectionstyle = "angle3,angleA=0,angleB=90"
    bbox = dict(boxstyle="round", fc="wheat")
    arrowprops = dict(
        arrowstyle="->", color="0.5",
        shrinkA=5, shrinkB=5,
        patchA=None, patchB=None,
        connectionstyle=connectionstyle
    )
    alpha_probe_err = alpha_probe*alpha_err
    alpha_probe_ltx = latex_float_with_error(value=alpha_probe*1E4, error=alpha_probe_err*1E4, digits=2)
    ax.annotate(
        rf"$\alpha$(${probe_wl_nm:.0f}$ nm) = ${alpha_probe_ltx}$ cm$^{{\mathregular{{-1}}}}$",
        xy=(probe_wl_nm, alpha_probe), xycoords='data',  # 'figure pixels', #data',
        xytext=(0, 2 * offset), textcoords='offset points',  # 'data',
        arrowprops=arrowprops,
        bbox=bbox,
        ha='center'
    )

    ax.set_xlim(200, 1000)
    ax.set_ylim(0., 20.)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.set_title('Smausz 2017 (ellipsometry)')

    fig.savefig(f'smausz_absorption_coefficient_wl_{probe_wl_nm:.0f}nm.png', dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
