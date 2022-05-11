import h5py
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os

from matplotlib import ticker, patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import json
from scipy import interpolate

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration'
data_file = 'reflectance_of_graphite_Taft&Philipp_PR1965'
laser_wavelength = 1.07  # um
band_pass_wavelength = 0.91 # um
interpolation_steps = 2000

def um_to_ev(energy: np.ndarray) -> np.ndarray:
    return 1.2398 / energy


if __name__ == '__main__':
    reflectivity_df = pd.read_csv(os.path.join(base_path, data_file + '.csv')).apply(pd.to_numeric)
    reflectivity_df.sort_values(by=['Photon Energy (eV)'], inplace=True)
    photon_energy = reflectivity_df['Photon Energy (eV)'].values
    wavelength = 1.2398 / photon_energy
    reflectivity = reflectivity_df['Reflectivity (%)'].values


    photon_energy = np.round(photon_energy, 3)

    e_min, e_max = photon_energy.min(), photon_energy.max()

    f1 = interpolate.interp1d(photon_energy, reflectivity, kind='slinear')
    photon_energy_interp = np.linspace(e_min, e_max, interpolation_steps)
    reflectivity_interp = f1(photon_energy_interp)

    f2 = interpolate.interp1d(wavelength, reflectivity, kind='slinear')
    wl_min, wl_max = np.round(wavelength.min(), 3), np.round(wavelength.max(), 3)
    print(wl_min, wl_min)
    wavelength_interp = np.linspace(wl_min, wl_max, interpolation_steps)
    wavelength_interp = wavelength_interp[3:-3]
    reflectivity_interp_wl = f2(wavelength_interp)

    reflectivity_at_laser_wl = f1(1.2398/laser_wavelength)
    reflectivity_at_bp_wl = f1(1.2398/band_pass_wavelength)
    print(f"Reflectance at {laser_wavelength:.3f} um: {reflectivity_at_laser_wl:4.1f} %")
    print(f"Reflectance at {band_pass_wavelength:.3f} um: {reflectivity_at_bp_wl:4.1f} %")

    reflectivity_df = pd.DataFrame(data={
        'Photon Energy (eV)': photon_energy_interp,
        'Reflectivity': reflectivity_interp
    })
    reflectivity_df.to_csv(
        os.path.join(base_path, data_file + '_eV.csv')
    )

    reflectivity_df = pd.DataFrame(data={
        'Wavelength (um)': wavelength_interp,
        'Reflectivity': reflectivity_interp_wl
    })
    reflectivity_df.to_csv(
        os.path.join(base_path, data_file + '_um.csv')
    )

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots()  # , constrained_layout=True)
    fig.set_size_inches(4.75, 3.75)
    ax.set_xlabel('Wavelength ($\\mathregular{\mu m}$)')
    ax.set_ylabel('Reflectance (%)')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.plot(
        wavelength_interp, reflectivity_interp_wl, ls='-', label=f'Taft & Phillip 1965',
        c='C0',
    )

    ax.axvline(
        x=laser_wavelength, ls=':', lw=1.25, color='tab:gray'
    )

    ax.plot(
        [laser_wavelength], [reflectivity_at_laser_wl], ls='none', marker='o',
        color='tab:red', fillstyle='none', mew=1.75
    )

    ax.plot(
        [band_pass_wavelength], [reflectivity_at_bp_wl], ls='none', marker='o',
        color='tab:red', fillstyle='none', mew=1.75
    )

    ax.set_xlim(0, 5.0)
    ax.set_ylim(30, 100.0)
    idx_25 = (np.abs(reflectivity_at_laser_wl - 25)).argmin()
    wl_range = np.array([5.0, wavelength_interp[idx_25]])
    pe_range = 1.2398 / wl_range
    idx_range = np.array([(np.abs(photon_energy_interp - e)).argmin() for e in pe_range], dtype=int)[::-1]
    r_range = np.array([reflectivity_interp[i] for i in idx_range])
    r_range[0] = 25.0
    print('Wavelength Range: ', wl_range)
    print('Photon Energy Range: ', pe_range)
    print('Index Range: ', idx_range)
    print('Reflectance Range: ', r_range)

    ax.ticklabel_format(useMathText=True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.5))

    # ax2 = ax.twiny()

    # print(xmin, xmax)
    #
    # ax2.set_xlim(um_to_ev(xmin), um_to_ev(xmax))
    # ax2.spines['top'].set_position(("axes", 1.0))
    # ax2.set_xlabel('Photon Energy (eV)')

    connectionstyle = "angle3,angleA=0,angleB=90"
    bbox = dict(boxstyle="round", fc="wheat", alpha=1.0)
    arrowprops = dict(
        arrowstyle="->", color="k",
        shrinkA=5, shrinkB=5,
        patchA=None, patchB=None,
        connectionstyle=connectionstyle
    )
    offset = 5
    x1 = laser_wavelength
    y1 = reflectivity_at_laser_wl
    txt = f"Laser WL: {laser_wavelength:.3f} $\mathregular{{\mu m}}$\nR = {reflectivity_at_laser_wl:.1f} %"
    ax.annotate(
        txt,
        xy=(x1, y1), xycoords='data',  # 'figure pixels', #data',
        xytext=(-8*offset, 8 * offset), textcoords='offset points',  # 'data',
        arrowprops=arrowprops,
        bbox=bbox, fontsize=9,
        ha='left'
    )

    offset = 3
    x1 = band_pass_wavelength
    y1 = reflectivity_at_bp_wl
    txt = f"IR Thermography WL: {band_pass_wavelength:.3f} $\mathregular{{\mu m}}$\nR = {reflectivity_at_bp_wl:.1f} %"
    ax.annotate(
        txt,
        xy=(x1, y1), xycoords='data',  # 'figure pixels', #data',
        xytext=(20 * offset, -2.5 * offset), textcoords='offset points',  # 'data',
        arrowprops=arrowprops, fontsize=9,
        bbox=bbox,
        ha='left'
    )

    ax.set_title('Reflectance of graphite (Taft & Philipp 1965)')

    axins = inset_axes(ax, width=1.95, height=1.2)
    axins.plot(photon_energy_interp, reflectivity_interp)
    axins.set_xlim(0,40)
    axins.set_ylim(0.1,100)
    axins.set_yscale('log')
    axins.set_xlabel('Photon Energy (eV)', fontsize=10)
    axins.set_ylabel('Reflectance (%)', fontsize=10)
    axins.tick_params(axis='both', labelsize=8)
    axins.xaxis.set_major_locator(ticker.MultipleLocator(10))
    axins.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))

    xy = (pe_range.min(), r_range[0])
    p_width = abs(pe_range[1] - pe_range[0])
    p_height = abs(r_range[1] - r_range[0])
    rect = patches.Rectangle(xy, p_width, p_height, linewidth=1, edgecolor='r', facecolor='none')
    axins.add_patch(rect)
    # ax.legend(loc='best', frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(base_path, data_file + '_plot.png'), dpi=600)
    fig.savefig(os.path.join(base_path, data_file + '_plot.svg'), dpi=600)
    plt.show()
