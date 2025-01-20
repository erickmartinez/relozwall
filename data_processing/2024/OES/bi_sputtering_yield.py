import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
import os
from scipy.optimize import least_squares, OptimizeResult

SPUTTERING_RATES_CSV = r'./data/bi_lorentzian.xlsx'
FOLDER_MAP_XLS = r'./PISCES-A_folder_mapping.xlsx'  # Folder name to plot label database

axes_mapping = {
    'echelle_20240815': 0, 'echelle_20240827': 1, 'echelle_20241003': 1, 'echelle_20241031': 0
}

lbl_mapping = {
    'echelle_20240815': 'Boron rod (~500 °C)',
    'echelle_20240827': 'Boron pebble rod (amorphous)',
    'echelle_20241003': 'Boron pebble rod (poly-C)',
    'echelle_20241031': 'Boron rod (~1000 °C)'
}

color_maping = {
    'echelle_20240815': 'C0', 'echelle_20240827': 'C1', 'echelle_20241003': 'C2', 'echelle_20241031': 'tab:red'
}

marker_mapping = {
    'echelle_20240815': 's', 'echelle_20240827': 'o', 'echelle_20241003': 'D', 'echelle_20241031': '^'
}

def bh_x_rate(T_e) -> np.ndarray:
    """
    Estimates the excitation rate coefficient from the
    ground state of B-H for the transition:

    .. math::\Chi^1 \Sigma^+ \to \mathrm{A}^1\Pi

    as a function of the electron temperature.

    This relationship corresponds to the modified Arrhenius function
    .. math:: k = A T_e^n\exp\left(-\frac{E_{\mathrm{act}}{T_e}\right)

    described in Kawate et al. Plasma Sources Sci. Technol. 32, 085006 (2023)
    doi: 10.1088/1361-6595/acec0c


    Parameters
    ----------
    T_e: np.ndarray
        The electron temperature in eV

    Returns
    -------
    np.ndarray:
        The excitation rate coefficient in cm^3/s

    """
    return 5.62E-8 * np.power(T_e, 0.021) * np.exp(-3.06 / T_e)

# Constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
m_B = 10.811 * 1.66053906660e-27  # Mass of boron atom (kg)
m_e = 9.10938356e-31  # Mass of electron (kg)
m_D = 2.014 * 1.66053906660e-27  # Mass of deuterium (kg)
def thermal_velocity(T, m):
    """
    Calculate thermal velocity
    T: temperature (K)
    m: mass of particle (kg)
    """
    global k_B
    return np.sqrt(8 * k_B * T / (np.pi * m)) * 100
    # return np.sqrt(2. * k_B * T / m)


def pec2flux(vth, intensity, n_e, pec, intensity_error):
    L = 1. # cm
    fb = vth * intensity / n_e / L / pec
    fb_err = np.abs(fb) * np.sqrt(np.power(intensity_error / intensity, 2.) + (0.05/L)**2. + (0.2*vth/vth)**2.)
    return fb, fb_err


sample_diameter = 1.016
sample_area = 0.25 * np.pi * sample_diameter ** 2.
flux_d = 0.23E18 # /cm^3/s
def flux2yield(flux_b):
    global flux_d
    return flux_b / flux_d

def yield2flux(x):
    global flux_d
    return x * flux_d


def model_poly(x, b) -> np.ndarray:
    n = len(b)
    r = np.zeros(len(x))
    for i in range(n):
        r += b[i] * x ** i
    return r


def res_poly(b, x, y, w=1.):
    return (model_poly(x, b) - y) * w


def jac_poly(b, x, y, w=1):
    n = len(b)
    r = np.zeros((len(x), n))
    for i in range(n):
        r[:, i] = w * x ** i
    return r


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}')


def load_folder_mapping():
    global FOLDER_MAP_XLS
    df = pd.read_excel(folder_map_xls, sheet_name=0)
    mapping = {}
    for i, row in df.iterrows():
        mapping[row['Echelle folder']] = row['Data label']
    return mapping


def main():
    global SPUTTERING_RATES_CSV, m_B
    # load the fitted lorentzian peaks
    bi_df = pd.read_excel(bi_lorentzian_xls, sheet_name=0)
    folder_mapping = load_folder_mapping()
    folders = bi_df['Folder'].unique()
    n_plots = len(folders)

    # fig_cols = max(int(n_plots * 0.5), 1)
    # fig_rows = max(int(n_plots / fig_cols), 1)

    load_plot_style()
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True, sharex=True)
    fig.set_size_inches(4.5, 5.)

    pec_bi = 5.1E-11
    n_e = 0.212E12  # 1/cm^3

    markers = ['^', 's', 'o', 'v']
    colors = ['C0', 'C1', 'C2', 'C3']
    for i, folder in enumerate(folders):
        ax = axes[axes_mapping[folder]]
        lbl = lbl_mapping[folder]
        marker = marker_mapping[folder]
        temperature_boron = 500 if folder != 'echelle_20241031' else 1010.
        vth_B = thermal_velocity(T=temperature_boron, m=m_B)
        pec_bd = bh_x_rate(T_e=temperature_boron)
        # Select the areas of the peaks for that folder
        idx_folder = bi_df['Folder'] == folder
        folder_df = bi_df[idx_folder].sort_values(by=['Elapsed time (s)'])
        time_s = folder_df['Elapsed time (s)'].values
        intensity = folder_df['BI H (photons/cm^2/s)'].values
        intensity_ub = folder_df['BI H ub (photons/cm^2/s)'].values
        intensity_err = np.abs(intensity_ub - intensity)
        fb, fb_err = pec2flux(vth=vth_B, intensity=intensity, n_e=n_e, pec=pec_bi,
                              intensity_error=intensity_err)

        eps = float(np.finfo(np.float64).eps)
        # ls_fit = least_squares(
        #     res_poly, x0=[0.01**i for i in range(5)], args=(time_s, fb),
        #     loss='soft_l1', f_scale=0.1,
        #     jac=jac_poly,
        #     xtol=eps,
        #     ftol=eps,
        #     gtol=eps,
        #     verbose=2,
        #     x_scale='jac',
        #     max_nfev=10000 * len(time_s)
        # )

        markers_b, caps_b, bars_b = ax.errorbar(
            time_s/60., fb, yerr=fb_err, capsize=2.75, mew=1.25, marker=marker, ms=8, elinewidth=1.25,
            color=color_maping[folder], fillstyle='none',
            ls='none',# lw=1.25,
            label=lbl,
        )

        # ax.plot(time_s/60., model_poly(time_s, ls_fit.x), color=colors[i], ls='--', lw=1)
        ax.set_yscale('log')
        ax.legend(loc='upper left', frameon=True, fontsize=10)
        ax.set_ylim(1E10, 1E16)
        secax = ax.secondary_yaxis('right', functions=(flux2yield, yield2flux))
        secax.set_ylabel(r'$Y_{\mathrm{B-H/D^+}}$', usetex=True)

        [bar.set_alpha(0.35) for bar in bars_b]
    axes[-1].set_xlabel(r'Time (min)', usetex=False)
    axes[-1].set_xlim(0, 100)
    axes[0].set_title('B-I sputtering')
    for ax in axes:
        ax.axhline(y=yield2flux(6E-3), ls='--', lw=1., color='0.5')
    fig.supylabel(r"$\Gamma_{\mathrm{B-D}}$ {\sffamily (cm\textsuperscript{-2} s\textsuperscript{-1})}", usetex=True)
    fig.savefig('./figures/bi_sputtering_yield.png', dpi=600)
    fig.savefig('./figures/bi_sputtering_yield.svg', dpi=600)
    plt.show()



if __name__ == '__main__':
    main()


