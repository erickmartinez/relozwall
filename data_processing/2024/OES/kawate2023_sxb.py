import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
import os
import numpy as np


def x_rate(T_e: np.ndarray) -> np.ndarray:
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

def s_rate(T_e: np.ndarray) -> np.ndarray:
    """
        Estimates the ionization rate coefficient from the
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
            The ionization rate coefficient in cm^3/s

        """
    return 1.46E-8 * np.power(T_e, 0.690) * np.exp(-9.38 / T_e)

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    plt.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}')

def main():
    temp_e = np.logspace(-1, 2, 500)
    k_x = x_rate(temp_e)
    k_s = s_rate(temp_e)
    branching_ratio = 1.
    sxb = k_s / k_x / branching_ratio
    wl = 432.9 # nm
    temp_e_linder = 15.
    ne_linder = 1E19
    sxb_linder = 1.2

    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3.)

    ax.plot(temp_e, sxb, color='C0', label='Kawate 2023')
    ax.plot([temp_e_linder], [sxb_linder], ls='none', marker='x', color='k', label='Linder 1994', mew=1.2)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'$T_{e}$ {\sffamily (eV)}', usetex=True)
    ax.set_ylabel(r'S/XB')

    ax.legend(loc='upper left', frameon=True)


    ax.set_xlim(0.1, 100)
    ax.set_ylim(1E-4, 1E1)

    plt.show()

if __name__ == '__main__':
    main()