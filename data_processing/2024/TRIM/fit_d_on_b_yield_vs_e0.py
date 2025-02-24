"""
This code fits the experimental data points from Eckstein's 1992 report [1] to
the Eckstein-Preuss model.

[1] E. Hechtl, A. Mazanec, W. Eckstein, J. Roth, C. Garcia-Rosales,
Sputtering behavior of boron and boron carbide,
Journal of Nuclear Materials,
Volumes 196–198,
1992,
Pages 713-716,
ISSN 0022-3115,
https://doi.org/10.1016/S0022-3115(06)80129-7.

Erick R Martinez Loran
erickrmartinez@gmail.com
"""
import numpy as np
import sputtering_yield as spty
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf # <- You might need to change this `import confidence as cf`
import json
import os
import matplotlib as mpl


"""
E. Hechtl, A. Mazanec, W. Eckstein, J. Roth, C. Garcia-Rosales,
Sputtering behavior of boron and boron carbide,
Journal of Nuclear Materials,
Volumes 196–198,
1992,
Pages 713-716,
ISSN 0022-3115,
https://doi.org/10.1016/S0022-3115(06)80129-7.
"""
DATA = {
    'Energy (eV)': [20., 30., 40., 80., 400., 1000., 2000., 8000.],
    'Y': [0.00460, 0.01140, 0.02460, 0.02730, 0.02840, 0.02750, 0.01630, 0.00510]
}

# Change the path to the right location in your system
PATH_TO_TRIM_YIELDS = r'./data/simulations/SDTrimSP/D_ON_B.csv'

E_b = 5.73 # Surface binding energy (eV)
Z1 = 1
Z2 = 5
M1 = 2.014
M2 = 10.811

def model(e, params):
    """
    Use Eckstein-Preuss model
    Parameters
    ----------
    e: np.ndarray
        The ion energies (eV)
    params: tuple, np.ndarray
        The parameters in Eckstein-Preuss model

    Returns
    -------
    np.ndarray
        The sputtering yields at the ion energies.
    """
    q, lam, mu, E_th, m1, m2, z1, z2, sbe = params
    y = spty.estimate_yield_eckstein_preuss(e, q, lam, E_th, mu, m1, m2, z1, z2, sbe)
    return y

def residual(b, energy, sputtering_yield, m1, m2, z1, z2, sbe):
    """
    The objective function to be minimized by the least_squares routine
    Parameters
    ----------
    b: tuple
        The parameters to be fitted
    energy: np.ndarray
        The energies of the ions (eV)
    sputtering_yield: np.ndarray
        The observed sputtering yields
    m1: float
        The mass of the ion (amu)
    m2: float
        The mass of the target atom (amu)
    z1: float
        The atomic number of the ion
    z2: float
        The atomic number of the target atom
    sbe: float
        The surface binding energy of the target atoms (eV)

    Returns
    -------
    np.ndarray
        The residuals at the parameters
    """
    q, lam, mu, E_th = b
    E_th_estimate = spty.estimate_eth(m1, m2, sbe)
    params = (q, lam, mu, E_th, m1, m2, z1, z2, sbe)
    sputtering_yield_ep = model(energy, params)
    res = (sputtering_yield_ep - sputtering_yield) / sputtering_yield
    if E_th > E_th_estimate:
        res *= 1000
    return res

def fit_eckstein(energy, sputtering_yield, m1, m2, z1, z2, sbe, x0=None, loss='soft_l1', f_scale=1.0, tol=None):
    """
    Attempts to fit the experimental values of the sputtering yield to the Eckstein-Preuss model

    Parameters
    ----------
    energy: np.ndarray
        The observed energies of the ions (eV)
    sputtering_yield: np.ndarray
        The observed sputtering yield at the ion energies
    m1: float
        The mass of the ion (amu)
    m2: float
        The mass of the target atom (amu)
    z1: float
        The atomic number of the ion
    z2: float
        The atomic number of the target atom
    sbe: float
        The surface binding energy of the target atom (eV)
    x0: list
        The initial guess for the parameters. If None, makes its own guess
    loss: str
        The loss to be used in optimize.least_squares minimizer
    f_scale: float
        The f_scale parameter used to discriminate outliers in the data
    tol: float
        The tolerance used to determine convergence of the minimizer. If not provided, use the machine epsilon.

    Returns
    -------

    """
    if tol is None:
        tol = float(np.finfo(np.float64).eps)

    if x0 is None:
        x0 = [0.13, 10, 0.5, 13.]
    result: OptimizeResult = least_squares(
        fun=residual,
        x0=x0,
        args=(energy, sputtering_yield, m1, m2, z1, z2, sbe),
        bounds=([1E-6, 1E-6, 1E-5, 1E-8], [1E5, 1E5, 10., 30]),
        loss=loss,
        f_scale=f_scale,
        xtol=tol,
        gtol=tol,
        ftol=tol,
        max_nfev=1000*len(x0),
        # diff_step=tol,
        x_scale='jac',
        jac='3-point',
        method='trf',
        verbose=2
    )
    return result

def load_plot_style():
    """
    Loads the style of the plot
    """
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')

def main(data, m1, m2, z1, z2, sbe, path_to_trim_yields):
    """
    Main function

    Parameters
    ----------
    data: dict
        The experimental sputtering yield
    m1: float
        The mass of the ion (amu)
    m2: float
        The mass of the target atom (amu)
    z1: float
        The atomic number of the ion
    z2: float
        The atomic number of the target atom
    sbe: float
        The surface binding energy of the target atom (eV)
    path_to_trim_yields: str
        The path to the sputtering yield calculated by SDTrimSP


    """
    df = pd.DataFrame(data=data) # Load the experimental data into a dataframe
    trim_df = pd.read_csv(path_to_trim_yields).apply(pd.to_numeric) # load the TRIM results
    energy = df['Energy (eV)'].values
    sputtering_yield = df['Y'].values
    # Fit the data
    fit_result = fit_eckstein(energy, sputtering_yield, m1, m2, z1, z2, sbe, loss='cauchy', f_scale=0.1)

    q, lam, mu, E_th = fit_result.x # Get the optimized parameters from the fit

    # Create an array of energies for the prediction
    energy_pred = np.logspace(start=np.log10(E_th+0.002), stop=np.log10(energy.max()), num=1000)

    # Define a model that only uses the optimized q, lam, mu, E_th from the fit and m1, m2, z1, z2, sbe set before
    def model_ep(e, b):
        q, lam, mu, E_th = b
        params = (q, lam, mu, E_th, m1, m2, z1, z2, sbe)
        return model(e, params)

    # Get the parameters confidence interval
    popt = fit_result.x
    ci = cf.confidence_interval(res=fit_result)
    popt_delta = np.abs(ci[:,1] - popt)

    print(f"q:\t {q:.4f} -/+ {popt_delta[0]:.5f}")
    print(f"lambda:\t {lam:.4f} -/+ {popt_delta[1]:.5f}")
    print(f"mu:\t {mu:.4f} -/+ {popt_delta[2]:.5f}")
    print(f"Eth:\t {E_th:.4f} -/+ {popt_delta[3]:.5f}")

    # Estimate the sputtering yield at the new points and their corresponding prediction bands
    eckstein_yield, delta = cf.prediction_intervals(model=model_ep, x_pred=energy_pred, ls_res=fit_result)

    # Get the prediction at the ion energies of interest for the beam composition
    ion_energies = np.array([11.67, 14., 16.67, 17.5, 21., 25.0, 35, 42., 50])
    yields_at_energies, yields_delta_at_energies = cf.prediction_intervals(model=model_ep, x_pred=ion_energies, ls_res=fit_result)
    # Save the predicted sputtering yield at the ion energies of interest for the beam composition
    output_df = pd.DataFrame(data={
        'Ion energy (eV)': ion_energies,
        'Sputtering yield': yields_at_energies,
        'Sputtering yield delta': yields_delta_at_energies
    })

    output_df.to_csv(r'./data/eckstein_preuss_yields.csv', index=False)

    # Plot the sputtering yield
    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(3.5, 4.0)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(energy, sputtering_yield, marker='D', ls='none', mfc='none', mec='C0', label='Experimental (Eckstein)')
    ax.plot(trim_df['Incident energy (eV)'].values, trim_df['Ysum'], marker='D', ls='none', mfc='C0', mec='C0', label='SDTrimSP')
    ax.plot(energy_pred, eckstein_yield, ls='-', color='tab:red', label='Eckstein-Preuss model')
    ax.fill_between(energy_pred, eckstein_yield-delta, eckstein_yield + delta, color='tab:red', alpha=0.2)

    ax.set_xlabel('Incident energy (eV)')
    ax.set_ylabel('Sputtering yield')

    ax.set_xlim(10, 1E4)
    ax.set_ylim(1E-3, 1E0)

    ax.legend(loc='upper left')
    ax.set_title(r"{\sffamily D\textsuperscript{+} → B}", usetex=True)

    fig.savefig(r'./figures/eckstein_preuss_sputtering_yield.png', dpi=600)

    plt.show()

if __name__ == '__main__':
    main(data=DATA, z1=Z1, z2=Z2, m1=M1, m2=M2, sbe=E_b, path_to_trim_yields=PATH_TO_TRIM_YIELDS)





