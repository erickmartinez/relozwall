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
import logging
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

GRAZING_SIM_CSV = r"erosion_simulations.csv"
ANGLE = 78.5
DATA = {
    'Energy (eV)': [20., 30., 40., 80., 400., 1000., 2000., 8000.],
    'Y': [0.00460, 0.01140, 0.02460, 0.02730, 0.02840, 0.02750, 0.01630, 0.00510]
}

# Change the path to the right location in your system
PATH_TO_TRIM_YIELDS = r'../D_ON_B_normal_incidence.csv'

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

def model_angular_dependence(e, params):
    """
    Use Eckstein-Preuss model with angular dependence
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
    q, lam, mu, E_th, m1, m2, z1, z2, sbe, angle, f, b, c = params
    y = spty.estimate_yield_eckstein_preuss_with_angle_dependence(
        e, q, lam, E_th, mu, m1, m2, z1, z2, sbe, angle, f, b, c
    )
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
    res = (sputtering_yield_ep - sputtering_yield) #/ sputtering_yield
    # if E_th > E_th_estimate:
    #     res *= 1000
    return res

def residual_angular_dependence(b, energy, sputtering_yield, m1, m2, z1, z2, sbe, angle):
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
    angle: float
        The angle of incidence

    Returns
    -------
    np.ndarray
        The residuals at the parameters
    """
    q, lam, mu, E_th, f, b, c = b
    E_th_estimate = spty.estimate_eth(m1, m2, sbe)
    params = (q, lam, mu, E_th, m1, m2, z1, z2, sbe, angle, f, b, c)
    sputtering_yield_ep = model_angular_dependence(energy, params)
    res = (sputtering_yield_ep - sputtering_yield)# / sputtering_yield
    # if E_th > E_th_estimate:
    #     res *= 1000
    return res

def fit_eckstein(energy, sputtering_yield, m1, m2, z1, z2, sbe, angle=0, x0=None, loss='soft_l1', f_scale=1.0, tol=None):
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
    angle: float
        The angle of incidence
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
        if angle != 0:
            x0.append(30.)  # < initial guess for f
            x0.append(15.)  # < initial guess for b
            x0.append(0.5)  # < initial guess for c


    residual_func= residual
    bounds = ([1E-6, 1E-6, 1E-5, 1E-8], [1E5, 1E5, 10., 30])
    args = (energy, sputtering_yield, m1, m2, z1, z2, sbe)
    if angle != 0:
        residual_func = residual_angular_dependence
        args = (energy, sputtering_yield, m1, m2, z1, z2, sbe, angle)
        bounds = ([1E-6, 1E-6, 1E-5, 1E-8, 1E-9, 1E-9, 1E-9], [1E5, 1E5, 10., 30, 1E3, 1E3, 1E3])


    result: OptimizeResult = least_squares(
        fun=residual_func,
        x0=x0,
        args=args,
        bounds=bounds,
        loss=loss,
        f_scale=f_scale,
        xtol=tol,
        gtol=tol,
        ftol=tol,
        max_nfev=1000*len(x0),
        # diff_step=tol**0.5,
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
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')

def main(data, m1, m2, z1, z2, sbe, path_to_trim_yields, grazing_sim_csv, angle):
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
    alpha = np.pi * angle / 180.
    experiment_df = pd.DataFrame(data=data) # Load the experimental data into a dataframe
    trim_normal_incidence_df = pd.read_csv(path_to_trim_yields).apply(pd.to_numeric) # load the TRIM results
    energy = experiment_df['Energy (eV)'].values
    sputtering_yield = experiment_df['Y'].values

    fit_energy = np.hstack([energy, trim_normal_incidence_df['Incident energy (eV)'].values])
    fit_spy = np.hstack([sputtering_yield, trim_normal_incidence_df['Ysum'].values])

    trim_grazing_incidence_df = pd.read_csv(grazing_sim_csv).apply(pd.to_numeric) # The results at grazing incidence

    # Fit the data
    fit_result = fit_eckstein(fit_energy, fit_spy, m1, m2, z1, z2, sbe, loss='soft_l1', f_scale=0.1)

    # Fit grazing incidence
    fit_grazing_indicence_result = fit_eckstein(
        energy=trim_grazing_incidence_df['E0 (eV)'].values,
        sputtering_yield=trim_grazing_incidence_df['Yield'].values, angle=alpha,
        m1=m1, m2=m2, z1=z1, z2=z2, sbe=sbe, loss='soft_l1', f_scale=0.1
    )

    q, lam, mu, E_th = fit_result.x # Get the optimized parameters from the fit

    q2, lam2, mu2, E_th2, f, b, c = fit_grazing_indicence_result.x  # Get the optimized parameters from the fit

    # Create an array of energies for the prediction
    energy_pred = np.logspace(start=np.log10(E_th+0.05), stop=np.log10(energy.max()), num=1000)
    energy_pred_angle = np.logspace(start=np.log10(E_th + 0.002), stop=4, num=1000)


    # Define a model that only uses the optimized q, lam, mu, E_th from the fit and m1, m2, z1, z2, sbe set before
    def model_ep(e, b):
        q, lam, mu, E_th = b
        params = (q, lam, mu, E_th, m1, m2, z1, z2, sbe)
        return model(e, params)

    def model_epa(e, b):
        q, lam, mu, E_th, f, b, c = b
        params = (q, lam, mu, E_th, m1, m2, z1, z2, sbe, alpha, f, b, c)
        return model_angular_dependence(e, params)

    # Get the parameters confidence interval
    popt = fit_result.x
    ci = cf.confidence_interval(res=fit_result)
    popt_delta = np.abs(ci[:,1] - popt)

    popt_a = fit_grazing_indicence_result.x
    ci_a = cf.confidence_interval(res=fit_grazing_indicence_result)
    popt_delta_a = np.abs(ci_a[:, 1] - popt_a)

    logger = logging.getLogger('eckstein_preuss_fit_logger')
    logger.setLevel(logging.DEBUG)  # Set the logger's overall level

    # Create a StreamHandler to send logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Set the handler's level

    file_handler = logging.FileHandler(filename='Eckstein_Preuss_fit_log.txt', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Add the handler to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("Normal incidence fit")
    logger.info("=" * 60)
    logger.info(f"q:\t {q:.4f} -/+ {popt_delta[0]:.5f}")
    logger.info(f"lambda:\t {lam:.4f} -/+ {popt_delta[1]:.5f}")
    logger.info(f"mu:\t {mu:.4f} -/+ {popt_delta[2]:.5f}")
    logger.info(f"Eth:\t {E_th:.4f} -/+ {popt_delta[3]:.5f}")
    logger.info("")

    logger.info("=" * 60)
    logger.info(f"Grazing angle incidence fit ⍺ = {angle:.1f} °")
    logger.info("=" * 60)
    logger.info(f"q:\t {q2:.4E} -/+ {popt_delta_a[0]:.5E}")
    logger.info(f"lambda:\t {lam2:.4f} -/+ {popt_delta_a[1]:.5f}")
    logger.info(f"mu:\t {mu2:.4f} -/+ {popt_delta_a[2]:.5f}")
    logger.info(f"Eth:\t {E_th2:.4f} -/+ {popt_delta_a[3]:.5f}")
    logger.info(f"f:\t {f:.4f} -/+ {popt_delta_a[4]:.5f}")
    logger.info(f"b:\t {b:.4f} -/+ {popt_delta_a[5]:.5f}")
    logger.info(f"c:\t {c:.4f} -/+ {popt_delta_a[6]:.5f}")

    # Estimate the sputtering yield at the new points and their corresponding prediction bands
    eckstein_yield, delta = cf.prediction_intervals(model=model_ep, x_pred=energy_pred, ls_res=fit_result)
    eckstein_yield_angle, delta_angle = cf.prediction_intervals(
        model=model_epa, x_pred=energy_pred_angle, ls_res=fit_grazing_indicence_result
    )


    # Get the prediction at the ion energies of interest for the beam composition
    ion_energies = np.array([11.67, 14., 16.67, 17.5, 21., 25.0, 35, 42., 50])
    yields_at_energies, yields_delta_at_energies = cf.prediction_intervals(model=model_ep, x_pred=ion_energies, ls_res=fit_result)
    # Save the predicted sputtering yield at the ion energies of interest for the beam composition
    output_df = pd.DataFrame(data={
        'Ion energy (eV)': ion_energies,
        'Sputtering yield': yields_at_energies,
        'Sputtering yield delta': yields_delta_at_energies
    })

    # output_df.to_csv(r'./data/eckstein_preuss_yields.csv', index=False)

    # Plot the sputtering yield
    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4, 4.5)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(energy, sputtering_yield, marker='D', ls='none', mfc='C0', mec='C0', label='Experimental (Eckstein)')
    ax.plot(trim_normal_incidence_df['Incident energy (eV)'].values, trim_normal_incidence_df['Ysum'], marker='D', ls='none', mfc='none', mec='C0', label='SDTrimSP')
    ax.plot(energy_pred, eckstein_yield, ls='-', color='C0', label='Eckstein-Preuss model')
    ax.fill_between(energy_pred, eckstein_yield-delta, eckstein_yield + delta, color='C0', alpha=0.2)

    ax.plot(
        trim_grazing_incidence_df['E0 (eV)'].values, trim_grazing_incidence_df['Yield'].values,
        marker='>', ls='none', mfc='none', mec='C1', label=f'SDTrimSP ({angle:.1f}°)'
    )

    ax.plot(energy_pred_angle, eckstein_yield_angle, ls='-', color='C1', label='Eckstein-Preuss model')
    ax.fill_between(energy_pred_angle, eckstein_yield_angle - delta_angle, eckstein_yield_angle + delta_angle, color='C1', alpha=0.2)

    ax.set_xlabel('Incident energy (eV)')
    ax.set_ylabel('Sputtering yield')

    ax.set_xlim(10, 1E4)
    ax.set_ylim(1E-3, 1E0)

    ax.legend(loc='lower center', fontsize=10)
    ax.set_title(r"{\sffamily D\textsuperscript{+} → B}", usetex=True)

    fig.savefig(r'./figures/eckstein_preuss_sputtering_yield.png', dpi=600)
    fig.savefig(r'./figures/eckstein_preuss_sputtering_yield.pdf', dpi=600)

    plt.show()

if __name__ == '__main__':
    main(
        data=DATA, z1=Z1, z2=Z2, m1=M1, m2=M2, sbe=E_b, path_to_trim_yields=PATH_TO_TRIM_YIELDS,
        grazing_sim_csv=GRAZING_SIM_CSV, angle=ANGLE
    )





