import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os
from scipy.stats.distributions import t
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf
from scipy.integrate import simpson, trapezoid
from data_processing.utils import latex_float_with_error, latex_float
from decimal import Decimal


path_to_data_csv = r'./data/PA_probe/20240827/langprobe_results/symmetrized/lang_results_gamma_ivdata0003_symmetrized.csv'
parent_path_figures = r'./figures/langmuir_probe_results'
mean_values_xlsx = r'./data/PA_probe_surface_mean.xlsx'

n_e_lorentzian = True
T_e_gaussian = True
symmetric_gaussian = True
poly_order = 8
area_err_pct = 25.

# https://webbook.nist.gov/cgi/cbook.cgi?ID=C14464472&Units=SI
m_d1 = 2.0135531979
# https://webbook.nist.gov/cgi/cbook.cgi?Formula=D2%2B&MatchIso=on&Units=SI
m_d2 = 4.0276549757
# https://webbook.nist.gov/cgi/cbook.cgi?ID=C12595969&Mask=8
m_d3 = 6.0417567535

w_d1 = 0.41 # The concentration of the d+ ions relative to n_e
w_d2 = 0.22 # The concentration of the d2+ ions relative to n_e
w_d3 = 0.37 # The concentration of the d3+ ions relative to n_e

mi = w_d1 * m_d1 + w_d2 * m_d2 + w_d3 * m_d3
Zion = 1. # ion charge state
Ti = 0.5 # ion temperature

sample_diameter_in = 0.4 #


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')


def gaussian(x, params):
    """
    Gaussian function with constant baseline: A * exp(-(x - mu)^2 / (2 * sigma^2)) + baseline

    Parameters:
    x: array-like, independent variable
    params: array-like (A, mu, sigma, baseline)
        A: amplitude
        mu: mean
        sigma: standard deviation
        baseline: constant offset
    """
    A, mu, sigma, baseline = params
    # A, sigma, baseline = params
    # mu = 0.
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + baseline


def residuals_gaussian(params, x, y, w=1):
    """Calculate residuals between observed data and the Gaussian model"""
    return (gaussian(x, params) - y) * w


def jacobian_gaussian(params, x, y, w=1.):
    """
    Analytical Jacobian matrix for the Gaussian function with baseline
    Returns partial derivatives with respect to (A, mu, sigma, baseline)
    """
    A, mu, sigma, baseline = params
    # A, sigma, baseline = params
    # mu=0.
    exp_term = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    # Partial derivatives
    d_A = w * exp_term
    d_mu = A * exp_term * (x - mu) / sigma ** 2
    d_sigma = w * A * exp_term * (x - mu) ** 2 / sigma ** 3
    d_baseline = w * np.ones_like(x)  # Derivative with respect to baseline

    return np.vstack([d_A, d_mu, d_sigma, d_baseline]).T
    # return np.vstack([d_A, d_sigma, d_baseline]).T


def gaussian_symmetric(x, params):
    """
    Gaussian function with constant baseline: A * exp(-(x - mu)^2 / (2 * sigma^2)) + baseline

    Parameters:
    x: array-like, independent variable
    params: array-like (A, mu, sigma, baseline)
        A: amplitude
        mu: mean
        sigma: standard deviation
        baseline: constant offset
    """
    # A, mu, sigma, baseline = params
    A, sigma, baseline = params
    mu = 0.
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + baseline


def residuals_gaussian_symmetric(params, x, y, w=1):
    """Calculate residuals between observed data and the Gaussian model"""
    return (gaussian_symmetric(x, params) - y) * w


def jacobian_gaussian_symmetric(params, x, y, w=1.):
    """
    Analytical Jacobian matrix for the Gaussian function with baseline
    Returns partial derivatives with respect to (A, mu, sigma, baseline)
    """
    # A, mu, sigma, baseline = params
    A, sigma, baseline = params
    mu=0.
    exp_term = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    # Partial derivatives
    d_A = w * exp_term
    d_mu = A * exp_term * (x - mu) / sigma ** 2
    d_sigma = w * A * exp_term * (x - mu) ** 2 / sigma ** 3
    d_baseline = w * np.ones_like(x)  # Derivative with respect to baseline

    # return np.vstack([d_A, d_mu, d_sigma, d_baseline]).T
    return np.vstack([d_A, d_sigma, d_baseline]).T


EPS = float(np.finfo(np.float64).eps)
def fit_gaussian(x, y, dy=None, p0=None, symmetric=False, loss='linear', f_scale=1.0, tol=EPS) -> OptimizeResult:
    """
    Fit Gaussian profile to data using least_squares with analytical Jacobian

    Parameters:
    x: array-like, independent variable
    y: array-like, dependent variable
    p0: initial guess for parameters (A, mu, sigma)
    symmetric: bool, if symmetric, make mu=0


    Returns:
    OptimizeResult object containing the fitted parameters
    """
    global residuals_gaussian, residuals_gaussian_symmetric, jacobian_gaussian, jacobian_gaussian_symmetric
    residuals:callable = residuals_gaussian
    jac:callable = jacobian_gaussian
    if p0 is None:
        # Make educated guesses for initial parameters
        baseline = np.min(y)  # Estimate baseline as minimum y value
        A = np.max(y) - baseline  # Estimate amplitude above baseline
        mu = x[np.argmax(y)]
        sigma = np.std(x) / 2
        p0 = np.array([A, mu, sigma, baseline])
        bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
        if symmetric:
            p0 = np.array([A, sigma, baseline])
            residuals:callable = residuals_gaussian_symmetric
            jac:callable = jacobian_gaussian_symmetric
            bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

    if dy is None:
        weights = 1
    else:
        weights = np.abs(1. / (dy + np.median(dy)/10))

    result = least_squares(
        residuals,
        x0=p0,
        jac=jac,
        bounds=bounds,
        args=(x, y, weights),
        method='trf',
        loss = loss,
        f_scale = f_scale,
        xtol = tol,
        ftol = tol,
        gtol = tol,
        verbose = 2,
        x_scale='jac',
        max_nfev = 10000 * len(p0)
    )

    return result


def lorentzian(x, params):
    """
    Lorentzian function centered at x=0
    Parameters:
    - x: x values
    - params: [amplitude, gamma]
        - amplitude: peak height
        - gamma: half-width at half-maximum (HWHM)
    """
    amplitude, gamma, offset = params
    return amplitude * (gamma ** 2 / (x ** 2 + gamma ** 2)) + offset


def lorentzian_jacobian(x, params):
    """
    Analytical Jacobian matrix for the Lorentzian function
    Parameters:
    - x: x values
    - params: [amplitude, gamma]
    Returns:
    - J: Jacobian matrix with derivatives [dF/dA, dF/dγ]
    """
    amplitude, gamma, offset = params
    denominator = (x ** 2 + gamma ** 2)

    # Derivative with respect to amplitude
    dF_dA = gamma ** 2 / denominator

    # Derivative with respect to gamma
    dF_dg = amplitude * (2 * gamma / denominator - 2 * gamma ** 3 / denominator ** 2)

    # Derivative with respect to offset
    dF_doffset = np.ones_like(x)

    return np.vstack([dF_dA, dF_dg, dF_doffset]).T


def lorentzian_residuals(params, x_data, y_data, weights):
    """
    Calculate residuals between data and model
    """
    model = lorentzian(x_data, params)
    return weights *  (model - y_data)


def lorentzian_residuals_jacobian(params, x_data, y_data, weights=None):
    """
    Jacobian of the residuals
    """
    if weights is None:
        return lorentzian_jacobian(x_data, params)
    else:
        return weights[:, np.newaxis] * lorentzian_jacobian(x_data, params)


def fit_lorentzian_peak(x_data, y_data, y_uncertainties=None, initial_guess=None, loss='soft_l1', f_scale=0.1, tol=EPS):
    """
    Fit a symmetric Lorentzian function to data using least_squares with soft_l1 loss
    and analytical Jacobian

    Parameters
    ----------
    x_data: np.ndarray
        The x data to fit
    y_data: np.ndarray
        The y data to fit
    y_uncertainties: np.ndarray
        The uncertainties in y_data
    initial_guess: list
        The initial guess for the parameters of the Lorentzian
    loss: str
        The loss to be used by the `least_squares` optimizer
    f_scale:
        The scaling factor for the outliers in the `least_squares` optimizer

    Returns
    -------
    OptimizeResult
        The result of the fit
    """

    # Estimate weights if uncertainties are provided, else use uniform weights
    if y_uncertainties is not None:
        weights = weights = np.abs(1. / (y_uncertainties + np.median(y_uncertainties)/10))
    else:
        weights = np.ones_like(y_data)

    if initial_guess is None:
        # Make educated guesses for initial parameters
        background = np.median(y_data[:10] + y_data[-10:]) / 2  # Estimate background from edges
        amplitude = np.max(y_data) - background
        # Estimate gamma from the width at half maximum
        half_max = amplitude / 2 + background
        indices = np.where(y_data >= half_max)[0]
        if len(indices) >= 2:
            gamma = abs(x_data[indices[-1]] - x_data[indices[0]]) / 2
        else:
            gamma = np.std(x_data)
        initial_guess = [amplitude, gamma, background]

    # Use least_squares with soft_l1 loss and analytical Jacobian
    result = least_squares(
        lorentzian_residuals,
        initial_guess,
        jac=lorentzian_residuals_jacobian,  # Analytical Jacobian
        loss=loss,
        f_scale=f_scale,
        args=(x_data, y_data, weights),
        method='trf',
        verbose=2, # Show optimization progress
        xtol=tol,
        ftol=tol,
        gtol=tol,
        max_nfev=10000*len(initial_guess)
    )

    return result

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

def fit_poly(x, y, dy=None, poly_order=poly_order, loss='soft_l1', f_scale=1.0, tol=EPS) -> OptimizeResult:
    x0 = np.array([0.5 ** i for i in range(poly_order+1)])
    if dy is None:
        weights = 1
    else:
        weights = np.abs(1. / (dy + np.median(dy)/10.))
    result = least_squares(
        res_poly,
        x0=x0,
        jac=jac_poly,
        args=(x, y, weights),
        method='trf',
        loss=loss,
        f_scale=f_scale,
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=2,
        x_scale='jac',
        max_nfev=10000 * poly_order
    )
    return result


m_p = 1.67262192369 # x1E-27 kg
e_v = 1.602176634 # x1E-19 J
cs_factor = np.power(e_v / m_p, 0.5) * 1E6

# print(f"{cs_factor:.3E}")

def calculate_ion_flux(n_e, T_e, T_i=Ti, ion_mass=mi, dn_e=None, dT_e=None, B_field=None):
    """
    Calculate ion flux in a linear plasma device

    The flux density was calculated according to the Bohm criterion assuming the density at the sheath
    edge is half the upstream density.

    Parameters:
    n_e : float
        Electron density in cm^-3
    T_e : float
        Electron temperature in eV
    T_i : float
        Ion temperature in eV
    ion_mass : float
        Ion mass in kg (e.g., 1.67e-27 for hydrogen)
    dn_e: float
        The uncertainty in the electron density
    dT_e: float
        The uncertainty in the electron temperature
    B_field : float, optional
        Magnetic field strength in Tesla

    Returns:
    float : Ion flux in cm^-2s^-1
    """
    global cs_factor, mi, Zion
    # Constants
    # e_charge = 1.602e-19  # Elementary charge in Coulombs

    # Calculate ion sound speed (Bohm velocity)
    # cs = sqrt(k*(T_e + gamma*T_i)/m_i), where gamma ≈ 1 for cold ions
    # c_s = np.sqrt(e_charge * (T_e + T_i) / ion_mass)
    # c_s = 9.79E5 * np.sqrt((Zion*T_e + T_i)/mi)
    c_s = cs_factor * np.sqrt((Zion * T_i + T_e) / mi)

    # Calculate sheath edge density
    # Typically n_se = n_e * exp(-1/2) at the sheath edge
    # n_se = n_e * np.exp(-0.5)
    n_se = 0.5 * n_e

    # Calculate ion flux
    # Γ_i = n_se * c_s
    ion_flux = n_se * c_s

    if (dT_e is None) or ( dn_e is None):
        ion_flux_delta = np.zeros_like(ion_flux)
    else:
        delta_cs = 0.5 * cs_factor * np.power((Zion * T_i + T_e) / mi, -0.5) * dT_e / mi
        if type(T_e) == np.ndarray:
            ion_flux_delta = ion_flux * np.linalg.norm(np.stack([delta_cs/c_s, dn_e / n_e]), axis=0)
        else:
            ion_flux_delta = ion_flux * np.linalg.norm(np.stack([delta_cs / c_s * np.ones_like(n_e), dn_e / n_e]), axis=0)
    # If magnetic field is present, consider magnetic pre-sheath effects
    if B_field is not None:
        # Approximate correction for magnetic pre-sheath
        # This is a simplified model - actual correction depends on field angle
        ion_flux *= np.cos(np.pi / 4)  # Typical angle correction

    return ion_flux, ion_flux_delta


def trapezoid_with_uncertainty(x, y, dy):
    """
    Perform numerical integration using the trapezoidal rule and propagate uncertainties.

    Parameters:
    -----------
    x : array_like
        Array of x values (independent variable)
    y : array_like
        Array of y values (dependent variable)
    dy : array_like
        Array of uncertainties in y values

    Returns:
    --------
    integral : float
        Result of the integration
    uncertainty : float
        Propagated uncertainty in the integral

    Notes:
    ------
    The uncertainty propagation follows from the general formula for error
    propagation in a sum/integral, where uncertainties are added in quadrature
    and weighted according to trapezoidal coefficients.
    """

    # Check inputs have same length
    if not (len(x) == len(y) == len(dy)):
        raise ValueError("All input arrays must have the same length")

    # Get trapezoidal coefficients (weights)
    n = len(x)

    # Trapezoidal coefficients: 1,2,2,...,2,1
    coeff = np.full(n, 2)
    coeff[0] = coeff[-1] = 1

    # Calculate step size (assuming uniform spacing)
    h = (x[-1] - x[0]) / (n - 1)

    # Calculate integral
    integral = trapezoid(y, x)

    # Propagate uncertainties
    # Each point's contribution to the uncertainty is weighted by its trapezoidal coefficient
    # and the step size, then added in quadrature
    weighted_variances = (coeff * dy * h / 2.0) ** 2
    uncertainty = np.sqrt(np.sum(weighted_variances))

    return integral, uncertainty


def format_latex_with_uncertainty(value, uncertainty):
    """
    Format a number with its uncertainty in LaTeX notation.

    Args:
        value (float): The main value (e.g., 1.234E16)
        uncertainty (float): The uncertainty value (e.g., 5.678E14)

    Returns:
        str: LaTeX formatted string with uncertainty

    Example:
        >>> format_latex_with_uncertainty(1.234e16, 5.678e14)
        '(1.234 \\pm 0.057) \\times 10^{16}'
    """
    # Convert to Decimal for better precision
    val = Decimal(str(value))
    unc = Decimal(str(uncertainty))

    # Get scientific notation components
    val_exp = val.adjusted()
    unc_exp = unc.adjusted()

    # Normalize both numbers to the same exponent (using the larger one)
    target_exp = val_exp

    # Normalize value
    val_coeff = val / Decimal(f'1e{val_exp}')

    # Normalize uncertainty to match value's exponent
    unc_coeff = unc / Decimal(f'1e{val_exp}')

    # Format the coefficients with appropriate precision
    # We'll use the uncertainty to determine precision
    unc_str = f"{unc_coeff:.3f}"
    val_str = f"{val_coeff:.3f}"

    # Construct LaTeX string
    latex_str = f"({val_str} \\pm {unc_str}) \\times 10^{{{val_exp}}}"

    return latex_str

ouput_columns = [
    'Folder', 'File',
    'T_e mean (eV)', 'T_e error (eV)', 'D_flux mean (/cm^2/s)', 'D_flux error (/cm^2/s)',
    'Shot #', 'Datetime'
]
def load_output_db(xlsx_source):
    global ouput_columns
    try:
        out_df: pd.DataFrame = pd.read_excel(xlsx_source, sheet_name='Probe data')
    except Exception as e:
        out_df = pd.DataFrame(data={
            col: [] for col in ouput_columns
        })
    return out_df

def update_df(db_df:pd.DataFrame, row_data):
    row = pd.DataFrame(data={key: [val] for key, val in row_data.items()})
    print(row_data)
    if len(db_df) == 0:
        return row
    folder = row_data['Folder']
    file = row_data['File']
    # Try finding the folder and file in db_df
    row_index = (db_df['Folder'] == folder) & (db_df['File'] == file)
    previous_row = db_df.index[row_index]
    if len(previous_row) == 0:
        return pd.concat([db_df, row], ignore_index=True).reset_index(drop=True)
    row_index = db_df.loc[row_index].index[0]
    for col, val in row_data.items():
        db_df.loc[row_index, col] = val
        # try:
        #     db_df.loc[row_index, col] = val
        # except Exception as e:
        #     print(e)
        #     print(row_index, col, val)
    db_df.sort_values(by=['Folder', 'File'], ascending=(True, True), inplace=True)
    return db_df

def main():
    global path_to_data_csv, parent_path_figures, sample_diameter_in, EPS, poly_order
    global symmetric_gaussian, n_e_lorentzian, T_e_gaussian, mean_values_xlsx
    parent_folder = os.path.dirname(path_to_data_csv)
    folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path_to_data_csv))))
    file_tag = os.path.splitext(os.path.basename(path_to_data_csv))[0]
    # Check if parent_path_figures exists
    if not os.path.exists(parent_path_figures):
        os.makedirs(parent_path_figures)
    # Create the folder to save the figures into if it does not exist already
    path_to_figures = os.path.join(parent_path_figures, folder)
    if not os.path.exists(path_to_figures):
        os.makedirs(path_to_figures)

    # load the data
    data_df: pd.DataFrame = pd.read_csv(filepath_or_buffer=path_to_data_csv).apply(pd.to_numeric)
    data_df = data_df[data_df['n_e (cm^{-3})']> 0]
    x = data_df['x (cm)'].values
    n_e = data_df['n_e (cm^{-3})'].values
    n_e_error = data_df['n_e error (cm^{-3})'].values
    T_e = data_df['T_e (eV)'].values
    T_e_error = data_df['T_e error (eV)'].values
    synthetic_msk = data_df['synthetic'].values

    weights_ne = 1. / (n_e_error + np.median(n_e_error))


    if symmetric_gaussian:
        model_te:callable = gaussian_symmetric
        jac_te:callable = jacobian_gaussian_symmetric
    else:
        model_te:callable = gaussian
        jac_te:callable = jacobian_gaussian

    if T_e_gaussian:
        lsq_result_Te: OptimizeResult = fit_gaussian(x=x, y=T_e, dy=None, symmetric=symmetric_gaussian,
                                                     loss='soft_l1', f_scale=1.)
    else:
        lsq_result_Te: OptimizeResult = fit_poly(x=x, y=T_e, dy=None, poly_order=poly_order,
                                                     loss='soft_l1', f_scale=1.0)
        model_te: callable = model_poly
        jac_te: callable = jac_poly

    if n_e_lorentzian:
        # lsq_result_ne: OptimizeResult = fit_gaussian(x=x, y=n_e, dy=n_e_error, symmetric=True, loss='soft_l1',
        #                                          f_scale=1.0)
        # model_ne: callable = gaussian_symmetric
        # jac_ne: callable = jacobian_gaussian_symmetric
        lsq_result_ne: OptimizeResult = fit_lorentzian_peak(
            x_data=x, y_data=n_e, y_uncertainties=None, loss='linear', f_scale=0.1
        )
        model_ne: callable = lorentzian
        jac_ne: callable = lorentzian_residuals_jacobian

    else:
        lsq_result_ne: OptimizeResult = fit_poly(x=x, y=n_e, dy=n_e_error, poly_order=poly_order, loss='soft_l1',
                                                 f_scale=1.0)
        model_ne: callable = model_poly
        jac_ne: callable = jac_poly

    # The 95 % confidence intervals for the n_e and T_e
    ci_ne = cf.confidence_interval(lsq_result_ne)
    ci_Te = cf.confidence_interval(lsq_result_Te)


    x_pred_1 = np.linspace(-4, 4, num=200)
    x_flux = np.linspace(-4, 4, num=500)

    y_pred_1, delta_1 = cf.prediction_intervals(
        model=model_ne, ls_res=lsq_result_ne, x_pred=x_pred_1, jac=jac_ne,
        new_observation=False
    )

    y_pred_2, delta_2 = cf.prediction_intervals(
        model=model_te, ls_res=lsq_result_Te, x_pred=x_pred_1, jac=jac_te,
        new_observation=False
    )

    # Estimate the mean flux using the fitted gaussian profiles
    r_sample = sample_diameter_in * 2.54 * 0.5
    a_sample = np.pi * (r_sample ** 2.)
    r_flux = np.linspace(0., r_sample, num=1000)

    n_e_r, n_e_r_delta = cf.prediction_intervals(
        model=model_ne, ls_res=lsq_result_ne, x_pred=r_flux, jac=jac_ne,
        new_observation=True
    )
    T_e_r, T_e_r_delta = cf.prediction_intervals(
        model=model_te, ls_res=lsq_result_Te, x_pred=r_flux, jac=jac_te,
        new_observation=True
    )

    n_e_x, n_e_x_delta = cf.prediction_intervals(
        model=model_ne, ls_res=lsq_result_ne, x_pred=x_flux, jac=jac_ne,
        new_observation=True
    )
    T_e_x, T_e_x_delta = cf.prediction_intervals(
        model=model_te, ls_res=lsq_result_Te, x_pred=x_flux, jac=jac_te,
        new_observation=True
    )

    # If T_e is greater than 6, then the probe analysis is unreliable. Previous estimates put T_e at ~5 eV
    # if np.mean(T_e_r) > 6.:
    #     T_e_r = np.ones_like(T_e_r) * 5.0
    #     T_e_r_delta = np.ones_like(T_e_r) * 5.0 * 0.15
    #     T_e_x = np.ones_like(T_e_x) * 5.0
    #     T_e_x_delta = np.ones_like(T_e_x) * 5.0 * 0.15

    gamma_d, gamma_d_error = calculate_ion_flux(n_e=n_e_r, T_e=T_e_r, dn_e=n_e_r_delta, dT_e=T_e_r_delta)
    # gamma_d, gamma_d_error = calculate_ion_flux(n_e=n_e_r, T_e=5.0 * np.ones_like(n_e_r), dn_e=n_e_r_delta,
    #                                             dT_e=0.75 * np.ones_like(n_e_r))
    # Integrate over the area of the sample
    gamma_mean, gamma_mean_error = trapezoid_with_uncertainty(x=r_flux, y=gamma_d*r_flux*2.*np.pi, dy=gamma_d_error*r_flux*2.*np.pi)

    gamma_mean /= a_sample
    # Assume that the sample is made of closely packed spheres (91% of the area is covered by spheres) of radius R
    # The number of spheres in the area A is n = 0.91*A/(pi R^2)
    # Each sphere contributes to the area 2*pi*R^2 (considering the upper hemisphere only)
    # The total surface area of the spheres is 0.91*A/(pi R^2) * 2*pi*R^2 = 2*0.91*A
    # Add the rest of the area that is not covered by the spheres (0.09A)
    # The total surface area is then
    # SA = 1.82A + 0.09A = 1.91 A
    # The difference between this area and the projected area is 1.91 or ~200%
    gamma_mean_error = gamma_mean * np.linalg.norm([gamma_mean_error/gamma_mean, area_err_pct/100.])
    print(f"Ion flux (mean): {gamma_mean:.3E} -/+ {gamma_mean_error:.3E} /cm^2/s")

    gamma_txt = rf"$\Gamma_{{\mathrm{{D}}}} = {format_latex_with_uncertainty(gamma_mean, gamma_mean_error)}$ "
    gamma_txt += r" {\sffamily s\textsuperscript{-1} cm\textsuperscript{-2}}"

    t_e_mean, t_e_mean_err = trapezoid_with_uncertainty(x=r_flux, y=T_e_r*r_flux*2.*np.pi, dy=T_e_r_delta*r_flux*2.*np.pi)
    t_e_mean /= a_sample
    t_e_mean_err = t_e_mean * np.linalg.norm([t_e_mean_err/t_e_mean, area_err_pct/100.])
    # t_e_mean, t_e_mean_err = np.mean(T_e), np.linalg.norm(T_e_error)
    # t_e_std = np.std(T_e, ddof=1)
    # num_t_e = len(T_e)
    # conf_level = 0.95
    # alpha = 1 - conf_level
    # tval_te = t.ppf(1 - alpha/2, num_t_e-1)
    # t_e_se = tval_te * t_e_std / np.sqrt(num_t_e)
    # t_e_mean_err = t_e_se # np.linalg.norm([t_e_se, t_e_mean_err])


    te_mean_txt = rf"$T_{{e}} = {t_e_mean:.1f} \pm {t_e_mean_err:.1f}$ "
    te_mean_txt += r" {\sffamily eV}"

    output_df = load_output_db(mean_values_xlsx)
    mean_data = {
        'Folder': 'PA_probe/' + folder,
        'File': file_tag,
        'T_e mean (eV)': t_e_mean,
        'T_e error (eV)': t_e_mean_err,
        'D_flux mean (/cm^2/s)': gamma_mean,
        'D_flux error (/cm^2/s)': gamma_mean_error
    }
    output_df = update_df(db_df=output_df, row_data=mean_data)
    output_df.to_excel(excel_writer=mean_values_xlsx, sheet_name='Probe data', index=False)
    print(output_df)

    load_plot_style()

    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True, sharex=True)
    fig.set_size_inches(3., 6.5)
    axes[0].set_xlim(-4, 4.)
    axes[0].set_xlabel(r"Position (cm)")
    axes[0].set_ylabel(r"n$_{\mathregular{e}}$ (x10$^{\mathregular{11}}$ cm$^{\mathregular{-3}}$)")

    markers, caps, bars = axes[0].errorbar(
        x, n_e / 1E11, yerr=np.abs(n_e_error) /1E11,
        marker='o', ms=6, mfc='none', mew=1.25, ls='none', color='C0',
        capsize=2.75,
        label='Experiment'
    )
    [bar.set_alpha(0.3) for bar in bars]


    axes[0].plot(x_pred_1, y_pred_1/1E11, color='tab:red', label='Fit')

    axes[0].fill_between(
        x_pred_1, (y_pred_1 - delta_1)/1E11, (y_pred_1 + delta_1)/1E11, color='C0', alpha=0.2,
        # label='Prediction interval'
    )
    axes[0].set_title('Density')
    axes[0].legend(loc='upper left', frameon=True, fontsize=9)

    markers, caps, bars = axes[1].errorbar(
        x, T_e, yerr=T_e_error,
        marker='s', ms=6, mfc='none', mew=1.25, ls='none', color='C1',
        capsize=2.75, elinewidth=1.25,
        label='Experiment'
    )
    [bar.set_alpha(0.3) for bar in bars]


    # axes[1].fill_between(
    #     x_pred_1, (y_pred_2 - delta_2), (y_pred_2 + delta_2), color='C1', alpha=0.2,
    #     # label='Prediction interval'
    # )
    axes[1].plot(x_pred_1, y_pred_2 , color='tab:red', label='Fit')

    axes[1].set_xlabel(r"Position (cm)")
    axes[1].set_ylabel(r"T$_{\mathregular{e}}$ (eV)")
    axes[1].set_title('Temperature')

    axes[1].legend(loc='upper left', frameon=True, fontsize=10)

    gamma_x, gamma_x_error = calculate_ion_flux(n_e=n_e_x, T_e=T_e_x, dn_e=n_e_x_delta, dT_e=T_e_x_delta)
    # gamma_x, gamma_x_error = calculate_ion_flux(n_e=n_e_x, T_e=5.0, dn_e=n_e_x_delta, dT_e=0.1)
    #
    axes[2].fill_between(
        x_flux,
        (gamma_x - gamma_x_error)/1e16, (gamma_x + gamma_x_error)/1E16, color='tab:purple', alpha=0.2,
    )
    axes[2].plot(x_flux, gamma_x*1E-16, color='tab:purple', label='Fit', ls='--')

    axes[2].set_xlabel(r"Position (cm)")
    axes[2].set_ylabel(r"$\Gamma_{\mathrm{D}}$ {\sffamily (x10\textsuperscript{16} /s/cm\textsuperscript{2})}", usetex=True)
    axes[2].set_title('Flux')

    axes[2].text(
        0.5, 0.025, gamma_txt, ha='center', va='bottom', transform=axes[2].transAxes,
        usetex=True, color='blue'
    )

    axes[1].text(
        0.95, 0.95, te_mean_txt, ha='right', va='top', transform=axes[1].transAxes,
        usetex=True, color='blue'
    )

    for ax in axes.flatten():
        ax.axvspan(xmin=-r_sample, xmax=r_sample, ls='--', color='0.5', lw=1., alpha=0.2)

    ax0_ylim_bottom = np.floor(0.9 * (y_pred_1 - delta_1).min() /1E11 / 2) * 2
    ax0_ylim_bottom = max(ax0_ylim_bottom, 0.)
    ax0_ylim_top = np.ceil(1.1 * (y_pred_1 + delta_1).max() / 1E11 / 1.5) * 1.5
    ax0_ylim_top = max(ax0_ylim_top, np.ceil(1.05 * (n_e).max() / 1E11 / 1.5) * 1.5)

    ax1_ylim_bottom = np.floor((y_pred_2 - delta_2).min() / 2) *2
    ax1_ylim_bottom = max(ax1_ylim_bottom, 0.)
    ax1_ylim_top = np.ceil(1.05 * (y_pred_2 + delta_2).max() / 2) * 2
    ax1_ylim_top = max(ax1_ylim_top, np.ceil(1.05 * (T_e).max() / 2) * 2)

    ax2_ylim_bottom = np.floor(gamma_x.min() / 2) * 2
    ax2_ylim_bottom = max(ax2_ylim_bottom, 0.)
    ax2_ylim_top = np.ceil(1.05 * gamma_x.max() / 2) * 2

    axes[0].set_ylim(0, 6)
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(2.))
    axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(1.))
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(2.))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(1.))
    axes[1].set_ylim(0, 8)
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(4.))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(2.0))
    axes[2].set_ylim(0, 50)
    axes[2].yaxis.set_major_locator(ticker.MultipleLocator(10.))
    axes[2].yaxis.set_minor_locator(ticker.MultipleLocator(5.))

    connectionstyle = "angle,angleA=-90,angleB=180,rad=0"
    # connectionstyle = "arc3,rad=0."
    bbox = dict(boxstyle="round", fc="wheat")
    arrowprops = dict(
        arrowstyle="->", color="k",
        shrinkA=5, shrinkB=0,
        patchA=None, patchB=None,
        connectionstyle=connectionstyle
    )
    axes[2].annotate(
        f"Sample diameter",
        xy=(0.55, 0.5), xycoords='axes fraction',  # 'figure pixels', #data',
        # transform=axes[1].transAxes,
        xytext=(0.95, 0.90), textcoords='axes fraction',
        ha='right', va='top',
        arrowprops=arrowprops,
        bbox=bbox,
        fontsize=10
    )

    for i, axi in enumerate(axes):
        # axi.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        # axi.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        panel_label = chr(ord('`') + i + 1) # starts from a
        # panel_label = chr(ord('`') + i + 3)
        axi.text(
            -0.125, 1.075, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )


    symmetrized_fit_df: pd.DataFrame = pd.DataFrame(data={
        'x (cm)': x_pred_1,
        'n_e (cm^{-3})': y_pred_1,
        'n_e error (cm^{-3})': delta_1,
        'T_e (eV)': y_pred_2,
        'T_e error (eV)': delta_2
    })


    output_csv = os.path.join(parent_folder, file_tag + '_fit.csv')
    popt_ne = lsq_result_ne.x
    popt_ne_delta = np.abs(ci_ne[:,1] - popt_ne)
    popt_Te = lsq_result_Te.x
    popt_Te_delta = np.abs(ci_Te[:, 1] - popt_Te)
    with open(output_csv, 'w') as f:
        f.write(f"# ******** Mean D flux on the sample **********\n")
        f.write(f"# Gamma_D_mean: {gamma_mean:.3E} -/+ {gamma_mean_error:.3E} 1/cm^2/s\n")
        f.write(f"# ******** Mean T_e on the sample **********\n")
        f.write(f"# T_e_mean: {t_e_mean:.3E} -/+ {t_e_mean_err:.3E} eV\n")
        if n_e_lorentzian:
            f.write(f"# ************ Lorentzian fit to n_e ************\n")
            f.write(f"# amplitude:  {popt_ne[0]:.3E} -/+ {popt_ne_delta[0]:.3E} (1/cm^3)\n")
            f.write(f"# gamma:      {popt_ne[1]:.3f} -/+ {popt_ne_delta[1]:.3f} (cm)\n")
            f.write(f"# center:     0.00 (cm)\n")
            f.write(f"# yoffset:     {popt_ne[2]:.3E} -/+ {popt_ne_delta[2]:.3E} (1/cm^3)\n")
        else:
            f.write(f"# *********** Polynomial fit to n_e ***********\n")
            f.write(f"# Order:      {poly_order:<2d}\n")
            for i in range(len(popt_ne)):
                f.write(f"# a_{i:>2d}: {popt_ne[i]:>4.3E} -/+ {popt_ne_delta[i]:>4.3E}\n")
        if T_e_gaussian:
            f.write(f"# ************ Gaussian fit to T_e ************\n")
            f.write(f"# A:          {popt_Te[0]:.3E} -/+ {popt_Te_delta[0]:.3E} (1/cm^3)\n")
            f.write(f"# sigma:      {popt_Te[1]:.3f} -/+ {popt_Te_delta[1]:.3f} (cm)\n")
            if not symmetric_gaussian:
                f.write(f"# mu:         {popt_Te[2]:.3f} -/+ {popt_Te_delta[1]:.3f} (cm)\n")
                f.write(f"# baseline:   {popt_Te[3]:.3E} -/+ {popt_Te_delta[3]:.3E} (1/cm^3)\n")
            else:
                f.write(f"# mu:         0.00 (cm)\n")
                f.write(f"# baseline:   {popt_Te[2]:.3E} -/+ {popt_Te_delta[2]:.3E} (1/cm^3)\n")
        else:
            f.write(f"# *********** Polynomial fit to T_e ***********\n")
            f.write(f"# Order:      {poly_order:<2d}\n")
            for i in range(len(popt_Te)):
                f.write(f"# a_{i:2d}: {popt_Te[i]:>4.3E} -/+ {popt_Te_delta[i]:>4.3E}\n")

        symmetrized_fit_df.to_csv(f, index=False)


    fig.savefig(os.path.join(path_to_figures, file_tag + '_ion_flux.png'), dpi=600)
    fig.savefig(os.path.join(path_to_figures, file_tag + '_ion_flux.svg'), dpi=600)
    fig.savefig(os.path.join(path_to_figures, file_tag + '_ion_flux.pdf'), dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
