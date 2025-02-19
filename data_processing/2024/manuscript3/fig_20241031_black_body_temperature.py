from typing import List, Dict

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf
import re
from decimal import Decimal

TEMPERATURE_CSV = r'./data/oes_black_body/20241031_bb_temp.xlsx'
SPECTRUM_CSV = r"./data/oes_black_body/echelle_20241031/MechelleSpect_006_data.csv"


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    plt.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}')

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

EPS = float(np.finfo(np.float64).eps)
def fit_polynomial(x, y, weights=1, poly_order:int=10, loss:str= 'soft_l1', f_scale:float=1.0, tol:float=EPS) -> OptimizeResult:
    """
    Fits the curve to a polynomial function
    Parameters
    ----------
    x: np.ndarray
        The x values
    y: np.ndarray
        The y values
    weights: float
        The weights for the residuals
    poly_order: int
        The degree of the polynomial to be used
    loss: str
        The type of loss to be used
    f_scale: float
        The scaling factor for the outliers
    tol: float
        The tolerance for the convergence

    Returns:
    -------
    OptimizeResult:
        The least squares optimized result
    """
    ls_res = least_squares(
        res_poly,
        x0=[(0.01) ** (i-1) for i in range(poly_order)],
        args=(x, y, weights),
        loss=loss, f_scale=f_scale,
        jac=jac_poly,
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=2,
        x_scale='jac',
        method='trf',
        tr_solver='exact',
        max_nfev=10000 * poly_order
    )
    return ls_res

def radiance_at_temperature(temperature: float, wavelength_nm: float, A=1.) -> float:
    hc = 6.62607015 * 2.99792458  # x 1E -34 (J s) x 1E8 (m/s) = 1E-26 (J m)
    hc2 = hc * 2.99792458  # x 1E -34 (J s) x 1E16 (m/s)^2 = 1E-18 (J m^2 s^{-1})
    factor = 2. * 1E14 * hc2 * np.power(wavelength_nm, -5.0)  # W / cm^2 / nm
    arg = 1E6 * hc / wavelength_nm / 1.380649 / temperature  # 26 (J m) / 1E-9 m / 1E-23 J/K
    return A * factor / (np.exp(arg) - 1.)


def model_bb(wavelength_nm: np.ndarray, b):
    temperature, factor = b[0], b[1]
    return factor * radiance_at_temperature(temperature=temperature, wavelength_nm=wavelength_nm)


def res_bb(b, x, y, w=1.):
    return (model_bb(wavelength_nm=x, b=b) - y)*w

all_tol = float(np.finfo(np.float64).eps)

def fit_black_body(
    wavelength: np.ndarray, radiance:np.ndarray, temperature_guess:float, scaling_factor_guess:float, tol=all_tol,
    f_scale=1., loss='soft_l1', lm_window_size=150
) -> OptimizeResult:
    """
    Tries to fit the spectrum to a black body spectrum
    Parameters
    ----------
    wavelength: np.ndarray
        The wavelength in nm
    radiance: np.ndarray
        The spectral radiance in W/m^2/s/nm
    temperature_guess: float
        The initial guess for the temperature in K
    scaling_factor_guess: float
        The initial guess of the scaling factor for the black body spectrum
    tol: float
        The tolerance used for the convergence of the least_squares
    f_scale: float
        The scaling factor for the outliers
    loss: str
        The type of loss to be used

    Returns
    -------
    OptimizeResult:
        The results from scipy.optimize.least_squares optimization
    """

    b0 = np.array([temperature_guess, scaling_factor_guess])

    # Find minima
    window_size = len(wavelength) // lm_window_size
    if window_size % 2 == 0:
        window_size += 1
    minima_data = find_local_minima(radiance.tolist(), window_size=window_size)
    indices = minima_data['minima_indices']
    wavelength = np.array([wavelength[i] for i in indices])
    radiance = np.array(minima_data['minima_values'])
    n = len(wavelength)
    result: OptimizeResult = least_squares(
        res_bb,
        b0,
        loss=loss,
        f_scale=f_scale,
        args=(wavelength, radiance),
        bounds=([all_tol, all_tol], [np.inf, np.inf]),
        xtol=tol,
        ftol=tol,
        gtol=tol,
        max_nfev=10000 * n,
        x_scale='jac',
        verbose=2
    )
    return result

def find_local_minima(signal: List[float], window_size: int = 3) -> Dict[str, List]:
    """
    Find local minima in a signal using a sliding window approach.

    Parameters:
    -----------
    signal : List[float]
        Input signal as a list of numerical values
    window_size : int, optional
        Size of the window to consider for finding local minima (default: 3)
        Must be odd number

    Returns:
    --------
    Dict with three keys:
        'minima_indices': List[int] - Indices where local minima occur in original signal
        'minima_values': List[float] - Values of the local minima
        'minima_pairs': List[Tuple[int, float]] - List of (index, value) pairs for compatibility

    Raises:
    -------
    ValueError
        If window_size is not odd or is larger than signal length
    """
    if len(signal) < window_size:
        raise ValueError("Signal length must be greater than window size")
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")

    # Convert to numpy array for easier processing
    signal_array = np.array(signal)
    half_window = window_size // 2

    # Initialize separate lists for indices and values
    minima_indices = []
    minima_values = []

    # Iterate through signal excluding edges
    for i in range(half_window, len(signal_array) - half_window):
        window = signal_array[i - half_window:i + half_window + 1]
        center_value = window[half_window]

        # Check if center point is minimum in window
        if center_value == np.min(window):
            # Ensure it's strictly less than at least one neighbor
            if np.sum(window == center_value) == 1:
                minima_indices.append(i)
                minima_values.append(center_value)

    # Return results in a dictionary for clarity
    return {
        'minima_indices': minima_indices,
        'minima_values': minima_values,
        'minima_pairs': list(zip(minima_indices, minima_values))
    }

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

def read_initial_params_from_comments(csv_file:str):
    """
    Tries to read the results from a previous fit in the data csv of the spectrum

    The header of the file must look like:
    ``
        # Temperature [K]: 922.3 -/+ 1.1
        # Scaling factor: 9.95E-01 -/+ 2.08E-02
        # Fit local minima: True
        # f_scale: 0.1
    ``

    Parameters
    ----------
    csv_file: str
        The path to the csv file containing the results from a previous fit

    Returns
    -------

    """
    p_t = re.compile(r"\#\s+Temperature\s+\[K\]\:\s(\d+\.?\d*)")
    p_s = re.compile(r"\#\s+Scaling.factor\:\s+(\d+\.?\d*[eE][\-\+]\d+)")
    p_fs = re.compile(r"\#\s+f.scale\:\s+(\d+\.?\d*)")
    p_ws = re.compile(r"\#\s+lm_window_size\:\s+(\d+\.?\d*)")
    matches = {}
    with open(csv_file, 'r') as f:
        for line in f:
            if not 'temperature' in matches:
                m_t = p_t.match(line)
                if m_t:
                    matches['temperature'] = float(m_t.group(1))
            if not 'scaling_factor' in matches:
                m_s = p_s.match(line)
                if m_s:
                    matches['scaling_factor'] = float(m_s.group(1))
            if not 'f_scale' in matches:
                m_fs = p_fs.match(line)
                if m_fs:
                    matches['f_scale'] = float(m_fs.group(1))
            if not 'lm_window_size' in matches:
                m_ws = p_ws.match(line)
                if m_ws:
                    matches['lm_window_size'] = int(m_ws.group(1))
            if len(matches) >= 4:
                break
    return matches

def main(temperature_csv, spectrum_csv):
    # Load the optical emission spectrum and fit the baseline to the black body to find the temperature
    spectrum_df = pd.read_csv(spectrum_csv, comment='#').apply(pd.to_numeric)
    spectrum_df = spectrum_df[spectrum_df['Radiance (W/cm^2/ster/nm)'] > 0.].reset_index(
        drop=True)  # Remove negative values
    initial_values = read_initial_params_from_comments(spectrum_csv)
    if (not 'temperature' in initial_values) and (not 'scaling_factor' in initial_values):
        initial_values['temperature'] = 1000.
        initial_values['scaling_factor'] = 1.
        initial_values['f_scale'] = 1.

    wavelength = spectrum_df['Wavelength (nm)'].values
    radiance = spectrum_df['Radiance (W/cm^2/ster/nm)'].values
    fit_result: OptimizeResult = fit_black_body(
        wavelength=wavelength, radiance=radiance,
        temperature_guess=initial_values['temperature'], scaling_factor_guess=initial_values['scaling_factor'],
        f_scale=initial_values['f_scale'], lm_window_size=initial_values['lm_window_size']
    )
    popt = fit_result.x
    ci = cf.confidence_interval(fit_result)
    popt_err = np.abs(ci[:, 1] - popt)
    wl_pred = np.linspace(wavelength.min(), wavelength.max(), num=2000)
    b_pred, b_delta = cf.prediction_intervals(
        model=model_bb, x_pred=wl_pred, ls_res=fit_result
    )
    """
    Load the dataset with the temperature vs time dependence
    """
    temperature_df = pd.read_excel(temperature_csv, sheet_name=0)
    temp_cols = [
        'Elapsed time (s)',
        'Temperature (K)',
        'Temperature error (K)'
    ]
    temperature_df[temp_cols] = temperature_df[temp_cols].apply(pd.to_numeric)
    time_s = temperature_df['Elapsed time (s)'].values
    temperature_k = temperature_df['Temperature (K)'].values
    temperature_err_k = temperature_df['Temperature error (K)'].values

    weights = 1. / (temperature_err_k + np.median(temperature_err_k) / 10) # Include measurement uncertainties
    fit_result: OptimizeResult = fit_polynomial(x=time_s, y=temperature_k, weights=weights, loss='soft_l1', f_scale=0.5, poly_order=6)
    popt = fit_result.x

    t_pred = np.linspace(0, time_s.max(), 1000)
    temp_pred, temp_pred_delta = cf.prediction_intervals(model_poly, x_pred=t_pred, ls_res=fit_result, jac=jac_poly, weights=weights)


    load_plot_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
    fig.set_size_inches(4., 5.5)

    # Plot the black body fit
    ax1.set_xlabel(r'$\lambda$ {\sffamily (nm)}', usetex=True)
    ax1.set_ylabel(r'$L_{\lambda}$ {\sffamily (W/cm\textsuperscript{2}/ster/nm)', usetex=True)
    ax1.set_title(r'Spectral radiance')

    ax1.plot(
        wavelength, radiance,
        marker='none', ms='4', mew=0.5, mfc='none',
        ls='-', color='0.5', alpha=1., label='OES data', lw=0.5
    )

    window_size = len(wavelength) // initial_values['lm_window_size']
    if window_size % 2 == 0:
        window_size += 1
    minima_data = find_local_minima(radiance.tolist(), window_size=window_size)
    indices = minima_data['minima_indices']
    wavelength_lm = np.array([wavelength[i] for i in indices])
    radiance_lm = np.array(minima_data['minima_values'])
    ax1.plot(wavelength_lm, radiance_lm, 'ro', ms=4, mfc='none', label='Baseline')

    ax1.plot(t_pred, temp_pred, color='tab:red', label='Best fit')
    ax1.fill_between(
        wl_pred, b_pred - b_delta, b_pred + b_delta, color='tab:red', alpha=0.5
    )

    ax1.ticklabel_format(axis='y', useMathText=True)
    # ax.set_xlim(wavelength.min(), wavelength.max())
    ax1.set_xlim(400, 860)
    ax1.set_ylim(0, 2E-6)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5E-7))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1E-7))

    ax1.legend(loc='upper left', fontsize=11)

    fitted_temp = popt[0] - 273.15
    fitted_temp_err = popt_err[0]
    results_txt = f'T = {round(fitted_temp/5)*5:.0f} ± {round(popt_err[0]/10)*10:.0f} °C\n'
    ax1.text(
        0.85, 0.95, results_txt, transform=ax1.transAxes, va='top', ha='right', c='r',
        fontsize=12
    )

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(20))

    # Plot time dependence

    ax2.errorbar(
        x=time_s/60, y=temperature_k-273.15, yerr=temperature_err_k, label='Data',
        capsize=2.75, mew=1.25, marker='o', ms=8, elinewidth=1.25,
        color='C0', fillstyle='none',
        ls='none',  # lw=1.25,
    )

    ax2.plot(t_pred/60, temp_pred-273.15, color='C0', label='Polynomial fit')
    ax2.fill_between(t_pred/60, temp_pred-273.15-temp_pred_delta, temp_pred-273.15+temp_pred_delta, color='C0', alpha=0.2)

    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('Surface temperature')

    ax2.legend(
        loc='upper right', frameon=True
    )

    ax2.set_xlim(0,100)
    ax2.set_ylim(400, 900)

    for i, axi in enumerate([ax1, ax2]):
        panel_label = chr(ord('`') + i + 1) # starts from a
        # panel_label = chr(ord('`') + i + 3)
        axi.text(
            -0.2, 1.05, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )

    model_df = pd.DataFrame(data={
        'Time (s)': t_pred,
        'Temperature (K)': temp_pred,
        'Temperature error (K)': temp_pred_delta
    })

    model_df.to_csv(r'./data/oes_black_body/echelle_20241031/20241031_temperature_model.csv', index=False)
    fig.savefig(r'./figures/fig_20241031_black_body_surface_temperature.pdf', dpi=600)
    fig.savefig(r'./figures/fig_20241031_black_body_surface_temperature.svg', dpi=600)
    fig.savefig(r'./figures/fig_20241031_black_body_surface_temperature.png', dpi=600)

    plt.show()

if __name__ == '__main__':
    main(TEMPERATURE_CSV, SPECTRUM_CSV)