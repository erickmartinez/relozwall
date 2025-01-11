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
from typing import List, Dict


spectrum_csv = r"./data/oes_black_body/echelle_20241031/MechelleSpect_029_data.csv"


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    plt.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}')


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
    f_scale=0.1, loss='cauchy'
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
    window_size = len(wavelength) // 250
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
            if len(matches) >= 3:
                break
    return matches

def main():
    global spectrum_csv
    spectrum_df = pd.read_csv(spectrum_csv, comment='#').apply(pd.to_numeric)
    spectrum_df = spectrum_df[spectrum_df['Radiance (W/cm^2/ster/nm)'] > 0.].reset_index(drop=True) # Remove negative values
    initial_values = read_initial_params_from_comments(spectrum_csv)
    if (not 'temperature' in initial_values) and ( not 'scaling_factor' in initial_values):
        initial_values['temperature'] = 1000.
        initial_values['scaling_factor'] = 1.
        initial_values['f_scale'] = 1.

    wavelength = spectrum_df['Wavelength (nm)'].values
    radiance = spectrum_df['Radiance (W/cm^2/ster/nm)'].values
    fit_result: OptimizeResult = fit_black_body(
        wavelength=wavelength, radiance=radiance,
        temperature_guess=initial_values['temperature'], scaling_factor_guess=initial_values['scaling_factor'],
        f_scale=initial_values['f_scale']
    )
    popt = fit_result.x
    ci = cf.confidence_interval(fit_result)
    popt_err = np.abs(ci[:, 1] - popt)
    x_pred = np.linspace(wavelength.min(), wavelength.max(), num=2000)
    y_pred, delta = cf.prediction_intervals(
        model=model_bb, x_pred=x_pred, ls_res=fit_result
    )

    load_plot_style()
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.5)
    ax.set_xlabel(r'$\lambda$ {\sffamily (nm)}', usetex=True)
    ax.set_ylabel(r'Spectral radiance (W/cm$^{\mathregular{2}}$/ster/nm)', usetex=False)

    ax.plot(
        wavelength, radiance,
        marker='+', ms='4', mew=0.5, mfc='none',
        ls='none', color='0.5', alpha=1., label='OES data'
    )

    window_size = len(wavelength) // 150
    if window_size % 2 == 0:
        window_size += 1
    minima_data = find_local_minima(radiance.tolist(), window_size=window_size)
    indices = minima_data['minima_indices']
    wavelength_lm = np.array([wavelength[i] for i in indices])
    radiance_lm = np.array(minima_data['minima_values'])
    ax.plot(wavelength_lm, radiance_lm, 'ro', ms=4, mfc='none', label='Baseline')


    ax.plot(x_pred, y_pred, color='tab:red', label='Best fit')
    ax.fill_between(
        x_pred, y_pred - delta, y_pred + delta, color='tab:red', alpha=0.5
    )


    ax.ticklabel_format(axis='y', useMathText=True)
    ax.set_xlim(wavelength.min(), wavelength.max())
    ax.set_ylim(0, 4E-7)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1E-7))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5E-8))

    ax.legend(loc='upper left', fontsize=11)

    fitted_temp = popt[0] - 273.15
    fitted_temp_err = popt_err[0]
    results_txt = f'T = {fitted_temp:.0f} °C\n'
    ax.text(
        0.05, 0.5, results_txt, transform=ax.transAxes, va='bottom', ha='left', c='r',
        fontsize=12
    )

    fig.savefig(r'./figures/fig_black_body_oes.pdf', dpi=600)
    fig.savefig(r'./figures/fig_black_body_oes.svg', dpi=600)
    fig.savefig(r'./figures/fig_black_body_oes.png', dpi=600)
    plt.show()



if __name__ == '__main__':
    main()