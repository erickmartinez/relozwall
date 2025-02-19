from decimal import Decimal
from typing import List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from edx_calibration import load_plot_style
import matplotlib.ticker as ticker
import re
import os
from scipy.optimize import least_squares, differential_evolution, OptimizeResult
from scipy.integrate import simpson
import data_processing.confidence as cf
from matplotlib.gridspec import GridSpec


PATH_TO_CSV_SPECTRUM = r"../OES/data/brightness_data_fitspy_wl-calibrated/echelle_20240827/MechelleSpect_022.csv"
PATH_TO_FITSPY_CD_RESULTS = r'../OES/data/fitspy_results_cd_bd/echelle_20240827/MechelleSpect_022.csv'
PATH_TO_ND_SPECTRUM = r"../OES/data/brightness_data_fitspy_wl-calibrated/echelle_20241003/MechelleSpect_001.csv"
PATH_TO_BB_SPECTRUM = r"./data/oes_black_body/echelle_20241031/MechelleSpect_006_data.csv"

BD_WL_RANGE = (429., 434.5)
BD_FIT_RANGE = (430, 433.5)


BI_WL_RANGE = (818, 824)
ND_WL_RANGE = (333., 338.)

CALIBRATION_LINE = {'center_wl': 434.0, 'label': r'D$_{\gamma}$'}
BI_CALIBRATION_LINE = {'center_wl': 821.2, 'label': r'B I'}
ND_REFERENCE_WL = 335.76

PEAKS_OF_INTEREST = [
    {'center_wl': 430.9, 'label': "C-D"},
    {'center_wl': 432.6, 'label': 'B-D (Q-branch)'}
]

Q_BRANCH_RANGE = (432.2, 433.8)
MIXED_RANGE = (429.6, 432.1)

f_pattern = re.compile(r'\s*\#\s*(.*)?\:\s+(.*)')

def lorentzian(x, h, mu, gamma):
    return 2.*gamma*h/(np.pi)/(4*(x-mu)**2. + gamma**2.)

def res_sum_lorentzians(b, x, y):
    return sum_lorentzians(x, b) - y

def sum_lorentzians(x, b):
    m = len(x)
    n3 = len(b)
    selector = np.arange(0, n3) % 3
    h = b[selector == 0]
    mu = b[selector == 1]
    gamma = b[selector == 2]
    n = len(h)
    res = np.zeros(m)
    for i in range(n):
        res += h[i]*gamma[i] / ( ((x-mu[i])**2.) + (0.25* gamma[i] ** 2.) )
    return 0.5 * res / np.pi

def jac_sum_lorentzians(b, x, y):
    m = len(x)
    n3 = len(b)
    selector = np.arange(0, n3) % 3
    h = b[selector == 0]
    mu = b[selector == 1]
    gamma = b[selector == 2]
    n = len(h)
    res = np.empty((m, n3), dtype=np.float64)
    for i in range(n):
        g = gamma[i]
        g2 = g ** 2.
        xm = x - mu[i]
        xm2 = xm ** 2.
        den = (4. * xm2 + g2) ** 2.
        res[:, 3*i] = 0.5 * g / (xm2 + 0.25*g2)
        res[:, 3*i+1] = 16. * g * h[i] * xm / den
        res[:, 3*i+2] = h[i] * (8. * xm2 - 2. * g2) / den
    return res / np.pi

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

def res_bb_de(b, x, y):
    res = res_bb(b, x, y)
    result = 0.5 * np.linalg.norm([res, res])
    return result

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
    # result_de: OptimizeResult = differential_evolution(
    #     func=res_bb_de,
    #     args=(wavelength, radiance),
    #     x0=b0,
    #     bounds=[(all_tol, 2000.), (all_tol, 1E20)],
    #     maxiter=10000 * len(b0),
    #     tol=tol,
    #     atol=tol,
    #     workers=-1,
    #     updating='deferred',
    #     recombination=0.1,
    #     strategy='best1bin',
    #     mutation=(0.5, 1.5),
    #     init='sobol',
    #     polish=False,
    #     disp=True
    # )
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

def load_brightness_file(path_to_file):
    params = {}
    with open(path_to_file, 'r') as f:
        for line in f:
            matches = f_pattern.match(line)
            if matches:
                params[matches.group(1)] = matches.group(2)
            if not line.startswith('#'):
                break
    df: pd.DataFrame = pd.read_csv(path_to_file, comment='#').apply(pd.to_numeric)
    return df, params

def get_peak_stats(line_number, path_to_stats_file):
    amount_pm_err_str = r"\s+(\d+\.?\d*[eE]?[\-\+]?\d*)\s+\+\/\-\s+(\d+\.?\d*[eE]?[\-\+]?\d*)"
    p_x0 = re.compile(rf"\s+m{line_number:02d}_x0.\s+{amount_pm_err_str}.*$")
    p_ampli = re.compile(rf"\s+m{line_number:02d}_ampli.\s+{amount_pm_err_str}.*$")
    p_gamma = re.compile(rf"\s+m{line_number:02d}_fwhm.\s+{amount_pm_err_str}.*$")
    x0_found = False
    ampli_found = False
    gamma_found = False
    output = {}
    with open(path_to_stats_file, 'r') as f:
        for line in f:
            if not x0_found:
                m = p_x0.match(line)
                if m:
                    output['x0'] = float(m.group(1))
                    output['x0_error'] = float(m.group(2))
                    ampli_found = True
            if not ampli_found:
                m = p_ampli.match(line)
                if m:
                    output['ampli'] = float(m.group(1))
                    output['ampli_error'] = float(m.group(2))
                    ampli_found = True
            if not gamma_found:
                m = p_gamma.match(line)
                if m:
                    output['fwhm'] = float(m.group(1))
                    output['fwhm_error'] = float(m.group(2))
                    ampli_found = True
            if ampli_found and gamma_found and x0_found:
                break
    area = 0.5 * output['ampli'] * np.pi * output['fwhm']
    error = area * np.linalg.norm([output['fwhm_error']/output['fwhm'], output['ampli_error']/output['ampli']])
    output['area'] = area
    output['area_error'] = error
    return output

def plot_bd_spectrum(
    ax, full_df:pd.DataFrame, model_df:pd.DataFrame, wl_range, calibration_line,
    path_to_stats_file, q_branch_range, mixed_range, d_gamma_shift=False
):
    # Focus only on the wavelength region defined in wl_range
    df = full_df[full_df['Wavelength (nm)'].between(wl_range[0], wl_range[1])].reset_index(drop=True)
    wavelength = df['Wavelength (nm)'].values
    photon_flux = df['Brightness (photons/cm^2/s/nm)'].values

    dx = 0.
    if d_gamma_shift: # If the spectrum is not calibrated, shift it according to the D_gamma peak
        b_peak = photon_flux.max()
        idx_dg = np.argmin(np.abs(photon_flux - b_peak))
        w_dg = wavelength[idx_dg]
        dx = 434.0 - w_dg
        full_df['Wavelength (nm)'] = full_df['Wavelength (nm)'] + dx
        df = full_df[full_df['Wavelength (nm)'].between(wl_range[0], wl_range[1])].reset_index(drop=True)
        wavelength = df['Wavelength (nm)'].values
        photon_flux = df['Brightness (photons/cm^2/s/nm)'].values

    ax.plot(
        wavelength, photon_flux, color='C0', marker='o', ms=3, mfc='none', mew=1., label='Data'
    )

    model_df = model_df.reset_index(drop=True)
    n_peaks = len(model_df)

    popt = np.empty(n_peaks * 3, dtype=np.float64)
    for i, row in model_df.iterrows():
        x0 = row['x0']
        ampli = row['ampli']
        g = row['fwhm']
        h = 0.5 * ampli * np.pi * g
        idx_h = 3 * i
        idx_m = idx_h + 1
        idx_g = idx_m + 1
        popt[idx_h] = h
        popt[idx_m] = x0
        popt[idx_g] = g

    yfit = sum_lorentzians(wavelength, popt)
    ax.plot(
        wavelength, yfit,
        color='red', lw=1.5, label='Fit'
    )

    ax.axvline(x=calibration_line['center_wl'], ls='--', lw=1., color='grey')

    connectionstyle = "angle,angleA=-90,angleB=180,rad=0"
    # connectionstyle = "arc3,rad=0."
    bbox = dict(boxstyle="round", fc="wheat")
    arrowprops = dict(
        arrowstyle="->", color="k",
        shrinkA=5, shrinkB=0,
        patchA=None, patchB=None,
        connectionstyle=connectionstyle
    )
    ax.annotate(
        f"{calibration_line['label']} ({calibration_line['center_wl']:.2f}) nm",
        xy=(calibration_line['center_wl'], photon_flux.max() * 0.075), xycoords='data',  # 'figure pixels', #data',
        # transform=axes[1].transAxes,
        xytext=(0.75, 0.90), textcoords='axes fraction',
        ha='right', va='top',
        arrowprops=arrowprops,
        bbox=bbox,
        fontsize=11
    )

    lorentzians_x0 = model_df['x0'].values + dx

    connectionstyle = "angle,angleA=0,angleB=-90,rad=0"
    # connectionstyle = "arc3,rad=0."
    bbox = dict(boxstyle="round", fc="honeydew")
    arrowprops = dict(
        arrowstyle="->", color="k",
        shrinkA=5, shrinkB=0,
        patchA=None, patchB=None,
        connectionstyle=connectionstyle
    )

    # Group peaks corresponding to the q-branch of B-D
    q_branch_peaks = model_df[model_df['x0'].between(q_branch_range[0], q_branch_range[1])].reset_index(drop=True)
    # Group peaks corresponding to the mixed range where B-D and C-D (and potentially other transitions) mix
    mixed_range_peaks = model_df[model_df['x0'].between(mixed_range[0]-2, mixed_range[1])].reset_index(drop=True)

    # Plot the Q-branch
    popt_qb = np.empty(len(q_branch_peaks) * 3, dtype=np.float64)
    qb_x0_mean = 0.
    bd_area_err = np.zeros(len(q_branch_peaks), dtype=np.float64)
    area_sum = 0
    for i, row in q_branch_peaks.iterrows():
        x0 = row['x0']
        qb_x0_mean += x0
        ampli = row['ampli']
        idx = np.argmin(np.abs(x0 - lorentzians_x0))
        peak_stats = get_peak_stats(idx + 1, path_to_stats_file)
        g = row['fwhm']
        h = 0.5 * ampli * np.pi * g
        idx_h = 3 * i
        idx_m = idx_h + 1
        idx_g = idx_m + 1
        popt_qb[idx_h] = h
        popt_qb[idx_m] = x0
        popt_qb[idx_g] = g
        area_sum += h
        bd_area_err[i] = peak_stats['ampli_error']


    qb_x0_mean /= len(q_branch_peaks)
    qb_shape = sum_lorentzians(x=wavelength, b=popt_qb)
    qb_y_max = qb_shape.max()

    ax.fill_between(
        wavelength, 0, qb_shape,
        alpha=0.25
    )

    ax.annotate(
        f"BD\nQ-branch",
        xy=(qb_x0_mean, qb_y_max), xycoords='data',  # 'figure pixels', #data',
        # transform=axes[1].transAxes,
        xytext=(-10, 40), textcoords='offset pixels',
        ha='center', va='bottom',
        # arrowprops=arrowprops,
        # bbox=bbox,
        fontsize=11
    )

    # Plot the mixed branch
    popt_mb = np.empty(len(mixed_range_peaks) * 3, dtype=np.float64)
    mb_x0_mean = 0.
    for i, row in mixed_range_peaks.iterrows():
        x0 = row['x0']
        mb_x0_mean += x0
        ampli = row['ampli']
        g = row['fwhm']
        h = 0.5 * ampli * np.pi * g
        idx_h = 3 * i
        idx_m = idx_h + 1
        idx_g = idx_m + 1
        popt_mb[idx_h] = h
        popt_mb[idx_m] = x0
        popt_mb[idx_g] = g

    mb_x0_mean /= len(mixed_range_peaks)
    mb_shape = sum_lorentzians(x=wavelength, b=popt_mb)
    mb_y_max = sum_lorentzians(x=[430.7], b=popt_mb)

    ax.fill_between(
        wavelength, 0, mb_shape,
        alpha=0.25
    )

    ax.annotate(
        f"BD + CD",
        xy=(mb_x0_mean, mb_y_max * 1.05), xycoords='data',  # 'figure pixels', #data',
        # transform=axes[1].transAxes,
        xytext=(-10, 50), textcoords='offset pixels',
        ha='center', va='bottom',
        # arrowprops=arrowprops,
        # bbox=bbox,
        fontsize=11
    )

    top_lim = photon_flux.max() * 0.1
    top_lim = np.round(top_lim / 0.5E12) * 0.5E12
    ax.set_ylim(bottom=0, top=top_lim)
    ax.set_xlim(wl_range)

    mf = ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(mf)
    ax.ticklabel_format(useMathText=True)
    ax.set_xlabel(r"$\lambda$ {\sffamily (nm)}", usetex=True)
    ax.set_ylabel(r"$B_{\lambda}$ {\sffamily (photons/cm\textsuperscript{2}/s/nm)}", usetex=True)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

def plot_bi_spectrum(ax, full_df, wl_range, calibration_line):
    df = full_df[full_df['Wavelength (nm)'].between(wl_range[0], wl_range[1])].reset_index(drop=True)
    wavelength = df['Wavelength (nm)'].values
    photon_flux = df['Brightness (photons/cm^2/s/nm)'].values

    # Fit a gaussian to the main line
    n_peaks = 1
    line_center = calibration_line['center_wl']
    delta_wl = 0.25
    msk_window = ((line_center - delta_wl) <= wavelength) & (wavelength <= (line_center + delta_wl))
    wl_window = wavelength[msk_window]
    intensity_window = photon_flux[msk_window]
    peak_height = intensity_window.max()
    idx_peak = np.argmin(np.abs(peak_height - intensity_window))
    wl_peak = wl_window[idx_peak]
    # recalculate the window around the peak from the data
    msk_window = ((wl_peak - delta_wl) <= wavelength) & (wavelength <= (wl_peak + delta_wl))
    wl_window = wavelength[msk_window]
    intensity_window = photon_flux[msk_window]
    area_window = simpson(y=intensity_window, x=wl_window)
    over_sqrt_pi = 1. / np.sqrt(2. * np.pi)
    c_window = area_window * over_sqrt_pi

    x0 = [c_window * 0.75, wl_peak, (wl_window.max() - wl_window.min())]
    all_tol = float(np.finfo(np.float64).eps)
    res_lsq = least_squares(
        res_sum_lorentzians, x0=x0, args=(wl_window, intensity_window), loss='linear', f_scale=0.1,
        jac=jac_sum_lorentzians,
        bounds=(
            [0., wl_peak - 0.15, 1E-5],
            [3. * c_window, wl_peak + 0.15, np.inf]
        ),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        verbose=2,
        # x_scale='jac',
        max_nfev=10000 * len(wl_window)
    )
    popt = res_lsq.x
    ci = cf.confidence_interval(res=res_lsq)
    popt_delta = ci[:, 1] - popt

    wl_fit = np.linspace(wl_window.min(), wl_window.max(), num=500)

    wl_extrapolate = np.linspace(wavelength.min(), wavelength.max(), num=5000)

    ax.plot(wavelength, photon_flux, marker='o', ms=3, mfc='none', mew=1., color='C0')
    ax.plot(wl_fit, lorentzian(x=wl_fit, h=popt[0], mu=popt[1], gamma=popt[2]), color='tab:red', lw='2.', ls='-')

    ax.plot(wl_extrapolate, lorentzian(x=wl_extrapolate, h=popt[0], mu=popt[1], gamma=popt[2]), color='tab:red',
            lw='1.5', ls='--')

    ax.set_xlabel(r'$\lambda$ {\sffamily (nm)}', usetex=True)
    ax.set_ylabel(r"$B_{\lambda}$ {\sffamily (photons/cm\textsuperscript{2}/nm)}", usetex=True)

    ax.axvline(x=calibration_line['center_wl'], ls='--', lw=1., color='grey')

    ax.set_xlim(wl_range)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))


    top_lim = photon_flux.max() * 1.1
    top_lim = np.round(top_lim / 0.2E13) * 0.2E13
    ax.set_ylim(bottom=0, top=top_lim)

    mf = ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(mf)
    ax.ticklabel_format(useMathText=True)

    ax.annotate(
        f"B-I\n({calibration_line['center_wl']:.1f} nm)",
        xy=(calibration_line['center_wl'], peak_height*0.85), xycoords='data',  # 'figure pixels', #data',
        # transform=axes[1].transAxes,
        xytext=(-40, 0), textcoords='offset pixels',
        ha='right', va='top',
        # arrowprops=arrowprops,
        # bbox=bbox,
        fontsize=11
    )

def plot_nd_spectrum(ax, full_df, wl_range, reference_wl):
    df = full_df[full_df['Wavelength (nm)'].between(wl_range[0], wl_range[1])].reset_index(drop=True)
    wavelength = df['Wavelength (nm)'].values
    photon_flux = df['Brightness (photons/cm^2/s/nm)'].values
    ax.plot(wavelength, photon_flux, marker='o', ms=3, mfc='none', mew=1., color='C0')
    ax.set_xlabel(r'$\lambda$ {\sffamily (nm)}', usetex=True)
    ax.set_ylabel(r"$B_{\lambda}$ {\sffamily (photons/cm\textsuperscript{2}/nm)}", usetex=True)

    ax.axvline(x=reference_wl, ls='--', lw=1., color='grey')

    ax.set_xlim(wl_range)

    top_lim = photon_flux.max() * 1.1
    top_lim = np.round(top_lim / 0.2E12) * 0.2E12
    ax.set_ylim(bottom=0, top=top_lim)

    mf = ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(mf)
    ax.ticklabel_format(useMathText=True)

    delta_wl = 0.25
    msk_window = ((reference_wl - delta_wl) <= wavelength) & (wavelength <= (reference_wl + delta_wl))
    intensity_window = photon_flux[msk_window]
    peak_height = intensity_window.max()

    ax.annotate(
        f"ND\n({reference_wl:.2f} nm)",
        xy=(reference_wl, peak_height * 0.85), xycoords='data',  # 'figure pixels', #data',
        # transform=axes[1].transAxes,
        xytext=(-40, 0), textcoords='offset pixels',
        ha='right', va='top',
        # arrowprops=arrowprops,
        # bbox=bbox,
        fontsize=11
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))

def plot_black_body(ax, full_df, initial_guess):
    if (not 'temperature' in initial_guess) and ( not 'scaling_factor' in initial_guess):
        initial_guess['temperature'] = 1000.
        initial_guess['scaling_factor'] = 1.
        initial_guess['f_scale'] = 1.

    wavelength = full_df['Wavelength (nm)'].values
    radiance = full_df['Radiance (W/cm^2/ster/nm)'].values
    fit_result: OptimizeResult = fit_black_body(
        wavelength=wavelength, radiance=radiance,
        temperature_guess=initial_guess['temperature'], scaling_factor_guess=initial_guess['scaling_factor'],
        f_scale=initial_guess['f_scale'], lm_window_size=initial_guess['lm_window_size']
    )
    popt = fit_result.x
    ci = cf.confidence_interval(fit_result)
    popt_err = np.abs(ci[:, 1] - popt)
    x_pred = np.linspace(wavelength.min(), wavelength.max(), num=2000)
    y_pred, delta = cf.prediction_intervals(
        model=model_bb, x_pred=x_pred, ls_res=fit_result
    )

    ax.set_xlabel(r'$\lambda$ {\sffamily (nm)}', usetex=True)
    ax.set_ylabel(r'$L_{\lambda}$ {\sffamily (W/cm\textsuperscript{2}/ster/nm)}', usetex=True)

    ax.plot(
        wavelength, radiance,
        marker='none', ms='4', mew=0.5, mfc='none',
        ls='-', color='0.5', alpha=1., label='OES data', lw=0.5
    )

    window_size = len(wavelength) // initial_guess['lm_window_size']
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
    # ax.set_xlim(wavelength.min(), wavelength.max())
    ax.set_xlim(400, 860)
    ax.set_ylim(0, 2E-6)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5E-7))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1E-7))

    ax.legend(loc='upper left', fontsize=11)

    fitted_temp = popt[0] - 273.15
    fitted_temp_err = popt_err[0]
    results_txt = f'T = {fitted_temp:.0f} °C\n'
    ax.text(
        0.85, 0.95, results_txt, transform=ax.transAxes, va='top', ha='right', c='r',
        fontsize=12
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(20))

    ax.set_xlim(400, 860)

def main(
    path_to_csv_spectrum: str, path_to_bi_results: str, path_to_fitspy_bd_results:str, q_branch_range, mixed_range,
    bd_wl_range: tuple, bd_fit_range: tuple, bi_wl_range,
    calibration_line, bi_calibration_line, path_to_nd_spectrum, nd_wl_range, nd_reference_wl,
    path_to_bb_spectrum
):
    spectrum_df = pd.read_csv(path_to_csv_spectrum, comment= '#').apply(pd.to_numeric)
    nd_spectrum_df = pd.read_csv(path_to_nd_spectrum, comment='#').apply(pd.to_numeric)
    bb_spectrum_df = pd.read_csv(path_to_bb_spectrum, comment='#').apply(pd.to_numeric)
    bb_spectrum_df = bb_spectrum_df[bb_spectrum_df['Radiance (W/cm^2/ster/nm)'] > 0.].reset_index(
        drop=True)  # Remove negative values

    bd_file = os.path.basename(path_to_fitspy_bd_results)
    file_tag = os.path.splitext(bd_file)[0]
    folder = os.path.basename(os.path.dirname(path_to_fitspy_bd_results))
    bd_model_df = pd.read_csv(path_to_fitspy_bd_results, sep=';', usecols=np.arange(0, 5)).set_index(['label'])
    num_cols = ['x0', 'ampli', 'fwhm']
    bd_model_df[num_cols] = bd_model_df[num_cols].apply(pd.to_numeric)
    model_df = bd_model_df.reset_index(drop=True)
    n_peaks = len(model_df)
    path_to_bd_stats_file = os.path.join(os.path.dirname(path_to_fitspy_bd_results), file_tag + '_stats.txt')

    initial_values_bb = read_initial_params_from_comments(path_to_bb_spectrum)

    load_plot_style()

    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    fig.set_size_inches(7., 6.)

    plot_bd_spectrum(
        ax=axes[0, 0], full_df=spectrum_df, model_df=bd_model_df, wl_range=bd_wl_range, calibration_line=calibration_line,
        path_to_stats_file=path_to_bd_stats_file, q_branch_range=q_branch_range,
        mixed_range=mixed_range
    )

    plot_bi_spectrum(ax=axes[0, 1], full_df=spectrum_df, wl_range=bi_wl_range, calibration_line=bi_calibration_line)
    plot_nd_spectrum(ax=axes[1, 0], full_df=nd_spectrum_df, wl_range=nd_wl_range, reference_wl=nd_reference_wl)
    plot_black_body(ax=axes[1, 1], full_df=bb_spectrum_df, initial_guess=initial_values_bb)

    for i, axi in enumerate(axes.flatten()):
        panel_label = chr(ord('`') + i + 1) # starts from a
        axi.text(
            -0.125, 1.075, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )

    fig.savefig(r'./figures/fig_oes_spectra.png', dpi=600)
    fig.savefig(r'./figures/fig_oes_spectra.svg', dpi=600)
    plt.show()

if __name__ == '__main__':
    main(
        path_to_csv_spectrum=PATH_TO_CSV_SPECTRUM,
        path_to_bi_results=PATH_TO_FITSPY_CD_RESULTS,
        path_to_fitspy_bd_results=PATH_TO_FITSPY_CD_RESULTS,
        q_branch_range=Q_BRANCH_RANGE,
        mixed_range=MIXED_RANGE,
        bd_wl_range=BD_WL_RANGE, bd_fit_range=BD_FIT_RANGE,
        bi_wl_range=BI_WL_RANGE,
        calibration_line=CALIBRATION_LINE,
        bi_calibration_line=BI_CALIBRATION_LINE,
        path_to_nd_spectrum=PATH_TO_ND_SPECTRUM,
        nd_wl_range=ND_WL_RANGE,
        nd_reference_wl=ND_REFERENCE_WL,
        path_to_bb_spectrum=PATH_TO_BB_SPECTRUM,
    )