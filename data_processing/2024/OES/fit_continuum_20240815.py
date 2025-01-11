import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import os
import json
from scipy.optimize import least_squares, OptimizeResult, minimize
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import data_processing.confidence as cf
from matplotlib.lines import Line2D
import matplotlib as mpl
from scipy.integrate import simpson


spectrum_csv = r'./data/brightness_data_fitspy_wl-calibrated/echelle_20240815/MechelleSpect_010.csv'
ref_wavelength = 410 # nm


lookup_lines = [
    {'center_wl': 410.06, 'label': r'D$_{\delta}$'},
    {'center_wl': 433.93, 'label': r'D$_{\gamma}$'},
    {'center_wl': 486.00, 'label': r'D$_{\beta}$'},
    {'center_wl': 656.10, 'label': r'D$_{\alpha}$'}
]

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    plt.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}')


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

over_sqrt_pi = 1. / np.sqrt(2. * np.pi)


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
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + baseline


def residuals_gaussian(params, x, y):
    """Calculate residuals between observed data and the Gaussian model"""
    return gaussian(x, params) - y


def jacobian_gaussian(params, x, y):
    """
    Analytical Jacobian matrix for the Gaussian function with baseline
    Returns partial derivatives with respect to (A, mu, sigma, baseline)
    """
    A, mu, sigma, baseline = params
    exp_term = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    # Partial derivatives
    d_A = exp_term
    d_mu = A * exp_term * (x - mu) / sigma ** 2
    d_sigma = A * exp_term * (x - mu) ** 2 / sigma ** 3
    d_baseline = np.ones_like(x)  # Derivative with respect to baseline

    return np.vstack([d_A, d_mu, d_sigma, d_baseline]).T


def fit_gaussian(x, y, p0=None, loss='linear', f_scale=1.0, tol=None) -> OptimizeResult:
    """
    Fit Gaussian profile to data using least_squares with analytical Jacobian

    Parameters:
    x: array-like, independent variable
    y: array-like, dependent variable
    p0: initial guess for parameters (A, mu, sigma)

    Returns:
    OptimizeResult object containing the fitted parameters
    """
    if tol is None:
        tol = float(np.finfo(np.float64).eps)
    if p0 is None:
        # Make educated guesses for initial parameters
        baseline = np.min(y)  # Estimate baseline as minimum y value
        A = np.max(y) - baseline  # Estimate amplitude above baseline
        mu = x[np.argmax(y)]
        sigma = np.std(x) / 2
        p0 = np.array([A, mu, sigma, baseline])


    result = least_squares(
        residuals_gaussian,
        x0=p0,
        jac=jacobian_gaussian,
        args=(x, y),
        method='trf',
        loss = loss,
        f_scale = f_scale,
        xtol = tol,
        ftol = tol,
        gtol = tol,
        verbose = 0,
        x_scale='jac',
        max_nfev = 10000 * len(x)
    )

    return result


def main():
    global spectrum_csv, ref_wavelength
    all_tol = float(np.finfo(np.float64).eps)
    spectrum_df = pd.read_csv(spectrum_csv).apply(pd.to_numeric)
    spectrum_df = spectrum_df[spectrum_df['Brightness (photons/cm^2/s/nm)']>0.] # Remove points with zero signal
    wl = spectrum_df['Wavelength (nm)'].values
    brightness = spectrum_df['Brightness (photons/cm^2/s/nm)'].values * 1E-12
    brightness_err = spectrum_df['Brightness error (photons/cm^2/s/nm)'].values * 1E-12

    n = len(brightness)
    # find the D_delta peak
    dd_wl = 409.992
    wl_del = 0.2
    msk_da = ((dd_wl - wl_del) <= wl) & (wl <= (dd_wl + wl_del))
    wl_win = wl[msk_da]
    y_win = brightness[msk_da]
    y_peak = y_win.max()
    idx_peak = np.argmin(np.argmin(y_win - y_peak))
    wl_peak = wl_win[idx_peak]

    wl_del = 0.75
    msk_da = ((dd_wl - wl_del) <= wl) & (wl <= (dd_wl + wl_del))
    wl_win = wl[msk_da]
    y_win = brightness[msk_da]
    # Fit Dalpha
    res_lsq = fit_gaussian(wl_win, y_win, loss='soft_l1', f_scale=10., tol=all_tol)
    popt_da = res_lsq.x
    ci_da = cf.confidence_interval(res=res_lsq)
    popt_delta_da = ci_da[:, 1] - popt_da
    x_pred_da = np.linspace(wl_win.min(), wl_win.max(), num=200)
    y_pred_da, delta_da = cf.prediction_intervals(gaussian, x_pred=x_pred_da, ls_res=res_lsq, jac=jacobian_gaussian)


    # Try to fit a polynomial to find the baseline
    percentile_threshold= 20
    # Normalize wavelength to prevent numerical issues
    wavelength_norm = (wl - wl.min()) / (wl.max() - wl.min())
    # Initial baseline points selection using percentile threshold
    window_size = len(wl) // 100  # 1% of data length
    rolling_min = pd.Series(brightness).rolling(window=window_size, center=True).quantile(percentile_threshold / 100)
    baseline_points = ~pd.isna(rolling_min)

    # Set weights (higher weight for lower intensity points)
    weights = 1 / (brightness**2 + np.median(brightness) / 10) # Add small value to prevent division by zero
    weights[~baseline_points] = 0  # Zero weight for non-baseline points
    # weights = weights / (brightness_err + np.median(brightness_err) / 10) # Include measurement uncertainties
    # weights[wl<260] = 1 / (brightness[wl<260] + np.median(brightness) / 10)


    poly_order = 30
    ls_res = least_squares(
        res_poly,
        x0=[(0.001) ** (i)  for i in range(poly_order)],
        args=(wl[baseline_points], brightness[baseline_points], weights[baseline_points]),
        loss='cauchy', f_scale=0.4,
        jac=jac_poly,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        verbose=2,
        x_scale='jac',
        method='trf',
        tr_solver='exact',
        max_nfev=10000 * poly_order
    )
    popt = ls_res.x
    ci = cf.confidence_interval(res=ls_res)
    xpred = np.linspace(wl.min(), wl.max(), 500)
    ypred, delta = cf.prediction_intervals(model=model_poly, x_pred=xpred, ls_res=ls_res, jac=jac_poly)

    load_plot_style()

    fit_points_df = pd.DataFrame(data={'x': [wl[0]], 'y': [brightness[0]]})

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, width_ratios=[1., 0.65])
    fig.set_size_inches(7., 4.)
    ax1.set_xlabel(r"$\lambda$ {\sffamily (nm)}", usetex=True)
    ax1.set_ylabel(r"B (photons/cm$^{\mathregular{2}}$/s/nm) $\times$10$^{\mathregular{12}}$")



    ax1.set_ylim(bottom=-0.1, top=4)

    line = ax1.plot(
        wl, brightness, ms=2, color='C0', mfc='none',
        # picker=True, pickradius=1,
        ls='-', marker='o', alpha=0.25, lw=0.5,
        label='Data'
    )
    line_fit, = ax1.plot(xpred, ypred, color='tab:red', label='Baseline fit')
    ax1.fill_between(xpred, ypred-delta, ypred+delta, color='yellow')
    base_filename = os.path.splitext(os.path.basename(spectrum_csv))[0]
    base_folder = os.path.basename(os.path.dirname(spectrum_csv))
    ax1.set_title(f"{base_folder} - {base_filename}")

    msk_da_plot = ((dd_wl - 3.) <= wl) & (wl <= (dd_wl + 3.))
    # print(brightness[msk_da_plot])
    ax2.plot(wl[msk_da_plot], brightness[msk_da_plot], ms=6, marker='o', mec='C0', mfc='none', label='data', ls='none')
    ax2.plot(x_pred_da, y_pred_da, color='tab:red', label='Fit')

    # baseline = baseline_als(brightness, lam=1e9, p=0.1)

    # ax1.plot(wl, baseline, color='tab:green')

    eq_txt = fr"$f(x) = "
    n_par = len(popt)
    for i in range(n_par):
        eq_txt += f"a_{{{i}}}"
        if i > 0:
            eq_txt += "\lambda"
            if i > 1:
                eq_txt += fr"^{{{i}}}"
        if i + 1< n_par:
            eq_txt += " + "
    eq_txt += r"$\\" + "\n"
    for i in range(n_par):
        deltai =  popt[i] - ci[i, 0]
        eq_txt += fr"$a_{{{i}}} = \num{{{popt[i]:.2E}}} \pm \num{{{deltai:.3E}}}$"
        if i+1 < n_par:
            eq_txt += r"\\" + "\n"
    # eq_txt += "$"
    baseline_df = pd.DataFrame(data={
        f'a_{i}': [popt[i]] for i in range(n_par)
    })

    print(baseline_df)

    # Find the intensity at the selected reference wavelength
    idx_ref = np.argmin(np.abs(wl - ref_wavelength))
    ref_intensity = brightness[idx_ref] * 1E12
    ref_wavelength = wl[idx_ref]

    # ax1.text(
    #     0.05, 0.95, eq_txt, transform=ax1.transAxes,
    #     ha='left', va='top', fontsize=11, color='tab:red', usetex=True
    # )
    with open(os.path.join('./data', f'baseline_{base_folder}_{base_filename}.csv'), 'w') as f:
        intensity_da = popt_da[0]*1E12
        txt = f"# REF: {popt_da[1]:.3f} -/+ {delta_da[1]:.4f} nm, Intensity: {intensity_da:.3E} (photons/cm^2/s/nm)\n"
        f.write(txt)
        print(txt)
        txt = f"# Intensity at: {ref_wavelength:.3f} nm: {ref_intensity:.3E} (photons/cm^2/s/nm)\n"
        f.write(txt)
        print(txt)
        baseline_df.to_csv(f, index=False)

    fig.savefig(os.path.join('./figures', f'baseline_{base_folder}_{base_filename}.png'), dpi=600)
    plt.show()

if __name__ == '__main__':
    main()


