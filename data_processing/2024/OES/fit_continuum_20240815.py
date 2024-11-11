import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import os
import json
from scipy.optimize import least_squares, OptimizeResult
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import data_processing.confidence as cf
from matplotlib.lines import Line2D
import matplotlib as mpl
from scipy.integrate import simpson

spectrum_csv = r'./data/brightness_data_fitspy_wl-calibrated/echelle_20240815/MechelleSpect_007.csv'


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
def gaussian(x, c, sigma, mu):
    global over_sqrt_pi
    p = c * over_sqrt_pi / sigma
    arg = 0.5 * np.power((x - mu) / sigma, 2)
    return p * np.exp(-arg)

def sum_gaussians(x, b):
    global over_sqrt_pi
    m = len(x)
    nn = len(b)
    # Assume that the new b contains a list ordered like
    # (c1, sigma_1, mu1, c_2, sigma_2, mu_2, ..., c_n, sigma_n, mu_n)
    selector = np.arange(0, nn) % 3
    msk_c = selector == 0
    msk_sigma = selector == 1
    msk_mu = selector == 2
    cs = b[msk_c]
    sigmas = b[msk_sigma]
    mus = b[msk_mu]
    n = len(mus)
    u = over_sqrt_pi * np.power(sigmas, -1) * cs
    u = u.reshape((1, len(u)))
    v = np.zeros((n, m), dtype=np.float64)
    for i in range(len(sigmas)):
        arg = 0.5*np.power((x-mus[i])/sigmas[i], 2.)
        v[i, :] = np.exp(-arg)
    res = np.dot(u, v)
    return res.flatten()


def res_sum_gauss(b, x, y):
    return sum_gaussians(x, b) - y

def jac_sum_gauss(b, x, y):
    m, nn  = len(x), len(b)
    # Assume that the new b contains a list ordered like
    # (c1, sigma_1, mu1, c_2, sigma_2, mu_2, ..., c_n, sigma_n, mu_n)
    selector = np.arange(0, nn) % 3
    msk_c = selector == 0
    msk_sigma = selector == 1
    msk_mu = selector == 2
    c = b[msk_c]
    s = b[msk_sigma]
    mu = b[msk_mu]
    r = np.zeros((m, nn), dtype=np.float64)
    for i in range(len(s)):
        k = 3 * i
        g = gaussian(x, c[i], s[i], mu[i])
        r[:, k] = g / c[i]
        r[:, k+1] = np.power(s[i], -1) * ( np.power( (x - mu[i]) / s[i], 2) - 1.) * g
        r[:, k+2] = np.power(s[i], -2) * ( x - mu[i]) * g

    return r

def main():
    global spectrum_csv
    all_tol = float(np.finfo(np.float64).eps)
    spectrum_df = pd.read_csv(spectrum_csv).apply(pd.to_numeric)
    wl = spectrum_df['Wavelength (nm)'].values
    brightness = spectrum_df['Brightness (photons/cm^2/s/nm)'].values * 1E-12
    n = len(brightness)
    # find the Dalpha peak
    dg_wl = 433.93
    wl_del = 0.2
    msk_da = ((dg_wl - wl_del) <= wl) & (wl <= (dg_wl + wl_del))
    wl_win = wl[msk_da]
    y_win = brightness[msk_da]
    y_peak = y_win.max()
    idx_peak = np.argmin(np.argmin(y_win - y_peak))
    wl_peak = wl_win[idx_peak]

    wl_del = 0.75
    msk_da = ((dg_wl - wl_del) <= wl) & (wl <= (dg_wl + wl_del))
    wl_win = wl[msk_da]
    y_win = brightness[msk_da]
    area_window = simpson(y=y_win, x=wl_win)
    c_window = area_window * over_sqrt_pi
    x0 = [1.8*c_window, 0.01, wl_peak]
    # Fit Dalpha
    res_lsq = least_squares(
        res_sum_gauss, x0=x0, args=(wl_win, y_win),
        loss='linear', f_scale=0.1,
        jac=jac_sum_gauss,
        bounds=(
            [1E-10, 1E-10, wl_peak - 0.2],
            [100. * c_window, np.inf, wl_peak + 0.2]
        ),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        verbose=2,
        # x_scale='jac',
        max_nfev=10000 * len(wl_win)
    )
    popt_da = res_lsq.x
    ci_da = cf.confidence_interval(res=res_lsq)
    popt_delta_da = ci_da[:, 1] - popt_da
    x_pred_da = np.linspace(wl_win.min(), wl_win.max(), num=200)
    y_pred_da, delta_da = cf.prediction_intervals(sum_gaussians, x_pred=x_pred_da, ls_res=res_lsq, jac=jac_sum_gauss)


    ls_res = least_squares(
        res_poly,
        x0=[(0.01) ** (i+1) for i in range(6)],
        args=(wl, brightness),
        loss='cauchy', f_scale=0.1,
        jac=jac_poly,
        xtol=all_tol ** 0.5,
        ftol=all_tol ** 0.5,
        gtol=all_tol ** 0.5,
        verbose=2,
        x_scale='jac',
        max_nfev=10000 * n
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
        ls='none', marker='o', alpha=0.25,
        label='Data'
    )
    line_fit, = ax1.plot(xpred, ypred, color='tab:red', label='Baseline fit')
    ax1.fill_between(xpred, ypred-delta, ypred+delta, color='yellow')
    base_filename = os.path.splitext(os.path.basename(spectrum_csv))[0]
    base_folder = os.path.basename(os.path.dirname(spectrum_csv))
    ax1.set_title(f"{base_folder} - {base_filename}")

    msk_da_plot = ((dg_wl - 3.) <= wl) & (wl <= (dg_wl + 3.))
    print(brightness[msk_da_plot])
    ax2.plot(wl[msk_da_plot], brightness[msk_da_plot], ms=6, marker='o', mec='C0', mfc='none', label='data', ls='none')
    ax2.plot(x_pred_da, y_pred_da, color='tab:red', label='Fit')

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

    ax1.text(
        0.05, 0.95, eq_txt, transform=ax1.transAxes,
        ha='left', va='top', fontsize=11, color='tab:red', usetex=True
    )
    with open(os.path.join('./data', f'baseline_{base_folder}_{base_filename}.csv'), 'w') as f:
        intensity_da = sum_gaussians([popt_da[2]], popt_da)
        f.write(f"# D_gamma: {popt_da[2]:.3f} -/+ {delta_da[2]:.4f} nm, Intensity: {intensity_da[0]:.3E} (photons/cm^2/s/nm)\n")
        baseline_df.to_csv(f, index=False)

    fig.savefig(os.path.join('./figures', f'baseline_{base_folder}_{base_filename}.png'), dpi=600)
    plt.show()

if __name__ == '__main__':
    main()


