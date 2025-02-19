"""
This code tries to fit the approximate TDS profile according to

E. Tomkova/Surface Science 351 (1996) 309-318

"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import os
import json
import numpy as np
from scipy.optimize import least_squares, OptimizeResult, differential_evolution
import data_processing.confidence as cf
from scipy.stats.distributions import t
import matplotlib as mpl
from scipy.integrate import simpson
from data_processing.utils import latex_float, latex_float_with_error
from matplotlib.font_manager import FontProperties
from data_processing.utils import lighten_color

TDS_FILE = r'./data/Checchetto/Checchetto Fig 2a.csv'
RESULTS_FOLDER = r'./data/Checchetto/fit_results'
TDS_RAMP_RATE = 0.5 # K/s


k_b = 8.617333262E-5 # eV/K


def get_t(n):
    return t.ppf(1 - 0.05/2., n-1)

def r1(TK, r_m, e_a, T_m):
    x = (TK - T_m) / T_m
    a = e_a / k_b / T_m
    xx = 1. + x
    return r_m * np.exp(a*x/xx - np.power(xx, 2.)*np.exp(a*x/xx) + 1.)


def u1(TK, r_m, e_a, T_m):
    x = (TK - T_m) / T_m
    a = e_a / k_b / T_m
    xx = 1. + x
    arg = a * x / xx
    r = arg - a * x * np.exp(arg)
    return r

def u2(TK, r_m, e_a, T_m):
    x = (TK - T_m) / T_m
    a = e_a / k_b / T_m
    xx = TK / T_m # (1 + x)
    arg = a * x / xx
    r = (a*xx + 2. * np.power(xx, 2.)) * np.exp(arg) - a
    return r


def v1(TK, r_m, e_a, T_m):
    x = (TK - T_m) / T_m
    a = 0.5 * e_a / k_b / T_m
    xx = 1. + x
    arg = a * x / xx
    r = arg * (np.exp(-arg) - np.power(xx, 2.) * np.exp(arg))
    return r

def v2(TK, r_m, e_a, T_m):
    x = (TK - T_m) / T_m
    a = 0.5 * e_a / k_b / T_m
    xx = TK / T_m
    arg = a * x / xx
    r = (2.+a) * np.power(xx, 2.) * np.exp(arg) - 2. * arg * np.exp(-arg)
    return r


def model_1(x, b):
    return r1(x, b[0], b[1], b[2])

def res_1(b, x, y):
    rm = b[0]
    ea = b[1]
    tm = b[2]
    return r1(x, rm, ea, tm) - y


def jac_1(b, x, y):
    rm = b[0]
    ea = b[1]
    tm = b[2]

    rr = r1(x, rm, ea, tm)

    u_0 = (1. / rm) * np.ones(len(x), dtype=np.float64)
    u_1 = (1. / ea) * u1(x, rm, ea, tm)
    u_2 = (1. / tm) * u2(x, rm, ea, tm)
    jm = (np.stack([u_0 , u_1 , u_2 ]) * rr).T
    return jm


def model_npeaks(x, b):
    n_pars = len(b)
    if not n_pars % 3 == 0:
        raise ValueError(f'Invalid number of fitting parameters, must be a multiple of 3, {n_pars} were given')
    n_peaks = n_pars // 3
    rm_arr = np.empty(n_peaks, dtype=float)
    ea_arr = np.empty(n_peaks, dtype=float)
    tm_arr = np.empty(n_peaks, dtype=float)

    start_indices = np.arange(0, n_pars, 3)
    for i, idx in enumerate(start_indices):
        rm_arr[i] = b[idx]
        ea_arr[i] = b[idx + 1]
        tm_arr[i] = b[idx + 2]

    r = 0
    for i in range(n_peaks):
        r += r1(x, rm_arr[i], ea_arr[i], tm_arr[i])

    return r

def jac_npeaks(b, x, y):
    n_pars = len(b)
    m = len(x)
    if not n_pars % 3 == 0:
        raise ValueError(f'Invalid number of fitting parameters, must be a multiple of 3, {n_pars} were given')
    n_peaks = n_pars // 3
    rm_arr = np.empty(n_peaks, dtype=float)
    ea_arr = np.empty(n_peaks, dtype=float)
    tm_arr = np.empty(n_peaks, dtype=float)

    start_indices = np.arange(0, n_pars, 3)
    for i, idx in enumerate(start_indices):
        rm_arr[i] = b[idx]
        ea_arr[i] = b[idx + 1]
        tm_arr[i] = b[idx + 2]

    result = np.zeros((m, n_pars), dtype=np.float64)
    for start_idx in start_indices:
        end_idx = start_idx + 4
        result[:, start_indices:end_idx] = jac_1(b[start_idx:end_idx], x, y)

    return result





def model_sum1(x, b):
    nn = len(b)
    selector = np.arange(0, nn) % 3
    msk_rm = selector == 0
    msk_ea = selector == 1
    msk_tm = selector == 2
    rms = b[msk_rm]
    eas = b[msk_ea]
    tms = b[msk_tm]
    n = len(rms)
    result = np.zeros_like(x, dtype=np.float64)
    for i in range(n):
        result += r1(x, rms[i], eas[i], tms[i])
    return result

def res_sum1(b, x, y):
    return model_sum1(x, b) - y

def res_sum1_de(b, x, y):
    return 0.5 * np.linalg.norm(res_sum1(b, x, y))

def jac_sum1(b, x, y):
    nn = len(b)
    m = len(x)
    selector = np.arange(0, nn) % 3
    msk_rm = selector == 0
    msk_ea = selector == 1
    msk_tm = selector == 2
    rms = b[msk_rm]
    eas = b[msk_ea]
    tms = b[msk_tm]
    n = len(rms)
    result = np.zeros((m, nn), dtype=np.float64)
    for i in range(n):
        cols = 3*i + np.arange(0, 3)
        result[:, cols] = jac_1(np.array([rms[i], eas[i], tms[i]]), x, y)
    return result

def get_deuterium_retention(tds_df: pd.DataFrame, tds_ramp_rate):
    temperature = tds_df['Temperature (K)'].values
    deuterium_flux = tds_df['Effusion rate (D2/m^2/s)'].values
    time_ramp = temperature / tds_ramp_rate


    integrated_d = simpson(y=deuterium_flux, x=time_ramp)
    return integrated_d

def load_plot_style():
    with open('./plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    rcParams.update(plot_style)
    rcParams['pgf.texsystem'] = 'pdflatex'
    rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{color}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')


def fit_tds(xdata, ydata, x0, bounds, loss='soft_l1', f_scale=0.1, tol=None) -> OptimizeResult:
    n_params = len(x0)
    if tol is None:
        tol = np.finfo(np.float64).eps

    lower_bounds, upper_bounds = bounds
    bounds_de = []
    for lb, ub in zip(lower_bounds, upper_bounds):
        if np.isinf(lb):
            lb = 1E40
        if np.isinf(ub):
            ub = 1E40
        bounds_de.append((lb, ub))

    result_de: OptimizeResult = differential_evolution(
            func=res_sum1_de,
            args=(xdata, ydata),
            x0=x0,
            bounds=bounds_de,
            maxiter=1000 * len(x0),
            # tol=tol,
            # atol=tol,
            workers=-1,
            updating='deferred',
            recombination=0.5,
            strategy='best1bin',
            mutation=(0.5, 1.5),
            init='sobol',
            polish=False,
            disp=True
        )

    result: OptimizeResult = least_squares(
        res_sum1,
        jac=jac_sum1,
        x0=result_de.x,
        bounds=bounds,
        args=(xdata, ydata),
        loss=loss,
        f_scale=f_scale,
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=2,
        # x_scale='jac',
        max_nfev=1000 * len(x0)
    )

    return result

def load_tds_data(path_to_csv) -> pd.DataFrame:
    df = pd.read_csv(path_to_csv, comment = '#').apply(pd.to_numeric)
    df['Effusion rate (D2/m^2/s)'] = df['Effusion rate (D2/cm^2/s)'] * 1E4
    return df


def plot_fit_model(ax:plt.Axes, fit_result: OptimizeResult, x_pred: np.ndarray):
    popt = fit_result.x
    n_peaks = len(popt) // 3
    selector = np.arange(0, len(popt)) % 3
    msk_rm = selector == 0
    msk_ea = selector == 1
    msk_tm = selector == 2

    rms = popt[msk_rm]
    eas = popt[msk_ea]
    tms = popt[msk_tm]
    y_sum = np.zeros_like(x_pred)
    for i in range(n_peaks):
        bi = np.array([rms[i], eas[i], tms[i]])
        y_pred_i = model_npeaks(x_pred, bi)
        ax.plot(x_pred, y_pred_i, lw=0.75, color='0.2')
        y_sum += y_pred_i
    ax.plot(x_pred, y_sum, lw=2., color='k', label='Model')


def save_fit_results(fit_result: OptimizeResult, path_to_results_file, d_retention):
    popt = fit_result.x
    n_peaks = len(popt) // 3
    ci = cf.confidence_interval(res=fit_result, level=0.95)


    selector = np.arange(0, len(popt)) % 3
    msk_rm = selector == 0
    msk_ea = selector == 1
    msk_tm = selector == 2
    rms, rms_e = popt[msk_rm], ci[msk_rm,1]-popt[msk_rm]
    eas, eas_e = popt[msk_ea], ci[msk_ea,1]-popt[msk_ea]
    tms, tms_e = popt[msk_tm], ci[msk_tm,1]-popt[msk_tm]

    fit_result_df = pd.DataFrame(data={
        'Peak': np.arange(len(rms)) + 1,
        'r_m': rms,
        'r_m_error': rms_e,
        'Ea': eas,
        'Ea_error': eas_e,
        'Tm': tms,
        'Tm_error': tms_e
    })

    with open(path_to_results_file, 'w') as f:
        f.write(f'# Total D retention: {d_retention:.3E} 1/m^2\n')
        fit_result_df.to_csv(f, index=False)
    return

def main(path_to_csv, path_to_results, tds_ramp_rate):
    checchetto_hbn_20keV_df = load_tds_data(path_to_csv)

    """
    Fit the data for D+ implanted on hBN at 20 keV  with single desorption peak of order 1
    """
    x_data_hbn_20keV = checchetto_hbn_20keV_df['Temperature (K)'].values
    y_data_hbn_20keV = checchetto_hbn_20keV_df['Effusion rate (D2/m^2/s)'].values
    d_max = y_data_hbn_20keV.max()  # get rm
    idx_max = np.argmin(np.abs(y_data_hbn_20keV - d_max))  # find the index of the max
    t_m = x_data_hbn_20keV[idx_max]  # explicitly get tm
    x0 = np.array([
        d_max * 0.1, 0.5, t_m * 0.8,
        d_max * 0.9, 1.0, t_m,
    ])
    bounds = (
        [
            0., 0.01, x_data_hbn_20keV.min(),
            0., 1.0, x_data_hbn_20keV.min()
        ],
        [
            np.inf, 3., x_data_hbn_20keV.max(),
            np.inf, 6., 2000.
        ]
    )

    x0 = np.array([
        2E16, 0.1, 800,
        1E16, 0.1, 860,
        1E15, 0.1, 1080,
    ])

    bounds = (
        [
            1E-5, 1E-2, 690,
            1E-5, 1E-2, 700,
            1E-5, 1E-2, 900,
        ],
        [
            d_max, 1E1, 800,
            d_max, 1E1, 1150,
            d_max, 1E1, 1150
        ]
    )

    fit_result_hbn_20keV: OptimizeResult = fit_tds(
        xdata=x_data_hbn_20keV, ydata=y_data_hbn_20keV, x0=x0, bounds=bounds, loss='soft_l1', f_scale=0.1
    )

    x_pred = np.linspace(300, 1500, num=1000)
    results_file = os.path.splitext(os.path.basename(path_to_csv))[0] + '_fit.csv'
    path_to_results_file = os.path.join(path_to_results, results_file)
    if not os.path.exists(path_to_results):
        os.makedirs(path_to_results)
    save_fit_results(
        fit_result=fit_result_hbn_20keV,
        path_to_results_file=path_to_results_file,
        d_retention=get_deuterium_retention(tds_df=checchetto_hbn_20keV_df, tds_ramp_rate=tds_ramp_rate)
    )

    y_pred, delta = cf.prediction_intervals(
        model=model_sum1, x_pred=x_pred, ls_res=fit_result_hbn_20keV, jac=jac_sum1
    )

    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.5)
    ax.plot(
        checchetto_hbn_20keV_df['Temperature (K)'].values,
        checchetto_hbn_20keV_df['Effusion rate (D2/m^2/s)'].values, marker='o',
        color='C0', mew=1., mfc='none', ls='none', ms=7,
        label='Total D'
    )

    ax.plot(x_pred, y_pred, color='tab:red')
    plot_fit_model(ax=ax, fit_result=fit_result_hbn_20keV, x_pred=x_pred)

    ax.set_title(r'{\sffamily D\textsuperscript{+} → hBN 20~keV}', usetex=True)

    ax.ticklabel_format(axis='y', useMathText=True)
    ax.tick_params(axis='y', which='both', right=True)

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(r'Desorption flux (m$^{\mathregular{-2}}$ s$^{\mathregular{-1}}$)')
    ax.set_xlim(300,1500)
    ax.set_ylim(9, 7E17)

    fig.savefig(os.path.splitext(path_to_results_file)[0] + '.svg', dpi=600)
    fig.savefig(os.path.splitext(path_to_results_file)[0] + '.png', dpi=600)

    plt.show()


if __name__ == '__main__':
    main(path_to_csv=TDS_FILE, path_to_results=RESULTS_FOLDER, tds_ramp_rate=TDS_RAMP_RATE)

