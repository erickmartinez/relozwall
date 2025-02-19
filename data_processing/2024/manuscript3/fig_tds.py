"""
This code tries to fit the approximate TDS profile according to

E. Tomkova/Surface Science 351 (1996) 309-318

"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from docutils.nodes import label
from matplotlib import rcParams
import os
import json
import numpy as np
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf
from scipy.stats.distributions import t
import matplotlib as mpl
from scipy.integrate import simpson
from data_processing.utils import latex_float, latex_float_with_error
from matplotlib.font_manager import FontProperties
from data_processing.utils import lighten_color

TDS_FILE_SBR = r'./data/TDS/20240909/Brod_mks.txt'
TDS_FILE_ABPR = r'./data/TDS/20240909/Bpebble_srs.txt'
TDS_FILE_PBPR = r'./data/TDS/20241014/Bpebble_crystalline_srs.txt'




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

def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    rcParams.update(plot_style)
    rcParams['pgf.texsystem'] = 'pdflatex'
    rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{color}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')

def draw_retention_table(ax, retained_dh, retained_d2):
    col_labels = [r'Retention m$^{\mathregular{-2}}$']
    row_labels = [r'DH ', r'D$_{\mathregular{2}}$ ', r'Total D ']
    table1_vals = [
        [fr'${latex_float(retained_dh, significant_digits=1)}$'],
        [fr'${latex_float(retained_d2, significant_digits=1)}$'],
        [fr'${latex_float(retained_dh + 2.*retained_d2, significant_digits=1)}$']
    ]
    row_colors = ['C1', 'C2', 'C0']
    table = ax.table(
        cellText=table1_vals,
        colLabels=col_labels,
        rowLabels=row_labels,
        rowColours=row_colors,
        loc='upper left',
        bbox=[0.28, 0.54, 0.48, 0.44],
        edges='closed',
        fontsize=12,
        rowLoc='left',
    )
    # table_1.PAD=-0.1

    cells = table.get_celld()

    for (row, col), cell in cells.items():
        if col == -1:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
            cell.set_text_props(color='w')
    table.set_fontsize(12)


def fit_tds(xdata, ydata, x0, bounds, loss='soft_l1', f_scale=0.1, tol=None) -> OptimizeResult:
    n_params = len(x0)
    if tol is None:
        tol = np.finfo(np.float64).eps

    result: OptimizeResult = least_squares(
        res_sum1,
        jac=jac_sum1,
        x0=x0,
        bounds=bounds,
        args=(xdata, ydata),
        loss=loss,
        f_scale=f_scale,
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=0,
        x_scale='jac',
        max_nfev=1000 * len(x0)
    )

    return result


def load_tds_data(path_to_csv) -> pd.DataFrame:
    df = pd.read_csv(path_to_csv, comment = '#', delimiter = r'\s+').apply(pd.to_numeric)
    df['[D/m^2/s]'] = np.where(df['[D/m^2/s]'] < 0, 0, df['[D/m^2/s]'])
    df['[HD/m^2/s]'] = np.where(df['[HD/m^2/s]'] < 0, 0, df['[HD/m^2/s]'])
    df['[D2/m^2/s]'] = np.where(df['[D2/m^2/s]'] < 0, 0, df['[D2/m^2/s]'])
    mean_heating_rate = np.mean(np.gradient(df['Temp[K]'].values, df['Time[s]'].values))
    print(f"file: {path_to_csv}, mean heating rate (K/s) {mean_heating_rate:.3E}")
    return df

def plot_d_retention(ax, tds_df, color, title=None):
    ax.plot(
        tds_df['Temp[K]'].values, tds_df['[D/m^2/s]'].values, marker='o',
        color=color, mew=1., mfc='none', ls='none', ms=7,
        label='Total D'
    )
    ax.plot(
        tds_df['Temp[K]'].values, tds_df['[D2/m^2/s]'].values, marker='^',
        color=lighten_color(color, 0.5), mew=1., mfc='none', ls='none', ms=7,
        label=r'D$_{\mathregular{2}}$'
    )
    if not title is None:
        ax.set_title(title)

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


def get_deuterium_retention(tds_df: pd.DataFrame):
    time_s = tds_df['Time[s]'].values
    d_total = tds_df['[D/m^2/s]'].values
    dh = tds_df['[HD/m^2/s]'].values
    d2 = tds_df['[D2/m^2/s]'].values

    integrated_d = simpson(y=d_total, x=time_s)
    integrated_dh = simpson(y=dh, x=time_s)
    integrated_d2 = simpson(y=d2, x=time_s)
    # print(f'Retained D: {integrated_d:.3E}')
    return integrated_d, integrated_dh, integrated_d2

def save_fit_results(fit_result: OptimizeResult, path_to_tds_file, d_retention):
    popt = fit_result.x
    n_peaks = len(popt) // 3
    ci = cf.confidence_interval(res=fit_result, level=0.95)
    popt_delta = np.empty(len(popt), dtype=np.float64)
    for i, p, lci, uci in zip(range(n_peaks), popt, ci[:, 0], ci[:, 1]):
        popt_delta[i] = uci - p

    selector = np.arange(0, len(popt)) % 3
    msk_rm = selector == 0
    msk_ea = selector == 1
    msk_tm = selector == 2
    rms, rms_e = popt[msk_rm], popt_delta[msk_rm]
    eas, eas_e = popt[msk_ea], popt_delta[msk_ea]
    tms, tms_e = popt[msk_tm], popt_delta[msk_tm]

    fit_result_df = pd.DataFrame(data={
        'Peak': np.arange(len(rms)) + 1,
        'r_m': rms,
        'r_m_error': rms_e,
        'Ea': eas,
        'Ea_error': eas_e,
        'Tm': tms,
        'Tm_error': tms_e
    })

    print(fit_result_df)

    (integrated_d, integrated_dh, integrated_d2) = d_retention

    filename_without_extension = os.path.splitext(path_to_tds_file)[0]
    with open(filename_without_extension + '_fit_results.csv', 'w') as f:
        f.write(f'# Total D retention: {integrated_d:.3E} 1/m^2\n')
        f.write(f'# DH retention: {integrated_dh:.3E} 1/m^2\n')
        f.write(f'# D2 retention: {integrated_d2:.3E} 1/m^2\n')
        fit_result_df.to_csv(f, index=False)


def main(tds_file_sbr, tds_file_abpr, tds_file_pbpr):
    sbr_df = load_tds_data(tds_file_sbr)
    abpr_df = load_tds_data(tds_file_abpr)
    pbpr_df = load_tds_data(tds_file_pbpr)

    load_plot_style()

    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True, sharex=True)
    fig.set_size_inches(4., 5.5)

    plot_d_retention(axes[0], sbr_df, color='C0', title='Boron rod (SBR)')
    plot_d_retention(axes[1], abpr_df, color='C1', title='Boron pebble rod (ABPR)')
    plot_d_retention(axes[2], pbpr_df, color='C2', title='Boron pebble rod (PBPR)')

    """
    Fit the data for boron rod total d2  with single desorption peak of order 1
    """
    x_data_sbr = sbr_df['Temp[K]'].values
    y_data_sbr = sbr_df['[D/m^2/s]'].values
    d_max = y_data_sbr.max()  # get rm
    idx_max = np.argmin(np.abs(y_data_sbr - d_max))  # find the index of the max
    t_m = x_data_sbr[idx_max]  # explicitly get tm
    x0 = np.array([
        d_max * 0.1, 0.5, t_m * 0.8,
        d_max * 0.9, 1.0, t_m,
    ])
    bounds = (
        [
            0., 0.01, x_data_sbr.min(),
            0., 1.0, x_data_sbr.min()
        ],
        [
            np.inf, 3., x_data_sbr.max(),
            np.inf, 6., 2000.
        ]
    )

    fit_result_sbr: OptimizeResult = fit_tds(
        xdata=x_data_sbr, ydata=y_data_sbr, x0=x0, bounds=bounds, loss='soft_l1', f_scale=10.
    )

    x_pred = np.linspace(300, 1200, 2000)

    save_fit_results(
        fit_result=fit_result_sbr, path_to_tds_file=tds_file_sbr, d_retention=get_deuterium_retention(sbr_df)
    )

    y_pred_sbr, delta_sbr = cf.prediction_intervals(
        model=model_sum1, x_pred=x_pred, ls_res=fit_result_sbr, jac=jac_sum1
    )

    # axes[0].plot(x_pred, y_pred_sbr, color='k', label='Model')
    plot_fit_model(ax=axes[0], fit_result=fit_result_sbr, x_pred=x_pred)

    """
    Fit the data for boron pebble rod d-h with multiple desorption peaks
    The initial guess for the least_squares
    """
    x_data_abpr = abpr_df['Temp[K]'].values
    y_data_abpr = abpr_df['[D/m^2/s]'].values

    x0 = np.array([
        5E16, 0.3, 380,
        4E16, 0.3, 490,
        1E14, 0.5, 700,
        4E16, 1.0, 810,
        2E17, 2.0, 1100,
    ])

    bounds = (
        [
            0, 0.1, 200,
            0., 0.1, 420,
            0., 0.1, 650,
            0., 0.1, 800,
            0., 0.1, 900,
        ],
        [
            1E17, np.inf, 450,
            1E17, np.inf, 600,
            1E17, np.inf, 830,
            1E17, np.inf, 950,
            1E18, np.inf, 2000,
        ]
    )

    fit_result_abpr: OptimizeResult = fit_tds(
        xdata=x_data_abpr, ydata=y_data_abpr, x0=x0, bounds=bounds, loss='soft_l1', f_scale=10.
    )


    y_pred_abpr, delta_abpr = cf.prediction_intervals(
        model=model_sum1, x_pred=x_pred, ls_res=fit_result_abpr, jac=jac_sum1
    )

    # axes[1].plot(x_pred, y_pred_abpr, color='k', label='Model')
    plot_fit_model(ax=axes[1], fit_result=fit_result_abpr, x_pred=x_pred)

    save_fit_results(
        fit_result=fit_result_abpr, path_to_tds_file=tds_file_abpr, d_retention=get_deuterium_retention(abpr_df)
    )

    """
    Fit the data for boron rod (PBPR) total d2 with single desorption peak of order 1
    """
    x_data_pbpr = pbpr_df['Temp[K]'].values
    y_data_pbpr = pbpr_df['[D/m^2/s]'].values
    d_max = y_data_pbpr.max()  # get rm
    idx_max = np.argmin(np.abs(y_data_pbpr - d_max))  # find the index of the max
    t_m = x_data_pbpr[idx_max]  # explicitly get tm
    x0 = np.array([
        d_max * 0.5, 1.0, t_m * 0.9,
        d_max * 0.9, 1.5, t_m* 1.1,
    ])
    bounds = (
        [
            0., 0.01, x_data_pbpr.min(),
            0., 0.1, x_data_pbpr.min()
        ],
        [
            np.inf, 3., 2000,
            np.inf, 10., 2000.
        ]
    )

    fit_result_pbpr: OptimizeResult = fit_tds(
        xdata=x_data_pbpr, ydata=y_data_pbpr, x0=x0, bounds=bounds, loss='soft_l1', f_scale=10.
    )


    save_fit_results(
        fit_result=fit_result_pbpr, path_to_tds_file=tds_file_pbpr, d_retention=get_deuterium_retention(pbpr_df)
    )

    y_pred_pbpr, delta_pbpr = cf.prediction_intervals(
        model=model_sum1, x_pred=x_pred, ls_res=fit_result_pbpr, jac=jac_sum1
    )

    # axes[2].plot(x_pred, y_pred_pbpr, color='k', label='Model')
    plot_fit_model(ax=axes[2], fit_result=fit_result_pbpr, x_pred=x_pred)

    for ax in axes:
        ax.ticklabel_format(axis='y', useMathText=True)
        ax.tick_params(axis='y', which='both', right=True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        ax.legend(loc='upper left', frameon=True, fontsize=10)

    axes[-1].set_xlim(200, 1200)
    axes[-1].set_xlabel('Temperature (K)')

    axes[0].set_ylim(0, 3E20)
    axes[1].set_ylim(0, 6E17)
    axes[2].set_ylim(0, 6E17)

    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(1E20))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(5E19))

    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(2E17))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(1E17))

    axes[2].yaxis.set_major_locator(ticker.MultipleLocator(2E17))
    axes[2].yaxis.set_minor_locator(ticker.MultipleLocator(1E17))

    fig.supylabel(r'Desorption flux (m$^{\mathregular{-2}}$ s$^{\mathregular{-1}}$)')

    for i, axi in enumerate(axes.flatten()):
        panel_label = chr(ord('`') + i + 1) # starts from a
        # panel_label = chr(ord('`') + i + 3)
        axi.text(
            -0.12, 1.1, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )


    fig.savefig(r'./figures/fig_tds_plots.svg', dpi=600)
    plt.show()



if __name__ == '__main__':
    main(tds_file_sbr=TDS_FILE_SBR, tds_file_abpr=TDS_FILE_ABPR, tds_file_pbpr=TDS_FILE_PBPR)