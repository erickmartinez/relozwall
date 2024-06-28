import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
from data_processing.utils import get_experiment_params, latex_float_with_error
from scipy.stats.distributions import t
from scipy.interpolate import interp1d
import json
from scipy.signal import savgol_filter
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf

data_csv = './data/thermocouple_response/tc2_20240526-3.csv'
skip_points = 10


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def poly(x, b):
    m, n = len(x), len(b)
    r = np.zeros(len(x))
    xi = np.ones(m)
    for i in range(n):
        r += xi * b[i]
        xi *= x
    return r


def res_poly(b, x, y, w=1.):
    return (poly(x, b) - y) * w


def jac_poly(b, x, y, w=1.):
    m, n = len(x), len(b)
    jac = np.ones((m, n), dtype=np.float64)
    jac[:, 0] = w
    xi = x.copy()
    for i in range(1, n):
        jac[:, i] = w * xi
        xi *= x
    return jac


def model(x, b):
    return b[0] * (1. - np.exp(-x / b[1]))


def res_model(b, x, y, w=1.):
    return (model(x, b) - y) * w


def jac_model(b, x, y, w=1.):
    m, n = len(x), len(b)
    jac = np.empty((m, n), dtype=np.float64)
    e = np.exp(-x / b[1])
    jac[:, 0] = w * (1. - e)
    jac[:, 1] = -w * x * b[0] * e / (b[1] ** 2.)
    return jac


def main():
    global data_csv
    df: pd.DataFrame = pd.read_csv(data_csv)
    time_s = df['Time (s)'].values
    temperature = df['TC2 (C)'].values

    # time_s = time_s[skip_points::]
    # temperature = temperature[skip_points::]
    # time_s -= time_s.min()

    load_plot_style()
    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True, height_ratios=[0.5, 1, 1])
    fig.set_size_inches(4.0, 6.0)

    dt = time_s[1] - time_s[0]
    n = len(time_s)
    k = int(n / 100)
    k = k - 1 if k % 2 == 0 else k
    k = max(k, 5)

    dTdt = savgol_filter(x=temperature, window_length=k, polyorder=2, deriv=1, delta=dt)

    max_change = dTdt.max()
    idx_max_change = np.argmin(np.abs(dTdt - max_change))

    t_max_change = time_s[idx_max_change]
    dTdt_max_change = dTdt[idx_max_change]

    time_start = time_s[0:idx_max_change]
    dTdt_start = dTdt[0:idx_max_change]
    msk_no_change = np.abs(dTdt_start) < 5.
    dTdt_no_change = dTdt_start[msk_no_change]
    time_no_change = time_start[msk_no_change]

    time_0 = time_no_change[-1]
    idx_t0 = np.argmin(np.abs(time_s - time_0))
    temp_0 = temperature[idx_t0]

    time_fit = time_s[idx_t0::]
    temperature_fit = temperature[idx_t0::]
    dT_fit = temperature_fit - temperature_fit[0]

    msk_no_zeros = dT_fit > 0.
    time_fit = time_fit[msk_no_zeros]
    temperature_fit = temperature_fit[msk_no_zeros]
    dT_fit = dT_fit[msk_no_zeros]
    time_fit -= time_fit[0]

    max_t_fit = 5.  # s
    msk_max_time = time_fit <= max_t_fit
    time_fit = time_fit[msk_max_time]
    temperature_fit = temperature_fit[msk_max_time]
    dT_fit = dT_fit[msk_max_time]
    time_fit -= time_fit[0]

    """
    Use the model dT = B + A*exp(-t/t_r) 
    """
    y = np.log(dT_fit)
    all_tol = float(np.finfo(np.float64).eps)
    weights = 1.
    res: OptimizeResult = least_squares(
        x0=[dT_fit.max() * 0.5, 0.25],
        fun=res_model,
        args=(time_fit, dT_fit, weights),
        jac=jac_model,
        # loss='soft_l1', f_scale=0.1,
        xtol=all_tol,  # ** 0.5,
        ftol=all_tol,  # ** 0.5,
        gtol=all_tol,  # ** 0.5,
        max_nfev=10000 * len(y),
        x_scale='jac',
        verbose=2
    )

    popt = res.x
    ci = cf.confidence_interval(res)
    delta_popt = np.abs(ci[:, 1] - popt)
    x_pred = np.linspace(time_fit.min(), time_fit.max(), num=500)
    y_pred, ydelta = cf.prediction_intervals(model=model, x_pred=x_pred, ls_res=res, jac=jac_model)
    poly_n = len(popt)
    fit_txt = fr'$A = {popt[0]:.1f}\pm{delta_popt[0]:.2f}~°\mathrm{{C}}$' + '\n'
    fit_txt += fr'$t_{{\mathrm{{s}}}} = {popt[1]:.3f}\pm{delta_popt[1]:.3f}~°\mathrm{{C}}$'

    axes[0].plot(
        time_s, dTdt, color='tab:purple'
    )

    # axes[0].plot(
    #     time_no_change, dTdt_no_change, color='k'
    # )

    axes[0].plot(
        [t_max_change], [dTdt_max_change], 'sg', ls='none', mfc='none', mew=1, ms=6
    )

    axes[0].plot(
        [time_0], [dTdt_no_change[-1]], 'or', ls='none', mfc='none', mew=1, ms=6
    )

    axes[1].errorbar(
        time_s, temperature, yerr=0.25, xerr=0.001,
        color='C0', marker='o',
        ms=9, mew=1.25, mfc='none', ls='none',
        capsize=2.75, elinewidth=1.25, lw=1.5,
        label='Data', zorder=0
    )

    axes[1].plot(
        [time_0], [temp_0], 'or', ls='none', mfc='none', zorder=1, mew=2
    )

    sqrt2 = np.sqrt(2)
    axes[2].errorbar(
        time_fit, dT_fit, yerr=0.25 * sqrt2, xerr=0.001 * sqrt2,
        color='C1', marker='s',
        ms=9, mew=1.25, mfc='none', ls='none',
        capsize=2.75, elinewidth=1.25, lw=1.5,
        label='Data', zorder=0
    )

    axes[2].plot(
        x_pred, y_pred, color='red', ls='-', label=r'$\Delta T = T_0\left[1 - \exp(-t/t_{\mathrm{r}})\right]$'
    )

    axes[0].set_ylabel(r'$d T / dt$')
    axes[1].set_xlabel('$t$ (s)')
    axes[1].set_ylabel('$T$ (°C)')

    axes[2].set_xlabel('$t - t_0$ (s)')
    axes[2].set_ylabel('$\Delta T$ (°C)')
    # axes[2].set_yscale('log')

    axes[2].text(
        0.5, 0.05, fit_txt,
        transform=axes[2].transAxes,
        ha='left', va='bottom'
    )

    axes[2].legend(
        loc='center right', frameon=True
    )

    for i in range(2):
        axes[i].set_xlim(0, 30)

    axes[2].set_xlim(-0.2, 5)
    axes[2].xaxis.set_major_locator(ticker.MultipleLocator(1.))
    axes[2].xaxis.set_minor_locator(ticker.MultipleLocator(0.2))

    base_name = os.path.splitext(os.path.basename(data_csv))[0]
    axes[0].set_title(base_name, fontsize=12)
    fig.savefig(f'figures/thermocouple_response_{base_name}.png', dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
