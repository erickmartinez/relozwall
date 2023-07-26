import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
from scipy.optimize import least_squares
import confidence as cf
from utils import lighten_color, specific_heat_of_graphite

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\thermal_conductivity\graphite'
csv_file = 'entegris_thermal_diffusivity_axf5q.csv'


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


def model(temperature_c, a0, a1, a2):
    return a0 * np.exp(-a1 * temperature_c) + a2


def model_bapat(temperature_k, b):
    """
    https://doi.org/10.1016/0008-6223(73)90073-0
    """
    return b[0] + 1.0 / (b[1] * temperature_k + b[2])


def fobj_bapat(b, temperature_k, alpha_exp):
    return model_bapat(temperature_k, b) - alpha_exp


def jac_bapat(b, temperature_k, alpha_exp):
    j0 = np.ones_like(temperature_k)
    r = -np.power(b[1] * temperature_k + b[2], -2.0)
    j1 = temperature_k * r
    j2 = r
    return np.stack([j0, j1, j2]).T


def poly(x, b):
    return b[0] + b[1] * x


def fobj_poly(b, temperature_k, alpha_exp):
    return poly(temperature_k, b) - alpha_exp


def jac_poly(b, temperature_k, alpha_exp):
    j0 = np.ones_like(temperature_k)
    j1 = temperature_k
    return np.stack([j0, j1]).T


def func(temperature_c, b):
    return model(temperature_c, b[0], b[1], b[2])


def fobj(b, temperature_c, alpha_exp):
    return model(temperature_c, b[0], b[1], b[2]) - alpha_exp


def jac(b, temperature_c, alpha_exp):
    j0 = np.exp(-b[1] * temperature_c)
    j1 = -temperature_c * b[0] * j0
    j2 = np.ones_like(temperature_c)
    return np.stack([j0, j1, j2]).T


if __name__ == '__main__':
    basename = os.path.splitext(csv_file)[0]
    df = pd.read_csv(os.path.join(base_path, csv_file), comment='#').apply(pd.to_numeric)
    print(df)
    load_plot_style()
    temperature = df['Temperature (°C)'].values
    temperature_k = temperature + 273.15
    alpha = df['Thermal diffusivity (cm^2/s)'].values

    b0 = [1.0, 0.1, 0.01]
    all_tol = np.finfo(np.float64).eps
    n = len(temperature)
    res = least_squares(
        fobj,
        b0,
        loss='soft_l1', f_scale=0.1,
        jac=jac,
        args=(temperature, alpha),
        bounds=([1E-5, 1E-5, 1E-5], [1E5, 1.0, 1.0]),
        xtol=all_tol ** 0.5,
        ftol=all_tol ** 0.5,
        gtol=all_tol ** 0.5,
        max_nfev=10000 * n,
        x_scale='jac',
        verbose=2
    )

    by_t = 1.0 / temperature_k
    thermal_conductivity = alpha * 1.7 * specific_heat_of_graphite(temperature, units='C')

    b0_poly = [0, 1.0]
    res_poly = least_squares(
        fobj_poly, b0_poly,
        loss='soft_l1', f_scale=0.1,
        jac=jac_poly,
        args=(by_t, thermal_conductivity),
        bounds=([0.0, 0.0], [np.inf, np.inf]),
        xtol=all_tol ** 0.5,
        ftol=all_tol ** 0.5,
        gtol=all_tol ** 0.5,
        max_nfev=10000 * n,
        x_scale='jac',
        verbose=2
    )

    res_bapat = least_squares(
        fobj_bapat,
        [alpha.min(), 0.1, 0.1],
        loss='soft_l1', f_scale=0.1,
        jac=jac_bapat,
        args=(temperature_k, thermal_conductivity),
        bounds=([1E-20, 1E-20, 1E-20], [np.inf, np.inf, np.inf]),
        xtol=all_tol ** 1.0,
        ftol=all_tol ** 1.0,
        gtol=all_tol ** 1.0,
        max_nfev=10000 * n,
        x_scale='jac',
        verbose=2
    )

    popt = res.x
    pcov = cf.get_pcov(res)
    ci = cf.confint(n=n, pars=popt, pcov=pcov)
    tpred = np.linspace(0, temperature.max(), 500)
    ypred, lpb, upb = cf.predint(x=tpred, xd=temperature, yd=alpha, func=func, res=res)

    popt_poly = res_poly.x
    pcov_poly = cf.get_pcov(res_poly)
    ci_poly = cf.confint(n=n, pars=popt_poly, pcov=pcov_poly)
    by_t_pred = np.linspace(by_t.min(), by_t.max(), 500)
    ypred_poly, lpb_poly, upb_poly = cf.predint(x=by_t_pred, xd=by_t, yd=thermal_conductivity, func=poly, res=res_poly)

    popt_bapat = res_bapat.x
    pcov_bapat = cf.get_pcov(res_bapat)
    ci_bapat = cf.confint(n=n, pars=popt_bapat, pcov=pcov_bapat)
    temperature_k_pred = np.linspace(temperature_k.min(), temperature_k.max(), 500)
    ypred_bapat, lpb_bapat, upb_bapat = cf.predint(x=temperature_k_pred, xd=temperature_k, yd=thermal_conductivity,
                                                   func=model_bapat, res=res_bapat)

    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    fig.set_size_inches(4.5, 5.5)
    ax[0].plot(temperature, alpha, marker='o', fillstyle='none', ls='none', color='C0', label='Data')
    ax[0].plot(tpred, ypred, marker='none', color='C0', label='Best fit')
    ax[0].fill_between(tpred, lpb, upb, color=lighten_color('C0', 0.5))

    ax[1].plot(temperature_k, thermal_conductivity, marker='s', fillstyle='none', ls='none', color='C0', label='Data')
    ax[1].plot(temperature_k_pred, ypred_bapat, marker='none', color='C0', label='Best fit')
    ax[1].fill_between(temperature_k_pred, lpb_bapat, upb_bapat, color=lighten_color('C0', 0.5))
    # ax[1].set_yscale('log')

    ax[0].set_xlabel('Temperature (°C)')
    ax[0].set_ylabel('$\\alpha$ (cm$^2$/s)')
    ax[0].set_xlim(10, 1800)
    ax[0].set_ylim(0, 1.2)

    ax[1].set_xlabel('Temperature (K)')
    ax[1].set_ylabel('$K$ (W/cm-K)')

    # xfmt = ticker.ScalarFormatter()
    # xfmt.useMathText = True
    # xfmt.set_powerlimits((-2,2))
    # ax[1].xaxis.set_major_formatter(xfmt)

    fit_txt = '$\\alpha = a_0 e^{a_1 T} + a_2$\n'
    fit_txt += f'$a_0$ = {popt[0]:.3f} 95% CI: [{ci[0][0]:.3f}, {ci[0][1]:.3f}]\n'
    fit_txt += f'$a_1$ = {popt[1]:.3E} 95% CI: [{ci[1][0]:.3E}, {ci[1][1]:.3E}]\n'
    fit_txt += f'$a_2$ = {popt[2]:.3E} 95% CI: [{ci[2][0]:.3E}, {ci[2][1]:.3E}]'

    fit_bapat_txt = '$K = a_0 + 1 / (a_0 T + a_2) $\n'
    fit_bapat_txt += f'$a_0$ = {popt_bapat[0]:.3f} 95% CI: [{ci_bapat[0][0]:.3f}, {ci_bapat[0][1]:.3f}]\n'
    fit_bapat_txt += f'$a_1$ = {popt_bapat[1]:.3E} 95% CI: [{ci_bapat[1][0]:.3E}, {ci_bapat[1][1]:.3E}]\n'
    fit_bapat_txt += f'$a_2$ = {popt_bapat[2]:.3E} 95% CI: [{ci_bapat[2][0]:.3E}, {ci_bapat[2][1]:.3E}]'

    ax[0].text(
        0.95, 0.95, fit_txt,
        ha='right', va='top',
        transform=ax[0].transAxes,
        fontsize=9
    )

    ax[1].text(
        0.95, 0.95, fit_bapat_txt,
        ha='right', va='top',
        transform=ax[1].transAxes,
        fontsize=9
    )

    plt.show()
