import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from scipy.optimize import least_squares
import confidence as cf
from utils import lighten_color

data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\thermal_conductivity\graphite'
csv_file = 'entegris_thermal_diffusivity_zxf-5q.csv'


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


def model(temperature_c, a0, a1, a2):
    return a0 * np.exp(-a1 * temperature_c) + a2


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
    df = pd.read_csv(os.path.join(data_path, csv_file), comment='#').apply(pd.to_numeric)
    print(df)
    load_plot_style()
    temperature = df['Temperature (°C)'].values
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

    popt = res.x
    pcov = cf.get_pcov(res)
    ci = cf.confint(n=n, pars=popt, pcov=pcov)
    tpred = np.linspace(0, temperature.max(), 500)
    ypred, lpb, upb = cf.predint(x=tpred, xd=temperature, yd=alpha, func=func, res=res)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.plot(temperature, alpha, marker='o', fillstyle='none', ls='none', color='C0', label='Data')
    ax.plot(tpred, ypred, marker='none', color='C0', label='Best fit')
    ax.fill_between(tpred, lpb, upb, color=lighten_color('C0', 0.5))

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('$\\alpha$ (cm$^2$/s)')
    ax.set_xlim(10, 1800)
    ax.set_ylim(0, 1.2)

    fit_txt = '$\\alpha = a_0 e^{a_1 T} + a_2$\n'
    fit_txt += f'$a_0$ = {popt[0]:.3f} 95% CI: [{ci[0][0]:.3f}, {ci[0][1]:.3f}]\n'
    fit_txt += f'$a_1$ = {popt[1]:.3E} 95% CI: [{ci[1][0]:.3E}, {ci[1][1]:.3E}]\n'
    fit_txt += f'$a_2$ = {popt[2]:.3E} 95% CI: [{ci[2][0]:.3E}, {ci[2][1]:.3E}]'

    ax.text(
        0.95, 0.95, fit_txt,
        ha='right', va='top',
        transform=ax.transAxes,
        fontsize=9
    )

    plt.show()
