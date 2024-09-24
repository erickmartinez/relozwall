from string import digits

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json
from scipy.optimize import least_squares, OptimizeResult
from scipy.interpolate import interp1d
import data_processing.confidence as cf
from scipy.integrate import simps
from data_processing.utils import lighten_color, latex_float_with_error, latex_float


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


def res_bb(b, x, y):
    return model_bb(wavelength_nm=x, b=b) - y

def main():
    labsphere_df = pd.read_csv(
        './data/PALabsphere_2014.txt', sep=' ', comment='#',
        usecols=[0], names=['Radiance (W/cm2/ster/nm)']
    ).apply(pd.to_numeric)
    radiance = labsphere_df['Radiance (W/cm2/ster/nm)']
    n = len(radiance)
    wl = 350. + np.arange(n) * 10.
    print(f'wl.min: {wl.min():.0f}, wl.max(): {wl.max():.0f}')

    radiated_power = simps(y=radiance, x=wl)

    radiance_interp = interp1d(x=wl, y=radiance)
    

    all_tol = float(np.finfo(np.float64).eps)
    b0 = np.array([2900., 1E-5])
    ls_res = least_squares(
        res_bb,
        b0,
        # loss='cauchy', f_scale=0.001,
        # loss='soft_l1', f_scale=0.1,
        args=(wl, radiance),
        bounds=([0., 0.], [np.inf, np.inf]),
        xtol=all_tol,  # ** 0.5,
        ftol=all_tol,  # ** 0.5,
        gtol=all_tol,  # ** 0.5,
        diff_step=all_tol,
        max_nfev=10000 * n,
        # x_scale='jac',
        verbose=2
    )

    popt = ls_res.x
    ci = cf.confidence_interval(ls_res)
    popt_err = np.abs(ci[:, 1] - popt)
    print(f"I0: {popt[1]:.3E} ± {popt_err[1]:.3E}, 95% CI: [{ci[1, 0]:.5E}, {ci[1, 1]:.5E}]")
    nn = wl.max() - wl.min()
    x_pred = 350. + np.arange(nn)
    y_pred, delta = cf.prediction_intervals(
        model=model_bb, x_pred=x_pred, ls_res=ls_res
    )


    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3.)

    ax.plot(wl, radiance, label='Data')
    ax.plot(x_pred, y_pred, color='tab:red', label='Fit')
    ax.fill_between(x_pred, y_pred-delta, y_pred+delta, color='tab:red', alpha=0.25)

    ax.set_xlabel(r'$\lambda$ {\sffamily (nm)}', usetex=True)
    ax.set_ylabel(r'Radiance (W/cm$^{\mathregular{2}}$/ster)', usetex=False)
    # (W/cm^{2}/ster/nm
    ax.legend(loc='upper right')
    radiated_power_txt = f"$P = {latex_float(radiated_power, significant_digits=3)}~ \mathrm{{(W/cm^2/ster)}}$"
    ax.text(
        0.98, 0.65, radiated_power_txt,
        transform=ax.transAxes,
        ha='right', va='top', fontsize=11, color='tab:blue',usetex=True
    )

    results_txt = f'$T = {popt[0]:.0f}\pm{popt_err[0]:.0f}~\mathrm{{K}}$\n'
    results_txt += f'$I_{{\mathrm{{0}}}} = {latex_float_with_error(value=popt[1], error=popt_err[1], digits=0, digits_err=0)}$'

    ax.text(
        0.25, 0.05, results_txt, transform=ax.transAxes, va='bottom', ha='left', c='r',

    )
    ax.ticklabel_format(axis='y', useMathText=True)
    ax.set_title('PA Labsphere')
    ax.set_xlim(300, wl.max())
    ax.set_ylim(0, 1E-4)
    mf = ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(mf)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))


    fig.savefig('./figures/PA_labsphere.png', dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
