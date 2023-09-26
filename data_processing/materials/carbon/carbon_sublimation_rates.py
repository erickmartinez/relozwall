import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import matplotlib.ticker as ticker
from scipy.optimize import least_squares, OptimizeResult
from data_processing.utils import latex_float, lighten_color
import data_processing.confidence as cf

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\manuscripts\paper2\free_pebble_power_balance'
rates_csv = 'sublimation_rate_arrhenius.csv'

kcalpermol2eV = 0.043364104241800934
kb_ev = 8.617333262E-5
by_na = 1. / 6.02214076E23
all_tol = np.finfo(np.float64).eps

deposition_data = {
    'temp_k': 2800,
    'rate [C/s/nm^2]': 1.02E6
}

free_pebble_data = {
    'temp_k': 2800,
    'rate [C/s/nm^2]': 78854.03
}

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


def tbyT2K(x):
    return 1000. / x

def K2tbyT(x):
    return 1000. / x

def poly(x, b):
    xx = np.ones_like(x)
    y = np.zeros_like(x)
    n = len(b)
    for i in range(n):
        y += b[i] * xx
        xx *= x
    return y


def fobj(b, x, y):
    return poly(x, b) - y


def jac(b, x, y):
    m, n = len(x), len(b)
    jc = np.ones((m, n), dtype=float)
    xx = x.copy()
    for i in range(1, n):
        jc[:, i] = xx
        xx *= x
    return jc


def main():
    df = pd.read_csv(os.path.join(base_dir, rates_csv), comment='#').apply(pd.to_numeric)
    tbyT = df['1000/T (1/K)'].values
    rate = df['ln(C/s/nm^2)'].values

    b0 = [np.log(18.), -1E5]
    res = least_squares(
        fobj,
        b0,
        # loss='soft_l1', f_scale=0.1,
        jac=jac,
        args=(tbyT, rate),
        # bounds=([0., 0.], [np.inf, np.inf]),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        diff_step=all_tol,
        max_nfev=10000 * len(rate),
        method='trf',
        x_scale='jac',
        verbose=2
    )

    popt = res.x
    ci = cf.confidence_interval(res)
    xp = np.linspace(0.275, 0.575, 500)
    yp, lpb, upb = cf.predint(x=xp, xd=tbyT, yd=rate, func=poly, res=res)

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 5.)

    sec_xax = ax.secondary_xaxis('top', functions=(tbyT2K, K2tbyT))
    sec_xax.set_xlabel('T [K]')

    ax.plot(
        tbyT, rate, marker='o', fillstyle='none', ls='none', label='Long et al.\nand references therein', zorder=2,
        color='C0'
    )
    ax.fill_between(xp, lpb, upb, color=lighten_color('C0', 0.25))
    ax.plot(xp, yp, color='k', label='fit')

    ax.plot(
        [1000. / deposition_data['temp_k']], [np.log(deposition_data['rate [C/s/nm^2]'])], marker='^', ls='none',
        color='C1', label='Carbon deposition'
    )

    ax.plot(
        [1000. / free_pebble_data['temp_k']], [np.log(free_pebble_data['rate [C/s/nm^2]'])], marker='v', ls='none',
        color='C2', label='Ejected pebble'
    )

    for i in range(len(popt)):
        print(f'popt[{i}] = {popt[i]:.3f}, 95% CI: [{ci[i,0]:.3f},{ci[i,1]:.3f}]')
    activation_energy = -popt[1] * kb_ev * 1000.
    activation_energy_ci = -ci[1, :] * kb_ev * 1000.
    activation_energy_err = np.max(np.abs(activation_energy_ci - activation_energy))
    r0 = np.exp(popt[0])
    r0_ci = np.exp(ci[0, :])
    r0_err = np.max(r0_ci - r0)

    num1_arr = [float(i) for i in f'{r0:.3E}'.split('E')]
    num2_arr = [float(i) for i in f'{r0_err:.3E}'.split('E')]

    num3_arr = [float(i) for i in f'{np.exp(ci[0, 0]):.3E}'.split('E')]
    num4_arr = [float(i) for i in f'{np.exp(ci[0, 1]):.3E}'.split('E')]

    pef = 10. ** (num1_arr[1] - num2_arr[1])
    pef2 = 10. ** (num3_arr[1] - num4_arr[1])
    r0_str = rf'({num1_arr[0]*pef:.2f} \pm {num2_arr[0]:.2f}) \times 10^{{{num2_arr[1]:.0f}}}'

    res_txt = "$E_{\mathrm{a}}$: " + f'{activation_energy:.1f}±{activation_energy_err:.1f} eV\n'
    res_txt += '$r_0$: ' + f'${latex_float(r0, significant_digits=3)}$' \
               + ' C/s/nm$^{\mathregular{2}}$\n'
    res_txt += f'95% CI: [{num3_arr[0]*pef2:.2f}, {num4_arr[0]:.2f}]$\\times 10^{{{num4_arr[1]:.0f}}}$'
    res_txt += ' C/s/nm$^{\mathregular{2}}$'
    # res_txt += '$r_0$: ' + f'${r0_str}$' + ' C/s/nm$^{\mathregular{2}}$'

    print(f'E_a: {activation_energy:.4f} eV')
    print(f'r_0: {r0:.4E} C/s/nm^2')
    ax.text(
        0.05, 0.05,
        res_txt,
        ha='left', va='bottom',
        transform=ax.transAxes,
        color='b'
    )

    T2800 = 1000. / 2800.
    lnrT2800 = poly(T2800, popt)
    ax.annotate(f"${latex_float(np.exp(lnrT2800), significant_digits=2)}$"+' C/s/nm$^{\mathregular{2}}$',
                xy=(T2800, lnrT2800), xycoords='data',
                xytext=(0.05, 0.25), textcoords='axes fraction',
                va='center', ha='left',
                arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=-90,angleB=180,rad=5"))

    ax.legend(
        loc='upper right',
        frameon=True,
        fontsize=10
    )

    ax.set_xlabel(r'1000/T [K$^{\mathregular{-1}}$]')
    ax.set_ylabel(r'ln(C/s/nm$^{\mathregular{2}}$)')

    ax.set_xlim(0.275, 0.575)
    ax.set_ylim(-17.5, 17.5)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.5))

    fig.savefig(os.path.join(base_dir, 'carbon_sublimation_rates.png'), dpi=600)
    plt.show()


if __name__ == '__main__':
    load_plot_style()
    main()
