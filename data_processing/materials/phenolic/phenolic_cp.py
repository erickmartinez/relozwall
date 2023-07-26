import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from scipy.optimize import least_squares
import data_processing.confidence as cf
from data_processing.utils import lighten_color

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\Literature\phenolic'
csv_file = 'Harwood -  J. Polym. Sci. Polym. Phys. Ed. 1978 - Specific Heat.csv'


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


def load_data():
    return pd.read_csv(os.path.join(base_dir, csv_file)).apply(pd.to_numeric)


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
    df = load_data()
    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)

    temperature_k = df['T (K)'].values
    cp = df['cp (J/g/K)'].values

    # fit the values to poly
    b0 = np.array([1., -1E-3, 0.01])
    all_tol = np.finfo(np.float64).eps
    n = len(cp)

    markers = ['o', 's', '^', 'v', 'd', '>', '<']
    colors = ['C0', 'C1', 'C2', 'C3']

    res = least_squares(
        fobj,
        b0,
        loss='soft_l1', f_scale=0.1,
        jac=jac,
        args=(temperature_k, cp),
        # bounds=([0., 0., 0., 0.], [brightness_fit.max(), np.inf, np.inf, np.inf]),
        xtol=all_tol,  # ** 0.5,
        ftol=all_tol,  # ** 0.5,
        gtol=all_tol,  # ** 0.5,
        max_nfev=10000 * n,
        x_scale='jac',
        verbose=2
    )

    popt = res.x
    pcov = cf.get_pcov(res)
    ci = cf.confint(n=n, pars=popt, pcov=pcov)
    xpred = np.linspace(temperature_k.min(), temperature_k.max(), 500)
    ypred, lpb, upb = cf.predint(x=xpred, xd=temperature_k, yd=cp, func=poly, res=res)

    x_extra = np.linspace(temperature_k.max(), 350, 500)
    y_extra = poly(x_extra, res.x)

    ax.fill_between(xpred, lpb, upb, color=lighten_color(colors[0], 0.5), alpha=0.5)
    ax.plot(temperature_k, cp, color=colors[0], ls='none', marker=markers[0], mfc='none', label=f'Experiment')
    ax.plot(xpred, ypred, color=lighten_color(colors[0], 1.25))
    ax.plot(x_extra, y_extra, color=lighten_color(colors[0], 1.25), ls='--', lw=1.25)

    t300 = 300.
    cp_300 = poly(np.array([t300]), res.x)
    cp_300 = cp_300[0]

    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('$\\alpha$ [x10$^{\mathregular{-6}}$ /K]')

    # ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(6, 350)
    ax.set_ylim(1E-3, 10.)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax.legend(
        loc='upper left',
        frameon=True
    )
    ax.set_title('Harwood 1978')

    print(f'CTE(T=300K), {cp_300:.3f}:.3f (10^{{-6}}/K)')

    cte_txt = f'$\\alpha$(T=300K), {cp_300.mean():.1f} x10$^{{\\mathregular{{-6}}}}$/K)'

    ax.text(
        0.95, 0.05, cte_txt, ha='right', va='bottom',
        transform=ax.transAxes, fontsize=11, fontweight='regular', color='b'
    )

    fig.savefig(os.path.join(base_dir, 'Harwood-phenolic_specific_heat.png'), dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
