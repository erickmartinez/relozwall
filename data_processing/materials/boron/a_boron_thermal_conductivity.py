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

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\Literature\boron'
cp_csv = 'Medwick AIP1991 - Thermal conductivity of amorphous boron.csv'



def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


def load_data():
    df = pd.read_csv(os.path.join(base_dir, cp_csv), comment='#').apply(pd.to_numeric)
    return df


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
    jc = np.ones((m,n), dtype=float)
    xx = x.copy()
    for i in range(1, n):
        jc[:, i] = xx
        xx *= x
    return jc

def main():
    df = load_data()
    load_plot_style()
    temperature_k = df['Temperature (K)'].values
    alpha = df['Thermal conductivity (W/cm-K)'].values

    msk_fit = temperature_k > 21.
    temperature_fit = temperature_k[msk_fit]
    alpha_fit = alpha[msk_fit]

    # fit the values to poly
    b0 = np.array([0.01, 0.1])
    all_tol = np.finfo(np.float64).eps
    n = len(alpha)
    log_t, log_a = np.log10(temperature_fit), np.log10(alpha_fit)
    res = least_squares(
        fobj,
        b0,
        loss='soft_l1', f_scale=0.1,
        jac=jac,
        args=(log_t, log_a),
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
    xpred = np.linspace(log_t.min(), log_t.max(), 500)
    ypred, lpb, upb = cf.predint(x=xpred, xd=temperature_fit, yd=alpha_fit, func=poly, res=res)
    x_extra = np.linspace(log_t.max(), np.log10(1000.), 200)
    y_extra = poly(x_extra, res.x)

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(3.5, 4.0)
    ax.fill_between(10**xpred, 10**lpb, 10**upb, color=lighten_color('C0', 0.5), label='Prediction band')
    ax.plot(temperature_k, alpha, color='C0', ls='none',  marker='o', mfc='none', label='Experiment')
    ax.plot(10**xpred, 10**ypred, color=lighten_color('C0', 1.25), label='Model')
    ax.plot(10 ** x_extra, 10 ** y_extra, ls=':', color='tab:red')

    t300 = 300.
    a_300 = 10**poly(np.log10(np.array([t300])), res.x)
    a_300 = a_300[0]

    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('$\\alpha$ [W/cm-K]')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(0.1, 1000.)
    ax.set_ylim(1E-6, 10)

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))


    ax.legend(
        loc='upper left',
        frameon=True
    )
    ax.set_title('Medwick 1991')



    print(f'a(T=300K): {a_300:.2E} (W/cm-K)')

    fig.savefig(os.path.join(base_dir, 'Medwick1991_a-boron_thermal_conductivity.png'), dpi=300)

    plt.show()

if __name__ == '__main__':
    main()