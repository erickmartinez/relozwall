import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
import os
from scipy.interpolate import interp1d
from data_processing.utils import latex_float
from scipy.optimize import least_squares
import data_processing.confidence as cf
from data_processing.utils import lighten_color

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\thermal_expansion'
target_temperature_k = 300.0  # K
data_csv = 'White - Cryogenics 1976 - Thermal Expansion at Low Temperatures of Glass-Ceramics and Glasses.csv'
lbl = 'White 1976'


def poly(x, b):
    n = len(b)
    r = 0.
    xx = np.ones_like(x)
    for i in range(n):
        r += xx * b[i]
        xx = xx * x
    return r

def model1(x, b):
    n = len(b)
    r = 0.
    xx = np.ones_like(x)
    for i in range(n):
        r += xx * b[i]
        xx = xx * np.log(x)
    return r

def fun(b, x, y):
    return poly(x, b) - y

def fun2(b, x, y):
    return model1(x, b) - y

def jac_model1(b, x, y):
    n = len(b)
    j = np.ones((len(x), n), dtype=np.float64)
    xx = np.log(x.copy())
    for i in range(1, n):
        j[:, i] = xx
        xx = xx * xx
    return j

def jac_poly(b, x, y):
    n = len(b)
    j = np.ones((len(x), n), dtype=np.float64)
    xx = x.copy()
    for i in range(1, n):
        j[:, i] = xx
        xx *= x
    return j




def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['xtick.top'] = True


def main():
    load_plt_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 5.0)
    all_tol = np.finfo(np.float64).eps


    df = pd.read_csv(os.path.join(base_dir, data_csv), comment='#').apply(pd.to_numeric)
    df.sort_values(by=['Temperature (K)'], inplace=True)
    df['alpha (1E-8 K^-1)'] = df['alpha (1E-8 K^-1)'] - df['alpha (1E-8 K^-1)'].min()
    temperature = df['Temperature (K)'].values
    alpha = df['alpha (1E-8 K^-1)'].values
    n = len(df)
    print(df)

    msk_fit = temperature >= 27.0
    t_fit = temperature[msk_fit]
    a_fit = alpha[msk_fit]

    b0 = np.array([0., 0.1, 0.1])
    res = least_squares(
        fun2,
        b0,
        # loss='soft_l1', f_scale=0.1,
        jac=jac_model1,
        args=(t_fit, a_fit),
        # bounds=([0., -np.inf, -np.inf, -np.inf, 0.], [np.inf, np.inf, np.inf, np.inf, np.inf]),
        xtol=all_tol, # ** 0.5,
        ftol=all_tol, # ** 0.5,
        gtol=all_tol, # ** 0.5,
        max_nfev=10000 * n,
        x_scale='jac',
        verbose=2
    )

    popt = res.x
    pcov = cf.get_pcov(res)
    ci = cf.confint(n=n, pars=popt, pcov=pcov)
    tpred = np.linspace(temperature.min(), temperature.max(), 500)
    ypred, lpb, upb = cf.predint(x=tpred, xd=t_fit, yd=a_fit, func=model1, res=res)
    t_extra = np.linspace(temperature.max(), 980, 500)
    y_extra = model1(t_extra, popt)

    a_target = model1(target_temperature_k, popt) * 0.01
    txt_a = f"$\\alpha$ (extraplated, 300 K) = {a_target:.1f} ($\\mathregular{{\\times 10^{{-6}} }}$/K)"

    ax.fill_between(tpred, lpb, upb, color=lighten_color('C0', 0.5))

    ax.plot(
        temperature,
        alpha,
        label=lbl,
        marker='o',
        mfc='none',
        markevery=5,
        color='C0',
        ls='none'
    )
    ax.plot(
        tpred, ypred, label='Fit', color='b'
    )
    ax.plot(
        t_extra, y_extra, label='Extrapolation', ls='--', color='tab:red'
    )

    ax.text(
        0.05, 0.95, txt_a, transform=ax.transAxes, fontsize=9, fontweight='regular',
        va='top', ha='left', color='b'
    )


    ax.set_xlabel('Temperature (K)')
    ax.legend(
        loc='lower right', frameon=True, fontsize=9
    )
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3), useMathText=True)
    ax.set_xlim(2, 500)
    ax.set_xscale('log')
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))

    ax.set_ylabel('$\\alpha$ ($\\mathregular{10^{-8}}$/K)')

    ax.set_ylim(0, 200)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.set_title('Glassy carbon CTE')

    # fig.savefig(os.path.join(base_dir, 'glassy_carbon_coefficient_of_thermal_expansion.png'), dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
