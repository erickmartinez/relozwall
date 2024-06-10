import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.ticker as ticker
import os
from scipy.optimize import least_squares, OptimizeResult
import json
import data_processing.confidence as cf

experiment_ids = [
    568, 569, 570, 571, 572, 573, 574, 575, 580
]


def poly(x, b):
    m, n = len(x), len(b)
    r = np.zeros(len(x))
    xi = np.ones(m)
    for i in range(n):
        r += xi * b[i]
        xi *= x
    return r


def res_poly(b, x, y):
    return poly(x, b) - y


def jac_poly(b, x, y):
    m, n = len(x), len(b)
    jac = np.ones((m, n))
    xi = x.copy()
    for i in range(1, n):
        jac[:, i] = xi
        xi *= x
    return jac


def mean_err(x):
    return np.linalg.norm(x) / len(x)


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def main():
    global experiment_ids
    outgassing_df: pd.DataFrame = pd.read_csv(
        'data/outgassing_rates_carbon_20240606.csv'
    )
    columns = outgassing_df.columns
    num_columns = list(set(columns) - set(['Sample ID', 'Filler']))
    outgassing_df[num_columns] = outgassing_df[num_columns].apply(pd.to_numeric)
    outgassing_df = outgassing_df[outgassing_df['Laser test ID'].isin(experiment_ids)]
    outgassing_df.reset_index(inplace=True, drop=True)

    outgassing_df['HL'] = np.round(outgassing_df['Heat load (MW/m^2)'].values / 5) * 5
    plot_df = outgassing_df.groupby(by=['HL']).agg({
        'Outgassing rate (Torr-L/s/m^2)': ['mean'],
        'Outgassing rate error (Torr-L/s/m^2)': [mean_err],
        'Recession rate (cm/s)': ['mean'],
        'Recession rate error (cm/s)': [mean_err]
    })
    # print(outgassing_df[['Heat load (MW/m^2)', 'HL']])
    # print(plot_df)
    load_plot_style()
    fig = plt.figure(constrained_layout=False)
    spec = fig.add_gridspec(nrows=2, ncols=1, hspace=0, wspace=0)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0])
    axes = [ax1, ax2]
    # fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, layout='constrained')
    fig.subplots_adjust(hspace=0)
    fig.set_size_inches(4.0, 5.0)

    heat_load = np.array(list(plot_df.index))
    carbon_emission = plot_df['Outgassing rate (Torr-L/s/m^2)']['mean'].values * 1E-3


    all_tol = float(np.finfo(np.float64).eps)
    res = least_squares(
        x0=[0.0, 1.0],
        fun=res_poly,
        args=(heat_load, carbon_emission),
        jac=jac_poly,
        loss='soft_l1', f_scale=0.1,
        xtol=all_tol,  # ** 0.5,
        ftol=all_tol,  # ** 0.5,
        gtol=all_tol,  # ** 0.5,
        max_nfev=10000 * len(carbon_emission),
        x_scale='jac',
        verbose=2
    )

    x_pred = np.linspace(heat_load.min(), heat_load.max(), num=100)
    ypred, delta = cf.prediction_intervals(model=poly, x_pred=x_pred, ls_res=res, jac=jac_poly)
    # ypred = poly(x_pred, res.x)

    axes[0].errorbar(
        plot_df.index, plot_df['Outgassing rate (Torr-L/s/m^2)']['mean'] * 1E-3,
        yerr=plot_df['Outgassing rate error (Torr-L/s/m^2)']['mean_err'] * 1E-3,
        color='C0', marker='o',
        ms=9, mew=1.25, mfc='none', ls='none',
        capsize=2.75, elinewidth=1.25, lw=1.5,
    )

    axes[0].plot(
        x_pred, ypred, color='C0', lw=1.25, ls='--'
    )

    axes[1].errorbar(
        plot_df.index, plot_df['Recession rate (cm/s)']['mean'],
        yerr=plot_df['Recession rate error (cm/s)']['mean_err'],
        color='C1', marker='s',
        ms=9, mew=1.25, mfc='none', ls='none',
        capsize=2.75, elinewidth=1.25, lw=1.5,
    )

    ax1.tick_params(axis='x', labelbottom=False)
    ax2.set_xlim(10, 45)
    ax1.set_ylim(0, 125)
    ax2.set_ylim(0, 0.55)

    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    axes[0].set_ylabel(r'C ($\mathregular{\times}$10$^{\mathregular{3}}$ Torr-L/s/m$^{\mathregular{2}}$)')
    axes[1].set_ylabel(r'$\nu$ (cm/s)')
    axes[1].set_xlabel(r'$q$ (MW/m$^{\mathregular{2}}$)')
    fig.tight_layout()
    fig.savefig('figures/carbon_emission_157.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
