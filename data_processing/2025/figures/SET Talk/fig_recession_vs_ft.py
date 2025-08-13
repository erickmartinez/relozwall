import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import os
import json
import data_processing.confidence as cf
from scipy.optimize import least_squares, OptimizeResult


DATA_PATH = r'./data'

# A list of simulation files with its corresponding labels, and plot styles
SIMULATED_DB = [
    {'lbl': 'Simulation (F$_{\mathregular{b}}$ = 1.3 N)', 'file': 'Ben_Fig10_dt_sim_ft_60.csv', 'marker': 's', 'ft': 60, 'ls':'-'},
    {'lbl': 'Simulation (F$_{\mathregular{b}}$ = 1.5 N)', 'file': 'Ben_Fig10_dt_sim_ft_70.csv', 'marker': 's', 'ft': 70, 'ls':'-'},
    {'lbl': 'Simulation (F$_{\mathregular{b}}$ = 1.7 N)', 'file': 'Ben_Fig10_dt_sim_ft_80.csv', 'marker': 's', 'ft': 80, 'ls':'-'},
]
EXPERIMENTAL_CSV = 'recession_vs_heat_load_30KPa.csv'




def load_plot_style():
    with open('./plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def model_exp(x, b):
    return b[0] * np.exp(-x / b[1])


def residual_exp(b, x, y, w=1.):
    return (model_exp(x, b) - y) * w


def jac_exp(b, x, y, w=1.):
    m, n = len(x), len(b)
    r = np.ones((m, n))
    ee = w * np.exp(-x / b[1])
    r[:, 0] = ee
    r[:, 1] = b[0] * x * ee / (b[1] ** 2.)
    return r

def model(t: np.ndarray, beta: np.ndarray):
    return beta[0] - t * beta[1]


def residual(b, x, y, w=1):
    return (model(x, b) - y) * w


def jac(b, t, r, w=1):
    n, p = len(r), len(b)
    j = np.empty((n, p), dtype=np.float64)
    j[:, 0] = np.ones(n, dtype=np.float64) * w
    j[:, 1] = - t * w
    return j

def main(data_path, simulated_db, experimental_csv):
    load_plot_style()


    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.5)
    colors_sim = ['C0', 'C3', 'C2', 'C3', 'C4']

    ax.set_yscale('log')
    ax.set_ylim(1E-5, 1)
    ax.set_xlabel(r'Heat load (MW/m$^{\mathregular{2}}$)')
    ax.set_ylabel(r'Recession rate (cm/s)')

    ax.set_xlim(0, 55)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))


    for i, r in enumerate(simulated_db):
        csv = os.path.join(data_path, r['file'])
        lbl = r['lbl']
        marker = r['marker']
        ls = r['ls']
        sim_df = pd.read_csv(csv, comment='#').apply(pd.to_numeric)
        sim_df['Q'] = np.round(sim_df['Heat load (MW/m2)']/5.) * 5. # < round the heat load to multiples
        heat_load = sim_df['Heat load (MW/m2)'].values
        recession_rate = sim_df['Recession rate (cm/s)'].values
        ax.plot(
            heat_load, recession_rate, c=colors_sim[i], marker=marker, fillstyle='full',
            ls=ls, mew=2., label=lbl, ms=6, lw=1.25
        )

    experiment_df = pd.read_csv(os.path.join(data_path, experimental_csv)).apply(pd.to_numeric)
    ax.errorbar(
        experiment_df['Heat load (MW/m2)'],
        experiment_df['Recession rate (cm/s)'],
        yerr=(experiment_df['Recession rate (cm/s)']*0.5, experiment_df['Recession rate (cm/s)']*1.5),
        marker='o', color='C0',
        ms=9, mew=1.25, mfc='none', ls='none',
        capsize=2.75, elinewidth=1.25, lw=1.5,
        label=r'Experiment (F$_{\mathregular{b}}$ = 1.5 N)'
    )


    ax.legend(loc='lower right', frameon=True, fontsize=11)



    fig.savefig('fig_recession_heat_load_ft.png', dpi=600)
    fig.savefig('fig_recession_heat_load_ft.pdf', dpi=600)
    plt.show()


if __name__ == '__main__':
    main(
        data_path=DATA_PATH, simulated_db=SIMULATED_DB, experimental_csv=EXPERIMENTAL_CSV
    )
