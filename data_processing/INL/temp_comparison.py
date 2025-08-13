import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np
import os
import json
import platform

data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\manuscripts\paper1\inl\data_and_script_for_figures\data_and_script_for_figures\thermal_test'
data_dir = 'gold'

def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

if __name__ == '__main__':
    if platform == 'Windows':
        data_path = r'\\?\\' + data_dir
    load_plt_style()
    TC1 = pd.read_csv(os.path.join(data_path, data_dir, 'TC1.csv'))
    TC1_t = TC1['time'].values
    TC1_T = TC1['TC1'].values
    TC1_res = np.ones_like(TC1_T) * 0.25
    TC1_T_err = np.maximum(TC1_res, TC1_T*0.01)
    TC2 = pd.read_csv(os.path.join(data_path, data_dir, 'TC2.csv'))
    TC2_t = TC2['time'].values
    TC2_T = TC2['TC2'].values
    TC2_res = np.ones_like(TC2_T) * 0.25
    TC2_T_err = np.maximum(TC2_res, TC2_T * 0.01)


    sim3d = pd.read_csv(os.path.join(data_path, 'cylinder_out.csv'))
    sim3d = sim3d.drop(sim3d.index[0])

    sim3d_t = sim3d['time'].values
    sim3d_TC1 = sim3d['TC1'].values - 273.15
    sim3d_TC2 = sim3d['TC2'].values - 273.15

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)


    ax.errorbar(
        TC1_t, TC1_T, yerr=TC1_T_err, ls='none', marker='o', c='r',  ms=8, mfc='None', mew=1.25, capsize=2.5,
        label='TC1 measured'
    )
    ax.errorbar(
        TC2_t, TC2_T, yerr=TC2_T_err, ls='none', marker='s', c='b', ms=8, mfc='None', mew=1.25, capsize=2.5,
        label='TC2 measured'
    )
    ax.plot(sim3d_t, sim3d_TC1, 'r-',  lw=1.5, label='Simulation-TC1')
    ax.plot(sim3d_t, sim3d_TC2, 'b-',  lw=1.5, label='Simulation-TC2')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlim(0,120)
    ax.set_ylim(15,75)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5.0))

    ax.legend(
        loc='best', prop={'size': 9}, ncol=1
    )
    fig.savefig(os.path.join(data_path, 'temp_cmp_curves.svg'), dpi=600, format='svg')
    fig.savefig(os.path.join(data_path, 'temp_cmp_curves.eps'), dpi=600, format='eps')
    fig.savefig(os.path.join(data_path, 'temp_cmp_curves.png'), dpi=600)
    plt.show()
