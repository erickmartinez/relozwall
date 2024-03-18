import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
import os
import tkinter as tk
from tkinter import filedialog
from data_processing.utils import get_experiment_params

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def main():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(title="Select csv file", filetypes=[("FS test", "*.csv")])
    rel_path = os.path.dirname(file_path)
    df = pd.read_csv(file_path, comment='#').apply(pd.to_numeric)
    df = df[df['Displacement (mm)'] > 0]
    df = df.sort_values(by=['Time (s)', 'Displacement (mm)'])
    filetag = os.path.splitext(os.path.basename(file_path))[0]
    params = get_experiment_params(relative_path=rel_path, filename=filetag)
    diameter = float(params['Sample diameter']['value'])
    test_id = params['Sample name']['value']
    support_span = float(params['Support span']['value'])
    force = df['Force (N)'].values
    time = df['Time (s)'].values

    # Find the breaking load
    fmax = force.max()
    idx_peak = (np.abs(fmax - force)).argmin()
    t_peak = time[idx_peak]
    msk_data = time <= t_peak
    df = df[df['Time (s)'] <= t_peak]

    force = df['Force (N)'].values
    time = df['Time (s)'].values
    displacement = df['Displacement (mm)'].values
    displacement_err = df['Displacement err (mm)'].values

    sigma_kpa = 8.0 * force * support_span / np.pi / (diameter ** 3.0) * 1E3
    sigma_kpa_err = np.zeros_like(sigma_kpa)
    for i, s in enumerate(sigma_kpa):
        sigma_kpa_err[i] = np.linalg.norm([0.5 / force[i], 0.002 / support_span, 3. * 0.6 / diameter])
    new_d = 9.18
    new_L = 40.00
    old_L = support_span
    old_d = diameter
    new_force = (old_L/new_L) * ((new_d / old_d) ** 3.) * force

    load_plot_style()

    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 6.)

    for ax in axes:
        # ax.set_xlim(0., 0.5)
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.02))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        ax.set_xlabel('Deformation (mm)')

    axes[0].errorbar(
        displacement, force, yerr=0.5,
        marker='o', mfc='none', capsize=2.0
    )

    axes[1].errorbar(
        displacement, new_force, yerr=0.5,
        marker='o', mfc='none', capsize=2.0
    )

    plt.show()




if __name__ == '__main__':
    main()