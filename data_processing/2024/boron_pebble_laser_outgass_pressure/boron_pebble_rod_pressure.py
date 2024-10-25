import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
from data_processing.utils import get_experiment_params, latex_float_with_error, specific_heat_of_graphite
from scipy.interpolate import interp1d
import json

data_dir = './data'

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

def main():
    global data_dir

    load_plot_style()
    files = [fn for fn in os.listdir(data_dir) if fn.endswith('.csv')]
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True, sharex=True)
    fig.set_size_inches(4., 4.)

    axes[1].set_xlabel('Time (s)')
    axes[0].set_ylabel('Pressure (mTorr)')
    axes[1].set_ylabel('Power (W)')
    cmap_name = 'cool'
    cmap = mpl.colormaps.get_cmap(cmap_name)
    n_files = len(files)
    norm = mpl.colors.Normalize(vmin=0, vmax=(n_files-1))
    colors = [cmap(norm(i)) for i in range(n_files)]

    for i, fn in enumerate(files):
        df = pd.read_csv(os.path.join(data_dir, fn), comment='#').apply(pd.to_numeric)
        laser_power = df['Laser output peak power (W)'].values
        msk_on = laser_power > laser_power.mean() * 0.1
        time_s = df['Measurement Time (s)'].values
        pressure_torr = df['Pressure (Torr)'].values

        time_msk = time_s[msk_on]
        power_on = laser_power[msk_on]
        mean_power = np.mean(power_on)
        t0 = time_msk[0]
        time_s -= t0

        lbl = fr'{mean_power:.0f} W'

        axes[0].plot(time_s, pressure_torr*1E3, c=colors[i], label=lbl, lw=1.25)
        axes[1].plot(time_s, laser_power, c=colors[i], label=lbl, lw=1.25)

    axes[0].legend(
        loc='upper right', frameon=True, fontsize=10
    )
    fig.savefig('boron_laser_test_pressure.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()

