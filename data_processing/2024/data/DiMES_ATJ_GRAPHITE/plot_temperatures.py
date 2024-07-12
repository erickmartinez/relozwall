import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
from data_processing.utils import get_experiment_params
from scipy.interpolate import interp1d
import json
from scipy.signal import savgol_filter
from scipy.stats.distributions import t

data_dir = './20240701/thermocouple_scatter'
output_data_dir = './output_data'

# thermocouple time constant
tc_time_constant = np.array([0.522, 0.454, 0.477]).mean()


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def mean_error(x):
    n = len(x)
    return np.linalg.norm(x) / np.sqrt(n)

def correct_thermocouple_response(measured_temperature, measured_time, tau):
    n = len(measured_time)
    k = int(n / 40)
    k = k + 1 if k % 2 == 0 else k
    k = max(k, 5)
    # T = savgol_filter(measured_temperature, k, 3)
    # dTdt = np.gradient(T, measured_time, edge_order=2)
    delta = measured_time[1] - measured_time[0]
    dTdt = savgol_filter(x=measured_temperature, window_length=k, polyorder=3, deriv=1, delta=delta)
    # dTdt = savgol_filter(dTdt, k - 2, 3)
    r = measured_temperature + tau * dTdt
    return savgol_filter(r, k, 3)


def main():
    global data_dir, output_data_dir
    global tc_time_constant
    load_plot_style()

    dt = 0.05
    n_points = int(100. / dt) + 1
    time_interp = dt * np.arange(0, n_points)

    data_dir = os.path.normpath(data_dir)
    list_files = [f for f in os.listdir(path=data_dir) if f.endswith('.csv')]
    n_files = len(list_files)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True)
    # fig.subplots_adjust(hspace=0)
    fig.set_size_inches(4., 5.)

    output_df = pd.DataFrame(data={
        'time (s)': time_interp
    })

    delta_df = pd.DataFrame(data={
        'time (s)': time_interp
    })

    for i, fn in enumerate(list_files):
        file_tag = os.path.splitext(fn)[0]
        params = get_experiment_params(relative_path=data_dir, filename=file_tag)
        data_df: pd.DataFrame = pd.read_csv(os.path.join(data_dir, fn), comment='#').apply(pd.to_numeric)
        time_s = data_df['Measurement Time (s)'].values
        laser_power = data_df['Laser output peak power (W)'].values
        msk_power = laser_power > 0.
        time_pulse = time_s[msk_power]
        t0 = time_pulse[0] - 0.005
        # idx_pulse = np.argmin(np.abs(time_s - t0))-1
        # t0 = time_s[idx_pulse]
        time_s -= t0
        temperature_raw = data_df['TC2 (C)'].values# + 273.15
        msk_t0 = time_s > 0.
        time_s = time_s[msk_t0]
        temperature_raw = temperature_raw[msk_t0]
        temperature = correct_thermocouple_response(
            measured_temperature=temperature_raw, measured_time=time_s, tau=tc_time_constant
        )

        f1 = interp1d(x=time_s, y=temperature_raw, bounds_error=False, fill_value='extrapolate')
        f2 = interp1d(x=time_s, y=temperature_raw-temperature_raw[0], bounds_error=False, fill_value='extrapolate')
        id = i + 1
        output_df[f'T_{id} (°C)'] = f1(time_interp)

        delta_df[f'dT_{id} (°C)'] = f2(time_interp)

        axes[0].plot(
            time_s, temperature_raw, color=colors[i], label=fr'Adjustment #{id}'
        )

        # axes[0].plot(
        #     time_s, temperature, ls='--', lw=1.25, color=colors[i], #label=fr'Adjustment #{id}'
        # )

        axes[1].plot(
            time_s, temperature_raw-temperature_raw[0], color=colors[i], label=fr'Adjustment #{id}'
        )

        # axes[1].plot(
        #     time_s, temperature - temperature[0], ls='--', lw=1.25, color=colors[i], label=fr'Adjustment #{id}'
        # )



    axes[1].set_xlabel(r'$t$ (s)')
    axes[0].set_ylabel(r'$T$ (°C)')
    axes[1].set_ylabel(r'$\Delta T$ (°C)')
    axes[0].legend(
        loc='lower right', frameon=True, fontsize=10
    )

    axes[0].set_ylim(20, 40.)
    axes[1].set_ylim(0, 10.)

    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(5))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(1))

    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(2))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    for ax in axes:
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20.))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5.))

    print(output_df)
    print(delta_df)
    output_df.to_csv(os.path.join(output_data_dir, 'raw_temperatures.csv'), index=False, encoding="utf-8-sig")
    columns = delta_df.columns
    descriptive_df = delta_df[columns[1::]].apply(pd.DataFrame.describe, axis=1)
    delta_df.to_csv(os.path.join(output_data_dir, 'del_temperature.csv'), index=False, encoding="utf-8-sig")
    print(descriptive_df)
    std = descriptive_df['std'].values
    tval = t.ppf(1. - 0.05*0.5, n_files - 1)
    se = std * tval / np.sqrt(n_files)

    total_error = np.linalg.norm([se]) / np.sqrt(len(se))
    print(f'Total error: {total_error:.3f} °C')

    plt.show()


if __name__ == '__main__':
    main()
