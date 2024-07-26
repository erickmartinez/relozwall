import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
from data_processing.utils import get_experiment_params, specific_heat_of_graphite
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


emission_time_s = 0.5
density = 1.76  # g/cm3
thickness_mm = 25.5
diameter_mm = 47.70
hp = 10.0
P = 26.3
rh = diameter_mm * 0.5

w_h, w_x = 1.3698, 0.3172

# cmap_name = 'jet'

cowan_corrections_df = pd.DataFrame(data={
    'Coefficient': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    '5 half times': [-0.1037162, 1.239040, -3.974433, 6.888738, -6.804883, 3.856663, -1.167799, 0.1465332],
    '10 half times': [0.054825246, 0.16697761, -0.28603437, 0.28356337, -0.13403286, 0.024077586, 0.0, 0.0]
})


beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707
# time response
tc_time_response_s = np.array([0.522, 0.454, 0.477]).mean()


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)

def main():
    global data_dir, output_data_dir
    global tc_time_constant
    global cowan_corrections_df, density, thickness_mm, rh
    load_plot_style()


    dt = 0.05
    n_points = int(100. / dt) + 1
    time_interp = dt * np.arange(0, n_points)

    data_dir = os.path.normpath(data_dir)
    list_files = [f for f in os.listdir(path=data_dir) if f.endswith('.csv')]
    n_files = len(list_files)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    """
    Load theoretical curve
    """
    path_to_theory = os.path.abspath('../../../thermal_conductivity')
    kval_df = pd.read_csv(os.path.join(path_to_theory, 'dimensionless_parameters.csv')).apply(pd.to_numeric)
    theory_df = pd.read_csv(os.path.join(path_to_theory, 'flash_curve_theory.csv')).apply(pd.to_numeric)

    kval_df = kval_df[kval_df['V (%)'].isin([25., 50., 75.])]
    kval_v = kval_df['V (%)'].values
    kval_k = kval_df['k(V)'].values

    omega = theory_df['w'].values
    v_theory = theory_df['V'].values
    t_by_th_theory = omega / w_h

    cowan_coefficients_5 = cowan_corrections_df['5 half times'].values
    cowan_coefficients_10 = cowan_corrections_df['10 half times'].values

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True)
    # fig.subplots_adjust(hspace=0)
    fig.set_size_inches(4., 6.)

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

        cp = specific_heat_of_graphite(temperature_raw[0])
        cp_err = cp * 0.05
        # temperature = temperature_raw
        temperature_max = temperature.max()
        idx_peak = np.argmin(np.abs(temperature - temperature_max))
        time_red = time_s[0:idx_peak]
        temperature_red = temperature[0:idx_peak]

        dT = temperature - temperature.min()
        dT_max = dT.max()
        v = dT / dT_max
        f_inv = interp1d(x=v, y=time_s, bounds_error=False, fill_value='extrapolate')
        f_inv_red = interp1d(x=v[0:idx_peak], y=time_red, fill_value='extrapolate')
        t_h = f_inv_red(0.5)
        # idx_h = np.argmin(np.abs(v[0:idx_peak] - 0.5))
        # t_h = time_s[idx_h]
        # print(f't_h = {t_h:.3f} s')
        t_by_th = time_s / t_h
        f_v = interp1d(x=t_by_th, y=v)

        t_flash = t_by_th_theory * t_h
        dtemp_max = np.max(temperature_max - temperature_raw[0])
        dT_flash = dtemp_max * v_theory


        f1 = interp1d(x=time_s, y=temperature_raw, bounds_error=False, fill_value='extrapolate')
        f2 = interp1d(x=time_s, y=temperature_raw-temperature_raw[0], bounds_error=False, fill_value='extrapolate')
        id = i + 1
        output_df[f'T{id} (C)'] = f1(time_interp)

        delta_df[f'dT{id} (C)'] = f2(time_interp)

        axes[0].plot(
            time_s, temperature_raw, color=colors[i], label=fr'Adjustment #{id}'
        )

        if i == 0:
            axes[1].plot(
                t_flash, dT_flash, color='k', label='Parker theory'
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
        ax.set_xlim(0, 30)
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
    fig.savefig('./figures/dimes_temperatures.png', dpi=600)
    print(f'Total error: {total_error:.3f} °C')

    plt.show()


if __name__ == '__main__':
    main()
