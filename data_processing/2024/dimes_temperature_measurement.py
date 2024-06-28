import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
from data_processing.utils import get_experiment_params, latex_float_with_error, specific_heat_of_graphite
from scipy.stats.distributions import t
from scipy.interpolate import interp1d
import json
from scipy.signal import savgol_filter
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf

data_dir = 'data/DiMES_ATJ_GRAPHITE'
input_files = [
    'LCT_GRAPHITE_PIECE_100PCT_2024-06-26_4.csv',
    'LCT_GRAPHITE_PIECE_100PCT_2024-06-27_1.csv',
    'LCT_GRAPHITE_PIECE_100PCT_2024-06-27_2.csv',
    'LCT_GRAPHITE_PIECE_REALIGNED_100PCT_2024-06-27_1.csv',
    'LCT_GRAPHITE_PIECE_REALIGNED_100PCT_2024-06-27_2.csv'
]

input_files = [
    {'csv': 'LCT_GRAPHITE_PIECE_100PCT_2024-06-26_4.csv', 'label':'T1L1'},
    {'csv': 'LCT_GRAPHITE_PIECE_100PCT_2024-06-27_1.csv', 'label':'T2L2'},
    {'csv': 'LCT_GRAPHITE_PIECE_100PCT_2024-06-27_2.csv', 'label':'T3L2'},
    {'csv': 'LCT_GRAPHITE_PIECE_REALIGNED_100PCT_2024-06-27_1.csv', 'label':'T4L3'},
    {'csv': 'LCT_GRAPHITE_PIECE_REALIGNED_100PCT_2024-06-27_2.csv', 'label':'T4L3'},
]

tc_time_response_s = np.array([0.522, 0.454, 0.477]).mean()
cmap_name = 'rainbow'

thickness_mm = 25.5
diameter_mm = 47.70
density = 1.76
density_error = 0.0066 * density

w_h, w_x = 1.3698, 0.3172

cowan_corrections_df = pd.DataFrame(data={
    'Coefficient': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    '5 half times': [-0.1037162, 1.239040, -3.974433, 6.888738, -6.804883, 3.856663, -1.167799, 0.1465332],
    '10 half times': [0.054825246, 0.16697761, -0.28603437, 0.28356337, -0.13403286, 0.024077586, 0.0, 0.0]
})
flash_method_theory_csv = '../thermal_conductivity/flash_curve_theory.csv'


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def correct_thermocouple_response(measured_temperature, measured_time, tau):
    n = len(measured_time)
    k = int(n / 30)
    k = k + 1 if k % 2 == 0 else k
    k = max(k, 5)
    T = savgol_filter(measured_temperature, k, 2)
    # dTdt = np.gradient(T, measured_time, edge_order=2)
    delta = measured_time[1] - measured_time[0]
    dTdt = savgol_filter(x=T, window_length=k, polyorder=2, deriv=1, delta=delta)
    # dTdt = savgol_filter(dTdt, k - 2, 3)
    r = measured_temperature + tau * dTdt
    return savgol_filter(r, k, 2)


def main():
    global data_dir, input_files, cmap_name
    global density, density_error
    global thickness_mm
    global flash_method_theory_csv
    load_plot_style()

    thickness_cm = 0.1 * thickness_mm
    l2 = thickness_cm ** 2.

    theory_df = pd.read_csv(flash_method_theory_csv).apply(pd.to_numeric)
    omega = theory_df['w'].values
    v_theory = theory_df['V'].values
    t_by_th_theory = omega / w_h

    cowan_coefficients_5 = cowan_corrections_df['5 half times'].values
    cowan_coefficients_10 = cowan_corrections_df['10 half times'].values

    cmap = mpl.colormaps.get_cmap(cmap_name)
    n_files = len(input_files)
    # Organize files by initial temperature
    map_files = {}
    for i, item in enumerate(input_files):
        fn = item['csv']
        path_to_csv = os.path.abspath(os.path.join(data_dir, fn))
        data_df: pd.DataFrame = pd.read_csv(path_to_csv, comment='#').apply(pd.to_numeric)
        time_s = data_df['Measurement Time (s)'].values
        temperature_raw = data_df['TC2 (C)'].values + 273.15
        laser_power = data_df['Laser output peak power (W)'].values
        msk_power = laser_power > 0.
        time_pulse = time_s[msk_power]
        t0 = time_pulse[0]
        time_s -= t0
        msk_t0 = time_s >= 0.
        time_s = time_s[msk_t0]
        temperature_raw = temperature_raw[msk_t0]
        map_files[int(temperature_raw[0])] = {'csv': fn, 'label':item['label']}
    keys = list(map_files.keys())
    keys.sort()

    sorted_list = []
    for k in keys:
        item = map_files[k]
        sorted_list.append({'csv': item['csv'], 'label':item['label']})

    norm = mpl.colors.Normalize(vmin=0, vmax=(n_files - 1))
    colors = [cmap(norm(i)) for i in range(n_files)]
    fig, axes = plt.subplots(nrows=4, ncols=1, constrained_layout=True, height_ratios=[1., 0.4, 1, 1])
    fig.set_size_inches(4.5, 7.5)

    axes[3].plot(t_by_th_theory, v_theory, color='k', label='Model')

    for i, item in enumerate(sorted_list):
        fn = item['csv']
        lbl = item['label']
        base_name = os.path.splitext(fn)[0]
        path_to_csv = os.path.abspath(os.path.join(data_dir, fn))
        data_df: pd.DataFrame = pd.read_csv(path_to_csv, comment='#').apply(pd.to_numeric)
        params = get_experiment_params(relative_path=data_dir, filename=base_name)
        emission_time = float(params['Emission Time']['value'])
        time_s = data_df['Measurement Time (s)'].values
        temperature_raw = data_df['TC2 (C)'].values + 273.15
        laser_power = data_df['Laser output peak power (W)'].values
        msk_power = laser_power > 0.
        time_pulse = time_s[msk_power]
        t0 = time_pulse[0]
        time_s -= t0
        msk_t0 = time_s >= 0.
        laser_power_pulse = laser_power[msk_power]
        laser_energy = laser_power_pulse.mean() * emission_time * 1E-3
        temperature = correct_thermocouple_response(
            measured_temperature=temperature_raw, measured_time=time_s, tau=tc_time_response_s
        )
        cp = specific_heat_of_graphite(temperature_raw[0], units='K')
        cp_err = cp * 0.05
        # temperature = temperature_raw
        n_test = i + 1
        time_s = time_s[msk_t0]
        temperature = temperature[msk_t0]
        temperature_raw = temperature_raw[msk_t0]

        if temperature_raw[0] > 370.:
            axes[0].plot(time_s, temperature, color=colors[i], ls='-', lw=1.5, label=f'{lbl}')
            axes[0].plot(time_s, temperature_raw, color=colors[i], ls='--', lw=1.25)
        else:
            axes[1].plot(time_s, temperature, color=colors[i], ls='-', lw=1.5, label=f'{lbl}')
            axes[1].plot(time_s, temperature_raw, color=colors[i], ls='--', lw=1.25)

        axes[2].plot(time_s, temperature-temperature[0], color=colors[i], ls='-', lw=1.5,
                     label=f'{lbl}')


        # Estimate thermal diffusivity
        temperature_max = temperature.max()
        idx_peak = np.argmin(np.abs(temperature - temperature_max))
        time_red = time_s[0:idx_peak]
        temperature_red = temperature[0:idx_peak]
        dT = temperature - temperature.min()
        dT_max = dT.max()
        v = dT / dT_max
        f_inv = interp1d(x=v, y=time_s, bounds_error=False, fill_value='extrapolate')
        f_inv_red = interp1d(x=v[0:idx_peak], y=time_red, fill_value='extrapolate')
        # t_h = f_inv_red(0.5)
        idx_h = np.argmin(np.abs(v[0:idx_peak] - 0.5))
        t_h = time_s[idx_h]
        t_by_th = time_s / t_h
        f_v = interp1d(x=t_by_th, y=v)

        alpha_05 = 0.138785 * l2 / t_h
        # Radiative losses Cowan
        # print(f'dt5: {dt5:.3f}')
        kc = cowan_coefficients_5[0]
        dt5 = f_v(5.) / 0.5
        xx = dt5
        for j in range(1, 8):
            # print(f'kc = {kc:.5f}, coeff_{j+1}: {cowan_coefficients[j]:.5f}, xx:{xx}')
            kc += xx * cowan_coefficients_5[j]
            xx *= dt5

        try:
            dt10 = f_v(10.) / 0.5
            kc = cowan_coefficients_10[0]
            xx = dt5
            for j in range(1, 8):
                # print(f'kc = {kc:.5f}, coeff_{j+1}: {cowan_coefficients[j]:.5f}, xx:{xx}')
                kc += xx * cowan_coefficients_10[j]
                xx *= dt10
        except Exception as err:
            print(err)

        alpha_c = alpha_05 * kc / 0.13885
        kappa_c = alpha_c * cp * density * 100.
        kappa_c_err = kappa_c * np.linalg.norm([density_error / density, cp_err / cp])

        axes[3].plot(
            t_by_th, v, ls='-', color=colors[i],
            label=fr'$\kappa = {kappa_c:.0f} \pm {kappa_c_err:.0f}$ (W/m-K)'
        )



    axes[1].set_xlabel('$t$ (s)')
    axes[0].set_ylabel('$T$ (K)')
    axes[1].set_xlim(0, 60)
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(10.))
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(2.))

    axes[0].set_ylim(375, 400)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(5.))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(1.))

    axes[1].set_ylim(300, 310)
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(5.))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(1.))

    # broken axis
    axes[0].xaxis.tick_top()
    axes[0].spines.bottom.set_visible(False)
    axes[1].spines.top.set_visible(False)
    axes[0].tick_params(labeltop=False)
    axes[1].xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axes[0].plot([0, 1], [0, 0], transform=axes[0].transAxes, **kwargs)
    axes[1].plot([0, 1], [1, 1], transform=axes[1].transAxes, **kwargs)

    axes[2].set_xlabel('$t$ (s)')
    axes[2].set_ylabel('$\Delta T$ (K)')
    axes[2].set_xlim(0, 60)
    axes[2].xaxis.set_major_locator(ticker.MultipleLocator(10.))
    axes[2].xaxis.set_minor_locator(ticker.MultipleLocator(2.))

    axes[2].set_ylim(0, 10)
    axes[2].yaxis.set_major_locator(ticker.MultipleLocator(2.))
    axes[2].yaxis.set_minor_locator(ticker.MultipleLocator(1.))

    axes[3].set_xlabel('$t/t_{\mathrm{h}}$ (s)')
    axes[3].set_ylabel('$\Delta T / T_{\mathrm{max}}$ (K)')
    axes[3].set_xlim(0, 10)
    axes[3].xaxis.set_major_locator(ticker.MultipleLocator(1.))
    axes[3].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    axes[3].set_ylim(0, 1.1)
    axes[3].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axes[3].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    axes[0].legend(loc='upper right', frameon=True, fontsize=9, ncols=2)
    axes[1].legend(loc='lower right', frameon=True, fontsize=9, ncols=1)
    # axes[2].legend(loc='lower center', frameon=True, fontsize=9, ncols=2)
    axes[3].legend(loc='lower right', frameon=True, fontsize=9, ncols=2)
    fig.savefig('figures/dimes_graphite_block_temperature.pdf', dpi=600)
    fig.savefig('figures/dimes_graphite_block_temperature.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
