import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import pandas as pd
import re
import ir_thermography.thermometry as irt
import matplotlib.ticker as ticker
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy import interpolate
import platform


base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration\IR Thermography Calibration'
# data_file = 'LT_GR008G_4mTorr-contact-shield_080PCT_60GAIN 2022-05-06_1'
data_file = 'LT_GR008G_6mTorr-contact-shield_100PCT_50GAIN 2022-05-04_1'

time_constant_1, time_constant_2 = 2.1148, 2.1148
emissivity = 1.0 - (36.9 / 100)
tc_min = 0.0



def correct_thermocouple_response(measured_temperature, measured_time, time_constant, debug=False):
    n = len(measured_time)
    k = int(n / 10)
    k = k + 1 if k % 2 == 0 else k
    T = savgol_filter(measured_temperature, k, 2)
    dTdt = np.gradient(T, measured_time, edge_order=2)
    if debug:
        for ti, Ti, Ts, dti in zip(measured_time, measured_temperature, T, dTdt):
            print(f'{ti:6.3f} s, T = {Ti:6.2f} °C, T_smooth: {Ts:6.3f} dT/dt = {dti:>5.3E}')
    # dTdt = savgol_filter(dTdt, k, 3)
    r = T + time_constant * dTdt
    r = savgol_filter(r, k - 4, 3)
    return r


def get_experiment_params(relative_path: str, filename: str):
    # Read the experiment parameters
    results_csv = os.path.join(relative_path, f'{filename}.csv')
    count = 0
    params = {}
    with open(results_csv) as f:
        for line in f:
            if line.startswith('#'):
                if count > 1:
                    l = line.strip()
                    print(l)
                    if l == '#Data:':
                        break
                    pattern1 = re.compile("\s+(.*?):\s(.*?)\s(.*?)$")
                    pattern2 = re.compile("\s+(.*?):\s(.*?)$")
                    matches1 = pattern1.findall(l)
                    matches2 = pattern2.findall(l)
                    if len(matches1) > 0:
                        params[matches1[0][0]] = {
                            'value': matches1[0][1],
                            'units': matches1[0][2]
                        }
                    elif len(matches2) > 0:
                        params[matches2[0][0]] = {
                            'value': matches2[0][1],
                            'units': ''
                        }
                count += 1
    return params


if __name__ == '__main__':
    if platform.system() == 'Windows':
        base_path = r'\\?\\' + base_path
    results_path = os.path.join(base_path, 'temperature_data')
    data_filetag = data_file
    print('results_path: ', results_path)
    main_csv = os.path.join(base_path, data_filetag + '.csv')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    measurements_df = pd.read_csv(main_csv, comment='#').apply(pd.to_numeric)
    experiment_params = get_experiment_params(relative_path=base_path, filename=data_file)
    photodiode_gain = experiment_params['Photodiode Gain']['value']
    laser_power_setting = experiment_params['Laser Power Setpoint']['value']
    sample_name = experiment_params['Sample Name']['value']
    output_path = os.path.join(results_path, f'{sample_name.upper()}_{laser_power_setting}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    thermometry = irt.PDThermometer()

    # print(measurements_df)

    measurement_time = measurements_df['Measurement Time (s)'].values
    trigger_voltage = measurements_df['Trigger (V)'].values
    photodiode_voltage = measurements_df['Photodiode Voltage (V)'].values

    # for i, p in enumerate(photodiode_voltage):
    #     if np.isnan(p):
    #         print(i, measurement_time[i], p)

    tc_csv = os.path.join(base_path, data_filetag + '_tcdata.csv')
    tc_df = pd.read_csv(tc_csv, comment='#').apply(pd.to_numeric)
    tc_time = tc_df['Time (s)'].values
    temperature_a = tc_df['TC1 (C)'].values
    # temperature_b = tc_df['TC2 (C)'].values

    ta_corrected = correct_thermocouple_response(
        measured_temperature=temperature_a, measured_time=tc_time, time_constant=time_constant_1
    )


    t_max_idx = measurement_time <= 3.0
    tc_max_idx = tc_time <= np.inf

    measurement_time = measurement_time[t_max_idx]
    trigger_voltage = trigger_voltage[t_max_idx]
    photodiode_voltage = photodiode_voltage[t_max_idx]
    tc_time = tc_time[tc_max_idx]
    temperature_a = temperature_a[tc_max_idx]
    ta_corrected = ta_corrected[tc_max_idx]

    trigger_voltage = savgol_filter(trigger_voltage, 5, 4)

    irradiation_time_idx = trigger_voltage > 1.5
    irradiation_time = measurement_time[irradiation_time_idx]
    reflection_signal = np.zeros_like(photodiode_voltage)

    sg_window = 9
    photodiode_voltage[irradiation_time_idx] = savgol_filter(photodiode_voltage[irradiation_time_idx], sg_window, 2)


    t_pulse_max = irradiation_time.max() + 0.2
    noise_level = np.abs(photodiode_voltage[measurement_time > t_pulse_max]).max()
    print(f"Noise Level: {noise_level:.4f} V")

    t0 = irradiation_time.min()

    # t0_idx = (np.abs(measurement_time - t0)).argmin() - 1
    # t0 = measurement_time[t0_idx]
    irradiation_time -= t0
    measurement_time -= t0
    n = 3
    reflective_signal_zero_idx = (np.abs(measurement_time)).argmin() + n
    reflection_signal_max_idx = (np.abs(measurement_time - 0.5)).argmin()
    reflection_signal[irradiation_time_idx] = photodiode_voltage[reflective_signal_zero_idx]
    reflection_signal[reflective_signal_zero_idx - n] = 0.0
    reflection_signal[reflection_signal_max_idx:reflection_signal_max_idx] = photodiode_voltage[reflection_signal_max_idx:reflection_signal_max_idx]
    print(f"Baseline signal: {photodiode_voltage[reflective_signal_zero_idx]:.3f} V")
    thermometry.gain = int(photodiode_gain)
    print(f"Calibration Factor: {thermometry.calibration_factor}")
    thermometry.emissivity = emissivity

    photodiode_corrected = photodiode_voltage - reflection_signal
    pd_corrected_min = photodiode_corrected[irradiation_time_idx].min()
    photodiode_corrected[irradiation_time_idx] -= pd_corrected_min + 0.5 * noise_level
    time_pd_idx = (photodiode_corrected > 0.0) & (measurement_time > noise_level)
    measurement_time_pd = measurement_time[time_pd_idx]
    photodiode_voltage_positive = photodiode_corrected[time_pd_idx]
    measurement_time_pd = measurement_time_pd[n:-2]
    photodiode_voltage_positive = photodiode_voltage_positive[n:-2]

    temperature_pd = thermometry.get_temperature(voltage=photodiode_voltage_positive) - 273.15
    temperature_pd = savgol_filter(temperature_pd, 15, 2)

    print(f't0 = {t0:.3f} s')

    fig_pd, ax_pd = plt.subplots()
    fig_pd.set_size_inches(5.0, 3.5)

    color_pd = 'C0'
    color_tr = 'C5'

    ax_pd.plot(measurement_time, photodiode_voltage, color=color_pd, lw=1.75, label='Photodiode Raw')
    ax_pd.set_xlim(measurement_time.min(), measurement_time.max())
    ax_tr = ax_pd.twinx()
    ax_pd.set_zorder(1)
    ax_tr.set_zorder(0)
    ax_pd.patch.set_visible(False)
    ax_tr.plot(measurement_time, trigger_voltage, color=color_tr, lw=1.75)
    ax_pd.set_xlabel('Time (s)')
    ax_pd.set_ylabel(f'Voltage (V)', color=color_pd)
    ax_pd.tick_params(axis='y', labelcolor=color_pd)
    ax_tr.set_ylabel(f'Trigger Signal (V)', color=color_tr)
    ax_tr.tick_params(axis='y', labelcolor=color_tr)
    ax_pd.plot(measurement_time, reflection_signal, color='k', lw=1.25, ls='--', label='Reflection Baseline')
    ax_pd.plot(measurement_time, photodiode_corrected, color='C1', lw=1.25, label='Corrected')

    ax_pd.ticklabel_format(useMathText=True)
    ax_pd.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax_pd.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    ax_pd.ticklabel_format(useMathText=True)
    ax_tr.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    ax_pd.set_title(f"Photodiode Signal at {photodiode_gain} dB Gain,\n{laser_power_setting}% Laser Power")
    ax_pd.legend(
        loc='upper right', frameon=False
    )

    tr_ylim = ax_tr.get_ylim()

    ax_pd.set_ylim(bottom=tr_ylim[0], top=max(1.25 * photodiode_corrected.max(), 1.0))
    ax_tr.set_ylim(bottom=tr_ylim[0], top=1.1 * trigger_voltage.max())


    fig_pd.tight_layout()

    fig_t, ax_t = plt.subplots()
    fig_t.set_size_inches(4.5, 3.0)

    tol = 0.25
    tc_0 = temperature_a[0]
    print(f'TC[t=0]: {tc_0:4.2f} °C')
    msk_onset = (ta_corrected - tc_0) > tol
    time_onset = tc_time[msk_onset]
    time_onset = time_onset[0]
    idx_onset = (np.abs(tc_time - time_onset)).argmin() - 5
    # print(idx_onset)

    tc_time = tc_time[idx_onset::]
    tc_time -= tc_time.min()
    temperature_a = temperature_a[idx_onset::]
    ta_corrected = ta_corrected[idx_onset::]
    tc_time_positive_idx = tc_time > 0
    tc_time = tc_time[tc_time_positive_idx]
    temperature_a = temperature_a[tc_time_positive_idx]
    ta_corrected = ta_corrected[tc_time_positive_idx]

    ax_t.plot(
        measurement_time_pd, temperature_pd, lw=1.5, label=f'Photodiode $\\epsilon = {thermometry.emissivity}$',
        color='C2'
    )
    ax_t.plot(tc_time, temperature_a, lw=1.5, label='TC(x = 1.0 cm)', color='C3')
    ax_t.plot(tc_time, ta_corrected, lw=1.5, label='TC(x = 1.0 cm) corrected ', color='C3', ls='--')
    ax_t.set_xlim(measurement_time.min(), measurement_time.max())

    ax_t.set_xlabel(f'Time (s)')
    ax_t.set_ylabel(f'Temperature (°C)')
    ax_t.set_title("Graphite type GR001CC")
    # ax_t.legend(loc='upper right', frameon=False)
    leg = ax_t.legend(frameon=False, loc='best', fontsize=10)
    colors = ['C2', 'C3', 'C3']
    for color, text in zip(colors, leg.get_texts()):
        text.set_color(color)
    fig_t.tight_layout()

    surface_temp_df = pd.DataFrame(data={
        'Time (s)':measurement_time_pd,
        'Surface Temperature (°C)': temperature_pd
    })

    surface_temp_df.to_csv(os.path.join(output_path, f'{data_filetag}_surface_temp.csv'), index=False)

    fig_pd.savefig(os.path.join(output_path, f'{data_filetag}_photodiode_voltage.png'), dpi=600)
    fig_t.savefig(os.path.join(output_path, f'{data_filetag}_measured_temperatures.png'), dpi=600)



    fig_t2, ax_tm = plt.subplots()
    fig_t2.set_size_inches(5.0, 3.5)

    ax_tm.plot(tc_time, temperature_a, lw=1.75, label='TCA @ x = 1.0 cm', color='C3')
    ax_tm.plot(tc_time, ta_corrected, lw=1.75, label='TCA @ x = 1.0 cm Corrected', color='C3', ls=":")

    ax_tm.set_xlabel(f'Time (s)')
    ax_tm.set_ylabel(f'Temperature (°C)')
    ax_tm.set_title("Graphite type GR008G")
    leg = ax_tm.legend(frameon=True, loc='best', fontsize=9)

    fig_t2.tight_layout()

    tc_smoothed_df = pd.DataFrame(
        data={
            'Time (s)': tc_time,
            'Temperature A (C)': temperature_a,
            'Corrected Temperature A (C)': ta_corrected
        }
    )
    smoothed_csv = os.path.join(output_path, f'{data_filetag}_corrected_temperature_data.csv')
    print('Destiantion of smoothed: ', smoothed_csv)
    tc_smoothed_df.to_csv(smoothed_csv, index=False)

    fig_t2.savefig(os.path.join(output_path, f'{data_filetag}_corrected_temperature.png'), dpi=600)

    plt.show()

