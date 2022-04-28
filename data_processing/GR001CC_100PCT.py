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

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration'
# data_file = 'LT_GR001CC_7mTorr_100PCT_60GAIN 2022-04-22_1'
data_file = 'LT_GR001CC_150mT_1cm_100PCT_60GAIN 2022-04-26_1'
time_constant_1, time_constant_2 = 2.9, 2.9


def correct_thermocouple_response(
        measured_time: np.ndarray, measured_temperature: np.ndarray, time_constant: float = 1.0,
):
    dts = np.diff(measured_time)
    dTs = np.diff(measured_temperature)
    T0 = measured_temperature[0]
    output_temperature = np.empty_like(measured_temperature, dtype=np.float64)
    output_temperature[0] = T0
    t_sum = T0
    for i in range(1, len(measured_temperature)):
        j = i - 1
        dT_correction = dTs[j] / (1.0 - np.exp(-dts[j] / time_constant))
        t_sum += dT_correction
        output_temperature[i] = t_sum
    return output_temperature


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
    results_path = os.path.join(base_path, 'results')
    data_filetag = os.path.join(base_path, data_file)
    main_csv = data_filetag + '.csv'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    measurements_df = pd.read_csv(main_csv, comment='#').apply(pd.to_numeric)
    experiment_params = get_experiment_params(relative_path=base_path, filename=data_filetag)
    photodiode_gain = experiment_params['Photodiode Gain']['value']
    laser_power_setting = experiment_params['Laser Power Setpoint']['value']

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    thermometry = irt.PDThermometer()

    measurement_time = measurements_df['Measurement Time (s)'].values
    trigger_voltage = measurements_df['Trigger (V)'].values
    photodiode_voltage = measurements_df['Photodiode Voltage (V)'].values
    temperature_a = measurements_df['TC1 (C)'].values
    temperature_b = measurements_df['TC2 (C)'].values

    t_max_idx = measurement_time <= 3.0

    measurement_time = measurement_time[t_max_idx]
    trigger_voltage = trigger_voltage[t_max_idx]
    photodiode_voltage = photodiode_voltage[t_max_idx]
    temperature_a = temperature_a[t_max_idx]
    temperature_b = temperature_b[t_max_idx]

    irradiation_time_idx = trigger_voltage > 0.5
    irradiation_time = measurement_time[irradiation_time_idx]
    reflection_signal = np.zeros_like(photodiode_voltage)

    t0 = irradiation_time.min()
    irradiation_time -= t0
    measurement_time -= t0
    reflective_signal_zero_idx = (np.abs(measurement_time)).argmin()
    base_line_msk = (0.0 < measurement_time) & (measurement_time < 0.5)
    # print(base_line_msk)
    reflection_signal[irradiation_time_idx] = photodiode_voltage[reflective_signal_zero_idx + 3]

    print(f"Baseline signal: {photodiode_voltage[reflective_signal_zero_idx + 1]:.3f} V")
    thermometry.gain = int(photodiode_gain)
    print(f"Calibration Factor: {thermometry.calibration_factor}")
    thermometry.emissivity = 0.8

    photodiode_corrected = photodiode_voltage - reflection_signal
    time_pd_idx = photodiode_corrected > 0.0
    measurement_time_pd = measurement_time[time_pd_idx]
    photodiode_voltage_positive = photodiode_corrected[time_pd_idx]
    temperature_pd = thermometry.get_temperature(voltage=photodiode_voltage_positive) - 273.15

    print(f't0 = {t0:.3f} s')

    fig_pd, ax_pd = plt.subplots()
    fig_pd.set_size_inches(5.0, 3.5)

    color_pd = 'C0'
    color_tr = 'C5'

    ax_pd.plot(measurement_time, photodiode_voltage, color=color_pd, lw=1.75, label='Photodiode Raw')
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
    ax_pd.set_ylim(bottom=tr_ylim[0], top=1.1 * photodiode_voltage.max())

    fig_pd.tight_layout()

    fig_t, ax_t = plt.subplots()
    fig_t.set_size_inches(4.5, 3.0)

    tc_time = measurement_time - 0.75
    tc_time_positive_idx = tc_time > 0
    tc_time = tc_time[tc_time_positive_idx]
    temperature_a = temperature_a[tc_time_positive_idx]
    temperature_b = temperature_b[tc_time_positive_idx]

    ax_t.plot(
        measurement_time_pd, temperature_pd, lw=1.75, label=f'Photodiode $\\epsilon = {thermometry.emissivity}$',
        color='C2'
    )
    ax_t.plot(tc_time, temperature_a, lw=1.75, label='TCA @ x = 0.3 cm', color='C3')
    ax_t.plot(tc_time, temperature_b, lw=1.75, label='TCB @ x = 1.0 cm', color='C4')

    ax_t.set_xlabel(f'Time (s)')
    ax_t.set_ylabel(f'Temperature (°C)')
    ax_t.set_title("Graphite type GR001CC")
    # ax_t.legend(loc='upper right', frameon=False)
    leg = ax_t.legend(frameon=False, loc='best', fontsize=10)
    colors = [f'C{i:d}' for i in range(2, 5)]
    for color, text in zip(colors, leg.get_texts()):
        text.set_color(color)
    fig_t.tight_layout()

    f1 = interpolate.interp1d(tc_time, temperature_a, kind='linear')
    f2 = interpolate.interp1d(tc_time, temperature_b, kind='linear')
    t_steps = 50
    t_max_interp = 1.0
    time_interp = np.linspace(0.0, t_max_interp, num=t_steps, dtype=np.float64)
    temperature_interp_x1 = f1(time_interp[1::])
    temperature_interp_x2 = f2(time_interp[1::])
    temperture_a_reduced = np.hstack((temperature_a[0], temperature_interp_x1))
    temperture_b_reduced = np.hstack((temperature_b[0], temperature_interp_x2))

    ta_smooth = savgol_filter(temperature_a, 61, 3)
    tb_smooth = savgol_filter(temperature_b, 61, 3)
    ta_corrected = correct_thermocouple_response(
        measured_time=tc_time, measured_temperature=ta_smooth, time_constant=time_constant_1
    )
    tb_corrected = correct_thermocouple_response(
        measured_time=tc_time, measured_temperature=tb_smooth, time_constant=time_constant_2
    )

    fig_tc, ax_tc = plt.subplots()
    fig_tc.set_size_inches(4.5, 3.0)
    ax_tc.plot(tc_time, temperature_a, lw=1.75, label='TCA @ x = 0.3 cm', color='C3')
    ax_tc.plot(tc_time, temperature_b, lw=1.75, label='TCB @ x = 1.0 cm', color='C4')
    # ax_tc.plot(tc_time, ta_corrected, lw=1.75, label='TCA @ x = 0.3 cm Corrected', color='C3', ls=":")
    # ax_tc.plot(tc_time, tb_corrected, lw=1.75, label='TCB @ x = 1.0 cm Corrected', color='C4', ls=":")
    ax_tc.plot(
        time_interp, temperture_a_reduced, marker='o', fillstyle='none', label='TCA Reduced', color='C3',
        ls='none'
    )
    ax_tc.plot(
        time_interp, temperture_b_reduced, marker='s', fillstyle='none', label='TCB Reduced', color='C4',
        ls='none'
    )
    ax_tc.set_xlabel(f'Time (s)')
    ax_tc.set_ylabel(f'Temperature (°C)')
    ax_tc.set_title("Graphite type GR001CC")
    # ax_t.legend(loc='upper right', frameon=False)
    leg = ax_tc.legend(frameon=True, loc='best', fontsize=9)
    # colors = [f'C{i:d}' for i in range(2, 5)]
    # for color, text in zip(colors, leg.get_texts()):
    #     text.set_color(color)
    fig_tc.tight_layout()

    fig_pd.savefig(os.path.join(results_path, 'GR001CC_photodiode_voltage.png'), dpi=600)
    fig_t.savefig(os.path.join(results_path, 'GR001CC_measured_temperatures.png'), dpi=600)
    fig_tc.savefig(os.path.join(results_path, 'GR001CC_measured_temperatures_tca_tcb.png'), dpi=600)

    reduced_tc_df = pd.DataFrame(
        data={
            'Time (s)': time_interp,
            'Temperature A (C)': temperture_a_reduced,
            'Temperature B (C)': temperture_b_reduced
        }
    )

    reduced_tc_df.to_csv(os.path.join(results_path, 'reduced_temperature_data.csv'), index=False)

    fig_t2, ax_tm = plt.subplots()
    fig_t2.set_size_inches(5.0, 3.5)

    ax_tm.plot(tc_time, temperature_a, lw=1.75, label='TCA @ x = 0.3 cm', color='C3')
    ax_tm.plot(tc_time, temperature_b, lw=1.75, label='TCB @ x = 1.0 cm', color='C4')
    ax_tm.plot(tc_time, ta_corrected, lw=1.75, label='TCA @ x = 0.3 cm Corrected', color='C3', ls=":")
    ax_tm.plot(tc_time, tb_corrected, lw=1.75, label='TCB @ x = 1.0 cm Corrected', color='C4', ls=":")

    ax_tm.set_xlabel(f'Time (s)')
    ax_tm.set_ylabel(f'Temperature (°C)')
    ax_tm.set_title("Graphite type GR001CC")
    leg = ax_tm.legend(frameon=True, loc='best', fontsize=9)
    # colors = [f'C{i:d}' for i in range(2, 5)]
    # for color, text in zip(colors, leg.get_texts()):
    #     text.set_color(color)
    fig_tc.tight_layout()

    tc_smoothed_df = pd.DataFrame(
        data={
            'Time (s)': tc_time,
            'Temperature A (C)': ta_smooth,
            'Temperature B (C)': tb_smooth
        }
    )

    tc_smoothed_df.to_csv(os.path.join(results_path, 'smoothed_temperature_data.csv'), index=False)

    plt.show()
