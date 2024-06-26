import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from typing import List
import pandas as pd
import re

from scipy.signal import savgol_filter

import ir_thermography.thermometry as irt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from scipy.optimize import least_squares
import confidence as cf

# base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\IR_VS_POWER'
base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\surface_temperature\equilibrium'
# csv_database = r'R3N21_firing_database.csv'
csv_database = 'graphite_equilibrium_files.csv'
chamber_volume = 31.57  # L
max_time = 10.0  # s

heat_flux_at_100pct = 25.2  # MW/m2
correct_reflection = True


def plot_pressure(base_path: str, filelist: List, legends: List, output_filename: str, colors, display=False,
                  plot_title=None):
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3.0)

    base_pressures = np.empty_like(filelist, dtype=np.float64)
    peak_pressures = np.empty_like(filelist, dtype=np.float64)
    peak_dt = np.empty_like(filelist, dtype=np.float64)

    for fn, leg, c, i in zip(filelist, legends, colors, range(len(filelist))):
        params = get_experiment_params(base_path, fn)
        pressure_csv = f'{fn}_pressure.csv'
        print(pressure_csv)
        pressure_data = pd.read_csv(filepath_or_buffer=os.path.join(base_path, pressure_csv))
        pressure_data = pressure_data.apply(pd.to_numeric)
        time_s = pressure_data['Time (s)'].values
        time_s -= time_s.min()
        pressure = 1000 * pressure_data['Pressure (Torr)'].values
        base_pressures[i] = pressure[0]
        peak_pressures[i] = pressure.max()
        idx_peak = (np.abs(pressure - peak_pressures[i])).argmin()
        peak_dt[i] = time_s[idx_peak]

        # title_str = 'Sample ' + params['Sample Name']['value'] + ', '
        # params_title = params
        # params_title.pop('Sample Name')
        #
        # for i, p in enumerate(params_title):
        #     title_str += f"{params_title[p]['value']}{params_title[p]['units']}"
        #     if i + 1 < len(params_title):
        #         title_str += ', '

        line = ax.plot(time_s, pressure, label=leg, color=c)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pressure (mTorr)')
        # colors.append(line[0].get_color())

    leg = ax.legend(frameon=True, loc='best', fontsize=8)
    for color, text in zip(colors, leg.get_texts()):
        text.set_color(color)

    if plot_title is not None:
        ax.set_title(plot_title)

    outgassing_rate = chamber_volume * (peak_pressures - base_pressures) * 1E-3 / peak_dt

    outgas_df = pd.DataFrame(data={
        'Sample': legends,
        'Base Pressure (mTorr)': base_pressures,
        'Peak Pressure (mTorr)': peak_pressures,
        'Peak dt (s)': peak_dt,
        'Outgassing Rate (Torr L / s)': outgassing_rate
    })

    print(outgas_df)
    outgas_df.to_csv(os.path.join(base_path, f'{output_filename}_OUTGASSING.csv'), index=False)

    fig.tight_layout()
    fig.savefig(os.path.join(base_path, f'{output_filename}_PRESSURE.png'), dpi=600)
    if display:
        fig.show()


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


def model_tanh(t, b):
    return b[0] * np.tanh(b[1] * t + b[2])

def model_exp(t, b):
    return b[0]*(1.0 - np.exp(-b[1]*t-b[2]))


def fobj_tanh(b, t, temp):
    return model_tanh(t, b) - temp

def fobj_exp(b, t, temp):
    return model_exp(t, b) - temp


def model_tanh_jac(b, t, temp):
    j1 = np.tanh(b[1] * t + b[2])
    # j2 = b[0] * t * np.power(np.cosh(b[1] * t + b[2]), -2)
    j3 = b[0] * np.power(np.cosh(b[1] * t + b[2]), -2)
    j2 = j3 * t
    return np.array([j1, j2, j3]).T

def model_exp_jac(b, t, temp):
    j1 = 1.0 - np.exp(-b[1]*t-b[2])
    j3 = b[0] * np.exp(-b[1]*t-b[2])
    j2 = j3 * t
    return np.array([j1, j2, j3]).T


if __name__ == '__main__':
    thermometry = irt.PDThermometer()
    database_df = pd.read_csv(
        os.path.join(base_path, csv_database), comment='#'
    )
    database_df['Laser Power Setting (%)'] = database_df['Laser Power Setting (%)'].apply(pd.to_numeric)
    database_df.sort_values(by='Laser Power Setting (%)', ascending=True)
    # print(database_df)
    filelist = database_df['csv']
    n = len(filelist)
    colors = plt.cm.jet(np.linspace(0, 1, n))

    base_pressures = np.empty(n, dtype=np.float64)
    peak_pressures = np.empty(n, dtype=np.float64)
    peak_dt = np.empty(n, dtype=np.float64)

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(5.0, 4.0)

    gs = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)  # , height_ratios=[1.618, 1.618, 1])

    ax1 = fig.add_subplot(gs[0])
    # ax2 = fig.add_subplot(gs[1])

    laser_power_setpoint_list = []
    max_temperature = np.empty(len(filelist), dtype=np.float64)
    time_to_equilibrium = np.empty_like(max_temperature)
    heat_flux = np.empty_like(max_temperature)

    for i, file in enumerate(filelist):
        file = file.strip()
        csv_file = file + '_irdata.csv'
        print(f"Processing file {i + 1:d} of {n}: {file}")
        experiment_params = get_experiment_params(relative_path=base_path, filename=file)
        photodiode_gain = experiment_params['Photodiode Gain']['value']
        laser_power_setting = experiment_params['Laser Power Setpoint']['value']
        laser_power_setpoint_list.append(laser_power_setting)
        heat_flux[i] = heat_flux_at_100pct * float(laser_power_setting) * 1E-2
        ir_df = pd.read_csv(os.path.join(base_path, csv_file)).apply(pd.to_numeric)
        ir_df = ir_df[ir_df['Time (s)'] <= max_time]
        ir_df = ir_df[ir_df['Voltage (V)'] > 0.0]
        measurements_df = pd.read_csv(os.path.join(base_path, file+'.csv'), comment='#').apply(pd.to_numeric)
        measurement_time = measurements_df['Measurement Time (s)'].values
        trigger_voltage = measurements_df['Trigger (V)'].values
        photodiode_voltage = measurements_df['Photodiode Voltage (V)'].values
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
        reflection_signal[reflection_signal_max_idx:reflection_signal_max_idx] = photodiode_voltage[
                                                                                 reflection_signal_max_idx:reflection_signal_max_idx]
        print(f"Baseline signal: {photodiode_voltage[reflective_signal_zero_idx]:.3f} V")

        thermometry.gain = int(photodiode_gain)
        print(f"Calibration Factor: {thermometry.calibration_factor}")
        # time_s = measurement_time  # ir_df['Time (s)'].values
        # dt = 0.103 if float(laser_power_setting) < 70 else 0.0270
        # time_s -= dt  # time_s.min()
        # if int(laser_power_setting) == 80:
        #     time_s -= 0.067
        if correct_reflection:
            photodiode_corrected = photodiode_voltage - reflection_signal
            pd_corrected_min = photodiode_corrected[irradiation_time_idx].min()
            photodiode_corrected[irradiation_time_idx] -= pd_corrected_min + 0.5 * noise_level
            time_pd_idx = (photodiode_corrected > 0.0) #& (measurement_time > noise_level)
            measurement_time_pd = measurement_time[time_pd_idx]
            photodiode_voltage_positive = photodiode_corrected[time_pd_idx]
            measurement_time_pd = measurement_time_pd[n:-2]
            photodiode_voltage_positive = photodiode_voltage_positive[n:-2]
            trigger_voltage_corrected = trigger_voltage[time_pd_idx]
            trigger_voltage_corrected = trigger_voltage_corrected[n:-2]
            irradiation_time_idx = trigger_voltage_corrected >= 1.5
        else:
            time_pd_idx = (photodiode_voltage > 0.0)
            photodiode_voltage_positive = photodiode_voltage[time_pd_idx]
            measurement_time_pd = measurement_time[time_pd_idx]
            trigger_voltage_corrected = trigger_voltage[time_pd_idx]
            irradiation_time_idx = trigger_voltage_corrected >= 1.5

        voltage = photodiode_voltage_positive # ir_df['Voltage (V)'].values
        temperature_c = thermometry.get_temperature(voltage=voltage) - 273.15

        print('len(measurement_time_pd):', len(measurement_time_pd))
        print('len(temperature_c):', len(temperature_c))
        print('len(irradiation_time_idx):', len(irradiation_time_idx))
        t_fit = measurement_time_pd[irradiation_time_idx]
        temp_fit = temperature_c[irradiation_time_idx]
        b_guess = [3000, 1E-8, 0.5]
        all_tol = np.finfo(np.float64).eps
        res = least_squares(
            fobj_tanh, b_guess, args=(t_fit, temp_fit),
            bounds=([0.0, 0.0, 0.0], [5000.0, 2, 2]),
            jac=model_tanh_jac,
            xtol=all_tol,
            ftol=all_tol,
            gtol=all_tol,
            loss='soft_l1', f_scale=0.1,
            verbose=2
        )
        popt = res.x
        pcov = cf.get_pcov(res)

        print(f'b[0]: {popt[0]:.3E}')
        print(f'b[1]: {popt[1]:.3E}')
        print(f'b[2]: {popt[2]:.3E}')
        # print(f'b[3]: {popt[3]:.3E}')

        time_extrapolate = np.linspace(measurement_time_pd[irradiation_time_idx].min(), 20.0, 200)
        temp_extrapolate = model_tanh(time_extrapolate, popt)
        tc = (3.0 - popt[2]) / popt[1]
        limit_temp = model_tanh(tc, popt)
        if len(temperature_c) > 0:
            max_temperature[i] = limit_temp#temperature_c.max()
            # print(temperature_c)
        else:
            max_temperature[i] = 20.0

        time_to_equilibrium[i] = tc

        lbl = f'{float(laser_power_setting):3.0f} % Power'

        ax1.plot(
            measurement_time_pd,
            temperature_c,
            color=colors[i], lw=1.75, marker='o', fillstyle='none', ls='none',
            label=lbl
        )

        ax1.plot(
            time_extrapolate, temp_extrapolate, color=colors[i], ls='--', lw=1.0
        )

        ax1.plot(tc, limit_temp, color='k', marker='o', ls='none')

        pressure_csv = f'{file}_pressure.csv'
        pressure_data = pd.read_csv(filepath_or_buffer=os.path.join(base_path, pressure_csv))
        pressure_data = pressure_data.apply(pd.to_numeric)
        time_s = pressure_data['Time (s)'].values
        time_s -= time_s.min() + 0.5
        time_msk = time_s >= 0.0
        time_s = time_s[time_msk]
        pressure = 1000 * pressure_data['Pressure (Torr)'].values
        pressure = pressure[time_msk]
        base_pressures[i] = pressure[0]
        peak_pressures[i] = pressure.max()
        idx_peak = (np.abs(pressure - peak_pressures[i])).argmin()
        peak_dt[i] = time_s[idx_peak]

        # ax2.plot(time_s, pressure, label=lbl, color=colors[i], lw=1.75)

    ax1.set_xlabel('Time (s)')
    # ax2.set_xlabel('Time (s)')
    # ax2.set_ylabel('Chamber pressure (mTorr)')

    ax1.set_ylabel('Surface temperature (°C)')
    ax1.set_ylim(top=3000, bottom=1000)
    # ax2.set_ylim(top=40)
    # ax1.set_xlabel('Time (s)')
    ax1.set_xlim(0.0, 15.0)
    # ax2.set_xlim(left=0.0, right=20.0)
    # ax1.legend(loc="upper right", prop={'size': 9}, frameon=False, ncol=3)
    ax1.legend(
        bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=4, mode="expand", borderaxespad=0.,
        prop={'size': 8}
    )

    ax1.ticklabel_format(useMathText=True)
    # ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    # ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.125))

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(250))

    # ax2.ticklabel_format(useMathText=True)
    # ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    # ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    # ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
    # ax2.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    # ax1.set_title('IR Thermography')
    # ax2.set_title('Chamber Pressure')
    filetag = os.path.splitext(csv_database)[0]

    outgassing_rate = chamber_volume * (peak_pressures - base_pressures) * 1E-3 / peak_dt

    outgas_df = pd.DataFrame(data={
        'Laser Power Setpoint': laser_power_setpoint_list,
        'Base Pressure (mTorr)': base_pressures,
        'Peak Pressure (mTorr)': peak_pressures,
        'Peak dt (s)': peak_dt,
        'Outgassing Rate (Torr L / s)': outgassing_rate
    })

    temperature_vs_power_df = pd.DataFrame(data={
        'Laser power setpoint (%)': laser_power_setpoint_list,
        'Heat flux (MW/m^2)': heat_flux,
        'Max surface temperature (C)': max_temperature,
        'Time to equilibrium (s)': time_to_equilibrium
    })

    temperature_vs_power_df.to_csv(os.path.join(base_path, f'{filetag}_surface_temperature.csv'), index=False)

    print(outgas_df)
    outgas_df.to_csv(os.path.join(base_path, f'{filetag}_OUTGASSING.csv'), index=False)

    fig.savefig(os.path.join(base_path, filetag + "_ir-pressure_plot.png"), dpi=600)
    fig.savefig(os.path.join(base_path, filetag + "_ir-pressure_plot.eps"), dpi=600)
    fig.savefig(os.path.join(base_path, filetag + "_ir-pressure_plot.svg"), dpi=600)
    plt.show()
