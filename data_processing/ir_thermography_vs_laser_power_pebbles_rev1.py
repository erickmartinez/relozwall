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
base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\surface_temperature\equilibrium_redone\pebble_sample'
# csv_database = r'R3N21_firing_database.csv'
csv_database = 'pebble_sample_equilibrium_redone_files.csv'
chamber_volume = 31.57  # L
max_time = 10.0  # s

heat_flux_at_100pct = 25.2  # MW/m2
correct_reflection = True


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
    return b[0] * (1.0 - np.exp(-b[1] * t + b[2]))


def model_asymptotic2(t, b):
    # return b[0] * ((t - b[1]) ** b[2]) / ((t - b[1]) ** b[2] + b[3])
    t_min = np.min(t)
    res = b[0] * np.power(1.0 - t_min / t, b[1])
    return res


def model_asymptotic1(t, b):
    x = t - np.min(t) + 1E-10
    return b[0] * (x ** b[1] / (x ** b[1] + b[2]))


def fobj_tanh(b, t, temp):
    temp_max = np.max(temp)
    return (model_tanh(t, b) - temp) * temp / temp_max


def fobj_exp(b, t, temp):
    temp_max = np.max(temp)
    return (model_exp(t, b) - temp)*temp/temp_max


def fobj_asymptotic1(b, t, temp):
    temp_max = np.max(temp)
    return (model_asymptotic1(t, b) - temp) * temp/temp_max


def fobj_asymptotic2(b, t, temp):
    temp_max = np.max(temp)
    return (model_asymptotic2(t, b) - temp) * temp/temp_max


def model_asymptotic1_jac(b, t, temp):
    x = t - np.min(t) + 1E-10
    xb = x ** b[1]
    den = 1.0 / (xb + b[2])
    den2 = den * den
    j1 = xb * den  # (x ** b[1] / (x ** b[1] + b[2]))
    j2 = b[0] * b[2] * np.log(x) * den2
    j3 = -b[0] * xb * den2
    return np.array([j1, j2, j3]).T


def model_asymptotic2_jac(b, t, temp):
    # tb = t - b[1]
    # tbb2 = tb ** b[2]
    # den = tbb2 + b[3]
    # den2 = 1.0 / den * den
    # j1 = tbb2 / den
    # j2 = -b[0] * b[2] * b[3] * tbb2 * den2 / tb
    # j3 = b[0] * b[3] * tbb2 * np.log(tb) * den2
    # j4 = -b[0] * tbb2 * den2
    # return np.array([j1, j2, j3, j4]).T
    a = 1.0 - np.min(t) / t
    a[0] = 1E-50
    j1 = np.power(a, b[1])
    j2 = b[0] * j1 * np.log(a)
    return np.array([j1, j2]).T


def model_tanh_jac(b, t, temp):
    j1 = np.tanh(b[1] * t + b[2])
    # j2 = b[0] * t * np.power(np.cosh(b[1] * t + b[2]), -2)
    j3 = b[0] * np.power(np.cosh(b[1] * t + b[2]), -2)
    j2 = j3 * t
    return np.array([j1, j2, j3]).T


def model_exp_jac(b, t, temp):
    ee = np.exp(-b[1] * t + b[2])
    j1 = 1.0 - ee
    j2 = b[0] * t * ee
    j3 = -b[0] * ee

    return np.array([j1, j2, j3]).T


if __name__ == '__main__':
    thermometry = irt.PDThermometer()
    database_df = pd.read_csv(
        os.path.join(base_path, csv_database), comment='#'
    )
    database_df['Laser Power Setting (%)'] = database_df['Laser Power Setting (%)'].apply(pd.to_numeric)

    database_df.sort_values(by='Laser Power Setting (%)', ascending=True)
    database_df.dropna(inplace=True)
    # print(database_df)
    filelist = database_df['csv']
    fit_data = np.array(database_df['fit'], dtype=bool)
    fit_model = database_df['model']
    show_plot = np.array(database_df['plot'], dtype=bool)
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
    fig.set_size_inches(5.5, 3.5)

    fig_p, ax2 = plt.subplots()
    fig_p.set_size_inches(4.5, 3.0)

    gs = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)  # , height_ratios=[1.618, 1.618, 1])

    ax1 = fig.add_subplot(gs[0])
    # ax2 = fig.add_subplot(gs[1])

    laser_power_setpoint_list = []
    max_temperature = np.empty(len(filelist), dtype=np.float64)
    time_to_equilibrium = np.empty_like(max_temperature)
    heat_flux = np.empty_like(max_temperature)
    emission_times = np.empty_like(max_temperature)

    for i, file in enumerate(filelist):
        file = file.strip()
        csv_file = file + '_irdata.csv'
        print(f"Processing file {i + 1:d} of {n}: {file}")
        experiment_params = get_experiment_params(relative_path=base_path, filename=file)
        photodiode_gain = experiment_params['Photodiode Gain']['value']
        laser_power_setting = experiment_params['Laser Power Setpoint']['value']
        laser_power_setpoint_list.append(laser_power_setting)
        emission_time = float(experiment_params['Emission Time']['value'])
        emission_times[i] = emission_time

        heat_flux[i] = heat_flux_at_100pct * float(laser_power_setting) * 1E-2
        ir_df = pd.read_csv(os.path.join(base_path, csv_file)).apply(pd.to_numeric)
        ir_df = ir_df[ir_df['Time (s)'] <= max_time]
        ir_df = ir_df[ir_df['Voltage (V)'] > 0.0]
        measurements_df = pd.read_csv(os.path.join(base_path, file + '.csv'), comment='#').apply(pd.to_numeric)
        measurement_time = measurements_df['Measurement Time (s)'].values
        trigger_voltage = measurements_df['Trigger (V)'].values
        photodiode_voltage = measurements_df['Photodiode Voltage (V)'].values
        trigger_voltage = savgol_filter(trigger_voltage, 5, 4)

        irradiation_time_idx = trigger_voltage > 1.0
        irradiation_time = measurement_time[irradiation_time_idx]
        reflection_signal = np.zeros_like(photodiode_voltage)

        sg_window = 9
        photodiode_voltage[irradiation_time_idx] = savgol_filter(photodiode_voltage[irradiation_time_idx], sg_window, 2)

        # t_pulse_max = irradiation_time.max() + 0.2
        noise_level = np.abs(photodiode_voltage[measurement_time > 0.75 * measurement_time.max()]).max()
        print(f"Noise Level: {noise_level:.4f} V")

        t0 = irradiation_time.max() - emission_time

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
            time_pd_idx = (photodiode_corrected > 0.0)  # & (measurement_time > noise_level)
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

        voltage = photodiode_voltage_positive  # ir_df['Voltage (V)'].values
        temperature_c = thermometry.get_temperature(voltage=voltage) - 273.15

        print('len(measurement_time_pd):', len(measurement_time_pd))
        print('len(temperature_c):', len(temperature_c))
        print('len(irradiation_time_idx):', len(irradiation_time_idx))

        lbl = f'{float(laser_power_setting):3.0f} % Power'

        if show_plot[i]:
            ax1.plot(
                measurement_time_pd,
                temperature_c,
                color=colors[i], lw=1.75, marker='o', fillstyle='none', ls='none',
                label=lbl
            )

        temp_peak = temperature_c.max()
        t_max_idx = (np.abs(temperature_c - temp_peak)).argmin()
        t_peak = measurement_time_pd[t_max_idx]
        print(f't_peak: {t_peak:.3f} s, v_peak: {temp_peak:.2f} °C')
        t0_fit = int(0.01 * t_peak)
        idx_fit = (measurement_time_pd >= t0_fit) & (measurement_time_pd <= t_peak)
        # idx_fit = irradiation_time_idx
        # idx_fit = measurement_time_pd <= t_peak

        # t_fit = measurement_time_pd[irradiation_time_idx]
        # temp_fit = temperature_c[irradiation_time_idx]
        t_fit = measurement_time_pd[idx_fit]
        temp_fit = temperature_c[idx_fit]
        print(f'len(temp_fit): {len(temp_fit)}')

        if fit_model[i] == 'tanh':
            fobj = fobj_tanh
            jac = model_tanh_jac
            model = model_tanh
            b_guess = [temp_peak, 0.5, 0.1]
            bounds = ([200.0, 1E-10, -10.0], [2800.0, 2.0, 10.0])

        elif fit_model[i] == 'asymptotic1':
            fobj = fobj_asymptotic1
            jac = model_asymptotic1_jac
            model = model_asymptotic1
            b_guess = [temp_peak, 1.1, 1E-5]
            bounds = ([1000.0, 0.001, 0.0], [3000.0, 1E5, 1E10])
        elif fit_model[i] == 'asymptotic2':
            fobj = fobj_asymptotic2
            jac = model_asymptotic2_jac
            model = model_asymptotic2
            b_guess = [temp_peak, 1.1]
            bounds = ([1000.0, 1.001], [3000.0, 1E5])
        elif fit_model[i] == 'exp':
            fobj = fobj_exp
            jac = model_exp_jac
            model = model_exp
            b_guess = [temp_peak, 1.0, 0.1]
            bounds = ([800.0, 1E-16, -10.0], [2800.0, 1E3, 10.0])


        all_tol = np.finfo(np.float64).eps
        if fit_data[i]:
            res = least_squares(
                fobj, b_guess, args=(t_fit, temp_fit),
                bounds=bounds,
                jac=jac,
                xtol=all_tol,
                ftol=all_tol,
                gtol=all_tol,
                # loss='soft_l1', f_scale=0.1,
                verbose=2
            )
            popt = res.x
            # pcov = cf.get_pcov(res)

            print(f'b[0]: {popt[0]:.3E}')
            print(f'b[1]: {popt[1]:.3E}')
            # print(f'b[2]: {popt[2]:.3E}')
            # print(f'b[3]: {popt[3]:.3E}')

            time_extrapolate = np.linspace(t_fit.min(), 20.0, 200)
            temp_extrapolate = model(time_extrapolate, popt)
            if fit_model[i] == 'tanh':
                tc = (3.0 - popt[2]) / popt[1]
                limit_temp = model_tanh(tc, popt)
            elif fit_model[i] == 'asymptotic1':
                tc = np.power(0.995*popt[2]/(1.0-0.995), 1.0/popt[1]) + t_fit.min()
                limit_temp = 0.995 * popt[0]
            elif fit_model[i] == 'asymptotic2':
                tc = t_fit.min() / (1.0 - (0.995 ** (1.0 / popt[1])) )
                limit_temp = 0.995 * popt[0]
            elif fit_model[i] == 'exp':
                tc = (popt[2] - np.log(1-0.995))/popt[1]
                limit_temp = 0.995*popt[0]

            print(f'99.5% of final temperature: {tc:g} s, {limit_temp:.0f} °C ({fit_model[i]})')
            if len(temperature_c) > 0:
                max_temperature[i] = limit_temp  # temperature_c.max()
                # print(temperature_c)
            else:
                max_temperature[i] = 20.0

            time_to_equilibrium[i] = tc

            if show_plot[i]:
                ax1.plot(
                    time_extrapolate, temp_extrapolate, color=colors[i], ls='--', lw=1.0
                )

                ax1.plot(
                    t_fit, model(t_fit, popt), color='k', ls='--', lw=1.0,
                )
        else:
            idx_avg = (np.abs(temperature_c - temp_peak)).argmin()
            limit_temp = temp_peak
            max_temperature[i] = limit_temp
            tc = t_peak
            time_to_equilibrium[i] = tc

        if show_plot[i]:
            ax1.plot(tc, limit_temp, color='k', marker='*', ls='none', ms=8)

        offset = 1
        connectionstyle = "angle3,angleA=0,angleB=90"
        bbox = dict(boxstyle="round", fc="w", alpha=1.0, ec=colors[i])  # , fc="wheat", alpha=1.0)
        """
        arrowprops = dict(
            arrowstyle="->", color="k",
            shrinkA=5, shrinkB=5,
            patchA=None, patchB=None,
            connectionstyle=connectionstyle
        )

        xy = t_peak, t_max
        ax1.annotate(
            f'{float(laser_power_setting):3.0f} %',
            xy=xy, xycoords='data',  # 'figure pixels', #data',
            xytext=(t_peak*np.random.uniform(low=0.75, high=2.5/(i+1)), t_max*1.15), textcoords='data',  # 'data','offset points'
            arrowprops=arrowprops,
            bbox=bbox,
            ha='left',
            color=colors[i],
        )
        """
        temperature_irradiation = temperature_c[irradiation_time_idx]
        if show_plot[i]:
            ax1.text(
                t0 + emission_time * np.random.uniform(low=(1.0 - 0.075 * i / n), high=(1.0 + 0.065 * i / n)),
                temperature_irradiation[-1] * np.random.uniform(low=(0.95 - 0.075 * i / n), high=(0.975 + 0.05 * i / n)),
                f'{float(laser_power_setting):3.0f} %',
                color=colors[i],
                ha='right', bbox=bbox
            )

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

        ax2.plot(time_s, pressure - pressure[0], label=lbl, color=colors[i], lw=1.75)

    ax1.set_xlabel('Time (s)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Pressure increase (mTorr)')

    ax1.set_ylabel('Surface temperature (°C)')
    ax1.set_ylim(top=3000, bottom=1000)
    # ax2.set_ylim(top=40)
    ax1.set_xlabel('Time (s)')
    ax1.set_xlim(-0.5, max_time)
    ax2.set_xlim(left=0.0, right=max_time)
    # ax1.legend(loc="upper right", prop={'size': 9}, frameon=False, ncol=3)
    # ax1.legend(
    #     bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=4, mode="expand", borderaxespad=0.,
    #     prop={'size': 8}
    # )

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

    ax1.set_title('Surface Temperature')
    ax2.set_title('Chamber Pressure')
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

    fig_p.tight_layout()

    fig.savefig(os.path.join(base_path, filetag + "_ir_thermography_plot.png"), dpi=600)
    fig.savefig(os.path.join(base_path, filetag + "_ir_thermography_plot.eps"), dpi=600)
    fig.savefig(os.path.join(base_path, filetag + "_ir_thermography_plot.svg"), dpi=600)

    fig_p.savefig(os.path.join(base_path, filetag + "_pressure_plot.png"), dpi=600)
    fig_p.savefig(os.path.join(base_path, filetag + "_pressure_plot.eps"), dpi=600)
    fig_p.savefig(os.path.join(base_path, filetag + "_pressure_plot.svg"), dpi=600)
    plt.show()
