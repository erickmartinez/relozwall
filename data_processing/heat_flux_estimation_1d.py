import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
import time

from matplotlib import ticker

from heat_flux_1D import simulate_1d_temperature
from scipy.signal import savgol_filter
import matplotlib as mpl
import json
import ir_thermography.thermometry as irt
import re

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration\results'
csv_file = 'GR001C_1cm_probe_smoothed_temperature_data'
# data_file = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration\LT_GR001CC_150mT_1cm_100PCT_60GAIN 2022-04-26_1.csv'
data_file = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration\LT_GR008G_6mTorr-contact-shield_100PCT_50GAIN 2022-05-04_1.csv'


q0 = 25.0
time_constant = 2.1148
reflectance = 40.4

# density_g = 1.76  # g / cm^3 # GR008G
# density_g = 1.81 # g / cm^3 # GR001CC
density_g = 1.698 # measured
""" 
It has been found that the values of heat capacity for all
types of natural and manufactured graphites are basically
the same, except near absolute-zero temperatures.

https://www.goodfellow.com/us/en-us/displayitemdetails/p/c-00-rd-000130/carbon-rod 

and 

https://poco.entegris.com/content/dam/poco/resources/reference-materials/brochures/brochure-graphite-properties-and-characteristics-11043.pdf
"""
# specific_heat_g = 0.712  # J / g / K
specific_heat_g = 0.6752  # Markelov, Volga, et al., 1973
k0_1 = 85E-2 # W / (cm K) https://www.graphitestore.com/core/media/media.nl?id=6310&c=4343521&h=Tz5uoWvr-nhJ13GL1b1lG8HrmYUqV1M_1bOTFQ2MMuiQapxt # GR001C
# k0_1 = 130E-2  # W / (cm K) https://www.graphitestore.com/core/media/media.nl?id=7164&c=4343521&h=8qpl24Kn0sh2rXtzPvd5WxQIPQumdO8SE5m3VRfVBFvLJZtj # GR008G
k0_2 = 16.2E-2  # W / (cm K)

# kappa_1 = 1.11 # Thermal diffusivity of copper in cm^2/s
# kappa_1 = 25E-2 # Thermal diffusivity of polycrystalline graphite in cm^2/s
kappa_1 = k0_1 / (density_g * specific_heat_g)
alpha = 1.0 - (reflectance / 100.0)
emissivity = 1.0 - (36.9 / 100)
T_a = 20.0
pulse_length = 0.5
t_max = 2.01
sample_length = 5.0


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


def correct_thermocouple_response(measured_temperature, measured_time, time_constant):
    n = len(measured_time)
    k = int(n / 8)
    k = k + 1 if k % 2 == 0 else k
    T = savgol_filter(measured_temperature, k, 3)
    dTdt = np.gradient(T, measured_time)
    return savgol_filter(T + time_constant * dTdt, k - 2, 3)


if __name__ == "__main__":
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    measurements_df = pd.read_csv(data_file, comment='#').apply(pd.to_numeric)
    data_filetag = os.path.splitext(data_file)[0]
    experiment_params = get_experiment_params(relative_path=os.path.dirname(data_file), filename=data_filetag)
    photodiode_gain = experiment_params['Photodiode Gain']['value']
    laser_power_setting = experiment_params['Laser Power Setpoint']['value']

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
    thermometry.emissivity = emissivity

    photodiode_corrected = photodiode_voltage - reflection_signal
    time_pd_idx = photodiode_corrected > 0.0
    measurement_time_pd = measurement_time[time_pd_idx]
    photodiode_voltage_positive = photodiode_corrected[time_pd_idx]
    temperature_pd = thermometry.get_temperature(voltage=photodiode_voltage_positive) - 273.15

    tic = time.time()
    t_sim_1d, temp_sim_1d = simulate_1d_temperature(
        q0=q0, length=sample_length, k0=k0_1 * 100.0,
        t_max=t_max, pulse_length=pulse_length, alpha=alpha, rho=density_g * 1000.0,
        z_tc_1=1.0, T0=T_a + 273.15, cp=specific_heat_g * 1000.0,
        debug=True, use_radiation=True,
        x_points=500, t_steps=1000
    )
    toc = time.time() - tic
    print(f"Simulation time: {toc:.4f} s")
    tc_data_df = pd.read_csv(os.path.join(base_path, csv_file + '.csv')).apply(pd.to_numeric)
    ta = tc_data_df['Temperature A (C)'].values
    time_tc = tc_data_df['Time (s)'].values
    idx_range = time_tc <= 2.0
    t_steps = ta.size
    ta -= ta.min() - 20.0
    time_tc = time_tc[idx_range]
    ta = ta[idx_range]

    ta_corrected = correct_thermocouple_response(
        measured_time=time_tc, measured_temperature=ta, time_constant=time_constant
    )

    fig, ax = plt.subplots(ncols=1)  # , constrained_layout=True)
    fig.set_size_inches(5.0, 3.5)
    ax.plot(
        measurement_time_pd, temperature_pd, ls='none', label=f'Photodiode',
        c='tab:red', marker='o', fillstyle='none'
    )
    ax.plot(time_tc, ta, label='T(x=1.0 cm)', ls='none', color='tab:green', marker='s', fillstyle='none')
    ax.plot(time_tc, ta_corrected, label=f'T(x=1.0 cm) (corrected)', ls='none',
            color='tab:olive', marker='^', fillstyle='none')
    ax.plot(t_sim_1d, temp_sim_1d[0, :], label=f'Q={q0:.1f} MW/m$^{{\\mathregular{{2}}}}$, x=0.0 cm', c='tab:red',
            ls='-')
    ax.plot(t_sim_1d, temp_sim_1d[1, :], label=f'Q={q0:.1f} MW/m$^{{\\mathregular{{2}}}}$, x=1.0 cm', c='tab:green',
            ls='-')

    leg = ax.legend(
        loc='upper right', ncol=1, frameon=False
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('1D Model')
    ax.set_xlim((0., 2.0))
    ax.set_ylim((0., 1900.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(400))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))
    fig.tight_layout()
    fig.savefig(os.path.join(base_path, f'{data_filetag}_raw_fit.eps'), dpi=600)
    fig.savefig(os.path.join(base_path, f'{data_filetag}_raw_fit.svg'), dpi=600)
    fig.savefig(os.path.join(base_path, f'{data_filetag}_raw_fit.png'), dpi=600)
    plt.show()
