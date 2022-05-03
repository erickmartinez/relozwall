import h5py
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os

from matplotlib import ticker

from heat_flux_adi import simulate_adi_temp
from scipy.signal import savgol_filter
import matplotlib as mpl
import json
import ir_thermography.thermometry as irt
import re
import shutil

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration\results'
csv_file = 'GR001C_1cm_probe_smoothed_temperature_data'
data_file = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration\LT_GR001CC_150mT_1cm_100PCT_60GAIN 2022-04-26_1.csv'
load_model = False
saved_h5 = 'ADI_k1_1.09E+00_chi_0.60_P4.50E+03'

time_constant = 2.1148
qmax = 4.5E3
emissivity = 1.0 - (36.9/100)
reflectance = 40.4

M = 200  # number of intervals in r
N = 400  # the number of intervals in x
R = 1.27  # The radius of the cylinder in cm
R_sample = 0.5 * 1.288  #
L = 5.0  # the length of the cylinder
holder_thickness = 1.27
dt = 1.0E-3
beam_diameter = 1.5 * 0.8165  # cm
probe_size = 3.0 # mm

thermography_spot_diameter = 0.4  # cm
# thermography_spot_diameter = R_sample
density_g = 1.76  # g / cm^3 # GR008G
# density_g = 1.81 # g / cm^3 # GR001CC
""" 
It has been found that the values of heat capacity for all
types of natural and manufactured graphites are basically
the same, except near absolute-zero temperatures.

https://www.goodfellow.com/us/en-us/displayitemdetails/p/c-00-rd-000130/carbon-rod 

and 

https://poco.entegris.com/content/dam/poco/resources/reference-materials/brochures/brochure-graphite-properties-and-characteristics-11043.pdf
"""
# specific_heat_g = 0.712 # J / g / K
specific_heat_g = 0.6752  # Markelov, Volga, et al., 1973
# k0_1 = 85E-2 # W / (cm K) https://www.graphitestore.com/core/media/media.nl?id=6310&c=4343521&h=Tz5uoWvr-nhJ13GL1b1lG8HrmYUqV1M_1bOTFQ2MMuiQapxt # GR001C
k0_1 = 130E-2  # W / (cm K) https://www.graphitestore.com/core/media/media.nl?id=7164&c=4343521&h=8qpl24Kn0sh2rXtzPvd5WxQIPQumdO8SE5m3VRfVBFvLJZtj # GR008G
k0_2 = 16.2E-2  # W / (cm K)

# kappa_1 = 1.11 # Thermal diffusivity of copper in cm^2/s
# kappa_1 = 25E-2 # Thermal diffusivity of polycrystalline graphite in cm^2/s
kappa_1 = k0_1 / (density_g * specific_heat_g)
kappa_2 = 4.5E-2  # Thermal diffusivity of steel in cm^2/s
chi = 1.0 - (reflectance / 100.0)
T_a = 20.0
pulse_length = 0.5
t_max = 2.01

x_tc_1 = 1.0
x_tc_2 = 2.0

# Kim Argonne National Lab 1965
def cp_ss304l(temperature):
    return 4.184 * (0.1122 + 3.222E-5 * temperature)


def rho_ss304l(temperature):
    return 7.9841 - 2.6506E-4 * temperature - 1.1580E-7 * temperature ** 2.0


def thermal_conductivity_ss304l(temperature):
    return 8.11E-2 + 1.618E-4 * temperature

k0_2 = thermal_conductivity_ss304l(T_a+273.15)
cp_2 = cp_ss304l(T_a+273.15)
rho_2 = rho_ss304l(T_a+273.15)
kappa_2 = k0_2 / (cp_2*rho_2)


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
    k = int(n / 6)
    k = k + 1 if k % 2 == 0 else k
    T = savgol_filter(measured_temperature, k, 3)
    dTdt = np.gradient(T, measured_time, edge_order=2)
    return savgol_filter(T + time_constant * dTdt, k - 2, 3)


if __name__ == "__main__":
    adi_data_dir = os.path.join(os.path.join(base_path, 'adi_data'))

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
    reflection_signal[irradiation_time_idx] = photodiode_voltage[reflective_signal_zero_idx + 2]

    print(f"Baseline signal: {photodiode_voltage[reflective_signal_zero_idx + 1]:.3f} V")
    thermometry.gain = int(photodiode_gain)
    print(f"Calibration Factor: {thermometry.calibration_factor}")
    thermometry.emissivity = emissivity

    photodiode_corrected = photodiode_voltage - reflection_signal
    time_pd_idx = photodiode_corrected > 0.0
    measurement_time_pd = measurement_time[time_pd_idx]
    photodiode_voltage_positive = photodiode_corrected[time_pd_idx]
    temperature_pd = thermometry.get_temperature(voltage=photodiode_voltage_positive) - 273.15

    if not load_model:
        hf_file = simulate_adi_temp(
            laser_power=qmax, r_holder=R, r_sample=R_sample,
            length=L, kappa_1=kappa_1, kappa_2=kappa_2,
            k0_1=k0_1, k0_2=k0_2, r_points=M, x_points=N,
            pulse_length=pulse_length, dt=dt, chi=chi, T_a=T_a, t_max=t_max,
            report_every=20, debug=True, holder_thickness_cm=holder_thickness,
            save_h5=True, beam_diameter=beam_diameter, x_tc_1=x_tc_1, x_tc_2=x_tc_2,
            emissivity=1.0
        )

        if not os.path.exists(adi_data_dir):
            os.makedirs(adi_data_dir)
        shutil.move(hf_file + '.h5', os.path.join(adi_data_dir, hf_file + '.h5'))
    else:
        hf_file = saved_h5

    dr = R / M
    dx = L / N

    probe_size_idx = int(probe_size * 0.1 / dx)
    probe_idx_delta = int(0.5 * probe_size_idx)

    x = dx * np.arange(0, N + 1)
    r = dr * np.arange(0, M + 1)
    msk_holder = r > R_sample
    idx_r = (np.abs(r - R_sample)).argmin()
    idx_pd_spot = (np.abs(r - thermography_spot_diameter * 0.5)).argmin()
    print(f'IR Thermography spot size: {thermography_spot_diameter * 10.0:.1f} mm')
    print(f'IDX of photodiode spot: {idx_pd_spot}, radius at index: {r[idx_pd_spot] * 10:.1f} mm')

    # Get the size of the time array
    elapsed_time = np.arange(0, t_max + dt, dt, dtype=np.float64)
    # The temperature at the surface of the rod closest to the light source
    tp1 = T_a * np.ones_like(elapsed_time)
    xp1 = x_tc_1
    idx_p1 = int(xp1 / dx)

    # The temperature at the surface of the rod farther from the light source
    tp2 = T_a * np.ones_like(elapsed_time)
    xp2 = x_tc_2
    idx_p2 = int(xp2 / dx)

    # Stefan-Boltzmann constant
    sb = 5.670374419E-12  # W cm^{-2} K^{-4}

    # The average temperature at the front surfacve
    t_front = T_a * np.ones_like(elapsed_time)
    # The temperature at the back surface
    t_back = T_a * np.ones_like(elapsed_time)
    # radiated power
    radiated_power = np.empty_like(t_back)
    hf_file_path = os.path.join(adi_data_dir, hf_file + '.h5')
    with h5py.File(hf_file_path, 'r') as hf:
        x = np.array(hf['data/x'])
        r = np.array(hf['data/r'])

    for i in range(len(tp1)):
        ds_name = f'data/T_{i:d}'
        with h5py.File(hf_file_path, 'r') as hf:
            u = np.array(hf.get(ds_name))
            tp1[i] = u[idx_r, idx_p1-probe_idx_delta:idx_p1+probe_idx_delta].mean()
            tp2[i] = u[idx_r, idx_p2]
            t_front[i] = u[0:idx_pd_spot, 0:3].mean()
            radiated_power[i] = sb * emissivity * ((t_front[i] + 273.15) ** 4.0 - (T_a + 273.15) ** 4.0)
            t_back[i] = u[0, -1]

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
    ax.plot(elapsed_time, t_front, label=f'Q={qmax * 0.01:.1f} MW/m$^{{\\mathregular{{2}}}}$, x=0.0 cm', c='tab:red',
            ls='-')
    ax.plot(elapsed_time, tp1, label=f'Q={qmax * 0.01:.1f} MW/m$^{{\\mathregular{{2}}}}$, x=1.0 cm', c='tab:green',
            ls='-')

    leg = ax.legend(
        loc='upper right', ncol=1, frameon=False
    )

    ax.tick_params(axis='y', right=True, zorder=10, which='both')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('3D Model')
    ax.set_xlim((0., 2.0))
    ax.set_ylim((0., 1600.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(400))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))

    fig.tight_layout()
    fig.savefig(os.path.join(base_path, 'adi_raw_fit.eps'), dpi=600)
    fig.savefig(os.path.join(base_path, 'adi_raw_fit.svg'), dpi=600)
    fig.savefig(os.path.join(base_path, 'adi_raw_fit.png'), dpi=600)
    print(f"Filename: {hf_file}")
    plt.show()
