import h5py
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
from utils import get_experiment_params
from matplotlib import ticker

from heat_flux_adi import simulate_adi_temp
from scipy.signal import savgol_filter
import matplotlib as mpl
import json
# import ir_thermography.thermometry as irt
import re
import shutil
import platform
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares
from scipy.linalg import svd
from scipy import interpolate

data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration\results'
data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration\IR Thermography Calibration'
# data_file = 'LT_GR008G_6mTorr-contact-shield_100PCT_50GAIN 2022-05-04_1'
data_file = 'LT_GR008G_100mTorr_100PCT_50GAIN 2022-05-03_1'
# data_file = 'LT_GR008G_4mTorr_contact_shield_020PCT_70GAIN 2022-05-05_1'
load_model = True
saved_h5 = 'ADI_k1_7.41E-01_chi_0.60_P4.70E+03'
# saved_h5 = 'ADI_k1_1.09E+00_chi_0.60_P3.61E+03'
# saved_h5 = 'ADI_k1_7.41E-01_chi_0.60_P1.20E+03'

time_constant = 1.5#2.1148
# time_constant = 0.5
qmax = 4.7E3
# qmax = 5.55E3 * 0.65
# qmax = 4.80E3 * 0.25
emissivity = 1.0 - (36.9 / 100)
reflectance = 40.4

M = 200  # number of intervals in r
N = 400  # the number of intervals in x
R = 1.27  # The radius of the cylinder in cm
R_sample = 0.5 * 1.288  #
L = 5.0  # the length of the cylinder
holder_thickness = 1.27
dt = 1.0E-3
beam_diameter = 1.5 * 0.8165  # cm
probe_size = 2.0  # mm

thermography_spot_diameter = 0.8  # cm
# thermography_spot_diameter = R_sample
# density_g = 1.76  # g / cm^3 # GR008G
# density_g = 1.81 # g / cm^3 # GR001CC
density_g = 1.698
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
k0_1 = 85E-2 # W / (cm K) https://www.graphitestore.com/core/media/media.nl?id=6310&c=4343521&h=Tz5uoWvr-nhJ13GL1b1lG8HrmYUqV1M_1bOTFQ2MMuiQapxt # GR001C
# k0_1 = 130E-2  # W / (cm K) https://www.graphitestore.com/core/media/media.nl?id=7164&c=4343521&h=8qpl24Kn0sh2rXtzPvd5WxQIPQumdO8SE5m3VRfVBFvLJZtj # GR008G
# k0_1 = 200E-2
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


k0_2 = thermal_conductivity_ss304l(T_a + 273.15)
cp_2 = cp_ss304l(T_a + 273.15)
rho_2 = rho_ss304l(T_a + 273.15)
kappa_2 = k0_2 / (cp_2 * rho_2)


def correct_thermocouple_response(measured_temperature, measured_time, tau):
    n = len(measured_time)
    k = int(n / 15)
    k = k + 1 if k % 2 == 0 else k
    k = max(k, 5)
    # T = savgol_filter(measured_temperature, k, 3)
    # dTdt = np.gradient(T, measured_time, edge_order=2)
    delta = measured_time[1] - measured_time[0]
    dTdt = savgol_filter(x=measured_temperature, window_length=k, polyorder=4, deriv=1, delta=delta)
    # dTdt = savgol_filter(dTdt, k - 2, 3)
    r = measured_temperature + tau * dTdt
    return savgol_filter(r, k - 4, 3)


def fobj(beta: np.ndarray, tc: np.ndarray, tc_time: np.ndarray, y: np.ndarray) -> np.ndarray:
    return correct_thermocouple_response(tc, tc_time, beta[0]) - y


def get_pcov(res: OptimizeResult) -> np.ndarray:
    popt = res.x
    ysize = len(res.fun)
    cost = 2 * res.cost  # res.cost is half sum of squares!
    s_sq = cost / (ysize - popt.size)

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s ** 2, VT)
    pcov = pcov * s_sq

    if pcov is None:
        # indeterminate covariance
        print('Failed estimating pcov')
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(np.inf)
    return pcov


if __name__ == "__main__":
    adi_data_dir = os.path.join(os.path.join(data_path, 'adi_data'))

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    data_filetag = data_file
    print(data_filetag)
    experiment_params = get_experiment_params(relative_path=data_path, filename=data_filetag)
    photodiode_gain = experiment_params['Photodiode Gain']['value']
    laser_power_setting = experiment_params['Laser Power Setpoint']['value']
    sample_name = experiment_params['Sample Name']['value']

    temperature_path = os.path.join(data_path, 'temperature_data', f'{sample_name.upper()}_{laser_power_setting}')
    if platform.system() == 'Windows':
        temperature_path = r'\\?\\' + temperature_path

    # thermometry = irt.PDThermometer()

    surface_temperature_df = pd.read_csv(os.path.join(temperature_path, f'{data_file}_surface_temp.csv'),
                                         comment='#').apply(pd.to_numeric)
    measurement_time = surface_temperature_df['Time (s)'].values
    surface_temperature = surface_temperature_df['Surface Temperature (°C)'].values



    # thermocouple_df = pd.read_csv(os.path.join(temperature_path, f'{data_file}_corrected_temperature_data.csv'), comment='#').apply(pd.to_numeric)
    # tc_time = thermocouple_df['Time (s)']
    # temperature_a = thermocouple_df['Temperature A (C)'].values

    # thermometry.emissivity = emissivity

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
            tp1[i] = u[idx_r, idx_p1 - probe_idx_delta:idx_p1 + probe_idx_delta].mean()
            tp2[i] = u[idx_r, idx_p2]
            t_front[i] = u[0:idx_pd_spot, 0:3].mean()
            radiated_power[i] = sb * emissivity * ((t_front[i] + 273.15) ** 4.0 - (T_a + 273.15) ** 4.0)
            t_back[i] = u[0, -1]

    tc_csv = os.path.join(data_path, data_filetag + '_tcdata.csv')
    tc_df = pd.read_csv(tc_csv, comment='#').apply(pd.to_numeric)
    tc_time = tc_df['Time (s)'].values
    temperature_a = tc_df['TC1 (C)'].values

    b_guess = np.array([time_constant])
    all_tol = np.finfo(np.float64).eps
    f = interpolate.interp1d(elapsed_time, tp1)
    msk_time = tc_time <= 2.0
    print('tc_time', tc_time)
    print('elapsed_time', elapsed_time)
    y = f(tc_time[msk_time])
    res = least_squares(
        fobj, b_guess, args=(temperature_a[msk_time], tc_time[msk_time], y),
        jac='3-point',
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        max_nfev=10000 * len(tc_time),
        loss='soft_l1', f_scale=0.1,
        verbose=2
    )
    popt = res.x
    print(f'time constant: {popt[0]}')

    ta_corrected = correct_thermocouple_response(
        measured_time=tc_time, measured_temperature=temperature_a, tau=time_constant
    )

    tol = 0.25
    tc_0 = temperature_a[0:5].mean()
    print(f'TC[t=0]: {tc_0:4.2f} °C')
    msk_onset = (temperature_a - tc_0) > tol
    time_onset = tc_time[msk_onset]
    time_onset = time_onset[0]
    idx_onset = (np.abs(tc_time - time_onset)).argmin() - 20
    # print(idx_onset)

    tc_time = tc_time[idx_onset::]
    tc_time -= tc_time.min()
    temperature_a = temperature_a[idx_onset::]
    ta_corrected = ta_corrected[idx_onset::]
    tc_time_positive_idx = tc_time > 0
    tc_time = tc_time[tc_time_positive_idx]
    temperature_a = temperature_a[tc_time_positive_idx]
    ta_corrected = ta_corrected[tc_time_positive_idx]

    fig, ax = plt.subplots(ncols=1)  # , constrained_layout=True)
    fig.set_size_inches(5.0, 3.5)

    p0 = 0.5 * np.pi * qmax * (0.5 * beam_diameter) ** 2.0
    # estimated_power_density = p0 * (1.0 - np.exp(-2.0 * (2.0 * R_sample / beam_diameter) ** 2.0))
    estimated_power_density = qmax * (1.0 - np.exp(-2.0 * (2.0 * R_sample / beam_diameter) ** 2.0)) / (np.pi * R_sample ** 2.0)
    ax.plot(
        measurement_time, surface_temperature, ls='none', label=f'Photodiode',
        c='tab:red', marker='o', fillstyle='none'
    )
    ax.plot(tc_time, temperature_a, label='T(x=1.0 cm)', ls='none', color='tab:green', marker='s', fillstyle='none')
    ax.plot(tc_time, ta_corrected, label=f'T(x=1.0 cm) (corrected)', ls='none',
            color='tab:olive', marker='^', fillstyle='none')
    ax.plot(elapsed_time, t_front,
            label=f'Fit', c='tab:red',
            ls='-')
    ax.plot(elapsed_time, tp1, label='Fit',
            c='tab:green',
            ls='-')

    ax.set_title(
        f'$P=${qmax / 1000.0:.1f} kW, $I=${estimated_power_density * 0.01:.1f} MW/m$^{{\\mathregular{{2}}}}$, x=1.0 cm',
        fontweight='regular'
    )

    leg = ax.legend(
        loc='upper right', ncol=1, frameon=False
    )

    ax.tick_params(axis='y', right=True, zorder=10, which='both')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    # ax.set_title(f'3D Model, {laser_power_setting} % Power')
    ax.set_xlim((0., 2.0))
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(400))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))

    fig.tight_layout()
    fig.savefig(os.path.join(data_path, f'{data_filetag}_adi_raw_fit.eps'), dpi=600)
    fig.savefig(os.path.join(data_path, f'{data_filetag}_adi_raw_fit.svg'), dpi=600)
    fig.savefig(os.path.join(data_path, f'{data_filetag}_adi_raw_fit.png'), dpi=600)
    print(f"Filename: {hf_file}")
    plt.show()
