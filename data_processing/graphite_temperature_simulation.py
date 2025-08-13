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
import platform
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares
from scipy.linalg import svd
from scipy import interpolate

data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\IR_VS_POWER\graphite'
filetag = 'graphite_simulated_surface_temperature'
load_model = True
saved_h5 = 'ADI_k1_7.41E-01_chi_0.60_P4.80E+03'

time_constant = 1.5#2.1148
# time_constant = 0.5
# qmax = 5.55E3 * 0.65
qmax = 4.80E3 * 1.0
emissivity = 1.0 - (36.9 / 100)
reflectance = 40.4

M = 200  # number of intervals in r
N = 400  # the number of intervals in x
R = 1.27*0.75  # The radius of the cylinder in cm
R_sample = 0.5 * (3.0/8.0) * 2.54 #1.288  #
L = 2.5  # the length of the cylinder
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
pulse_length = 2.0
t_max = 4.01

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




    fig, ax = plt.subplots(ncols=1)  # , constrained_layout=True)
    fig.set_size_inches(5.0, 3.5)

    p0 = 0.5 * np.pi * qmax * (0.5 * beam_diameter) ** 2.0
    estimated_power_density = p0 * (1.0 - np.exp(-2.0 * (2.0 * R_sample / beam_diameter) ** 2.0))

    ax.plot(elapsed_time, t_front,
            label=f'Q={estimated_power_density * 0.01:.1f} MW/m$^{{\\mathregular{{2}}}}$, x=0.0 cm', c='tab:red',
            ls='-')


    leg = ax.legend(
        loc='best', ncol=1, frameon=False
    )

    ax.tick_params(axis='y', right=True, zorder=10, which='both')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlim((0., 4.0))
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(400))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))

    fig.tight_layout()
    fig.savefig(os.path.join(data_path, f'{filetag}_adi_raw_fit.png'), dpi=600)
    print(f"Filename: {hf_file}")
    plt.show()
