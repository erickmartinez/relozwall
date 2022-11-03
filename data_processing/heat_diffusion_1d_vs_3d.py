import h5py
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
from matplotlib import ticker
from heat_flux_adi import simulate_adi_temp
import heat_flux_fourier as hff
import matplotlib as mpl
import json
import platform
import shutil

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\heat equation\model_comparison'

incident_power = 3.18E3  # kW
time_constant = 1.681  # 0.256  #1.681  # 2.1148
reflectance = 40.4
pulse_length = 0.5
t_max = 2.01
sample_length = 5.0
x_probe_1 = 1.0

load_model = False
saved_h5 = 'ADI_k1_7.41E-01_chi_0.60_P3.37E+03'

M = 200  # number of intervals in r
N = 400  # the number of intervals in x
R = 1.27  # The radius of the cylinder in cm
R_sample = 0.5 * 1.288  #
holder_thickness = 1.27
dt = 1.0E-3
beam_diameter = 1.5 * 0.8165  # cm
probe_size = 2.0  # mm
sample_area = np.pi * (R_sample ** 2.0)

thermography_spot_diameter = 0.8  # cm

# density_g = 1.76  # g / cm^3 # GR008G
# density_g = 1.81 # g / cm^3 # GR001CC
density_g = 1.698  # measured
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
k0_1 = 85E-2  # W / (cm K) https://www.graphitestore.com/core/media/media.nl?id=6310&c=4343521&h=Tz5uoWvr-nhJ13GL1b1lG8HrmYUqV1M_1bOTFQ2MMuiQapxt # GR001C
# k0_1 = 130E-2  # W / (cm K) https://www.graphitestore.com/core/media/media.nl?id=7164&c=4343521&h=8qpl24Kn0sh2rXtzPvd5WxQIPQumdO8SE5m3VRfVBFvLJZtj # GR008G
k0_2 = 16.2E-2  # W / (cm K)

# kappa_1 = 1.11 # Thermal diffusivity of copper in cm^2/s
# kappa_1 = 25E-2 # Thermal diffusivity of polycrystalline graphite in cm^2/s
kappa_1 = k0_1 / (density_g * specific_heat_g)
alpha = 1.0 - (reflectance / 100.0)
emissivity = 1.0 - (36.9 / 100)
T_a = 20.0


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


def model_1d(t: np.ndarray, b: np.ndarray):
    x = np.array([0.0, x_probe_1])
    k0 = b[1] * density_g * specific_heat_g
    flux = b[0] * alpha / k0
    u = hff.get_ut(
        x=x, diffusion_time=t, rod_length=sample_length, diffusivity=b[1], emission_time=pulse_length,
        flux=flux, T0=T_a
    )
    return u


if __name__ == '__main__':
    chi = 1.0 - 0.01 * reflectance
    incident_irradiance = incident_power / sample_area
    gaussian_beam_aperture_factor = (1.0 - np.exp(-2.0 * (2.0 * R_sample / beam_diameter) ** 2.0))
    print(f'Aperture factor: {gaussian_beam_aperture_factor:.3E}')
    ps = incident_power / gaussian_beam_aperture_factor
    print(f'Incident beam power: {ps:.3E} W/cm^2')
    print(f'Gaussian beam total power: {incident_power:.3E} W/cm^2')

    # if platform.system() == 'Windows':
    #     base_path = r'\\?\\' + base_path

    adi_data_dir = os.path.join(os.path.join(base_path, 'adi_data'))

    if not load_model:
        hf_file = simulate_adi_temp(
            laser_power=ps, r_holder=R, r_sample=R_sample,
            length=sample_length, kappa_1=kappa_1, kappa_2=kappa_2,
            k0_1=k0_1, k0_2=k0_2, r_points=M, x_points=N,
            pulse_length=pulse_length, dt=dt, chi=chi, T_a=T_a, t_max=t_max,
            report_every=20, debug=True, holder_thickness_cm=holder_thickness,
            save_h5=True, beam_diameter=beam_diameter, x_tc_1=x_probe_1, x_tc_2=sample_length,
            emissivity=1.0
        )
        if not os.path.exists(adi_data_dir):
            os.makedirs(adi_data_dir)
        shutil.move(hf_file + '.h5', os.path.join(adi_data_dir, hf_file + '.h5'))
    else:
        hf_file = saved_h5

    time_sim = np.arange(0, t_max + dt, dt, dtype=np.float64)
    u_1d = model_1d(time_sim, np.array([incident_irradiance, kappa_1]))

    temperature_3d_p1 = T_a * np.ones_like(time_sim)
    dx = sample_length / N
    dr = R / M
    idx_p1 = int(x_probe_1 / dx)
    r = dr * np.arange(0, M + 1)
    idx_r = (np.abs(r - R_sample)).argmin()
    idx_pd_spot = (np.abs(r - thermography_spot_diameter * 0.5)).argmin()
    # The average temperature at the front surfacve
    temperature_3d_surface = T_a * np.ones_like(time_sim)

    hf_file_path = os.path.join(adi_data_dir, hf_file + '.h5')
    with h5py.File(hf_file_path, 'r') as hf:
        x = np.array(hf['data/x'])
        r = np.array(hf['data/r'])

    for i in range(len(time_sim)):
        ds_name = f'data/T_{i:d}'
        with h5py.File(hf_file_path, 'r') as hf:
            u = np.array(hf.get(ds_name))
            temperature_3d_p1[i] = u[idx_r, idx_p1].mean()
            temperature_3d_surface[i] = u[0:idx_r, 0].mean()

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1)  # , constrained_layout=True)
    fig.set_size_inches(4.5, 3.5)

    me = 80
    ms = 5
    ax.plot(
        time_sim, temperature_3d_surface, ls='-', label=f'3D model (surface)',
        c='tab:red', marker='o', markevery=(0, me), fillstyle='full', ms=ms
    )

    ax.plot(
        time_sim, u_1d[:, 0], ls='-', label=f'1D model (x = {x_probe_1:.1f} cm)',
        c='tab:purple', marker='^', markevery=(30, me), fillstyle='full', ms=ms
    )

    ax.plot(
        time_sim, temperature_3d_p1, ls='-', label=f'3D model (surface)',
        c='tab:blue', marker='s', markevery=(0, me), fillstyle='full', ms=ms
    )

    ax.plot(
        time_sim, u_1d[:, 1], ls='-', label=f'1D model (x = {x_probe_1:.1f} cm)',
        c='cyan', marker='v', markevery=(30, me), fillstyle='full', ms=ms
    )

    ax.tick_params(axis='y', right=True, zorder=10, which='both')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')

    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 2000.0)
    ax.legend(
        loc='best', frameon=True, prop={'size': 9}
    )

    ax.set_title(
        f'Incident irradiance: {incident_irradiance * 1E-2:.1f} MW/m$^{{\\mathregular{{2}}}}$', fontweight='regular'
    )

    file_tag = f'model_comparison_{incident_irradiance*0.01:02.1f}_MWpm2'

    fig.tight_layout()
    fig.savefig(
        os.path.join(base_path, file_tag + '.png'), dpi=600
    )

    plt.show()
