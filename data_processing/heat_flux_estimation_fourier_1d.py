import platform
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
from matplotlib import ticker
import heat_flux_fourier as hff
from scipy.signal import savgol_filter
import matplotlib as mpl
import json
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import confidence as cf
from utils import get_experiment_params
import re

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration\results'
data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration\IR Thermography Calibration'

# data_file = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration\LT_GR001CC_150mT_1cm_100PCT_60GAIN 2022-04-26_1.csv'
data_file = r'LT_GR008G_6mTorr-contact-shield_100PCT_50GAIN 2022-05-04_1'


q0 = 3.2E3#4.7E3
time_constant = 1.681 # 0.256  # 2.1148
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
x_probe_1 = 1.0
R_sample = 0.5 * 1.288  #
sample_area = np.pi * (R_sample ** 2.0)


def model(t:np.ndarray, b:np.ndarray):
    x = np.array([0.0, x_probe_1])
    # k0 = b[1] * density_g * specific_heat_g
    # flux = b[0] * alpha / k0 / sample_area
    k0 = k0_1
    flux = b[0] * alpha / k0 / sample_area
    u = hff.get_ut(
        x=x, diffusion_time=t, rod_length=sample_length,
        diffusivity=kappa_1, #b[1],
        emission_time=pulse_length,
        flux=flux, T0=T_a
    )
    return u


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


if __name__ == "__main__":
    if platform.system() == 'Windows':
        data_path = r'\\?\\' + data_path
    # Get the experiment params
    experiment_params = get_experiment_params(relative_path=data_path, filename=data_file)
    photodiode_gain = experiment_params['Photodiode Gain']['value']
    laser_power_setting = experiment_params['Laser Power Setpoint']['value']
    sample_name = experiment_params['Sample Name']['value']
    # Get the photodiode temperature
    surface_temperature_path = os.path.join(
        data_path, 'temperature_data', f'{sample_name.upper()}_{laser_power_setting}'
    )

    surface_temperature_df = pd.read_csv(os.path.join(surface_temperature_path, f'{data_file}_surface_temp.csv'),
                                         comment='#').apply(pd.to_numeric)
    time_pd = surface_temperature_df['Time (s)'].values
    temperature_pd = surface_temperature_df['Surface Temperature (°C)'].values

    tc_csv = os.path.join(data_path, data_file + '_tcdata.csv')
    tc_df = pd.read_csv(tc_csv, comment='#').apply(pd.to_numeric)
    time_tc = tc_df['Time (s)'].values
    temperature_tc1 = tc_df['TC1 (C)'].values
    temperature_tc1_corrected = correct_thermocouple_response(
        measured_time=time_tc, measured_temperature=temperature_tc1, tau=time_constant
    )

    tol = 0.25
    tc_0 = temperature_tc1[0:5].mean()
    print(f'TC[t=0]: {tc_0:4.2f} °C')
    msk_onset = (temperature_tc1 - tc_0) > tol
    time_onset = time_tc[msk_onset]
    time_onset = time_onset[0]
    idx_onset = (np.abs(time_tc - time_onset)).argmin() - 20
    # print(idx_onset)

    time_tc = time_tc[idx_onset::]
    time_tc -= time_tc.min()
    temperature_tc1 = temperature_tc1[idx_onset::]
    temperature_tc1_corrected = temperature_tc1_corrected[idx_onset::]
    tc_time_positive_idx = time_tc > 0
    time_tc = time_tc[tc_time_positive_idx]
    temperature_tc1 = temperature_tc1[tc_time_positive_idx]
    temperature_tc1_corrected = temperature_tc1_corrected[tc_time_positive_idx]



    n = len(time_tc)
    T_a = temperature_tc1[0]
    print(f'T_a = {T_a:.2f} °C')

    # b_guess = np.array([time_constant])
    # all_tol = np.finfo(np.float64).eps
    # f = interp1d(time_tc, temperature_tc1)
    # y = f(time_tc)


    # def fobj_tc(beta: np.ndarray, tc: np.ndarray, tc_time: np.ndarray) -> np.ndarray:
    #     return correct_thermocouple_response(tc, tc_time, beta[0]) - y
    #
    # res = least_squares(
    #     fobj_tc, b_guess, args=(temperature_tc1, time_tc),
    #     bounds=(1E-5, 1E2),
    #     jac='3-point',
    #     xtol=all_tol,
    #     ftol=all_tol,
    #     gtol=all_tol,
    #     max_nfev=10000 * n,
    #     loss='soft_l1', f_scale=0.1,
    #     verbose=2
    # )
    # popt = res.x
    # print(f'time constant: {popt[0]}')


    f = interp1d(time_pd, temperature_pd)
    temperature_interp_pd = np.ones_like(time_tc) * T_a
    msk_pd = (time_pd.min() <= time_tc) & (time_tc <= time_pd.max())
    temperature_interp_pd[msk_pd] = f(time_tc[msk_pd])
    u_exp = np.vstack((temperature_interp_pd, temperature_tc1_corrected)).T

    def func(diffusion_time, b):
        return model(diffusion_time, b).flatten()


    def fobj(b, t, u_e):
        u_model = model(t, b)
        u1 = u_model[:, 0]
        u1 = u1[msk_pd]
        u1_exp = u_e[:, 0]
        u1_exp = u1_exp[msk_pd]
        r = np.hstack((u1 - u1_exp, u_model[:, 1] - u_e[:, 1]))
        return r

    all_tol = (np.finfo(np.float64).eps)**0.5
    # b0 = np.array([q0, kappa_1])
    b0 = np.array([q0])
    res = least_squares(
        fobj, b0,
        loss='linear', f_scale=0.1,
        jac='3-point',
        args=(time_tc, u_exp),
        # bounds=([200, 1E-5], [1E4, 10]),
        bounds=([200], [1E5]),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        max_nfev=10000 * len(u_exp.flatten()),
        x_scale='jac',
        verbose=2
    )

    popt = res.x
    print(popt)

    pcov = cf.get_pcov(res)
    ci = cf.confint(n=n, pars=popt, pcov=pcov)
    tpred = np.linspace(0, time_tc.max(), 1000)
    # u_pred, lpb, upb = cf.predint(x=tpred, xd=time_pd, yd=temperature_pd, func=func, res=res)
    u_pred = model(tpred, popt)

    new_shape = (int(0.5 * u_pred.size), 2)
    u_pred = u_pred.reshape(new_shape)

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1)  # , constrained_layout=True)
    fig.set_size_inches(4.5, 3.5)
    ax.plot(
        time_pd, temperature_pd, ls='none', label=f'Photodiode',
        c='tab:red', marker='o', fillstyle='none'
    )
    # ax.plot(time_tc, temperature_tc1, label='T(x=1.0 cm)', ls='none', color='tab:green', marker='s', fillstyle='none')
    ax.plot(time_tc, temperature_tc1_corrected, label=f'T(x=1.0 cm) (corrected)', ls='none',
            color='tab:olive', marker='^', fillstyle='none')
    ax.plot(tpred, u_pred[:, 0], label=f'Q={popt[0] * 0.01:.1f} MW/m$^{{\\mathregular{{2}}}}$, x=0.0 cm', c='tab:red',
            ls='-')
    ax.plot(tpred, u_pred[:, 1], label=f'Q={popt[0] * 0.01:.1f} MW/m$^{{\\mathregular{{2}}}}$, x={x_probe_1:.1f} cm', c='tab:green',
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
    fig.savefig(os.path.join(base_path, f'{data_file}_fourier_fit.eps'), dpi=600)
    fig.savefig(os.path.join(base_path, f'{data_file}_fourier_fourier_fit.svg'), dpi=600)
    fig.savefig(os.path.join(base_path, f'{data_file}_fourier_raw_fit.png'), dpi=600)
    plt.show()
