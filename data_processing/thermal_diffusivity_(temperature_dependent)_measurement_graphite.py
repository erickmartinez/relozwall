"""
This script uses leasts squares to determine the thermal diffusivity of a  rod by
fitting the temperature measured at two points along the axis of the rod.

It is assumed that the rod is under vacuum and that it receives a pulse of energy
for certain amount of time before letting the temperature relax.

A Fourier solution to the heat equation is used with the thermal diffusivity and the
flux due to the pulse as free parameters.

Erick R Martinez Loran
erm013@ucsd.edu

--------------

Copyright 2022 Erick R Martinez Loran

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib import patches
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
import confidence as cf
import utils
import json
import os
import logging
import sys
from utils import specific_heat_of_graphite

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\thermal_conductivity\graphite'
csv_file = 'LT_GRAPHITE_ROD_015PCT_2022-12-14_1.csv'
tc_time_constant = 0.256
density_g = 1.661

# pulse_length = 8.0
# sample_length_cm = 2.7
# x_probe_1 = 0.65
# x_probe_2 = sample_length_cm

pulse_length = 10.0
sample_length_cm = 4.922
x_probe_1 = 5 / 10.0
# x_probe_2 = 5.04-(0.275*2.54)
x_probe_2 = 14.0 / 10.0

dx = 0.5  # 0.2 #0.5 * 8.04 / 10

debug = False
N = 100

by_p2 = np.power(np.pi, -2.0)

a0 = 0.419
a1 = 3.603E-3
a2 = 9.803E-2


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


def thermal_diffusivity(temperature_c, a0, a1):
    temperature = temperature_c + 273.15
    cp = specific_heat_of_graphite(temperature)
    k = 1.0 / (a0 * temperature + a1)
    return k / cp / density_g  # a0*np.exp(-a1*temperature_c) + a2


def u_1(x: np.ndarray, diffusion_time: np.ndarray, rod_length: float, diffusivity: float, flux: float):
    L = rod_length
    F = flux
    return 0.5 * F * (x ** 2.0) / L - (F * x) + diffusivity * (flux / L) * diffusion_time


def u_laser_on(x: np.ndarray, diffusion_time: np.ndarray, rod_length: float, a_0, a_1, flux: float,
               T0: float):
    L = rod_length
    # x = np.hstack([x, rod_length])
    u = T0 * np.ones((diffusion_time.size, x.size))
    a = np.pi / L
    b = 2.0 * flux * L * by_p2

    for i, t in zip(range(1, diffusion_time.size), diffusion_time[1:]):
        s0 = flux * L / 3.0
        s = 0
        diffusivity = thermal_diffusivity(u[i - 1], a_0, a_1)
        for n in range(1, N + 1):
            an = 1.0 / (n * n)
            s += an * np.cos(n * a * x) * np.exp(-diffusivity * t * (n * a) ** 2.0)
        u[i] += s0 - b * s + u_1(x, t, rod_length, diffusivity, flux)
        if debug:
            logger.info(f'[on] - ({i:4d}) T[0,t={t:>6.3f}s] = {u[i, 0]:>6.3E}, T[L,t={t:>6.3f}s] = {u[i, -1]:>6.3E}')
    return u # [:, 0:-1]


def get_an(n: int, rod_length: float, diffusivity: float, emission_time: float, flux: float, T0: float):
    L = rod_length
    if n == 0:
        return T0 + diffusivity * flux * emission_time / L
    return (2.0 * flux * L / ((n * np.pi) ** 2.0)) * (
            1.0 - np.exp(-diffusivity * emission_time * (n * np.pi / L) ** 2.0))


def get_ut(x: np.ndarray, diffusion_time: np.ndarray, rod_length: float, a_0, a_1, emission_time: float,
           flux: float, T0: float):
    L = rod_length
    # x = np.hstack([x, rod_length])
    u = np.zeros((diffusion_time.size, x.size))
    msk_on = diffusion_time <= emission_time
    msk_off = diffusion_time > emission_time
    time_off = diffusion_time[msk_off]
    idx_off = len(u[msk_on])
    u_on = u_laser_on(x=x, diffusion_time=diffusion_time[msk_on], rod_length=L, a_0=a_0, a_1=a_1, flux=flux,
                      T0=T0)
    u[msk_on, :] = u_on.copy()
    # After the laser pulse consider the solution of the heat equation for a 1D rod with insulated ends
    for i, ti in enumerate(time_off):
        diffusivity = thermal_diffusivity(u[i + idx_off - 1], a_0, a_1)
        for n in range(N + 1):
            arg = ((n * np.pi / L) ** 2.0) * diffusivity * (
                    ti - emission_time)  # (diffusion_time[i+idx_off]-pulse_length)
            a_n = get_an(n, rod_length=L, diffusivity=diffusivity, emission_time=emission_time, flux=flux, T0=T0)
            if n == 0:
                u[i + idx_off] = a_n
            else:
                u[i + idx_off] += a_n * np.cos(n * np.pi * x / L) * np.exp(-arg)
        if debug:
            logger.info(
                f'[off] - ({i + idx_off:3d}) T[0,t={ti:>6.3f}s] = {u[i + idx_off, 0]:>6.3E}, T[L,t={ti:>6.3f}s] = {u[i + idx_off, -1]:>6.3E}')

    return u # [:, 0:-1]


def get_mean_ut(x_center: np.ndarray, diffusion_time: np.ndarray, rod_length: float, diffusivity: float,
                emission_time: float,
                flux: float, T0: float, dx_probe):
    x1 = np.linspace(max(0, x_center[0] - dx_probe), min(x_center[0] + dx_probe, rod_length), 10)
    x2 = np.linspace(max(0, x_center[1] - dx_probe), min(x_center[1] + dx_probe, rod_length), 10)
    u1 = get_ut(x1, diffusion_time, rod_length, diffusivity, emission_time, flux, T0).mean(axis=1)
    u2 = get_ut(x2, diffusion_time, rod_length, diffusivity, emission_time, flux, T0).mean(axis=1)

    return np.stack([u1, u2]).T


def fobj(b, diffusion_time, temperature, x: np.ndarray, rod_length: float,
         emission_time: float, T0: float):
    r = get_ut(x=x, diffusion_time=diffusion_time, rod_length=rod_length, a_0=a0,
               a_1=a1, a_2=a2, emission_time=emission_time, flux=b[0], T0=T0) - temperature
    return r.flatten()


def fobj_log(b, diffusion_time, temperature, x: np.ndarray, rod_length: float,
             emission_time: float, T0: float):
    r = get_ut(x=np.array([10 ** b[1], 10 ** b[2]]), diffusion_time=diffusion_time, rod_length=rod_length, a_0=a0,
               a_1=a1, a_2=a2, emission_time=emission_time, flux=10 ** b[0], T0=T0) - temperature
    return r.flatten()


def fobj2(b, diffusion_time, temperature, x: np.ndarray, rod_length: float,
          emission_time: float, T0: float):
    r = get_ut(x=x, diffusion_time=diffusion_time, rod_length=rod_length, a_0=b[1], a_1=b[2],
               emission_time=emission_time, flux=b[0], T0=T0) - temperature
    return r.flatten()


def fobj2_log(b, diffusion_time, temperature, x: np.ndarray, rod_length: float,
              emission_time: float, T0: float):
    r = get_ut(x=x, diffusion_time=diffusion_time, rod_length=rod_length, a_0=10 ** b[1], a_1=10 ** b[2],
               a_2=10 ** b[3],
               emission_time=emission_time, flux=b[0], T0=T0) - temperature
    return r.flatten()


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


if __name__ == '__main__':
    file_tag = os.path.splitext(csv_file)[0]
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    log_file = os.path.join(base_dir, file_tag + '_fitting.log')
    ch = logging.StreamHandler()
    fh = logging.FileHandler(log_file)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setLevel(logging.DEBUG)
    fh.setLevel(logging.DEBUG)
    # ch.setFormatter(c_format)
    fh.setFormatter(f_format)
    logger.addHandler(ch)
    logger.addHandler(fh)
    # old_stdout = sys.stdout
    # sys.stdout = logger

    x = np.array([x_probe_1, x_probe_2])
    # Load the data from csv file
    laser_test_df = pd.read_csv(os.path.join(base_dir, csv_file), comment='#').apply(pd.to_numeric)
    time_s = laser_test_df['Measurement Time (s)'].values
    laser_output_power = laser_test_df['Laser output peak power (W)'].values

    tc_df = pd.read_csv(os.path.join(base_dir, file_tag + '_tcdata.csv')).apply(pd.to_numeric)
    time_s = tc_df['Time (s)'].values
    tc1 = tc_df['TC1 (C)'].values
    tc2 = tc_df['TC2 (C)'].values
    # dt12 = tc1[0] - tc2[0]
    # tc1 -= dt12

    temp_t0 = tc1[0]

    tc1_corrected = correct_thermocouple_response(
        measured_temperature=tc1,
        measured_time=time_s, tau=tc_time_constant
    )

    tc2_corrected = correct_thermocouple_response(
        measured_temperature=tc2,
        measured_time=time_s, tau=tc_time_constant
    )

    msk_delay = (tc1 - temp_t0) >= 0.25
    t_msk = time_s[msk_delay]
    t0 = t_msk[0]
    idx_t0 = (np.abs(time_s - t0)).argmin()
    t0 = time_s[idx_t0]
    logger.info(f't0: {t0:.3f} s')
    msk_delay = time_s >= t0
    time_s = time_s[msk_delay]
    tc1 = tc1[msk_delay]
    tc2 = tc2[msk_delay]
    tc1_corrected = tc1_corrected[msk_delay]
    tc2_corrected = tc2_corrected[msk_delay]
    n = tc1.size
    time_s -= time_s.min()
    temp_t0 = tc1[0]

    msk_laser_on = laser_output_power > 0.0
    laser_mean_power = laser_output_power[msk_laser_on].mean()
    logger.info(f'Laser mean power: {laser_mean_power:.1f} W')

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


    def func(diffusion_time, b):
        r = get_ut(x=x, diffusion_time=diffusion_time, rod_length=sample_length_cm, a_0=b[1],
                   a_1=b[2], emission_time=pulse_length, flux=b[0], T0=tc2[0])
        return r.flatten()


    u_exp = np.vstack((tc1, tc2)).T
    all_tol = np.finfo(np.float64).eps
    # b0_log = np.array([100, 0.5, 1.46])
    b0 = np.array([400, 1E-3, 0.9])
    x = np.array([x_probe_1, x_probe_2])
    # x = np.array([0.39, 1.24])
    # b0 = np.array([2000])
    # logger.info('Starting least squares (log)')
    # res_log = least_squares(
    #     fobj2_log, np.log10(b0_log),
    #     loss='soft_l1', f_scale=0.1,
    #     jac='2-point',
    #     # args=(time_s, u_exp, sample_length_cm, pulse_length, tc1[0]),
    #     args=(time_s, u_exp, x, sample_length_cm, pulse_length, tc1[0]),
    #     bounds=([-10, -5, -10, -10], [3, 1, -1, 0]),
    #     # bounds=([-5, -1, 0], [10, np.log10(sample_length_cm), np.log10(sample_length_cm)]),
    #     xtol=all_tol**0.3,
    #     ftol=all_tol**0.3,
    #     gtol=all_tol**0.3,
    #     max_nfev=10000 * len(u_exp.flatten()),
    #     x_scale='jac',
    #     verbose=2
    # )
    # logger.info(f'res_log.x: {10**res_log.x}')
    # logger.info('Starting least squares (linear)')
    # popt_log = res_log.x
    # x = 10**popt_log[1:3]
    res = least_squares(
        fobj2, b0,
        # loss='soft_l1', f_scale=0.1,
        jac='3-point',
        # args=(time_s, u_exp, sample_length_cm, pulse_length, tc1[0]),
        args=(time_s, u_exp, x, sample_length_cm, pulse_length, tc1[0]),
        bounds=([1E-10, 1E-20, 1E-20], [10000, 1E10, 1E10]),
        # bounds=([1E-5], [np.inf]),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        max_nfev=10000 * len(u_exp.flatten()),
        x_scale='jac',
        verbose=2
    )

    popt = res.x
    print(f'popt: {popt}')
    flux_fit = popt[0]
    a_fit = thermal_diffusivity(tc1[0], popt[1], popt[2])
    pcov = cf.get_pcov(res)
    ci = cf.confint(n=n, pars=popt, pcov=pcov)
    tpred = np.linspace(0, time_s.max(), 1000)
    ypred, lpb, upb = cf.predint(x=tpred, xd=time_s, yd=u_exp, func=func, res=res)
    temperature_fit = get_ut(x=x, diffusion_time=time_s, rod_length=sample_length_cm, a_0=a0,
                             a_1=a1, emission_time=pulse_length, flux=popt[0], T0=tc2_corrected[0])

    output_df = pd.DataFrame(data={
        'Time (s)': time_s,
        'TC1 (C)': tc1,
        'TC2 (C)': tc2,
        'TC1 corrected (C)': tc1_corrected,
        'TC2 corrected (C)': tc2_corrected,
        'TC1 fit (C)': temperature_fit[:, 0],
        'TC2 fit (C)': temperature_fit[:, 1],
    })

    output_df.to_csv(os.path.join(base_dir, os.path.splitext(csv_file)[0] + '_corrected.csv'), index=False)

    new_shape = (int(0.5 * ypred.size), 2)
    u_fit = ypred.reshape(new_shape)
    lpb, upb = lpb.reshape(new_shape), upb.reshape(new_shape)

    # u_fit = get_ut(x=x, diffusion_time=t, rod_length=length, diffusivity=k_fit, emission_time=pulse_length,
    #                flux=flux_fit,
    #                T0=inital_temperature)

    fig, ax = plt.subplots(ncols=1)  # , constrained_layout=True)
    fig.set_size_inches(5.0, 3.5)

    ax.fill_between(tpred, lpb[:, 0], upb[:, 0], color='C0', alpha=0.3)
    ax.fill_between(tpred, lpb[:, 1], upb[:, 1], color='C1', alpha=0.3)

    ax.plot(
        time_s, u_exp[:, 0], label=f'TC1 x={x[0]:.2f} cm', zorder=2,
        marker='o', ms=8, fillstyle='none', mew=1.0, ls='none', c='C0'
    )

    ax.plot(
        time_s, u_exp[:, 1], label=f'TC2 x={x[1]:.2f} cm', zorder=3,
        marker='s', ms=8, fillstyle='none', mew=1.0, ls='none', c='C1'
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlim(left=0.0, right=time_s.max())
    ax.set_ylim(bottom=0.0)
    ax.legend(
        loc='best', frameon=True, prop={'size': 9}
    )

    ax_t_xlim = ax.get_xlim()
    ax_t_ylim = ax.get_ylim()

    p_height = ax_t_ylim[1]
    p_width = pulse_length
    xy = (ax_t_xlim[0], ax_t_ylim[0])
    rect = patches.Rectangle(
        xy, p_width, p_height, linewidth=1, edgecolor='tab:purple', facecolor=mpl.colors.CSS4_COLORS['lavender'],
        alpha=0.5,
        zorder=1
    )
    ax.add_patch(rect)
    ax.text(
        0.025, 0.95, 'Laser on',
        ha='left', va='top',
        transform=ax.transAxes,
        fontsize=9
    )

    ax.plot(
        tpred, u_fit[:, 0], label=f'Fit x={x[0]:.2f} cm', zorder=2, color='b',
        # marker='o', ms=8, fillstyle='none', mew=1.0
    )

    ax.plot(
        tpred, u_fit[:, 1], label=f'Fit x={x[1]:.2f} cm', zorder=3, color='r',
    )

    # title_txt = rf'$\alpha_{{fit}} = {utils.latex_float(a_fit, 3)}$, 95% CI: [${utils.latex_float(ci[0][0], 4)}, {utils.latex_float(ci[0][1], 4)}$] cm$^2$/s'
    # title_txt += '\n'
    title_txt = rf'$F_{{fit}} = {utils.latex_float(flux_fit, 3)}$, 95% CI: [${utils.latex_float(ci[0][0], 4)}, {utils.latex_float(ci[0][1], 4)}$] K/cm'

    ax.set_title(title_txt, fontweight='regular')

    K0 = specific_heat_of_graphite(tc1[0], units='C') * density_g * a_fit
    heat_flux = K0 * flux_fit

    for i, p, c in zip(range(len(popt)), popt, ci):
        logger.info(f'popt_{i}: {p:.3E} 95% CI: [{c[0]:.3E}, {c[1]:.3E}]')

    logger.info('Thermal diffusivity model: alpha = 1.0 / (a_0*T + a_1)')
    logger.info(f'F_fit: {popt[0]:.4f} 95 CI: [{ci[0][0]:.5f},{ci[0][1]:.5f}]')
    logger.info(f'a: {a_fit:.3E} cm^2/s')
    # logger.info(f'F_fit: {flux_fit:.3E}')
    logger.info(f'K: {K0:.3f} W/cm/K')
    logger.info(f'Heat flux: {heat_flux:.3E} W/cm^2')

    # if len(popt) > 3:
    #     logger.info(f'z0: {popt[2]:.3f} cm')
    #     logger.info(f'z1: {popt[3]:.3f} cm')

    # ax_t.xaxis.set_minor_locator(ticker.MultipleLocator(2.0))
    # ax_t.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))

    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, f'{file_tag}_simulated_fit_a={a_fit:.2E}_f={flux_fit:.2E}.png'), dpi=600)
    fig.savefig(os.path.join(base_dir, f'{file_tag}_simulated_fit_a={a_fit:.2E}_f={flux_fit:.2E}.svg'), dpi=600)
    plt.show()

    # fig, ax = plt.subplots(ncols=1, nrows=1)#, constrained_layout=True)
    # fig.set_size_inches(5.0, 3.5)
    #
    # ax.plot(
    #     time_s, tc1, ls='none', marker='o', color=lighten_color('tab:red'), label=f'TC1 (raw)', lw=1.25,
    # )
    #
    # ax.plot(
    #     time_s, tc2, ls='none', marker='s', color=lighten_color('tab:blue'), label=f'TC2 (raw)', lw=1.25,
    # )
    #
    # ax.plot(
    #     time_s, tc1_corrected, ls='--', color='tab:red', label=f'TC1 (corrected)', lw=1.25,
    # )
    #
    # ax.plot(
    #     time_s, tc2_corrected, ls='--', color='tab:blue', label=f'TC2 (corrected)', lw=1.25,
    # )
    #
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Temperature (°C)')
    #
    # fig.tight_layout()
    # sys.stdout = old_stdout

    plt.show()
