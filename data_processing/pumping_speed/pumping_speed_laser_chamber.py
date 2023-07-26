import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
from matplotlib import ticker, gridspec
import matplotlib as mpl
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import json
from data_processing.utils import latex_float
from scipy.optimize import least_squares
import data_processing.confidence as cf
import warnings

warnings.filterwarnings("error")

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\laser_chamber_pumping_speed\turbo'
pumpdown_data_csv = 'DEGASSING_TRURBO_PUMPDOWN_2023-03-09_1'
# outgassing_data_csv = 'DEGASSING_LASER_CHAMBER_OUTGASSING_2022-07-13_1'
outgassing_data_csv = 'DEGASSING_LASER_CHAMBER_TURBO_OUTGASSING_2023-03-13_3'
file_tag = 'LASER_CHAMBER_VACUUM_PARAMETERS_TURBO'

xlim = (0.0, 0.2)

"""
Volume and surface of the laser chamber
---------------------------
V = L1 x L2 x L3 + 5 x pi x r^2 * Lc
S = 2 x (L1 x L2 + L2 x L3 + L3 x L1)  + 5 x (pi * r^2 + pi * d * Lc)
"""
L1, L2, L3, d = 2.54 * np.array([12.0, 13.5, 10.5, 8.0])  # cm
Lc = 2.28  # cm
r = 0.5 * d

volume_laser_chamber = L1 * L2 * L3 + 5.0 * (np.pi * r * r * Lc)
surface_laser_chamber = 2.0 * (L1 * L2 + L2 * L3 + L3 * L1) + 5.0 * np.pi * (d * Lc)
chamber_dimension_cm = L1

print('********* Laser Chamber *********')
print(f'V = {volume_laser_chamber:.2f} cm^3 = {volume_laser_chamber * 1E-3:.2f} L')
print(f'S = {surface_laser_chamber:.2f} cm^2 = {surface_laser_chamber * 1E-4:.2f} m^2')

air_n2_fraction = 0.80
air_o2_fraction = 0.20

kinetic_diameter_n2_pm = 364.0  # x 1E-12 m
kinetic_diameter_o2_pm = 346.0  # x 1E-12 m


def pump_down_initial_model(t, b):
    return b[0] * np.exp(-b[1] * t / volume_laser_chamber * 1E3)


def fobj_initial(b, t, p):
    return np.log10(pump_down_initial_model(t, b)) - np.log10(p)


def jac_initial(b, t, p):
    j1 = np.exp(-b[1] * t / volume_laser_chamber * 1E3)
    j2 = -(b[0] * t / volume_laser_chamber * 1E3) * np.exp(-b[1] * t / volume_laser_chamber * 1E3)
    return np.array([j1, j2]).T


def model_ultimate(t, b):
    return b[0] * np.power(t, -b[1])


def model_ultimate_derivative(t, b):
    return -b[0] * b[1] * np.power(t, -b[1] - 1.0)


def fobj_ultimate(b, t, p):
    return model_ultimate(t, b) - p


def jac_ultimate(b, t, p):
    j1 = np.power(t, -b[1])
    j2 = -b[0] * np.power(t, -b[1]) * np.log(t)
    return np.array([j1, j2]).T


def get_mean_free_path(temperature_c: np.ndarray = np.array([20.0]), pressure_pa: np.ndarray = np.array([101325.0])):
    """
    Estimates the mean free path in cm for air composed of 80% N2 and 20% O2
    """
    kB = 1.380649  # x 1E-23 J/K
    T = temperature_c + 273.15
    p = pressure_pa
    return 4.0E3 * kB * T / (np.sqrt(2.0) * np.pi * ((air_n2_fraction * kinetic_diameter_n2_pm +
                                                      air_o2_fraction * kinetic_diameter_o2_pm) ** 2.0) * p)


def outgassing_model(x, b):
    # result = np.empty_like(x)
    # msk_0 = x == 0
    result = b[0] + b[1] * np.power(x, b[2])
    return result


def outgassing_model2(t, b):
    result = b[0] - b[1] * np.exp(-b[2] * t)
    result[t == 0.] = b[0] - b[1]
    return result


def fobj_outgassing_model2(b2, t, y, b0, b1):
    return np.exp(outgassing_model2(t, [b0, b1, b2])) - np.exp(y)


def jac_outgassing2(b, t, y, b0, b1):
    # m, n = len(t), len(b)
    # jac = np.empty((m, n))

    bb = [b0, b1, b[0]]
    e = np.exp(-bb[2] * t)
    p = outgassing_model2(t, bb)
    # jac[:, 0] = np.ones(m) * p
    # jac[:, 1] = -e * p
    # jac[:, 2] = b[1] * t * e * p
    jac = np.empty((len(t), 1))
    jac[:, 0] = b1 * t * e * p
    return jac


def outgass_model2_dpdt(t, b):
    return b[1] * b[2] * np.exp(-b[2] * t)


def outgassing_model_dpdt(x, b):
    msk_0 = x == 0
    result = np.empty_like(x)
    result[~msk_0] = np.log(10) * np.power(10, outgassing_model(x[~msk_0], b)) * b[1] * b[2] * np.power(x[~msk_0],
                                                                                                        b[2] - 1.0)
    result[msk_0] = np.log(10) * np.power(10, outgassing_model(x[1], b)) * b[1] * b[2] * np.power(x[1], b[2] - 1.0)
    return result


def pumping_speed_model_viscous(x, b):
    return -b[1]


def model_ultimate_dpdt_over_p(t, b_trans):
    msk_0 = t == 0
    result = np.empty_like(t)
    result[~msk_0] = -b_trans[1] / t[~msk_0]
    result[msk_0] = -b_trans[1] / t[1]
    return result * volume_laser_chamber * 1E-3


def fobj_outgassing(b: np.ndarray, x: np.ndarray, y: np.ndarray):
    return outgassing_model(x, b) - np.log10(y)


def outgassing_jac(b: np.ndarray, x: np.ndarray, y: np.ndarray):
    j1 = np.ones_like(x)
    j2 = np.power(x, b[2])
    j3 = b[1] * np.power(x, b[2]) * np.log(x)
    return np.array([j1, j2, j3]).T


if __name__ == '__main__':
    pumpdown_df = pd.read_csv(os.path.join(base_dir, pumpdown_data_csv + '.csv'), comment="#").apply(pd.to_numeric)
    venting_df = pd.read_csv(os.path.join(base_dir, outgassing_data_csv + '.csv'), comment="#").apply(pd.to_numeric)
    # print(pumpdown_df)
    measurement_time_pumpdown = pumpdown_df['Measurement Time (h)'].values * 3600.0
    measurement_time_venting = venting_df['Measurement Time (h)'].values * 3600.0
    pressure_pumpdown = pumpdown_df['Pressure (Torr)'].values
    pressure_outgassing = venting_df['Pressure (Torr)'].values
    temperature_pumpdown = pumpdown_df['TC2 (C)'].values
    temperature_outgassing = venting_df['TC2 (C)'].values

    time_pumpdown_final = measurement_time_pumpdown.max()
    len_t = len(measurement_time_pumpdown)

    # idx_pump_down_le0 = pressure_pumpdown <= 0.
    #
    # pressure_pumpdown[idx_pump_down_le0] = 1E-3

    idx_pump_down_gt0 = pressure_pumpdown > 0.
    measurement_time_pumpdown = measurement_time_pumpdown[idx_pump_down_gt0]
    pressure_pumpdown = pressure_pumpdown[idx_pump_down_gt0]
    temperature_pumpdown = temperature_pumpdown[idx_pump_down_gt0]

    idx_outgassing_gt0 = pressure_outgassing > 0.
    measurement_time_venting = measurement_time_venting[idx_outgassing_gt0]
    pressure_outgassing = pressure_outgassing[idx_outgassing_gt0]
    temperature_outgassing = temperature_outgassing[idx_outgassing_gt0]

    # Find the range of measurements before the vacuum pump started running
    p0 = pressure_pumpdown[0]
    idx_p0 = len(pressure_pumpdown) - len(pressure_pumpdown[pressure_pumpdown < p0]) - 2
    print(f"Starting index: {idx_p0}")
    t0 = measurement_time_pumpdown[idx_p0]
    measurement_time_pumpdown = measurement_time_pumpdown[idx_p0::] - t0
    pressure_pumpdown = pressure_pumpdown[idx_p0::]
    temperature_pumpdown = temperature_pumpdown[idx_p0::]

    # idx_time = (measurement_time_pumpdown <= 0.6) & (measurement_time_pumpdown > 0.0)
    # measurement_time_pumpdown = measurement_time_pumpdown[idx_time] * 3600.0
    # pressure_pumpdown = pressure_pumpdown[idx_time]
    # temperature_pumpdown = temperature_pumpdown[idx_time]

    mean_free_path = get_mean_free_path(temperature_c=temperature_pumpdown, pressure_pa=pressure_pumpdown * 133.22)
    kn = mean_free_path / chamber_dimension_cm
    # transitional_idx = kn > 2E-1
    # viscous_idx = kn <= 2E-1
    ultimate_idx = pressure_pumpdown <= 0.1#7.5E-3
    viscous_idx = ~ultimate_idx
    print(f'len(viscous regime): {len(pressure_pumpdown[viscous_idx])}')

    time_initial = measurement_time_pumpdown[viscous_idx]
    pressure_initial = pressure_pumpdown[viscous_idx]
    # time_viscous = time_viscous[1::]
    # pressure_viscous = pressure_viscous[1::]

    # Outgassing
    all_tol = np.finfo(np.float64).eps
    bb0 = pressure_outgassing.max()
    bb1 = bb0 - pressure_outgassing.min()
    bb2 = np.exp(-1.) / measurement_time_venting.max()
    b_guess = [bb0, bb1, bb2]
    res = least_squares(
        fobj_outgassing_model2, [bb2], args=(measurement_time_venting[::], pressure_outgassing[::], bb0, bb1),
        # bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
        bounds=([0.0], [np.inf]),
        jac=jac_outgassing2,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        loss='soft_l1', f_scale=0.1,
        max_nfev=len(pressure_outgassing) * 10000,
        # x_scale='jac',
        verbose=2
    )
    popt = res.x
    pcov = cf.get_pcov(res)
    popt_real = [bb0, bb1, popt[0]]
    ci = cf.confint(len(measurement_time_venting[1::]), popt, pcov)
    xpred = np.linspace(measurement_time_venting[1], measurement_time_venting.max())
    # ypred, lpb, upb = cf.predint(xpred, measurement_time_venting[1::], pressure_venting[1::], venting_model, res)
    p_venting_model = outgassing_model2(measurement_time_venting, popt_real)
    # Print outgassing model parameters
    print(f'b0: {popt_real[0]:.4E}')
    print(f'b1: {popt_real[1]:.4E}')
    print(f'b2: {popt_real[2]:.4E}')

    # dt_outgassing = np.gradient(measurement_time_venting, edge_order=2)
    # dt_outgassing = dt_outgassing.mean()

    # dPvdt = outgassing_model_dpdt(measurement_time_venting, popt)
    dPvdt = outgass_model2_dpdt(measurement_time_venting, popt_real)
    q_v = dPvdt * volume_laser_chamber * 1E-3
    q_v_mean = q_v.mean()
    q_v_std = q_v.std()

    time_outgassing_extrapolate = np.logspace(-10, np.log10(96.0 * 3600.0), 100000)

    p_outgass_extrapolate = outgassing_model2(time_outgassing_extrapolate, popt_real)
    # outgassing_extrapolate = outgassing_model_dpdt(time_outgassing_extrapolate, popt) * volume_laser_chamber * 1E-3
    outgassing_extrapolate = outgass_model2_dpdt(time_outgassing_extrapolate, popt_real) * volume_laser_chamber * 1E-3

    f_s = interp1d(p_outgass_extrapolate, outgassing_extrapolate, bounds_error=False)
    f_o = interp1d(time_outgassing_extrapolate, outgassing_extrapolate, bounds_error=False)

    print(f'p_out.min: {p_outgass_extrapolate.min():.3E}, p.max = {p_outgass_extrapolate.max():.3E}')

    outgassing_model_df = pd.DataFrame(data={
        'Time (s)': time_outgassing_extrapolate,
        'Pressure (Torr)': p_outgass_extrapolate,
        'Outgassing (Torr L / s)': outgassing_extrapolate,
    })

    outgassing_model_df.to_csv(os.path.join(base_dir, 'outgassing_model.csv'), index=False)

    all_tol = np.finfo(np.float64).eps
    b_guess = [pressure_initial.mean(), 20.1]
    print('time_initial:', time_initial)
    print('len(time_initial):', len(time_initial))
    res_initial = least_squares(
        fobj_initial, b_guess, args=(time_initial, pressure_initial),
        bounds=([1E-16, 1E-16], [1E4, 1E5]),
        jac=jac_initial,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        diff_step=all_tol,
        # loss='soft_l1', f_scale=0.1,
        max_nfev=len(time_initial) * 10000,
        x_scale='jac',
        verbose=2
    )

    popt_initial = res_initial.x
    print(f'popt initial:')
    print(f'p0: {popt_initial[0]:.3E} Torr')
    print(f'S: {popt_initial[1]:.3E} L/s')
    pcov_initial = cf.get_pcov(res_initial)
    ci = cf.confint(time_initial.size, popt_initial, pcov_initial)
    tpred_initial = np.linspace(time_initial.min(), time_initial.max(), 1000)
    ypred_initial = pump_down_initial_model(tpred_initial, popt_initial)
    # ypred_initial, lpb, upb = cf.predint(tpred_initial, time_initial, pressure_initial, pump_down_initial_model, res_initial)

    time_ultimate = measurement_time_pumpdown[ultimate_idx]
    pressure_ultimate = pressure_pumpdown[ultimate_idx]
    b_guess = [1E-3, 0.01]

    res_u = least_squares(
        fobj_ultimate, b_guess, args=(time_ultimate, pressure_ultimate),
        bounds=([1E-16, 1E-16], [1E16, 1E5]),
        jac=jac_ultimate,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        # loss='soft_l1', f_scale=0.1,
        max_nfev=len(time_ultimate) * 10000,
        diff_step=all_tol,
        verbose=2
    )
    popt_ultimate = res_u.x
    pcov_ultimate = cf.get_pcov(res_u)
    ci = cf.confint(time_ultimate.size, popt_ultimate, pcov_ultimate)
    tpred_ult = np.linspace(time_ultimate.min(), 0.5 * 3600., 1000)
    ypred_ult, lpb_trans, upb_trans = cf.predint(tpred_ult, time_ultimate, pressure_ultimate,
                                                 model_ultimate, res_u)
    for i, p in enumerate(popt_ultimate):
        print(f'popt_ultimate[{i}]: {p:.3E}')

    dt = np.gradient(time_ultimate).mean()
    t_extrapolated_max = 10.0 * 3600.0
    n_extrapolated = int((t_extrapolated_max - time_ultimate[0]) / dt) + 1
    time_ultimate_extrapolated = dt * np.arange(0, n_extrapolated) + time_ultimate[0]

    p_v = pressure_pumpdown[viscous_idx]

    # # find the intercept between the pressure determined from the initial and ultimate model:
    p_i = pump_down_initial_model(measurement_time_pumpdown[viscous_idx], popt_initial)
    p_u = model_ultimate(time_ultimate_extrapolated, popt_ultimate)
    dpdt_u = model_ultimate_derivative(time_ultimate_extrapolated, popt_ultimate)
    print(f'Extrapolated p_u.min: {p_u.min():.3E} Torr, p.max: {p_u.max():.3E} Torr')
    print('Extrapolated venting pressures:')
    # idx_overlap = (np.abs(p_i <= p_u[0])).argmin() - 10
    # t_overlap = measurement_time_pumpdown[viscous_idx]
    # t_overlap = t_overlap[idx_overlap::]
    # p_o = model_ultimate(t_overlap, popt_ultimate)
    # p_i[idx_overlap::] = p_o
    # pressure_smooth[viscous_idx] = p_i
    # pressure_smooth[transitional_idx] = p_u
    time_join = np.hstack((time_initial, time_ultimate_extrapolated))
    pressure_join = np.hstack((p_v, p_u))

    pumping_speed = np.empty_like(pressure_join)
    pumping_speed[0:p_v.size] = popt_initial[1] * np.ones(p_v.size)
    # pumping_speed[transitional_idx] = (q_v_mean - dpdt_ultimate * volume_laser_chamber * 1E-3) / pressure_transitional_smooth
    pumping_speed[p_v.size:] = (f_s(p_u) - dpdt_u * volume_laser_chamber * 1E-3) / p_u

    # pumping_speed[transitional_idx] = savgol_filter(pumping_speed[transitional_idx], window_length=101, polyorder=4)

    pumping_speed_df = pd.DataFrame(data={
        'Pressure (Torr)': pressure_join,
        'Pumping Speed (L/s)': pumping_speed
    })

    pumping_speed_df.to_csv(
        path_or_buf=os.path.join(base_dir, pumpdown_data_csv + '_pumping_speed.csv'),
        index=False
    )

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig_v, ax_v = plt.subplots(ncols=1)  # , constrained_layout=True)
    fig_v.set_size_inches(5.0, 3.5)

    ax_v.plot(
        measurement_time_venting / 3600.0, pressure_outgassing, marker='o', fillstyle='none', mew=1.25, ms=5,
        c='C0', ls='none'
    )

    ax_v.plot(
        measurement_time_venting / 3600.0, p_venting_model, color='b', ls=':'
    )

    ax_v.set_yscale('log')
    ax_v.set_ylabel('Pressure (Torr)', color='C0')
    ax_v.set_xlabel('Time (h)')
    ax_v.set_xlim(0, 1)
    ax_v.set_ylim(bottom=1E-3, top=1E0)
    ax_v.set_title('Laser chamber outgassing')
    ax_v.tick_params(axis='y', labelcolor='C0')

    vent_model_txt = f'$P = b_0 - b_1 \\exp\\left(-{{b_2}}t\\right)$\n'
    vent_model_txt += f'$b_0: {latex_float(popt_real[0])}$\n'
    vent_model_txt += f'$b_1: {latex_float(popt_real[1])}$\n'
    vent_model_txt += f'$b_2: {latex_float(popt_real[2])}$'

    ax_v.text(
        0.95, 0.05,
        vent_model_txt,
        fontsize=11,
        transform=ax_v.transAxes,
        va='bottom', ha='right',
        color='b'
    )

    ax_q = ax_v.twinx()
    ax_q.plot(
        measurement_time_venting / 3600.0, q_v,
        c='C1',
    )
    ax_q.set_ylabel(r'Outgassing Rate (Torr $\cdot$ L / s )', color='C1')
    ax_q.tick_params(axis='y', labelcolor='C1')
    # ax_q.set_yscale('log')

    q_v_mean_txt = rf'$Q_{{\mathrm{{mean}}}} = {latex_float(q_v_mean, 2)}$ (Torr $\cdot$ L / s)'
    ax_q.text(
        0.95, 0.95,
        q_v_mean_txt,
        fontsize=11,
        transform=ax_q.transAxes,
        va='top', ha='right',
    )

    fig_v.tight_layout()
    fig_v.savefig(os.path.join(base_dir, file_tag + '_venting.png'), dpi=600)

    fig_p = plt.figure(tight_layout=True)
    fig_p.set_size_inches(4.5, 6.0)

    gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig_p)  # , height_ratios=[1.618, 1.618, 1])

    ax_p = fig_p.add_subplot(gs[0])
    ax_dp = fig_p.add_subplot(gs[1])
    ax_s = fig_p.add_subplot(gs[2])

    colors = ['C0', 'C1']

    ax_p.plot(
        measurement_time_pumpdown / 3600.0,
        pressure_pumpdown, ls='none', label=f'Pressure', marker='o', fillstyle='none', mew=1.25, ms=5,
        c=colors[0],
    )

    # ax_p.plot(
    #     tpred / 3600.0,
    #     ypred, ls=':', label=r'$p = p_0 e^{-\frac{S}{V}t}$',
    #     c='tab:red', lw=1.5
    # )

    ax_p.plot(
        tpred_initial/3600., ypred_initial, ls=':', label=r'$p = p_0 e^{-\frac{S}{V}t}$',
        c='k', lw=1.5
    )

    ax_p.plot(
        time_ultimate_extrapolated / 3600.0,
        p_u, ls=':', label='$p=kt^{-n}$',
        c='tab:red', lw=1.5
    )

    ax_p.plot(
        tpred_ult / 3600.0, ypred_ult, ls=':', #label='$p=kt^{-n}$',
        c='tab:red', lw=1.5
    )

    ax_p.patch.set_visible(False)

    ax_p.set_yscale('log')
    ax_s.set_xlabel('Time (h)')
    ax_p.set_ylabel('$p$ (Torr)')  # , color=colors[0])
    ax_p.set_xlim(xlim)
    ax_p.set_ylim(bottom=1E-6, top=1E3)
    # ax_p.tick_params(axis='y', labelcolor=colors[0])

    dp_initial = popt_initial[1] * p_i
    ax_dp.plot(
        measurement_time_pumpdown[viscous_idx] / 3600,
        dp_initial * volume_laser_chamber * 1E-3,
        color='C2', label='$-V\cdot dp/dt$'
    )
    # ax_dp.plot(
    #     measurement_time_pumpdown[ultimate_idx] / 3600,
    #     -model_ultimate_derivative(measurement_time_pumpdown[ultimate_idx],
    #                                popt_ultimate) * volume_laser_chamber * 1E-3,
    #     color='C2'
    # )

    ax_dp.plot(
        time_ultimate_extrapolated / 3600,
        -model_ultimate_derivative(time_ultimate_extrapolated,
                                   popt_ultimate) * volume_laser_chamber * 1E-3,
        color='C2'
    )

    ax_dp.plot(
        time_ultimate_extrapolated / 3600,
        f_s(p_u), ls='-',
        color='C5', label=r'$Q_{\mathrm{out}}$'
    )

    ax_dp.set_ylabel(r'Torr L / s')
    # ax_dp.tick_params(axis='y', labelcolor='C2')
    ax_dp.set_yscale('log')
    ax_dp.set_ylim(bottom=1E-5, top=1E3)
    ax_dp.set_xlim(xlim)
    ax_dp.legend(
        loc='upper right', frameon=False
    )

    ax_p.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_p.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax_p.yaxis.set_major_locator(ticker.LogLocator(base=10))
    # ax_p.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))

    ax_p.legend(
        loc='upper right', frameon=False
    )

    ax_p.set_title('Laser chamber pump down')

    ax_s.plot(
        time_join / 3600.0,
        pumping_speed, label=f'Pumping Speed',
        c=colors[1]
    )

    ax_s.set_ylabel('$S$ (L / s)')  # , color=colors[1])
    # ax_s.tick_params(axis='y', labelcolor=colors[1])
    ax_s.set_xlim(xlim)
    ax_s.set_ylim(bottom=1E0, top=100.0)
    ax_s.set_yscale('log')

    ax_s.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_s.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    # ax_s.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    # ax_s.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    estimation_txt = r'$S = \dfrac{1}{p}\left(Q_{\mathrm{out}} - V\dfrac{dp}{dt}\right)$'
    ax_s.text(
        0.95,
        0.95,
        estimation_txt,
        fontsize=11,
        transform=ax_s.transAxes,
        va='top', ha='right',
        # bbox=props
    )

    fig_p.tight_layout()

    # basename = os.path.splitext(os.path.basename(input_file))[0]
    # path = os.path.dirname(input_file)
    fig_p.savefig(os.path.join(base_dir, file_tag + '_pump_down.png'), dpi=600)

    fig_s, ax_s = plt.subplots(ncols=1)  # , constrained_layout=True)
    fig_s.set_size_inches(5.25, 3.25)

    ax_s.plot(
        measurement_time_pumpdown / 3600.0, mean_free_path, color='C2'
    )

    ax_s.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax_s.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax_s.set_xlim(xlim)

    ax_k = ax_s.twinx()
    ax_k.plot(
        measurement_time_pumpdown / 3600.0, kn, color='C2'
    )

    ax_s.set_yscale('log')
    ax_k.set_yscale('log')

    ax_s.set_xlabel('Time (h)')
    ax_s.set_ylabel('Mean free path (cm)', color='k')
    ax_k.set_ylabel('$Kn$', color='k')
    ax_s.tick_params(axis='y', labelcolor='k')
    ax_k.tick_params(axis='y', labelcolor='k')
    ax_s.set_title('Laser chamber')
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(base_dir, file_tag + '_mean_free_path.png'), dpi=600)

    fig_s2, ax_s2 = plt.subplots(ncols=1)  # , constrained_layout=True)
    fig_s2.set_size_inches(5.0, 3.5)

    ax_s2.plot(
        pressure_join, pumping_speed, color='C3'
    )

    ax_s2.set_yscale('log')
    ax_s2.set_xscale('log')
    ax_s2.set_ylabel('$S$ (L / s)')
    ax_s2.set_xlabel('$p$ (Torr)')
    ax_s2.set_xlim(left=1E-3, right=1E3)
    ax_s2.set_ylim(bottom=1E-3, top=10.0)
    # ax_s2.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    # ax_s2.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax_s2.set_title('Laser chamber')

    fig_s2.tight_layout()
    fig_s2.savefig(os.path.join(base_dir, file_tag + '_pumping_speed.png'), dpi=600)

    plt.show()
