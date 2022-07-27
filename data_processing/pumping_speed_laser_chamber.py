import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
from matplotlib import ticker, gridspec
import matplotlib as mpl
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import json
from utils import latex_float
from scipy.optimize import least_squares
import confidence as cf
import warnings

warnings.filterwarnings("error")

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\laser_chamber_pumping_speed'
pumpdown_data_csv = 'DEGASSING_LASER_CHAMBER_PUMPDOWN_2022-07-13_1'
# outgassing_data_csv = 'DEGASSING_LASER_CHAMBER_OUTGASSING_2022-07-13_1'
outgassing_data_csv = 'DEGASSING_LASER_CHAMBER_OUTGASSING_2022-07-14_1'
file_tag = 'LASER_CHAMBER_VACUUM_PARAMETERS'

xlim = (0.0, 1.0)

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

print( '********* Laser Chamber *********')
print(f'V = {volume_laser_chamber:.2f} cm^3 = {volume_laser_chamber*1E-3:.2f} L')
print(f'S = {surface_laser_chamber:.2f} cm^2 = {surface_laser_chamber*1E-4:.2f} m^2')

air_n2_fraction = 0.80
air_o2_fraction = 0.20

kinetic_diameter_n2_pm = 364.0  # x 1E-12 m
kinetic_diameter_o2_pm = 346.0  # x 1E-12 m


def pump_down_initial_model(t, b):
    return b[0] * np.exp(-b[1] * t / volume_laser_chamber * 1E3)


def fobj_initial(b, t, p):
    return pump_down_initial_model(t, b) - p


def jac_initial(b, t, p):
    j1 = np.exp(-b[1] * t / volume_laser_chamber * 1E3)
    j2 = -(b[0] * t / volume_laser_chamber * 1E3) * np.exp(-b[1] * t / volume_laser_chamber * 1E3)
    return np.array([j1, j2]).T


def model_ultimate(t, b):
    return b[0] * np.power(t, -b[1])


def model_ultimate_derivative(t, b):
    return -b[0] * b[1] * np.power(t, -b[1]-1.0)


def fobj_ultimate(b, t, p):
    return model_ultimate(t, b) - p


def jac_ultimate(b, t, p):
    j1 = np.power(t, -b[1])
    j2 = -b[0] * np.power(t, -b[1])*np.log(t)
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


def outgassing_model_dpdt(x, b):
    msk_0 = x == 0
    result = np.empty_like(x)
    result[~msk_0] = np.log(10) * np.power(10, outgassing_model(x[~msk_0], b)) * b[1] * b[2] * np.power(x[~msk_0], b[2] - 1.0)
    result[msk_0] = np.log(10) * np.power(10, outgassing_model(x[1], b)) * b[1] * b[2] * np.power(x[1], b[2] - 1.0)
    return result

def pumping_speed_model_viscous(x, b):
    return -b[1]

def model_transitional_dpdt_over_p(t, b_trans):
    msk_0 = t == 0
    result = np.empty_like(t)
    result[~msk_0] = b_trans[1] / t[~msk_0]
    result[msk_0] = b_trans[1] / t[1]
    return result * volume_laser_chamber * 1E-3


def fobj(b: np.ndarray, x: np.ndarray, y: np.ndarray):
    return outgassing_model(x, b) - np.log10(y)


def model_jac(b: np.ndarray, x: np.ndarray, y: np.ndarray):
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
    transitional_idx = kn > 1E-3
    viscous_idx = kn <= 1E-3
    print(f'len(viscous regime): {len(pressure_pumpdown[viscous_idx])}')

    time_viscous = measurement_time_pumpdown[viscous_idx]
    pressure_viscous = pressure_pumpdown[viscous_idx]
    # time_viscous = time_viscous[1::]
    # pressure_viscous = pressure_viscous[1::]

    # Outgassing
    all_tol = np.finfo(np.float64).eps
    b_guess = [np.log10(pressure_outgassing.min()), 1.0, 0.5]
    res = least_squares(
        fobj, b_guess, args=(measurement_time_venting[1::], pressure_outgassing[1::]),
        # bounds=([-10, 0.0, 0.0], [3, 1E10, 1.0]),
        jac=model_jac,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        # loss='soft_l1', f_scale=0.1,
        max_nfev=len(pressure_outgassing)*10000,
        verbose=2
    )
    popt = res.x
    pcov = cf.get_pcov(res)
    ci = cf.confint(len(measurement_time_venting[1::]), popt, pcov)
    xpred = np.linspace(measurement_time_venting[1], measurement_time_venting.max())
    # ypred, lpb, upb = cf.predint(xpred, measurement_time_venting[1::], pressure_venting[1::], venting_model, res)
    p_venting_model = 10 ** outgassing_model(measurement_time_venting, popt)
    # Print outgassing model parameters
    print(f'b0: {popt[0]:.4E}')
    print(f'b1: {popt[1]:.4E}')
    print(f'b2: {popt[2]:.4E}')

    # dt_outgassing = np.gradient(measurement_time_venting, edge_order=2)
    # dt_outgassing = dt_outgassing.mean()

    dPvdt = outgassing_model_dpdt(measurement_time_venting, popt)
    q_v = dPvdt * volume_laser_chamber * 1E-3
    q_v_mean = q_v.mean()
    q_v_std = q_v.std()

    time_outgassing_extrapolate = np.logspace(-5, np.log10(96.0 * 3600.0), 100000)

    pressure_extrapolate = 10 ** outgassing_model(time_outgassing_extrapolate, popt)
    outgassing_extrapolate = outgassing_model_dpdt(time_outgassing_extrapolate, popt) * volume_laser_chamber * 1E-3

    f_s = interp1d(pressure_extrapolate, outgassing_extrapolate)
    #
    outgassing_model_df = pd.DataFrame(data={
        'Time (s)': time_outgassing_extrapolate,
        'Pressure (Torr)': pressure_extrapolate,
        'Outgassing (Torr L / s)': outgassing_extrapolate,
    })

    outgassing_model_df.to_csv(os.path.join(base_dir, 'outgassing_model.csv'), index=False)

    all_tol = np.finfo(np.float64).eps
    b_guess = [pressure_viscous[0], 1.0E-2]
    res_v = least_squares(
        fobj_initial, b_guess, args=(time_viscous, pressure_viscous),
        bounds=([1E-16, 1E-16], [1E4, 1E3]),
        jac=jac_initial,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        loss='soft_l1', f_scale=0.1,
        max_nfev=len(time_viscous)*10000,
        verbose=2
    )

    popt_initial = res_v.x
    pcov_initial = cf.get_pcov(res_v)
    ci = cf.confint(time_viscous.size, popt_initial, pcov_initial)
    tpred = np.linspace(time_viscous.min(), time_viscous.max(), 100)
    ypred, lpb, upb = cf.predint(tpred, time_viscous, pressure_viscous, pump_down_initial_model, res_v)

    time_transitional = measurement_time_pumpdown[transitional_idx]
    pressure_transitional = pressure_pumpdown[transitional_idx]
    b_guess = [1E-3, 10.0]

    res_u = least_squares(
        fobj_ultimate, b_guess, args=(time_transitional, pressure_transitional),
        bounds=([1E-16, 1E-16], [1E50, 1E50]),
        jac=jac_ultimate,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        loss='soft_l1', f_scale=0.1,
        max_nfev=len(time_transitional)*10000,
        verbose=2
    )
    popt_ultimate = res_u.x
    pcov_ultimate = cf.get_pcov(res_u)
    ci = cf.confint(time_transitional.size, popt_ultimate, pcov_ultimate)
    tpred_trans = np.linspace(time_transitional.min(), time_transitional.max(), 1000)
    ypred_trans, lpb_trans, upb_trans = cf.predint(tpred_trans, time_transitional, pressure_transitional,
                                                   model_ultimate, res_u)

    dt = np.gradient(time_transitional).mean()
    t_extrapolated_max = 10.0*3600.0
    n_extrapolated = int((t_extrapolated_max - time_transitional[100]) / dt) + 1
    time_transitional_extrapolate = dt*np.arange(0, n_extrapolated) + time_transitional[100]

    p_v = pressure_pumpdown[viscous_idx]

    # # find the intercept between the pressure determined from the initial and ultimate model:
    p_i = pump_down_initial_model(measurement_time_pumpdown[viscous_idx], popt_initial)
    p_u = model_ultimate(time_transitional_extrapolate, popt_ultimate)
    dpdt_u = model_ultimate_derivative(time_transitional_extrapolate, popt_ultimate)
    print(f'Extrapolated p_u.min: {p_u.min():.3E} Torr, p.max: {p_u.max():.3E} Torr')
    print('Extrapolated venting pressures:')
    print(f'p.min: {pressure_extrapolate.min():.3E}, p.max = {pressure_extrapolate.max():.3E}')
    # idx_overlap = (np.abs(p_i <= p_u[0])).argmin() - 10
    # t_overlap = measurement_time_pumpdown[viscous_idx]
    # t_overlap = t_overlap[idx_overlap::]
    # p_o = model_ultimate(t_overlap, popt_ultimate)
    # p_i[idx_overlap::] = p_o
    # pressure_smooth[viscous_idx] = p_i
    # pressure_smooth[transitional_idx] = p_u
    time_join = np.hstack((time_viscous, time_transitional_extrapolate))
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

    with open('plot_style.json', 'r') as file:
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
    ax_v.set_xlim(xlim)
    ax_v.set_ylim(bottom=1E-3, top=1E-1)
    ax_v.set_title('Laser chamber outgassing')
    ax_v.tick_params(axis='y', labelcolor='C0')

    vent_model_txt = f'$\log_{{10}}(P) = a_0 + a_1 t^{{a_2}}$\n'
    vent_model_txt += f'$a_0: {latex_float(popt[0])}$\n'
    vent_model_txt += f'$a_1: {latex_float(popt[1])}$\n'
    vent_model_txt += f'$a_2: {latex_float(popt[2])}$'

    ax_v.text(
        0.05, 0.95,
        vent_model_txt,
        fontsize=11,
        transform=ax_v.transAxes,
        va='top', ha='left',
        color='b'
    )

    ax_q = ax_v.twinx()
    ax_q.plot(
        measurement_time_venting / 3600.0, q_v,
        c='C1',
    )
    ax_q.set_ylabel(r'Outgassing Rate (Torr $\cdot$ L / s )', color='C1')
    ax_q.tick_params(axis='y', labelcolor='C1')
    ax_q.set_yscale('log')

    q_v_mean_txt = rf'$Q_{{\mathrm{{mean}}}} = {latex_float(q_v_mean, 2)}$ (Torr $\cdot$ L / s)'
    ax_q.text(
        0.95, 0.05,
        q_v_mean_txt,
        fontsize=11,
        transform=ax_q.transAxes,
        va='bottom', ha='right',
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

    ax_p.plot(
        tpred / 3600.0,
        ypred, ls=':', label=r'$p = p_0 e^{-\frac{S}{V}t}$',
        c='tab:red', lw=1.5
    )

    ax_p.plot(
        tpred_trans / 3600.0, ypred_trans, ls=':', label='$p=kt^{-n}$',
        c='tab:purple', lw=1.5
    )

    ax_p.patch.set_visible(False)

    ax_p.set_yscale('log')
    ax_s.set_xlabel('Time (h)')
    ax_p.set_ylabel('$p$ (Torr)')# , color=colors[0])
    ax_p.set_xlim(xlim)
    ax_p.set_ylim(bottom=1E-2, top=1E3)
    # ax_p.tick_params(axis='y', labelcolor=colors[0])

    dp_initial = popt_initial[1] * p_i
    ax_dp.plot(
        measurement_time_pumpdown[viscous_idx]/3600,
        dp_initial,
        color='C2', label='$V\cdot dp/dt$'
    )
    ax_dp.plot(
        measurement_time_pumpdown[transitional_idx]/3600,
        -model_ultimate_derivative(measurement_time_pumpdown[transitional_idx], popt_ultimate)*volume_laser_chamber*1E-3,
        color='C2'
    )

    ax_dp.plot(
        time_transitional_extrapolate / 3600,
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
    ax_s.set_ylim(bottom=1E-3, top=3.0)
    ax_s.set_yscale('log')

    ax_s.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_s.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    # ax_s.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    # ax_s.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    estimation_txt = r'$S = \dfrac{1}{p}\left(Q_{\mathrm{cold}} - V\dfrac{dp}{dt}\right)$'
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
    ax_s2.set_xlim(left=4E-3, right=1E3)
    ax_s2.set_ylim(bottom=2E-3, top=2.0)
    # ax_s2.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    # ax_s2.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax_s2.set_title('Laser chamber')

    fig_s2.tight_layout()
    fig_s2.savefig(os.path.join(base_dir, file_tag + '_pumping_speed.png'), dpi=600)

    plt.show()