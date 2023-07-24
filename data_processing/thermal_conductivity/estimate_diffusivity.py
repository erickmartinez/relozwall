import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
import logging
from scipy.optimize import least_squares, OptimizeResult
from data_processing.utils import specific_heat_of_graphite
from data_processing.utils import get_experiment_params

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\thermal_conductivity\graphite\laser_flash\CALIBRATION_20230719'
data_csv = 'LCT_GRAPHITE_100PCT_2023-07-21_1.csv'
sample_length_cm = 4.92
sample_length_cm_err = 0.16
sample_diameter = 1.276
sample_diameter_err = 0.003
density_g = 1.710
density_g_err = 0.018
specific_heat_g = 0.6752  # Markelov, Volga, et al., 1973
specific_heat_g = 0.714  # ± 0.009 J/g-K
reflectance = 40.4
# time_constant = 1.681  # 0.256  # 2.1148
time_constant = 0.256  # 2.1148

PI2 = np.pi ** 2.


def poly(x, b):
    n = len(b)
    xx = np.ones_like(x)
    r = np.zeros_like(x)
    for i in range(n):
        r += xx * b[i]
        xx *= x
    return r


def fobj(b, x, y):
    return poly(x, b) - y


def jac(b, x, y):
    n = len(b)
    jj = np.zeros((len(x), n), dtype=float)
    for i in range(n):
        jj[:, i] = np.power(x, i)
    return jj


def load_dimensionless_params():
    df = pd.read_csv('dimensionless_parameters.csv').apply(pd.to_numeric)
    return df


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


def correct_thermocouple_response(measured_temperature, measured_time, tau):
    n = len(measured_time)
    k = int(n / 12)
    k = k + 1 if k % 2 == 0 else k
    k = max(k, 5)
    # T = savgol_filter(measured_temperature, k, 3)
    # dTdt = np.gradient(T, measured_time, edge_order=2)
    delta = measured_time[1] - measured_time[0]
    dTdt = savgol_filter(x=measured_temperature, window_length=k, polyorder=4, deriv=1, delta=delta)
    # dTdt = savgol_filter(dTdt, k - 2, 3)
    r = measured_temperature + tau * dTdt
    return savgol_filter(r, k - 4, 3)


def get_v(ww):
    s = 0
    for i in range(1, 502):
        sign = -1. if (i % 2) == 1 else 1.
        n2 = i * i
        s += sign * np.exp(-n2 * ww)
    r = 1.0 + 2.0 * s
    return r


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    L2 = sample_length_cm ** 2.

    params_df = load_dimensionless_params()
    file_tag = os.path.splitext(data_csv)[0]

    experiment_params = get_experiment_params(relative_path=base_dir, filename=file_tag)
    try:
        pulse_length = float(experiment_params["Emission Time"]["value"])
        laser_set_point = float(experiment_params["Laser Power Setpoint"]["value"])
    except KeyError as e:
        print(e)
        pulse_length = float(experiment_params["Emission time"]["value"])
        laser_set_point = float(experiment_params["Laser power setpoint"]["value"])

    fh = logging.FileHandler(os.path.join(base_dir, file_tag + '_flash_td.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    logger.addHandler(ch)

    data_df = pd.read_csv(os.path.join(base_dir, data_csv), comment='#').apply(pd.to_numeric)
    time_s = data_df['Measurement Time (s)'].values
    temperature = data_df['TC2 (C)'].values
    laser_peak_power = data_df['Laser output peak power (W)'].values

    specific_heat_g = specific_heat_of_graphite(temperature=temperature[0], units='C')

    msk = laser_peak_power > 0
    t_msk = time_s[msk]
    t0 = t_msk[0]
    msk_init = time_s >= t0
    time_s = time_s[msk_init] - t0
    temperature = temperature[msk_init]
    temperature = correct_thermocouple_response(measured_temperature=temperature, measured_time=time_s,
                                                tau=time_constant)

    dT = temperature - temperature[0]

    v_exp = dT / dT.max()
    laser_power = laser_peak_power[msk].mean()

    dt = time_s[1] - time_s[0]
    t_model = np.linspace(time_s.min() + dt, time_s.max(), 1000)
    f = interp1d(v_exp, time_s)
    g = interp1d(time_s, v_exp, bounds_error=False, fill_value='extrapolate')

    th = f(0.5)
    ah = 0.138785 * L2 / th
    # w = PI2 * ah * time_s / L2
    w = 1.369756 * t_model / th

    t025, t075 = f(0.25), f(0.75)
    rt7525 = t075 / t025
    logger.info(f"t_0.75/t0.25; {rt7525:.3f}")
    KR = -0.3461467 + 0.361578 * rt7525 - 0.06520543 * rt7525 ** 2.
    a_corr_hl = ah * KR / 0.138785

    all_tol = np.finfo(np.float64).eps
    tth = time_s / th
    msk_v = tth >= 3.
    tth_fit = tth[msk_v]
    v_fit = v_exp[msk_v]

    tth_extrapolate = np.linspace(tth_fit.min(), 5.5, num=100)

    b0 = [1, -1]
    res = least_squares(
        fun=fobj, x0=b0, args=(tth_fit, v_fit), jac=jac, ftol=all_tol, xtol=all_tol, gtol=all_tol,
        max_nfev=10000 * len(tth_fit),
        loss='soft_l1', f_scale=0.1,
        verbose=0
    )

    popt = res.x

    v_extrapolate = poly(x=tth_extrapolate, b=popt)

    # t5 = 5. * th
    # t10 = 10. * th
    # v5 = g(t5)
    # v10 = g(t10)
    # vh = g(th)

    v5 = poly(5., popt)
    v10 = poly(10., popt)
    vh = g(th)

    r5 = v5 / vh
    r10 = v10 / vh

    logger.info(f"dt5: {r5:.3f}, dt10: {r10:.3f}, dt5/dt10 = {r5 / r10:.3f}")
    kk = np.array([
        [-0.1037162, 1.239040, -3.974433, 6.888738, -6.804883, 3.856663, -1.167799, 0.1465332],
        [0.054825246, 0.16697761, -0.28603437, 0.28356337, -0.13403286, 0.024077586, 0., 0.]
    ]).T

    rt = np.array([r5 ** i for i in range(8)])
    kc = np.dot(kk[:, 0], rt)
    a_corr_rad = ah * kc / 0.13885

    v_model = get_v(w)
    # ah = 0.138785 * L2 / th
    th_corr = 0.138785 * L2 / a_corr_rad
    w_corr = 1.369756 * t_model / th_corr
    v_model_corr = get_v(w_corr)

    thermal_conductivity_rad = a_corr_rad * density_g * specific_heat_g
    thermal_conductivity_hl = a_corr_hl * density_g * specific_heat_g
    R_sample = 0.5 * sample_diameter
    area = np.pi * R_sample ** 2.
    area_err = np.pi * sample_diameter_err
    aperture_factor = 1. - np.exp(-2. * (R_sample / (0.5 * 0.8164)) ** 2.)
    Q = sample_length_cm * dT.max() * density_g * specific_heat_g / pulse_length
    input_power = Q * area / aperture_factor

    n = len(params_df)
    alpha_list = np.empty(n, dtype=float)
    tx_list = np.empty(n, dtype=float)

    for i, r in params_df.iterrows():
        if i == 0:
            continue
        vx = 0.01 * r['V (%)']
        tx = f(vx)
        alpha_list[i] = r['k(V)'] * L2 / tx
        tx_list[i] = tx

    out_df = pd.DataFrame(data={
        'V(%)': params_df['V (%)'].values,
        'tx (s)': tx_list,
        'alpha (cm^2/s)': alpha_list
    })

    # print(out_df)

    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.75, 3.25)
    # ax.plot(time_s, v_exp, c='C0', marker='o', ls='none', fillstyle='none', label='Experiment')
    ax.plot(time_s / th, v_exp, c='C0', marker='o', ls='none', fillstyle='none', label='Experiment')
    ax.plot(t_model / th, v_model, c='k', fillstyle='none', label='Model')
    # ax.plot(t_model / th_corr, v_model_corr, c='g', fillstyle='none', label='Model (corrected)')
    ax.plot(tth_extrapolate, v_extrapolate, c='tab:purple', ls='--', lw=1.25)
    ax.plot([5], [v5], ls='none', marker='s', c='blue')

    ax.set_title(f"Laser current {laser_set_point:.0f} %")

    fm = interp1d(t_model / th, v_model, bounds_error=False, fill_value='extrapolate')

    curve_df = pd.DataFrame(
        data={
            't/t_h': time_s / th,
            'DT/T_max': v_exp,
            'DT/T_max (model)': fm(time_s / th)
        }
    )

    curve_df.to_csv(os.path.join(base_dir, file_tag + '_ftd_curves.csv'), index=None)

    ax.set_xlabel('$t/t_{1/2}$')
    ax.set_ylabel('$\Delta T/ \Delta T_{\mathrm{max}}}$')
    ax.legend(
        loc='upper left', frameon=True
    )

    ax.set_xlim(0., 5.)
    ax.set_ylim(0., 1.)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    alpha_txt = f'$\\alpha_{{0.5}} = ${ah:.3f}±{0.05 * ah:0.3f} cm$^{{\\mathregular{{2}}}}$/s\n'
    alpha_txt += f'$\\alpha_{{\mathrm{{corrected, rad}}}} = ${a_corr_rad:.3f}±{0.05 * a_corr_rad:.3f} cm$^{{\\mathregular{{2}}}}$/s\n'
    alpha_txt += f'$\\alpha_{{\mathrm{{corrected, hl}}}} = ${a_corr_hl:.3f}±{0.05 * a_corr_hl:.3f} cm$^{{\\mathregular{{2}}}}$/s\n\n'
    alpha_txt += f'$K_{{\mathrm{{rd}}}}$ = {thermal_conductivity_rad:.3f}±{thermal_conductivity_rad * 0.05:.3f} W/cm-K\n'
    alpha_txt += f'$K_{{\mathrm{{hl}}}}$ = {thermal_conductivity_hl:.3f}±{thermal_conductivity_rad * 0.05:.3f} W/cm-K\n'
    alpha_txt += f'$P_{{\mathrm{{in}}}}$ = {input_power:.3f} W\n'
    alpha_txt += f"$P_{{\mathrm{{laser}}}}$ = {laser_power:.2f} W\n"
    alpha_txt += f"$Q_{{\mathrm{{in}}}}$ = {Q:.3f} W/cm$^{{\mathregular{{2}}}}$\n"
    alpha_txt += f"$\\Delta T_{{\mathrm{{m}}}}$ = {dT.max():.1f} °C ($T_0$ = {temperature[0]:.1f} °C)"

    logger.info(f"Emission time: {pulse_length:.3f} s")
    logger.info(f"Laser power: {laser_power:.3f} W")
    logger.info(f"Sample area: {area:.3f} ± {area_err:.4f} cm^2")
    logger.info(f"Sample diameter: {sample_diameter:.3f} ± {sample_diameter_err:.4f} cm")
    logger.info(f"Sample length: {sample_length_cm:.3f} cm")
    logger.info(f"Sample density: {density_g:.3f} ± {density_g_err} g/cm^3")
    logger.info(f"Sample's specific heat: {specific_heat_g:.3f} J/g-K")
    logger.info(f"dT.max: {dT.max():.3f} °C")
    logger.info(f"T max: {temperature.max():.3f} °C")
    logger.info(f"alpha_0.5: {ah:.4E} cm^2/s")
    logger.info(f"alpha_c (radiation): {a_corr_rad:.4f} cm^2/s")
    logger.info(f"alpha_c (heat loss): {a_corr_hl:.4f} cm^2/s")
    logger.info(f"Kc: {kc:.3E}")
    logger.info(f"Q_in: {Q:.3E} W/cm^2")

    ax.text(
        0.95, 0.05, alpha_txt, ha='right', va='bottom',
        transform=ax.transAxes,
        fontsize=10
    )

    ax.axvline(x=1, c='tab:red', ls='--', lw=1.25)
    ax.axhline(y=0.5, c='tab:red', ls='--', lw=1.25)

    fig.savefig(os.path.join(base_dir, file_tag + '_flash_td.png'), dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
