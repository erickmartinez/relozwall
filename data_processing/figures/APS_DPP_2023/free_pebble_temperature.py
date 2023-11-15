"""
This code fits the temperature of the pebble released after a laser heat load exposure
to a radiative cooling process. See:
http://hyperphysics.phy-astr.gsu.edu/hbase/thermo/cootime.html
"""
import pandas as pd
import numpy as np
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, OptimizeResult, newton, differential_evolution, brentq
import matplotlib.ticker as ticker
import data_processing.confidence as cf
from data_processing.utils import lighten_color, latex_float, latex_float_with_error
import scipy.special as sc
from scipy.integrate import simps
from scipy.interpolate import interp1d

base_dir = r"C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\DPP 2023\figures\temperature_stats"
pid = 4
emissivity = 0.8
cp = 0.714  # J / g / K
cp_err = 0.022
rho = 1.372  # g / cm^3
rho_err = 0.003
thermal_conductivity = 0.067  # W/cm-K
thermal_conductivity_err = 0.003  # W/cm-K
thermal_diffusivity = thermal_conductivity / (cp * rho)  # cm^2 / s
thermal_diffusivity_err = thermal_diffusivity * np.linalg.norm(
    [thermal_conductivity_err / thermal_conductivity, cp_err / cp, rho_err / rho])  # cm^2 / s
pebble_exposure_time = 0.1  # s
pebble_exposure_time_err = 0.01  # s
heat_of_sublimation = 170.39  #
activation_energy = 8.2  # eV
sample_diameter_cm = 1.0


def poly(x, b):
    xx = np.ones_like(x, dtype=np.float64)
    r = np.zeros_like(x, dtype=np.float64)
    n = len(b)
    for i in range(n):
        r += xx * b[i]
        xx *= x
    return r


def fobj(b, x, y):
    return poly(x, b) - y


def jac(b, x, y):
    xx = np.ones_like(x, dtype=np.float64)
    m, n = len(x), len(b)
    jj = np.zeros((m, n), dtype=np.float64)
    for i in range(n):
        jj[:, i] = xx
        xx *= x
    return jj


def exponential(x, b):
    return b[0] * np.exp(b[1] * x)


kcalpermol2eV = 0.043364104241800934
kb_ev = 8.617333262E-5
by_na = 1. / 6.02214076E23
all_tol = np.finfo(np.float64).eps
ea_by_kb = activation_energy / kb_ev

sphere_diameter_cm = 0.09
diffusion_length = np.sqrt(thermal_diffusivity * pebble_exposure_time)
temp0 = 2800.

delta_temp = 0.002
TSTEPS = 3000
trial_temp = np.linspace(298, temp0, 96230)

d_average_cm = 0.09
hs_jpermol = heat_of_sublimation * 4184.
subl_f = hs_jpermol * by_na


def f_msr(temp_k, t, r0, d, ld, cp_, rho_, temp_0, ea=activation_energy):
    global hs_jpermol
    global subl_f
    v_hot = (np.pi / 12.) * (d ** 3. - (d - 2. * ld) ** 3.)
    # if ea is None:
    #     ea = kcalpermol2eV * hs_kcal
    eaperkt = ea / (temp_k * kb_ev)
    eaperkt0 = ea / (temp_0 * kb_ev)
    cp_rho_vhot = cp_ * rho_ * v_hot
    r = -cp_rho_vhot * (temp_k * np.exp(eaperkt) - ea / kb_ev * sc.expi(eaperkt))
    r += cp_rho_vhot * (temp_0 * np.exp(eaperkt0) - ea / kb_ev * sc.expi(eaperkt0))
    r -= (subl_f * r0) * t
    return np.real(r)


def fobj_exp(b, x, y):
    return exponential(x, b) - y


def jac_exp(b, x, y):
    m, n = len(x), len(b)
    j0 = np.exp(b[1] * x)
    j1 = b[0] * j0 * x
    jj = np.empty((m, n), dtype=np.float64)
    jj[:, 0] = j0
    jj[:, 1] = j1
    return jj


def load_plot_style():
    with open('../../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def model_sublimation_rate(t_, b):
    global sphere_diameter_cm
    global diffusion_length
    global heat_of_sublimation
    global cp
    global rho
    global temp0
    r0 = 10. ** b[0]
    y = np.empty(len(t_), dtype=np.float64)
    for i, ti in enumerate(t_):
        x0 = 298.
        y[i] = newton(
            func=f_msr, x0=x0, args=(ti, r0, sphere_diameter_cm, diffusion_length, cp, rho, temp0),
            tol=all_tol ** 0.5, maxiter=100000, rtol=all_tol ** 0.5, disp=True
        )
    return y


def fobj_sublimation(b, t_, temp_k):
    return model_sublimation_rate(t_, b) - temp_k


def fobj_sublimation_de(b, t_, temp_k):
    return np.linalg.norm(model_sublimation_rate(t_, b) - temp_k) * 0.5


sbc = 5.67034419E-12  # W/cm^2/K^4
rad_f = np.pi * sbc * d_average_cm ** 2.
tm_f = cp * rho * (np.pi / 12.) * (sphere_diameter_cm ** 3. - (sphere_diameter_cm - 2. * diffusion_length) ** 3.)


def f_radsub(T, rate_0, t_):
    global temp0
    global delta_temp
    global rad_f
    global subl_f
    global activation_energy
    m = int(abs(temp0 - T) / delta_temp)
    x = T + delta_temp * np.arange(0, m + 1)
    y = rad_f * np.power(x, 4.)
    y += subl_f * rate_0 * np.exp(-ea_by_kb / x)
    y = np.power(y, -1.)
    return tm_f * simps(y, x) - t_


def model_radiation_sublimation(t_, b):
    global trial_temp
    r0 = 10 ** b[0]
    y = np.empty(len(t_), dtype=np.float64)
    # x0 = 3000.
    for i, ti in enumerate(t_):
        x0 = trial_temp[i]
        yi = newton(
            func=f_radsub, x0=x0, args=(r0, ti),
            tol=all_tol ** 0.4, maxiter=1000000, rtol=all_tol ** 0.4, disp=True
        )
        y[i] = yi
    return y


def fobj_radsub(b, t_, temp_k):
    global temp0
    global delta_temp
    global rad_f
    global subl_f
    global activation_energy
    r0 = 10 ** b[0]
    n = len(temp_k)
    y = np.empty(n)
    for i in range(n):
        # if temp0 == temp_k[i]:
        #     yy = rad_f * np.power(temp0, 4) + subl_f * r0 * np.exp(-ea_by_kb / temp0)
        #     yy = np.power(y, - 1)
        #     y[i] = yy * delta_temp
        #     continue
        m = int(abs(temp0 - temp_k[i]) / delta_temp)
        x = temp_k[i] + delta_temp * np.arange(0, m + 1)
        yy = rad_f * np.power(x, 4.)
        yy += subl_f * r0 * np.exp(-ea_by_kb / x)
        # print('x:', x)
        # print('rad_f:', rad_f)
        # print('subl_f * r0', subl_f * r0)
        # print(yy)
        yy = np.power(yy, -1.)
        y[i] = tm_f * simps(yy, x)
    return y - t_


def fobj_radsub_de(b, t_, temp_k):
    return 0.5 * np.linalg.norm(fobj_radsub(b, t_, temp_k))


def main():
    global temp0
    global TSTEPS
    global trial_temp
    sample_area = 0.25E14 * np.pi * (sample_diameter_cm) ** 2.
    temp_df = pd.read_csv(os.path.join(base_dir, 'selected_cooling_data.csv')).apply(pd.to_numeric)
    # temp_df = temp_df[temp_df['PID'] == pid]
    t, temp, temp_lb, temp_ub = temp_df['t [s]'].values, temp_df['T [K]'].values, temp_df['T_lb [K]'].values, temp_df[
        'T_ub [K]'].values
    temp, temp_lb, temp_ub = temp, temp_lb, temp_ub
    byT3 = np.power(temp, -3.)
    byT3_0 = np.power(temp.max(), -3.)
    byT3_lb, byT3_ub = np.power(temp_lb, -3), np.power(temp_ub, -3)
    yerr_byT3 = np.abs(np.array([byT3 - byT3_lb, byT3_ub - byT3]))
    yerr_T = np.abs(np.array([temp - temp_lb, temp_ub - temp]))

    R = 8.31446261815324  # J / K / mol
    m_c = 12.011  # g/mol
    # sbc = 5.67034419E-8 # W/m^2/K^4
    sbc = 5.67034419E-12  # W/cm^2/K^4
    diffusion_length_err = 0.5 * diffusion_length * np.linalg.norm(
        [thermal_diffusivity_err / thermal_diffusivity, pebble_exposure_time_err / pebble_exposure_time])
    print(f"Diffusion length: {diffusion_length:.3E}±{diffusion_length_err} cm")

    b0 = [byT3_0, -1.]

    n = len(t)
    res: OptimizeResult = least_squares(
        fobj,
        b0,
        loss='soft_l1', f_scale=0.1,
        jac=jac,
        args=(t, byT3),
        # bounds=([0., 0.], [np.inf, np.inf]),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        diff_step=all_tol,
        max_nfev=10000 * n,
        method='trf',
        x_scale='jac',
        verbose=2
    )

    intercept, slope = res.x
    # slope = 12 * sbc * m_c / d / rho / R
    # d = 12 * sbc * m_c / slope / rho / R
    # [cm] = 12 x [5.6703 x10^(-12) W/cm^2/K^4] x [12.011 g/mol] / [8.314 J/mol/K] / [1.372 g/cm^3])/[slope K^{-3}s^{-1}]
    # d_cm = 12. * sbc * m_c / R / rho / slope
    # d_mm = d_cm * 10.
    # print('len(t)', len(t), 'len(temp)', len(temp))
    temp0 = temp[0]
    TSTEPS = int(temp0 / delta_temp)

    print(f"Temp[0]: {temp0:.0f} (K)")

    ci = cf.confidence_interval(res=res)
    # ci_sublimation = cf.confidence_interval(res=res_sublimation)

    d_slope = max(np.abs(ci[1, :] - slope))
    d_intercept = max(np.abs(ci[0, :] - intercept))

    a_cal = 6. * sbc * emissivity / slope / cp / rho / diffusion_length
    c_cal = 1. - a_cal
    print(f"a_cal: {a_cal:.3E}")
    print(f"c_cal: {c_cal:.3E}")
    d_cm = diffusion_length * (1. - np.sqrt(1. - (4. / 3.) * c_cal)) / c_cal

    b_cal = 1. - (4. / 3.) * c_cal
    d_cm_err = diffusion_length * (1. + np.sqrt(b_cal) + (4. / 3.) * c_cal / np.sqrt(b_cal)) * (b_cal ** 2.)
    d_cm_err *= np.linalg.norm([diffusion_length_err / diffusion_length, d_slope / slope, cp_err / cp, rho_err / rho])
    d_mm = 10. * d_cm
    d_mm_err = 10. * d_cm_err

    xp = np.linspace(t.min(), t.max(), 500)
    yp, lpb, upb = cf.predint(x=xp, xd=t, yd=byT3, func=poly, res=res)

    f_interp = interp1d(t, temp)
    trial_temp = f_interp(xp)

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)


    ax.errorbar(
        t, temp, yerr=yerr_T, marker='o', ms=7, fillstyle='none', ls='none', mew=1.5, color='C0', mfc='none',
        capsize=2.75, elinewidth=1.25,
        lw=1.5, label='Experiment', zorder=2
    )


    rt3 = -1. / 3.
    ax.fill_between(xp, np.power(lpb, rt3), np.power(upb, rt3), color=lighten_color('C0', 0.3), zorder=1)

    # ax_exp.fill_between(xp, lpb_sub, upb_sub, color=lighten_color('C0', 0.3), label='95% Prediction', zorder=1)

    model_lbl = r'$\dfrac{1}{T^3}=\dfrac{1}{T_0^3} + \dfrac{36\sigma\epsilon d^2}{c_{\mathrm{p}}\rho\left[d^2-(d-2L_{\mathrm{d}})^3\right]} t$'
    ax.plot(xp, np.power(yp, rt3), color='C0', ls='-', lw=1.5, label=model_lbl,
            zorder=3)


    ax.set_xlabel('t [s]')
    ax.set_ylabel(r'T [K]')
    ax.set_title('Free pebble temperature')

    ax.set_xlim(0., 0.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.005))

    ax.ticklabel_format(axis='y', useMathText=True)

    ax.legend(loc='upper right', frameon=True, fontsize=11)

    # ax.set_ylim(1250, 2750)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(250))
    temp_0 = np.power(intercept, -1. / 3.)
    results_txt = f"$T_0$: {temp_0:.0f} K\n"
    results_txt += f"Slope: $\\mathregular{{{latex_float_with_error(slope, d_slope, 2)}}}$ " + "1/$(\\mathregular{K^3 s})$\n"
    results_txt += f"Pebble diameter: {d_mm:.3f}±{d_mm_err:.3f} mm"
    ax.text(
        0.95, 0.4,
        results_txt,
        horizontalalignment='right',
        verticalalignment='bottom',
        color='b',
        transform=ax.transAxes,
        fontsize=9
    )

    fig.savefig(os.path.join(base_dir, f"radiative_cooling_R4N85_PID-{pid}.png"), dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
