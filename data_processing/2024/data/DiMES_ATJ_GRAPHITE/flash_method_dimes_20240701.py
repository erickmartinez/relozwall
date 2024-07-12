import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
from data_processing.utils import get_experiment_params, latex_float_with_error, specific_heat_of_graphite
from scipy.stats.distributions import t
from scipy.interpolate import interp1d
import json
from scipy.signal import savgol_filter
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf

data_dir = './20240701/flash_method'

emission_time_s = 0.5
density = 1.76  # g/cm3
thickness_mm = 25.5
diameter_mm = 47.70
hp = 10.0
P = 26.3
rh = diameter_mm * 0.5

w_h, w_x = 1.3698, 0.3172

cmap_name = 'jet'

cowan_corrections_df = pd.DataFrame(data={
    'Coefficient': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    '5 half times': [-0.1037162, 1.239040, -3.974433, 6.888738, -6.804883, 3.856663, -1.167799, 0.1465332],
    '10 half times': [0.054825246, 0.16697761, -0.28603437, 0.28356337, -0.13403286, 0.024077586, 0.0, 0.0]
})

def g(x):
    global rh, hp, P
    return x * np.sin(x) - rh / P

def gp(x):
    return x * np.cos(x) + np.sin(x)



beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707
# time response
tc_time_response_s = np.array([0.522, 0.454, 0.477]).mean()


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def mean_and_standard_error(values: np.ndarray, confidence: float = 0.95) -> tuple:
    n = len(values)
    if n <= 2:
        raise ValueError(f'Cannot perform statistics with n = {n}')
    alpha = 1. - confidence
    tval = t.ppf(1. - 0.5 * alpha, n - 1)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    se = std * tval / np.sqrt(n)
    return mean, se


def correct_thermocouple_response(measured_temperature, measured_time, tau):
    n = len(measured_time)
    k = int(n / 40)
    k = k + 1 if k % 2 == 0 else k
    k = max(k, 5)
    # T = savgol_filter(measured_temperature, k, 3)
    # dTdt = np.gradient(T, measured_time, edge_order=2)
    delta = measured_time[1] - measured_time[0]
    dTdt = savgol_filter(x=measured_temperature, window_length=k, polyorder=3, deriv=1, delta=delta)
    # dTdt = savgol_filter(dTdt, k - 2, 3)
    r = measured_temperature + tau * dTdt
    return savgol_filter(r, k, 3)


def poly(x, b):
    m, n = len(x), len(b)
    r = np.zeros(len(x))
    xi = np.ones(m)
    for i in range(n):
        r += xi * b[i]
        xi *= x
    return r


def linear(x, b):
    return x * b[0]


def res_linear(b, x, y, w=1.):
    return (linear(x, b) - y) * w


def jac_linear(b, x, y, w=1.):
    jac = np.empty((len(x), 1), dtype=np.float64)
    jac[:, 0] = x * w
    return jac


def res_poly(b, x, y, w=1.):
    return (poly(x, b) - y) * w


def jac_poly(b, x, y, w=1.):
    m, n = len(x), len(b)
    jac = np.ones((m, n), dtype=np.float64)
    jac[:, 0] *= w
    xi = x.copy()
    for i in range(1, n):
        jac[:, i] = w * xi
        xi *= x
    return jac


def e_to_q(x):
    global emission_time_s
    return (x / emission_time_s) * 1E-2


def q_to_e(x):
    global emission_time_s
    return x * emission_time_s * 100.


def main():
    global data_dir, cmap_name, cowan_corrections_df
    global density, thickness_mm, rh
    global beam_radius, tc_time_response_s, diameter_mm
    thickness_cm = thickness_mm * 0.1
    thickness_err_cm = 0.05 * thickness_cm
    density_error = density * 0.02
    l2 = thickness_cm ** 2.
    diameter_cm = 0.1 * diameter_mm

    # find the radius of curvature of the front hemisphere
    theta_0 = 0.5 * np.pi
    all_tol = float(np.finfo(np.float64).eps)
    for i in range(100):
        print(f'theta_i: {theta_0*180./np.pi:.6f}')
        theta_n = theta_0 - g(theta_0) / gp(theta_0)
        if np.abs(theta_n - theta_0) < all_tol:
            break
        theta_0 = theta_n
    print(f'Theta: {theta_n/np.pi:.3} pi ({theta_n*180./np.pi:.2f} °)')
    R = 0.1* rh / np.sin(theta_n)
    print(f'R: {R:.1f} cm')
    area_cm = 2. * np.pi * (R ** 2.) * (1. - np.cos(theta_n))
    print(f'Front surface area: {area_cm:.2f} cm^2')

    # Load the dimensionless parameters
    path_to_theory = os.path.abspath('../../../thermal_conductivity')
    kval_df = pd.read_csv(os.path.join(path_to_theory, 'dimensionless_parameters.csv')).apply(pd.to_numeric)
    theory_df = pd.read_csv(os.path.join(path_to_theory, 'flash_curve_theory.csv')).apply(pd.to_numeric)

    kval_df = kval_df[kval_df['V (%)'].isin([25., 50., 75.])]
    kval_v = kval_df['V (%)'].values
    kval_k = kval_df['k(V)'].values

    omega = theory_df['w'].values
    v_theory = theory_df['V'].values
    t_by_th_theory = omega / w_h

    load_plot_style()

    data_dir = os.path.normpath(data_dir)
    list_files = [f for f in os.listdir(path=data_dir) if f.endswith('.csv')]
    n_files = len(list_files)
    cmap = mpl.colormaps.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=0, vmax=n_files - 1)
    colors = [cmap(norm(i)) for i in range(n_files)]

    n_cols = 1
    n_rows = 5  # max(int(n_files / 2) + 1, 2)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, constrained_layout=True)
    # fig.subplots_adjust(hspace=0)
    fig.set_size_inches(4., 7.0)

    # axes[-1, 0].set_xlabel('$t/t_{1/2}$')
    # axes[-1, 1].set_xlabel('$t/t_{1/2}$')
    # ax.set_ylabel(r'$\Delta T/ \Delta T_{\mathrm{max}}$')
    fig.supylabel(r'$\Delta T/ \Delta T_{\mathrm{max}}$')
    fig.supxlabel(r'$t/t_{1/2}$')

    cowan_coefficients_5 = cowan_corrections_df['5 half times'].values
    cowan_coefficients_10 = cowan_corrections_df['10 half times'].values

    alpha_df = pd.DataFrame(columns=[
        'Laser power setting (%)', 'Laser power (W)', 'Laser power error (W)',
        'Laser power density (MW/m^2)', 'Laser power density error (MW/m^2)',
        'Emission time (s)',
        'alpha_0.5 (cm^2/s)',
        'alpha_c (cm^2/s)', 'kappa_c (W/m-K)', 'kappa_c error (W/m-K)',
        'T_min (K)', 'DT_max (K)',
    ])

    emission_time_s = np.empty(n_files, dtype=np.float64)

    for i, fn in enumerate(list_files):
        file_tag = os.path.splitext(fn)[0]
        params = get_experiment_params(relative_path=data_dir, filename=file_tag)
        data_df: pd.DataFrame = pd.read_csv(os.path.join(data_dir, fn), comment='#').apply(pd.to_numeric)
        time_s = data_df['Measurement Time (s)'].values
        laser_power = data_df['Laser output peak power (W)'].values
        msk_power = laser_power > 0.
        time_pulse = time_s[msk_power]
        t0 = time_pulse[0]
        # idx_pulse = np.argmin(np.abs(time_s - t0))-1
        # t0 = time_s[idx_pulse]
        time_s -= t0
        temperature_raw = data_df['TC2 (C)'].values + 273.15
        msk_t0 = time_s > 0.
        time_s = time_s[msk_t0]
        temperature_raw = temperature_raw[msk_t0]
        temperature = correct_thermocouple_response(
            measured_temperature=temperature_raw, measured_time=time_s, tau=tc_time_response_s
        )

        cp = specific_heat_of_graphite(temperature_raw[0])
        cp_err = cp * 0.05
        # temperature = temperature_raw
        temperature_max = temperature.max()
        idx_peak = np.argmin(np.abs(temperature - temperature_max))
        time_red = time_s[0:idx_peak]
        temperature_red = temperature[0:idx_peak]

        laser_setpoint = float(params['Laser Power Setpoint']['value'])
        emission_time = float(params['Emission Time']['value'])
        emission_time_s[i] = emission_time

        laser_power = laser_power[msk_power]
        laser_power_mean, laser_power_se = mean_and_standard_error(values=laser_power)

        dT = temperature - temperature.min()
        dT_max = dT.max()
        v = dT / dT_max
        f_inv = interp1d(x=v, y=time_s, bounds_error=False, fill_value='extrapolate')
        f_inv_red = interp1d(x=v[0:idx_peak], y=time_red, fill_value='extrapolate')
        t_h = f_inv_red(0.5)
        # idx_h = np.argmin(np.abs(v[0:idx_peak] - 0.5))
        # t_h = time_s[idx_h]
        # print(f't_h = {t_h:.3f} s')
        t_by_th = time_s / t_h
        f_v = interp1d(x=t_by_th, y=v)

        # Find the time required to reach 0.1, ... 0.9 rise in V from the experiment
        n_k = len(kval_v)
        alpha_x = np.zeros(n_k)
        for j in range(0, n_k):
            vj = kval_v[j] / 100.
            kj = kval_k[j]
            tj = f_inv(vj) * t_h
            alpha_x[j] = kj * l2 / tj
            # print(f'Setpoint: {laser_setpoint:>3.0f}, alpha(t{vj:.2f}): {alpha_x[j]:>6.3E}')

        alpha_05 = 0.138785 * l2 / t_h
        # Radiative losses Cowan
        # print(f'dt5: {dt5:.3f}')
        kc = cowan_coefficients_5[0]
        dt5 = f_v(5.) / 0.5
        xx = dt5
        for j in range(1, 8):
            # print(f'kc = {kc:.5f}, coeff_{j+1}: {cowan_coefficients[j]:.5f}, xx:{xx}')
            kc += xx * cowan_coefficients_5[j]
            xx *= dt5

        try:
            dt10 = f_v(10.) / 0.5
            kc = cowan_coefficients_10[0]
            xx = dt5
            for j in range(1, 8):
                # print(f'kc = {kc:.5f}, coeff_{j+1}: {cowan_coefficients[j]:.5f}, xx:{xx}')
                kc += xx * cowan_coefficients_10[j]
                xx *= dt10
        except Exception as err:
            print(err)

        alpha_c = alpha_05 * kc / 0.13885

        # Heat loss Clark Taylor
        # v_reduced = v[t_by_th <= 2.]
        # t_reduced = time_s[t_by_th <= 2.]
        # f_inv_red = interp1d(x=v_reduced, y=t_reduced)
        # # idx_75 = np.argmin(np.abs(v - 0.75))
        # t075 = f_inv_red(0.75)
        # t025 = f_inv_red(0.25)
        # dt_ct = t075 / t025
        # # print(f't0.75/t_h: {t075/t_h:.3f}, v(t_0.75) = {f_v(t075/t_h):.3f}, t0.75/t0.25 = {dt_ct}')
        # kr = -0.3461467 + 0.361578 * dt_ct - 0.06520543 * dt_ct ** 2.
        # alpha_c = alpha_05 * kr / 0.13885

        kappa_c = alpha_c * cp * density * 100.
        kappa_c_err = kappa_c * np.linalg.norm([density_error / density, cp_err / cp])

        # q = density * cp * thickness_cm * temperature.max() / emission_time * 1E-2
        # dT_err = np.sqrt(2.) * 0.25
        # q_err = q * np.linalg.norm(
        #     [density_error / density, cp_err / cp, dT_err / dT_max, thickness_err_mm / thickness_mm])
        af = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.5 * diameter_cm)

        # area = 0.25 * np.pi * diameter_cm ** 2.
        area = area_cm
        area_err = area * 0.01

        laser_power_aperture = af * laser_power_mean / area * 1E-2
        laser_power_aperture_err = laser_power_aperture * np.linalg.norm([
            laser_power_se / laser_power_mean, area_err / area
        ])

        # print(f'Aperture factor: {af:.3f}, Laser heat load: {laser_power_aperture:.1f} MW/m^2')

        results_row = pd.DataFrame(data={
            'Laser power setting (%)': [laser_setpoint],
            'Laser power (W)': [laser_power_mean],
            'Laser power error (W)': [laser_power_se],
            'Emission time (s)': [emission_time],
            'Laser power density (MW/m^2)': [laser_power_aperture],
            'Laser power density error (MW/m^2)': [laser_power_aperture_err],
            'alpha_0.5 (cm^2/s)': [alpha_05],
            'alpha_c (cm^2/s)': [alpha_c],
            'kappa_c (W/m-K)': [kappa_c],
            'kappa_c error (W/m-K)': [kappa_c_err],
            'T_min (K)': [temperature.min()],
            'DT_max (K)': [dT_max],
        })

        alpha_df = pd.concat([alpha_df, results_row]).reset_index(drop=True)

        # idx_c = i % n_cols
        idx_r = int(i / n_cols)
        # print(f'idx_r: {idx_r}, idx_c: {idx_c}, axes.shape: {axes.shape}')

        axes[idx_r].plot(
            t_by_th, v, ls='-', color=colors[i],
            label=fr'$\kappa = {kappa_c:.0f} \pm {kappa_c_err:.0f}~\mathrm{{W/m/K}}$'
        )

        axes[idx_r].set_title(
            fr'$q_{{\mathrm{{L}}}} = {laser_power_mean:.0f}~\mathrm{{W}}~({laser_setpoint:>2.0f}\%)$')
        axes[idx_r].plot(t_by_th_theory, v_theory, color='k', label='Model')

        axes[idx_r].legend(
            loc='lower right', frameon=True, fontsize=9
        )

        # axes[i].set_ylabel(r'$\Delta T/ \Delta T_{\mathrm{max}}$')
        axes[idx_r].set_xlim(0, 10)
        axes[idx_r].set_ylim(0, 1.05)
        axes[idx_r].xaxis.set_major_locator(ticker.MultipleLocator(1))
        axes[idx_r].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    alpha_df.to_csv('flash_method_graphite_20240625.csv', index=False)

    """
    Find the relationship between q_laser and q_absorbed
    """
    t_emission = alpha_df['Emission time (s)'].values
    e_laser = alpha_df['Laser power density (MW/m^2)'].values * emission_time_s * 100.
    e_laser_err = alpha_df['Laser power density error (MW/m^2)'].values * emission_time_s * 100.
    dT_max = alpha_df['DT_max (K)'].values
    dT_max_err = 0.25 * np.sqrt(2.)
    weights = np.power(e_laser_err, -2)  # + np.power(dT_max_err, -2)
    # weights = 1.  # weights / weights.max()

    all_tol = float(np.finfo(np.float64).eps)
    res: OptimizeResult = least_squares(
        x0=[1.],
        fun=res_linear,
        args=(dT_max, e_laser, weights),
        jac=jac_linear,
        loss='soft_l1', f_scale=0.1,
        xtol=all_tol,  # ** 0.5,
        ftol=all_tol,  # ** 0.5,
        gtol=all_tol,  # ** 0.5,
        max_nfev=10000 * len(e_laser),
        # x_scale='jac',
        verbose=2
    )
    x_pred = np.linspace(dT_max.min(), dT_max.max(), num=100)
    ypred, delta = cf.prediction_intervals(model=linear, x_pred=x_pred, ls_res=res, jac=jac_linear, weights=weights)
    fig_q, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig_q.set_size_inches(4., 3.)

    ax.errorbar(
        dT_max, e_laser, yerr=e_laser_err,  # xerr=dT_max_err,
        color='C0', marker='o',
        ms=9, mew=1.25, mfc='none', ls='none',
        capsize=2.75, elinewidth=1.25, lw=1.5,
        label='Experiment'
    )
    ax.fill_between(x_pred, ypred - delta, ypred + delta, color='C0', alpha=0.25)
    ax.plot(
        x_pred, ypred, color='k', lw=1.25, ls='--', label=r'$f(x) = a_0 x $'
    )

    secaxy = ax.secondary_yaxis('right', functions=(e_to_q, q_to_e))

    ax.set_xlabel(r'$\Delta T_{\mathrm{max}}$ (K)')
    ax.set_ylabel(r'$E_{\mathrm{L}}$ (J/cm$^{\mathregular{2}}$)')
    secaxy.set_ylabel(r'$q_{\mathrm{L}}$ (MW/m$^{\mathregular{2}}$)')

    popt = res.x
    pcov = cf.get_pcov(res)
    ci = cf.confidence_interval(res)
    delta_popt = np.abs(ci[:, 1] - popt)
    fit_txt = ''
    poly_n = len(popt)
    for i in range(poly_n):
        fit_txt += fr'$a_{{{i}}} = {popt[i]:.1f} \pm {delta_popt[i]:.1f}$'
        if i + 1 < poly_n:
            fit_txt += '\n'
    absorbance = density * cp * thickness_cm / popt[0]
    absorbance_err = absorbance * np.linalg.norm(
        [delta_popt[0] / popt[0], density_error / density, cp_err / cp, thickness_err_cm / thickness_cm])
    fit_txt += '\n'
    fit_txt += r"$E_{\mathrm{L}} = (LDC_{\mathrm{p}}/\xi) \Delta T_{\mathrm{max}}$" + '\n'
    fit_txt += fr'$\xi = {absorbance:.2f} \pm {absorbance_err:.2f}$'

    ax.text(
        0.05, 0.95, fit_txt,
        transform=ax.transAxes,
        ha='left', va='top'
    )

    ax.legend(
        loc='lower right', frameon=True
    )

    ax.set_xlim(0., 12.)
    ax.set_ylim(0., 100.)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2.))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20.))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10.))
    secaxy.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    secaxy.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    fig.savefig('./figures/dimes_flash_method_20240701.png', dpi=600)
    fig_q.savefig('./figures/atj_graphite_absorption_20240701.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
