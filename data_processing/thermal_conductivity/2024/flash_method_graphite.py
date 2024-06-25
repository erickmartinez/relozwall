import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
from data_processing.utils import get_experiment_params, latex_float_with_error
from scipy.stats.distributions import t
from scipy.interpolate import interp1d
import json
from scipy.signal import savgol_filter

data_dir = './data'

thickness_mm = 4.0
thickness_err_mm = 0.05

w_h, w_x = 1.3698, 0.3172

cmap_name = 'jet'

cowan_corrections_df = pd.DataFrame(data={
    'Coefficient': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    '5 half times': [-0.1037162, 1.239040, -3.974433, 6.888738, -6.804883, 3.856663, -1.167799, 0.1465332],
    '10 half times': [0.054825246, 0.16697761, -0.28603437, 0.28356337, -0.13403286, 0.024077586, 0.0, 0.0]
})

"""
For density estimation
"""
rod_diameters_mm = np.array([
    9.70, 9.69, 9.65, 9.65, 9.57, 9.62, 9.56, 9.57, 9.60, 9.65, 9.63, 9.74, 9.75, 9.74, 9.56, 9.81, 9.65, 9.60,
    9.63, 9.57
])
rod_mass_g = 14.978
rod_mass_error_g = 0.002
rod_length_mm = 113.93
cp = 0.710  # J /g-K
cp_err = 0.1 * cp

beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707
# time response
# https://www.omega.com/en-us/resources/thermocouples-response-time?utm_source=google&utm_medium=cpc&utm_campaign=Omega_NA_US_Core_Performance_Max_Pressure_Transducers&utm_content=undefined&utm_term=go_cmp-17887736118_adg-_ad-__dev-c_ext-_prd-_mca-_sig-Cj0KCQjwsuSzBhCLARIsAIcdLm50zO4D1j9JLKN5_kIJjNoaOKwwTJMFCt3Boh_xnvb5G4opHLSSHloaAuc4EALw_wcB&gad_source=1&gclid=Cj0KCQjwsuSzBhCLARIsAIcdLm50zO4D1j9JLKN5_kIJjNoaOKwwTJMFCt3Boh_xnvb5G4opHLSSHloaAuc4EALw_wcB
tc_time_response_s = 0.25

def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)

def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def correct_thermocouple_response(measured_temperature, measured_time, tau):
    n = len(measured_time)
    k = int(n / 20)
    k = k + 1 if k % 2 == 0 else k
    k = max(k, 5)
    # T = savgol_filter(measured_temperature, k, 3)
    # dTdt = np.gradient(T, measured_time, edge_order=2)
    delta = measured_time[1] - measured_time[0]
    dTdt = savgol_filter(x=measured_temperature, window_length=k, polyorder=4, deriv=1, delta=delta)
    # dTdt = savgol_filter(dTdt, k - 2, 3)
    r = measured_temperature + tau * dTdt
    return savgol_filter(r, k - 4, 3)

def main():
    global data_dir, cmap_name, thickness_mm, thickness_err_mm, cowan_corrections_df
    global rod_length_mm, rod_mass_g, rod_mass_error_g, rod_diameters_mm
    global cp, cp_err, beam_radius, tc_time_response_s
    thickness_cm = thickness_mm * 0.1
    l2 = thickness_cm ** 2.
    diameter_cm = rod_diameters_mm.mean() * 0.1
    diameter_std = rod_diameters_mm.std(ddof=1) * 0.1
    n_d = rod_diameters_mm.size
    confidence = 0.95
    alpha = 1 - confidence
    t_val = t.ppf(1. - 0.5 * alpha, n_d - 1)
    diameter_se = diameter_std * t_val / np.sqrt(n_d)

    rod_length_cm = rod_length_mm * 0.1
    rod_area = 0.25 * np.pi * diameter_cm ** 2.
    rod_area_err = 2. * rod_area * diameter_se / diameter_cm
    volume = rod_area * rod_length_cm
    volume_error = volume * np.linalg.norm([rod_area_err/rod_area, 0.005/rod_length_cm])
    density = rod_mass_g / volume
    density_error = density * np.linalg.norm([rod_mass_error_g/rod_mass_g, volume_error/volume])
    print(f'Volume: {volume:.3f} -/+ {volume_error:.4f} cm^3')
    print(f'Density: {density:.3f} -/+ {density_error:.4f} g/cm^3')

    # Load the dimensionless parameters
    kval_df = pd.read_csv('../dimensionless_parameters.csv').apply(pd.to_numeric)
    theory_df = pd.read_csv('../flash_curve_theory.csv').apply(pd.to_numeric)

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

    n_cols = 2
    n_rows = max(int(n_files / 2), 1)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, constrained_layout=True)
    # fig.subplots_adjust(hspace=0)
    fig.set_size_inches(5.5, 6.0)

    # axes[-1, 0].set_xlabel('$t/t_{1/2}$')
    # axes[-1, 1].set_xlabel('$t/t_{1/2}$')
    # ax.set_ylabel(r'$\Delta T/ \Delta T_{\mathrm{max}}$')
    fig.supylabel(r'$\Delta T/ \Delta T_{\mathrm{max}}$')
    fig.supxlabel(r'$t/t_{1/2}$')

    cowan_coefficients = cowan_corrections_df['5 half times'].values

    alpha_df = pd.DataFrame(columns=[
        'Laser power setting (%)', 'Laser power (W)', 'Laser power error (W)',
        'Laser power density (MW/m^2)', 'Laser power density error (MW/m^2)',
        'Emission time (s)',
        'alpha_0.5 (cm^2/s)',
        'alpha_c (cm^2/s)', 'kappa_c (W/m-K)', 'kappa_c error (W/m-K)', 'q (MW/m^2)', 'q error (MW/m^2)',
        'absorption_coefficient'
    ])

    for i, fn in enumerate(list_files):
        file_tag = os.path.splitext(fn)[0]
        params = get_experiment_params(relative_path=data_dir, filename=file_tag)
        data_df: pd.DataFrame = pd.read_csv(os.path.join(data_dir, fn), comment='#').apply(pd.to_numeric)
        time_s = data_df['Measurement Time (s)'].values
        temperature_raw = data_df['TC1 (C)'].values #+ 273.15
        temperature = correct_thermocouple_response(
            measured_temperature=temperature_raw, measured_time=time_s, tau=tc_time_response_s
        ) + 273.15
        # temperature = temperature_raw
        temperature_max = temperature.max()
        idx_peak = np.argmin(np.abs(temperature - temperature_max))
        time_red = time_s[0:idx_peak]
        temperature_red = temperature[0:idx_peak]

        laser_power = data_df['Laser output peak power (W)'].values
        laser_setpoint = float(params['Laser Power Setpoint']['value'])
        emission_time = float(params['Emission Time']['value'])

        msk_power = laser_power > 0.
        laser_power = laser_power[msk_power]
        laser_power_mean = np.mean(laser_power)
        laser_power_std = np.std(laser_power, ddof=1)
        n_p = len(laser_power)
        t_val_lp = t.ppf(1. - 0.5 * alpha, n_p - 1)
        laser_power_se = laser_power_std * t_val_lp / np.sqrt(n_p)

        dT = temperature - temperature.min()
        dT_max = dT.max()
        v = dT / dT_max
        f_inv = interp1d(x=v, y=time_s, bounds_error=False, fill_value='extrapolate')
        f_inv_red = interp1d(x=v[0:idx_peak], y=time_red, fill_value='extrapolate')
        t_h = f_inv_red(0.5)
        # idx_h = np.argmin(np.abs(v[0:idx_peak] - 0.5))
        # t_h = time_s[idx_h]
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
        dt5 = f_v(5.) / 0.5
        # print(f'dt5: {dt5:.3f}')
        kc = cowan_coefficients[0]
        xx = dt5
        for j in range(1, 8):
            # print(f'kc = {kc:.5f}, coeff_{j+1}: {cowan_coefficients[j]:.5f}, xx:{xx}')
            kc += xx * cowan_coefficients[j]
            xx *= dt5
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
        kappa_c_err = kappa_c * np.linalg.norm([density_error/density, cp_err/cp])

        q = density * cp * thickness_cm * temperature.max() / emission_time * 1E-2
        dT_err = np.sqrt(2.) * 0.25
        q_err = q * np.linalg.norm([density_error/density, cp_err/cp, dT_err/dT_max, thickness_err_mm/thickness_mm])
        af = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.5*diameter_cm)
        laser_power_aperture = af * laser_power_mean / rod_area * 1E-2
        laser_power_aperture_err = laser_power_aperture * np.linalg.norm([
            laser_power_se / laser_power_mean, rod_area_err / rod_area
        ])

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
            'q (MW/m^2)': [q],
            'q error (MW/m^2)': [q_err],
            'absorption_coefficient': [q / laser_power_aperture]
        })

        alpha_df = pd.concat([alpha_df, results_row]).reset_index(drop=True)

        idx_c = int(i / n_rows)
        idx_r = i % n_rows

        axes[idx_r, idx_c].plot(
            t_by_th, v, ls='-', color=colors[i], label=fr'$\kappa = {kappa_c:.2f} \pm {kappa_c_err:.3f}~\mathrm{{W/m/K}}$'
        )

        axes[idx_r, idx_c].set_title(fr'$q_{{\mathrm{{L}}}} = {laser_power_mean:.0f}~\mathrm{{W}}$')

        axes[idx_r, idx_c].legend(
            loc='lower right', frameon=True, fontsize=10
        )

        axes[idx_r, idx_c].plot(t_by_th_theory, v_theory, color='k')

        # axes[i].set_ylabel(r'$\Delta T/ \Delta T_{\mathrm{max}}$')
        axes[idx_r, idx_c].set_xlim(0, 10)
        axes[idx_r, idx_c].set_ylim(0, 1.05)
        axes[idx_r, idx_c].xaxis.set_major_locator(ticker.MultipleLocator(1))

    alpha_df.to_csv('flash_method_graphite_20240624.csv', index=False)
    print(alpha_df)
    plt.show()


if __name__ == '__main__':
    main()
