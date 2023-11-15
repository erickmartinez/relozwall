"""
This code plots the individual sphere temperature vs time for a pebble sample subject to a laser heat load
"""
import pandas as pd
import numpy as np
from data_processing.utils import get_experiment_params, latex_float, lighten_color
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import json
from scipy.optimize import least_squares, OptimizeResult, differential_evolution
from mpl_toolkits.axes_grid1 import make_axes_locatable
from data_processing import confidence as cf

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\DPP 2023\figures'
save_dir = 'temperature_stats'
tracking_csv = r'LCT_R4N85_manual_tracking.xlsx'
sheet_name = 'R4N85'
info_csv = r'LCT_R4N85_ROW375_100PCT_2023-08-18_1.csv'
calibration_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\CALIBRATION_20231010\calibration_20231010_4us.csv'
laser_power_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN\laser_output'

sample_diameter_cm = 1.025
beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707

pixel_size = 20.4215  # pixels/mm
px2mm = 1. / pixel_size
px2cm = 0.1 * px2mm
center_mm = np.array([12.05, 30.78])

sensor_pixel_size_cm = 3.45E-4

CAMERA_ANGLE = 18.
CAMERA_DISTANCE = 5.

DEG2RAD = np.pi / 180.
RAD2DEG = 1. / DEG2RAD
GO2 = 0.5 * 9.82E2

X_MIN, X_MAX = -0.5, 0.5
Y_MIN, Y_MAX = -0.5, 0.5
THETA_MAX = 90.

OBJECTIVE_F_CM = 9.  # The focal length of the objective
WD_CM = 34.  # The working distance

all_tol = np.finfo(np.float64).eps
frame_rate = 200.

laser_settings_plot = np.array([100, 80, 60])

emissivity = 1.
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

sphere_diameter_cm = 0.09
diffusion_length = np.sqrt(thermal_diffusivity * pebble_exposure_time)
temp0 = 3500.

def load_plot_style():
    with open('../../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def load_calibration(calibration_csv=calibration_csv):
    df = pd.read_csv(calibration_csv).apply(pd.to_numeric)
    return df['Temperature [K]'].values


def convert_to_temperature(adc, cal):
    if (type(adc) == list) or (type(adc) == np.ndarray):
        r = []
        for a in adc:
            r.append(cal[int(a)])
        return np.array(r)
    return cal[int(adc)]


def map_laser_power_settings():
    rdir = os.path.join(base_path, laser_power_dir)
    file_list = os.listdir(rdir)
    mapping = {}
    for i, f in enumerate(file_list):
        if f.endswith('.csv'):
            params = get_experiment_params(relative_path=rdir, filename=os.path.splitext(f)[0])
            laser_setpoint = int(params['Laser power setpoint']['value'])
            df = pd.read_csv(os.path.join(rdir, f), comment='#').apply(pd.to_numeric)
            laser_power = df['Laser output peak power (W)'].values
            laser_power = laser_power[laser_power > 0.0]
            mapping[laser_setpoint] = laser_power.mean()

    keys = list(mapping.keys())
    keys.sort()
    return {i: mapping[i] for i in keys}


class DictClass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)


def map_heat_loads():
    laser_power_map = map_laser_power_settings()
    sample_area = 0.25 * np.pi * sample_diameter_cm ** 2.0
    aperture_factor = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.5 * sample_diameter_cm)
    keys = list(laser_power_map.keys())
    keys.sort()
    return {i: aperture_factor * laser_power_map[i] / sample_area / 100.0 for i in keys}


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

def main():
    cal = load_calibration()
    laser_power_mapping = map_laser_power_settings()
    heat_loads_mapping = map_heat_loads()
    laser_setting_color_map = {100: 'tab:red', 80: 'darkorchid', 60: 'olivedrab'}
    file_tag = os.path.splitext(info_csv)[0]
    if not os.path.exists(os.path.join(base_path, save_dir)):
        os.makedirs(os.path.join(base_path, save_dir))
    params = get_experiment_params(relative_path=base_path, filename=file_tag)
    pulse_length = float(params['Emission Time']['value'])
    sample_name = params['Sample Name']['value']
    tracking_df: pd.DataFrame = pd.read_excel(io=os.path.join(base_path, tracking_csv), sheet_name=sheet_name).apply(
        pd.to_numeric)
    pids = tracking_df['PID'].unique()
    n = len(pids)
    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=0, vmax=n)
    norm2 = mpl.colors.LogNorm(vmin=60, vmax=100)
    cmap_v = plt.get_cmap('jet')

    cooling_data = pd.DataFrame(columns=[
        'PID', 't [s]', 'T [K]', 'T_lb [K]', 'T_ub [K]'
    ])

    fig, axes = plt.subplots(nrows=3, ncols=1, gridspec_kw=dict(hspace=0), sharex=True, sharey=True)
    fig.set_size_inches(4.0, 4.5)

    fig2, ax2 = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig2.set_size_inches(4.0, 3.0)

    selected_cooling_df = pd.DataFrame(columns=['t [s]', 'T [K]', 'T_lb [K]', 'T_ub [K]'])

    for k, p in enumerate(pids):
        pid_df = tracking_df.loc[tracking_df['PID'] == p]
        t = pid_df['t (s)'].values
        t0 = t[0]
        adc_raw = pid_df['Mean gray'].values
        adc_corrected = pid_df['Corrected gray'].values
        adc_delta = pid_df['95% corrected delta'].values
        laser_setting = pid_df['Laser power setting (%)'].values[0]
        laser_power = laser_power_mapping[laser_setting]
        adc_lb, adc_ub = adc_corrected - adc_delta, adc_corrected + adc_delta
        # temperature_raw = convert_to_temperature(adc=adc_raw, cal=cal)
        temperature = convert_to_temperature(adc=adc_corrected, cal=cal)
        temperature_lb = convert_to_temperature(adc=adc_lb, cal=cal)
        temperature_ub = convert_to_temperature(adc=adc_ub, cal=cal)

        # sample_area = 0.25 * np.pi * sample_diameter_cm ** 2.0
        # aperture_factor = gaussian_beam_aperture_factor(beam_radius=beam_radius, sample_radius=0.5 * sample_diameter_cm)
        # incident_heat_load = aperture_factor * laser_power / sample_area / 100.0
        # incident_heat_load_rnd = np.round(incident_heat_load / 5.) * 5.

        # for i in range(len(temperature)):
        #     print(f'T_lb: {temperature_lb[i]:>3.0f}, T: {temperature[i]:>3.0f}, T_ub: {temperature_ub[i]:>3.0f}')

        s = pid_df['d_proj (px)'].values
        msk_takeoff = s < 5.

        s_takeoff = s[msk_takeoff]
        t_takeoff = t[msk_takeoff]
        temp_takeoff = temperature[msk_takeoff]
        s_takeoff = s_takeoff[-1]
        t_takeoff = t_takeoff[-1]
        temp_takeoff = temp_takeoff[-1]

        # Get the position and temperature after the pebble has been ejected
        t_threshold = t_takeoff + 0. * 0.005
        msk_ejected = t >= t_threshold
        t_ejected = t[msk_ejected]

        temp_ejected = temperature[msk_ejected]
        temp_ejected_lb = temperature_lb[msk_ejected]
        temp_ejected_ub = temperature_ub[msk_ejected]

        v_ejection = np.nan

        if len(t_ejected) > 3:
            # for j in range(len(t_ejected)):
            cooling_data = pd.concat([
                cooling_data,
                pd.DataFrame(data={
                    'PID': [p for ii in range(len(t_ejected))],
                    't [s]': t_ejected - t_takeoff,
                    'T [K]': temp_ejected,
                    'T_lb [K]': temp_ejected_lb,
                    'T_ub [K]': temp_ejected_ub,
                })
            ])

        # Determine in which ax to plot
        ax_idx = (np.abs(laser_setting - laser_settings_plot)).argmin()
        if temperature[0] < 2500.:
            axes[ax_idx].plot(t - t0, temperature, lw=1.5, color=laser_setting_color_map[laser_setting], alpha=0.75)
            axes[ax_idx].plot(
                [t_takeoff - t0], [temp_takeoff], marker='o', markersize=4, mew=0.75, fillstyle='full', color='k',
                mec=laser_setting_color_map[laser_setting], alpha=0.5
            )

        yerr_neg = np.array([max(yy, 0.) for yy in (temp_ejected - temp_ejected_lb)])
        yerr_pos = temp_ejected_ub - temp_ejected
        if (laser_setting == 100) and (temp_ejected.min()>= 2000.):
            lbl = 'Data' if len(selected_cooling_df) == 0 else None
            ax2.errorbar(
                x=t_ejected - t_takeoff,
                y=temp_ejected,
                yerr=(yerr_neg, yerr_pos),
                ls='none', color='C0', marker='o', ms=7, mew=1.25, mfc='none',
                capsize=2.75, elinewidth=1.25, lw=1.5, zorder=3, label=lbl
            )
            selected_cooling_df = pd.concat([
                selected_cooling_df,
                pd.DataFrame(data={
                    't [s]': t_ejected - t_takeoff,
                    'T [K]': temp_ejected,
                    'T_lb [K]': temp_ejected_lb,
                    'T_ub [K]':temp_ejected_ub
                })
            ])

    selected_cooling_df = selected_cooling_df.apply(pd.to_numeric)

    """
    Fit the trend for the cooling 
    """
    time_cooling = selected_cooling_df['t [s]'].values
    temperature_cooling = selected_cooling_df['T [K]'].values
    byT3 = np.power(temperature_cooling, -3.)
    byT3_0 = byT3[0]
    R = 8.31446261815324  # J / K / mol
    m_c = 12.011  # g/mol
    # sbc = 5.67034419E-8 # W/m^2/K^4
    sbc = 5.67034419E-12  # W/cm^2/K^4
    temperature_cooling_t0 = temperature_cooling[0]
    diffusion_length_err = 0.5 * diffusion_length * np.linalg.norm(
        [thermal_diffusivity_err / thermal_diffusivity, pebble_exposure_time_err / pebble_exposure_time])
    print(f"Diffusion length: {diffusion_length:.3E}±{diffusion_length_err} cm")

    b0 = [byT3_0, -1.]

    n = len(temperature_cooling)
    all_tol = np.finfo(np.float64).eps
    res: OptimizeResult = least_squares(
        fobj,
        b0,
        # loss='soft_l1', f_scale=0.1,
        jac=jac,
        args=(time_cooling, byT3),
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
    ci = cf.confidence_interval(res=res)

    d_slope = max(np.abs(ci[1, :] - slope))
    d_intercept = max(np.abs(ci[0, :] - intercept))

    print(f"Temp[0]: {temp0:.0f} (K)")

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

    xp = np.linspace(time_cooling.min(), time_cooling.max(), 500)
    yp, lpb, upb = cf.predint(x=xp, xd=time_cooling, yd=byT3, func=poly, res=res)


    for i, ax in enumerate(axes):
        ax.set_xlim(0, 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.tick_params(which='both', axis='y', labelright=False, right=True, direction='in')


    axes[0].set_ylim(1900, 3600)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(500))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(100))


    axes[0].set_title('Pebble temperature')

    axes[2].set_xlabel('Time [s]')

    rt3 = -1. / 3.
    ax2.fill_between(xp, np.power(lpb, rt3), np.power(upb, rt3), color=lighten_color('C0', 0.25), zorder=1)
    model_lbl = r'$\dfrac{1}{T^3}=\dfrac{1}{T_0^3} + \dfrac{36\sigma\epsilon d^2}{c_{\mathrm{p}}\rho\left[d^2-(d-2L_{\mathrm{d}})^3\right]} t$'
    ax2.plot(xp, np.power(yp, rt3), color='k', ls='-', lw=1.25, label='Radiative cooling',
            zorder=2)

    ax2.set_xlabel(r'$\Delta$t [s]')
    ax2.set_ylabel('T [K]')
    ax2.set_title('Free pebble cooling')

    ax2.set_xlim(0, 0.05)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.005))

    ax2.set_ylim(2400,4000)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(100))


    legend_elements0 = [
        Line2D(
            [0], [0], color=laser_setting_color_map[100], lw=1.5,
            label=f'{round(heat_loads_mapping[100] / 5.) * 5.} MW/m$^{{\mathregular{{2}}}}$'
        )
    ]
    # legend_elements1 = [Line2D([0], [0], color='C0', label='Temperature'),
    #                     Patch(facecolor=lighten_color('C0', 0.5), edgecolor='C0', lw=1.5,
    #                           label='95% CI')]
    legend_elements1 = [
        Line2D(
            [0], [0], color=laser_setting_color_map[80], lw=1.5,
            label=f'{round(heat_loads_mapping[80] / 5.) * 5.} MW/m$^{{\mathregular{{2}}}}$'
        )
    ]
    legend_elements2 = [
        Line2D(
            [0], [0], color=laser_setting_color_map[60], lw=1.5,
            label=f'{round(heat_loads_mapping[60] / 5.) * 5.} MW/m$^{{\mathregular{{2}}}}$'
        )
    ]

    axes[0].legend(handles=legend_elements0, loc='lower right', fontsize=11)
    axes[1].legend(handles=legend_elements1, loc='lower right', fontsize=11)
    axes[2].legend(handles=legend_elements2, loc='lower right', fontsize=11)

    ax2.legend(loc='upper right', fontsize=10, frameon=True)

    fig.supylabel('Temperature [K]', fontweight='regular', fontsize=12)
    fig.tight_layout()

    fig.savefig(os.path.join(base_path, save_dir, 'pebble_temperature.png'), dpi=600)
    fig2.savefig(os.path.join(base_path, save_dir, 'free_pebble_temperature.png'), dpi=600)

    print(cooling_data)

    cooling_data.to_csv(os.path.join(base_path, save_dir, 'cooling_data.csv'), index=False, encoding='utf-8-sig')

    plt.show()


if __name__ == '__main__':
    load_plot_style()
    main()
