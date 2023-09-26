"""
This code plots the temperature vs time (and energy) of a pebble subject to a 35 MW/m^2 heat load
"""
import pandas as pd
import numpy as np
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, OptimizeResult
import matplotlib.ticker as ticker
import data_processing.confidence as cf
from data_processing.utils import lighten_color, latex_float, get_laser_power_mapping
from data_processing.camera.sphere_temperatures.sphere_temperature_analysis import load_calibration, \
    convert_to_temperature

base_dir = r"C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC\R4N85_stats"
pids = [6, 7, 11, 12, 13, 14]
sample_diameter_cm = 1.025  # cm
sample_diameter_cm_err = 0.015  # cm
beam_diameter = 0.8164  # cm
absorption_coefficient = 0.6
pebble_diameter_mm = 0.9

tracking_xls = r"C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC\LCT_R4N85_manual_tracking.xlsx"
sheet_name = 'R4N85'
calibration_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\CALIBRATION_20230726\GAIN5dB\adc_calibration_curve.csv'

rho = 1.372  # g / cm^3
m_c = 12.011  # g/mol

pixel_size = 20.4215  # pixels/mm
px2mm = 1. / pixel_size
px2cm = 0.1 * px2mm
center_mm = np.array([12.05, 30.78])


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)


def load_plot_style():
    with open('../../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def gaussian_beam_intensity(r, w, p):
    return (2. * p / np.pi / w ** 2.) * np.exp(-2.*(r / w) ** 2.)


def main():
    # load the particle tracking data
    tracking_df: pd.DataFrame = pd.read_excel(io=os.path.join(base_dir, tracking_xls), sheet_name=sheet_name).apply(
        pd.to_numeric)
    # load the IR camera ADC calibration
    cal = load_calibration(calibration_csv=calibration_csv)
    laser_power_mapping = get_laser_power_mapping()
    laser_powers = np.array(list(laser_power_mapping.values()))

    sample_area_cm = 0.25 * np.pi * sample_diameter_cm ** 2.
    aperture_factor = gaussian_beam_aperture_factor(beam_radius=beam_diameter, sample_radius=sample_diameter_cm)
    beam_diameter_err = 0.01 * beam_diameter
    d_aperture = 4. * (sample_diameter_cm / beam_diameter) ** 2. * np.linalg.norm(
        [sample_diameter_cm_err / sample_diameter_cm, beam_diameter_err / beam_diameter])

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 2.5)

    norm = mpl.colors.Normalize(vmin=laser_powers.min(), vmax=laser_powers.max())
    cmap = plt.get_cmap('jet')

    for pid in pids:
        pid_df = tracking_df[tracking_df['PID'] == pid]
        laser_power_setting = pid_df['Laser power setting (%)'].values[0]
        laser_power = laser_power_mapping[int(laser_power_setting)]
        c = cmap(norm(laser_power))

        heat_load = laser_power / sample_area_cm / 100 * aperture_factor
        print(f'Laser power setting: {laser_power_setting} % -> {laser_power} W, Heat load: {heat_load:.2f} MW/m^2')
        t = pid_df['t (s)'].values
        x0, y0 = pid_df['x'].values[0], pid_df['y'].values[0]
        x0 *= px2mm
        y0 *= px2mm
        d_from_center_cm = np.linalg.norm([y0 - center_mm[1], x0 - center_mm[0]]) * 0.1
        # print(f'The center of the pebble is {d_from_center_cm:.3f} cm from the center of the beam.')

        pebble_front_area_mm2 = 0.5 * np.pi * pebble_diameter_mm ** 2.
        power_in = laser_power * absorption_coefficient * aperture_factor * (pebble_front_area_mm2 / sample_area_cm / 100.)
        # power_in = gaussian_beam_intensity(r=d_from_center_cm, w=beam_diameter, p=laser_power) * absorption_coefficient * pebble_front_area_mm2 / 100.

        energy = power_in * t
        nn = (1E-3 / 12.) * np.pi * pebble_diameter_mm ** 3. * rho / m_c
        energy_density = energy / nn

        def energy2time(ee):
            return ee / power_in

        def time2energy(tt):
            return tt * power_in

        t0 = t[0]
        adc_raw = pid_df['Mean gray'].values
        adc_corrected = pid_df['Corrected gray'].values
        adc_delta = pid_df['95% corrected delta'].values
        adc_lb, adc_ub = adc_corrected - adc_delta, adc_corrected + adc_delta
        temperature_raw = convert_to_temperature(adc=adc_raw, cal=cal)
        temperature = convert_to_temperature(adc=adc_corrected, cal=cal)
        temperature_lb = convert_to_temperature(adc=adc_lb, cal=cal)
        temperature_ub = convert_to_temperature(adc=adc_ub, cal=cal)
        ax.plot(t, temperature, color=c, zorder=2, label=f'{heat_load:.1f} MW/m$^{{\\mathregular{{2}}}}$')
        ax.fill_between(t, temperature_lb, temperature_ub, color=lighten_color(c, 0.5), zorder=1)

    ax.set_xlim(0, 0.5)
    ax.set_xlabel(r't [s]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.set_ylabel('T [°C]')
    ax.set_ylim(1400, 2600)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))


    # ax_e = ax.secondary_xaxis('top', functions=(time2energy, energy2time))
    # ax_e.set_xlabel('Average input energy [J]')
    # ax_e.xaxis.set_major_locator(ticker.MultipleLocator(2))
    # ax_e.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.ticklabel_format(axis='x', useMathText=True)
    ax.legend(loc='lower right', fontsize=9, frameon=True)
    ax.set_title('Pebble temperature')
    fig.savefig(os.path.join(base_dir, 'pebble_temperature_plots.svg'), dpi=600)
    fig.savefig(os.path.join(base_dir, 'pebble_temperature_plots.png'), dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
