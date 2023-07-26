import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import pandas as pd
import mpmath as mp
from scipy.interpolate import interp1d
from scipy.integrate import simps
import ir_thermography.thermometry as irt
import matplotlib.ticker as ticker
from data_processing.utils import get_experiment_params, latex_float

emissivity = 0.8
# emissivity = 1.0 - (36.9 / 100)

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration'
csv = 'LT_GRAPHITE-photodiode_100PCT_2023-02-22_1.csv'

camera_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\BFS-U3-16S2M_QE.csv'
filter_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\86117_Transmission.csv'

reflection_level = 0.0

mp.dps = 15
mp.pretty = True

np_poly_log = np.frompyfunc(mp.polylog, 2, 1)
np_log = np.frompyfunc(mp.log, 1, 1)
np_exp = np.frompyfunc(mp.exp, 1, 1)

h = 6.62607015  # x 1E-34 (J*s)
c = 2.99792458  # x 1E8 (m/s)
kB = 1.380649  # x E-23 (J/K)


def get_brightness(wl_nm: np.ndarray, temperature: np.ndarray, factor:np.ndarray):
    twohcsq = 2. * h * (c ** 2.) # x 1E-18 J * m^2 / s
    a1 = twohcsq * np.power(wl_nm, -5.)  # x 1E27 J / m^3 / s

    result = np.zeros_like(temperature)
    for i, t in enumerate(temperature):
        arg = 1E6 * h * c / kB / t / wl_nm
        b = a1 * (1. / (np.exp(arg) - 1.))
        y = b * factor
        integral = 1E14 * simps(y, wl_nm)
        result[i] = integral
    return result


def evaulate_integral(lambda_nm: float, temperature_k: float) -> np.ndarray:
    a = h / kB / temperature_k  # x 1E-11 (s)
    v = c / lambda_nm  # x 1E17 (1/s)
    av = a * v * 1E6
    # h2bycsq = 2. * h / (c * c)  # x 1E-50 J * s^3 / m^2
    h2bycsq = 2. * h / (c * c)  # x 1E-54 J * s^3 / cm^2
    a1 = (v ** 3.) / a  # x 1E62 s^{-4}
    a2 = 3. * (v ** 2.) / (a ** 2.)  # x 1E56 s^{-4}
    a3 = 6. * v / (a ** 3.)  # x 1E50 s^{-4}
    a4 = 6.0 / (a ** 4.)  # x 1E44 s^{-4}
    b1 = h2bycsq * a1 * 1E8
    b2 = h2bycsq * a2 * 1E2
    b3 = h2bycsq * a3 * 1E-4
    b4 = h2bycsq * a4 * 1E-10
    ex = np.array([np_exp(-x) for x in av])
    result = b1 * np.array([np_log(1. - e) for e in ex])
    result -= b2 * np.array([np_poly_log(2, e) for e in ex])
    result -= b3 * np.array([np_poly_log(3, e) for e in ex])
    result -= b4 * np.array([np_poly_log(4, e) for e in ex])
    return result


def main():
    qe_df = pd.read_csv(camera_csv, comment='#').apply(pd.to_numeric)
    filter_df = pd.read_csv(filter_csv, comment='#').apply(pd.to_numeric)
    qe_df.sort_values(by=['Wavelength (nm)'], inplace=True)
    filter_df.sort_values(by=['Wavelength (nm)'], inplace=True)
    wl_qe = qe_df['Wavelength (nm)'].values
    wl_trans = filter_df['Wavelength (nm)'].values
    qe = qe_df['Quantum efficiency (%)'].values
    transmission = filter_df['Transmission (%)'].values

    # interpolate the quantum efficiency spectrum in steps of 0.125 nm
    dl = 0.125
    N = int((1100. - 300.) / dl) + 1

    f1 = interp1d(wl_qe, qe, kind='linear', bounds_error=False)
    f2 = interp1d(
        wl_trans, transmission, kind='linear', bounds_error=False, fill_value=0.0#transmission[0]
    )
    wl_qe_interp = dl * np.arange(0, N) + 300.
    qe_interp = f1(wl_qe_interp)
    trans_interp = f2(wl_qe_interp)

    brightness_factor = qe_interp * trans_interp * 1E-4

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig1, axes1 = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
    fig1.set_size_inches(4.5, 6.0)

    axes1[0].plot(wl_qe_interp, qe_interp, label='QE', color='teal')
    axes1[1].plot(wl_qe_interp, trans_interp, label='Transmission', color='slateblue')
    axes1[2].plot(wl_qe_interp, brightness_factor, label='QE x Transmission', color='steelblue')

    for i, ax in enumerate(axes1):
        ax.set_xlabel('Wavelength (nm)')
        ax.set_xlim(300., 1100.)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(20))

    axes1[0].set_ylabel('QE (%)')
    axes1[1].set_ylabel(r'Transmission (%)')
    axes1[2].set_ylabel('QE x Transmission')

    axes1[0].set_title('Sensor quantum efficiency')
    axes1[1].set_title('Filter transmission')
    axes1[2].set_title('Product')

    axes1[0].set_ylim(0,70)
    axes1[0].yaxis.set_major_locator(ticker.MultipleLocator(20))
    axes1[0].yaxis.set_minor_locator(ticker.MultipleLocator(10))

    axes1[1].set_ylim(0, 100)
    axes1[1].yaxis.set_major_locator(ticker.MultipleLocator(20))
    axes1[1].yaxis.set_minor_locator(ticker.MultipleLocator(10))

    axes1[2].set_ylim(0, 0.7)
    axes1[2].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axes1[2].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    fig1.savefig(os.path.join(base_dir, 'camera_responsivity.png'), dpi=600)

    thermometry = irt.PDThermometer()
    df = pd.read_csv(os.path.join(base_dir, csv), comment='#').apply(pd.to_numeric)
    measured_time = df['Measurement Time (s)'].values
    trigger_voltage = df['Trigger (V)'].values
    pd_voltage = df['Photodiode Voltage (V)'].values

    noise_idx = pd_voltage < 0
    noise_level = pd_voltage[noise_idx]
    noise_max = np.abs(noise_level.mean())

    # Get the indices where the trigger is on
    irradiation_time_idx = trigger_voltage > 1.0
    # construct the reflection signal with a level provided upon inspection of the plot
    reflection_signal = np.zeros_like(pd_voltage)
    reflection_signal[irradiation_time_idx] = reflection_level
    # make the time start when the pulse starts
    t_on = measured_time[irradiation_time_idx]
    t0 = t_on[0]
    idx_on = measured_time >= t0
    measured_time = measured_time[idx_on] - t0
    pd_voltage = pd_voltage[idx_on]
    reflection_signal = reflection_signal[idx_on]

    experiment_params = get_experiment_params(relative_path=base_dir, filename=os.path.splitext(csv)[0])
    photodiode_gain = float(experiment_params['Photodiode Gain']['value'])
    laser_power_setting = experiment_params['Laser power setpoint']['value']
    emission_time = float(experiment_params['Emission time']['value'])
    pd_voltage_basedline = pd_voltage - reflection_signal
    idx_small_or_negative = pd_voltage_basedline <= 0.0
    pd_voltage_basedline[idx_small_or_negative] = noise_max

    thermometry.gain = photodiode_gain
    thermometry.emissivity = emissivity
    temperature_k = thermometry.get_temperature(voltage=pd_voltage_basedline)
    temperature_c = temperature_k - 273.15
    print(f'Gain: {thermometry.gain} dB')
    print(f'Factor: {thermometry.calibration_factor}')
    print(f'Emissivity: {thermometry.emissivity}')

    brightness_camera = get_brightness(wl_nm=wl_qe_interp, temperature=temperature_k, factor=brightness_factor)
    brightness = pd_voltage_basedline * thermometry.calibration_factor

    fig2, axes2 = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
    fig2.set_size_inches(4.5, 6.0)

    axes2[0].plot(measured_time, pd_voltage, label=f'PD voltage {photodiode_gain:.0f} dB Gain', color='r')
    axes2[0].plot(measured_time, reflection_signal, label='Reflection baseline', color='C0')
    c1 = 'C1'
    c2 = 'C2'
    axes2[1].plot(measured_time, brightness, label='Photodiode', color=c1)
    axes2[1].tick_params(axis='y', labelcolor=c1)
    ax_t = axes2[1].twinx()
    ax_t.plot(measured_time, brightness_camera, label='Camera', color=c2)
    ax_t.tick_params(axis='y', labelcolor=c2)
    axes2[2].plot(measured_time, temperature_c, label='Surface temperature', color='tab:purple')

    for i, ax in enumerate(axes2):
        ax.set_xlabel('Time (s)')
        ax.set_xlim(0, 2.0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    axes2[0].legend(loc='upper right', frameon=True, fontsize=10.)
    # axes2[1].legend(loc='upper right', frameon=True)

    axes2[0].set_ylabel('$V$ (V)')
    axes2[1].set_ylabel(r'$B_{\lambda=900}$ (W/ster-cm$^{\mathregular{2}}$)', color=c1)
    ax_t.set_ylabel(r'$B_{\mathregular{cam}}$ (W/ster-cm$^{\mathregular{2}}$)', color=c2)
    axes2[2].set_ylabel('$T$ (°C)')

    axes2[0].set_title('Photodiode voltage')
    axes2[1].set_title('Brightness')
    axes2[2].set_title('Temperature')

    axes2[0].set_ylim(0, 2.5)
    axes2[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axes2[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    axes2[1].set_ylim(0, 0.035)
    axes2[1].yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    axes2[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
    axes2[1].ticklabel_format(style='sci', axis='y', scilimits=(-2,2), useMathText=True)

    ax_t.set_ylim(0, 6.)
    ax_t.yaxis.set_major_locator(ticker.MultipleLocator(2.))
    ax_t.yaxis.set_minor_locator(ticker.MultipleLocator(1.))

    axes2[2].set_ylim(1000, 2500)
    axes2[2].yaxis.set_major_locator(ticker.MultipleLocator(500))
    axes2[2].yaxis.set_minor_locator(ticker.MultipleLocator(250))

    fig2.savefig(os.path.join(base_dir, 'brightness_and_temperature.png'), dpi=600)

    # save a table with the brightness and temperature data at each time point
    out_df = pd.DataFrame(data={
        'Time (s)': measured_time,
        'Brightness at 900 nm (W/ster/cm^2)': brightness,
        'Brightness at sensor (W/ster/cm^2)': brightness_camera,
        'Temperature (°C)': temperature_c
    })

    out_df.to_csv(os.path.join(base_dir, 'temperature_data.csv'), index=False, encoding='utf-8-sig')

    plt.show()


if __name__ == '__main__':
    main()
