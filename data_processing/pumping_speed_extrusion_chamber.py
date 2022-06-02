import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
from matplotlib import ticker, gridspec
import matplotlib as mpl
from scipy.signal import savgol_filter
import json

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\extrusion setup\pumping_speed'
data_csv = 'DEGASSING_FRONT_CHAMBER_2022-05-17_1'

"""
Volume and surface area of the outgassing chamber in the extruder system
"""
d, L = 2.54 * np.array([6.0, 5.866])  # cm
r = 0.5 * d  # cm
volume_extruder_chamber = np.pi * r * r * L
surface_extruder_chamber = np.pi * (d * L + 2.0 * r * r)
outlet_diameter_in = 1.335
outlet_diameter_cm = outlet_diameter_in * 2.54

print('********* Extruder Chamber *********')
print(f'V = {volume_extruder_chamber:.2f} cm^3 = {volume_extruder_chamber * 1E-3:.2f} L')
print(f'S = {surface_extruder_chamber:.2f} cm^2 = {surface_extruder_chamber * 1E-4:.2f} m^2')

air_n2_fraction = 0.80
air_o2_fraction = 0.20

kinetic_diameter_n2_pm = 364.0  # x 1E-12 m
kinetic_diameter_o2_pm = 346.0  # x 1E-12 m


def get_mean_free_path(temperature_c: np.ndarray = np.array([20.0]), pressure_pa: np.ndarray = np.array([101325.0])):
    """
    Estimates the mean free path in cm for air composed of 80% N2 and 20% O2
    """
    kB = 1.380649  # x 1E-23 J/K
    T = temperature_c + 273.15
    p = pressure_pa
    return 4.0E3 * kB * T / (np.sqrt(2.0) * np.pi * ((air_n2_fraction * kinetic_diameter_n2_pm +
                                                       air_o2_fraction * kinetic_diameter_o2_pm) ** 2.0) * p)


def latex_float(f, significant_digits=2):
    significant_digits += 1
    float_str_str = f"{{val:7.{significant_digits}g}}"
    float_str = float_str_str.format(val=f).lower()

    if "e" in float_str:
        base, exponent = float_str.split("e")
        # return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
        if exponent[0] == '+':
            exponent = exponent[1::]
        return rf"{base} \times 10^{{{int(exponent)}}}"
    else:
        return float_str

if __name__ == '__main__':
    pressure_df = pd.read_csv(os.path.join(base_dir, data_csv + '.csv'), comment="#").apply(pd.to_numeric)
    print(pressure_df)
    measurement_time = pressure_df['Measurement Time (h)'].values
    measurement_time = measurement_time[1:] * 3600.0
    pressure = pressure_df['Pressure (Torr)'].values
    pressure = pressure[1:]
    temperature_c = pressure_df['TC2 (C)'].values
    temperature_c = temperature_c[1:]
    mean_free_path = get_mean_free_path(temperature_c=temperature_c, pressure_pa=pressure)
    kn = mean_free_path / outlet_diameter_cm

    t1 = 0.6 * 3600.0
    t2 = 0.98 * 3600.0
    dt = t2 - t1
    idx_t1 = (np.abs(measurement_time - t1)).argmin()
    idx_t2 = (np.abs(measurement_time - t2)).argmin()
    p1, p2 = pressure[idx_t1], pressure[idx_t2]
    print(f'p1: {p1:.3E}, p2: {p2:.3E} Torr')
    S = (volume_extruder_chamber * 1E-3 / dt) * np.log(p1 / p2)

    n = len(pressure)
    window_length = int(n / 25)
    if window_length % 2 == 0:
        window_length - 1

    pressure_smooth = savgol_filter(pressure, window_length=71, polyorder=3)
    dt = np.gradient(measurement_time).mean()
    print(np.gradient(measurement_time))

    dPdt = savgol_filter(pressure, window_length=71, polyorder=3, deriv=1, delta=dt)
    # dPdt = savgol_filter(dPdt, window_length=41, polyorder=1)
    pumping_speed = -dPdt * volume_extruder_chamber * 1E-3 / pressure_smooth
    # pumping_speed = savgol_filter(pumping_speed, window_length=51, polyorder=2)

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(4.5, 4.5)

    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)  # , height_ratios=[1.618, 1.618, 1])

    ax_p = fig.add_subplot(gs[0])
    ax_s = fig.add_subplot(gs[1])

    colors = ['C0', 'C1']

    ax_p.plot(
        measurement_time / 3600.0,
        pressure, ls='-', label=f'Pressure',
        c=colors[0],
    )

    ax_p.plot(
        measurement_time / 3600.0,
        pressure_smooth, ls=':', label=f'Pressure (SG Filter)',
        c='tab:grey', lw=1.25
    )

    ax_p.plot(
        [t1 / 3600.0, t2 / 3600.0], [p1, p2], marker='o', c='tab:red', fillstyle='none',  # ls='none'
    )

    ax_p.set_yscale('log')
    ax_s.set_xlabel('Time (h)')
    ax_p.set_ylabel('Pressure (Torr)')  # , color=colors[0])
    ax_p.set_xlim(left=0.0, right=1.0)
    ax_p.set_ylim(bottom=1E-3, top=1E3)
    # ax_p.tick_params(axis='y', labelcolor=colors[0])

    ax_p.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_p.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax_p.yaxis.set_major_locator(ticker.LogLocator(base=10))
    ax_p.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))

    offset = 1
    connectionstyle = "angle3,angleA=0,angleB=90"
    bbox = dict(boxstyle="round", fc="wheat", alpha=1.0)
    arrowprops = dict(
        arrowstyle="->", color="r",
        shrinkA=5, shrinkB=5,
        patchA=None, patchB=None,
        connectionstyle=connectionstyle
    )
    ax_p.annotate(
        f"S = $latex_float(S)$ L/s",
        xy=(t1 / 3600.0, p1), xycoords='data',  # 'figure pixels', #data',
        xytext=(0.15, 1.0), textcoords='data',  # 'data','offset points'
        arrowprops=arrowprops,
        bbox=bbox,
        ha='left'
    )

    estimation_txt = r'$S = \dfrac{V}{\Delta t} \ln\left(\dfrac{p_0}{p_{\mathrm{f}}}\right)$'
    ax_p.text(
        0.95,
        0.95,
        estimation_txt,
        fontsize=11,
        transform=ax_p.transAxes,
        va='top', ha='right',
        # bbox=props
    )

    ax_p.set_title('Laser Chamber')

    ax_s.plot(
        measurement_time / 3600.0,
        pumping_speed, label=f'Pumping Speed',
        c=colors[1]
    )

    ax_s.set_ylabel('Pumping Speed (L / s)')  # , color=colors[1])
    # ax_s.tick_params(axis='y', labelcolor=colors[1])
    ax_s.set_xlim(left=0.0, right=1.0)
    ax_s.set_ylim(bottom=-0.25, top=3.0)

    ax_s.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax_s.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    estimation_txt = r'$S = \dfrac{V}{P(t)}\dfrac{d}{dt}P(t)$'
    ax_s.text(
        0.95,
        0.95,
        estimation_txt,
        fontsize=11,
        transform=ax_s.transAxes,
        va='top', ha='right',
        # bbox=props
    )

    fig.tight_layout()

    # basename = os.path.splitext(os.path.basename(input_file))[0]
    # path = os.path.dirname(input_file)
    fig.savefig(os.path.join(base_dir, data_csv + '_pumping_speed.png'), dpi=600)
    plt.show()
