import numpy as np
from utils import latex_float, get_experiment_params
import matplotlib.pylab as plt
import pandas as pd
import os
from matplotlib import ticker, gridspec
import matplotlib as mpl
from scipy.signal import savgol_filter
import json
from scipy.interpolate import interp1d

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\extrusion setup\outgassing'
pumping_speed_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\extrusion setup\pumping_speed\pumpdown_venting\DEGASSING_EXTRUDER_FRONT_CHAMBER_PUMPDOWN_2022-06-08_1_pumping_speed'
data_csv = 'EXTRUSION_R3N49_1_40V_350C_350C2022-06-02_1'
surface_cm2 = 0.25 * np.pi * 1.5 ** 2.0
print(f'Surface Area: {surface_cm2}')
label = '330 °C'
label = '330 °C'

"""
Volume and surface area of the outgassing chamber in the extruder system
"""
d, L = 2.54 * np.array([6.0, 5.866])  # cm
r = 0.5 * d  # cm
volume_extruder_chamber = np.pi * r * r * L
surface_extruder_chamber = np.pi * (d * L + 2.0 * r * r)
vacuum_chamber_diameter = 6.0
outlet_diameter_cm = vacuum_chamber_diameter * 2.54
chamber_outgassing_mean = 2.6E-4 # Torr L / (m^2 s)

# Parameters for the outgassing model
b_outgassing = np.array([-2.3309, 1.8101E-2, 4.8413E-1])


print('********* Extruder Chamber *********')
print(f'V = {volume_extruder_chamber:.2f} cm^3 = {volume_extruder_chamber * 1E-3:.2f} L')
print(f'S = {surface_extruder_chamber:.2f} cm^2 = {surface_extruder_chamber * 1E-4:.2f} m^2')

air_n2_fraction = 0.80
air_o2_fraction = 0.20

kinetic_diameter_n2_pm = 364.0  # x 1E-12 m
kinetic_diameter_o2_pm = 346.0  # x 1E-12 m


def get_mean_free_path(temp_c: np.ndarray = np.array([20.0]), pressure_pa: np.ndarray = np.array([101325.0])):
    """
    Estimates the mean free path in cm for air composed of 80% N2 and 20% O2
    """
    kB = 1.380649  # x 1E-23 J/K
    T = temp_c + 273.15
    p = pressure_pa
    return 4.0E3 * kB * T / (np.sqrt(2.0) * np.pi * ((air_n2_fraction * kinetic_diameter_n2_pm +
                                                      air_o2_fraction * kinetic_diameter_o2_pm) ** 2.0) * p)


if __name__ == '__main__':
    pressure_df = pd.read_csv(os.path.join(base_dir, data_csv + '.csv'), comment="#").apply(pd.to_numeric)
    pumping_speed_df = pd.read_csv(pumping_speed_csv + '.csv').apply(pd.to_numeric)
    measurement_time = pressure_df['Time (s)'].values
    pressure_1 = pressure_df['Baking Pressure (Torr)'].values
    pressure_2 = pressure_df['Outgassing Pressure (Torr)'].values
    pressure_2_baselined = chamber_outgassing_mean * surface_extruder_chamber - pressure_2
    temperature_c = pressure_df['Baking Temperature (C)'].values

    idx_p0 = 0
    # # Find the range of measurements before the vacuum pump started running
    # p0 = pressure_2[0]
    # idx_p0 = len(pressure_2) - len(pressure_2[pressure_2 < p0]) - 1
    # t0 = measurement_time[idx_p0]
    # measurement_time = measurement_time[idx_p0::] - t0
    # pressure_2 = pressure_2[idx_p0::]
    # temperature_c = temperature_c[idx_p0::]
    # pressure_1 = pressure_1[idx_p0::]

    dt = np.gradient(measurement_time).mean()
    # print(np.gradient(measurement_time))

    mean_free_path = get_mean_free_path(temp_c=temperature_c, pressure_pa=pressure_2 * 133.22)
    kn = mean_free_path / outlet_diameter_cm
    transitional_idx = kn >= 1E-2
    viscous_idx = kn < 1E-2
    viscous_window_length = len(pressure_2[viscous_idx])
    transitional_window_length = len(pressure_2[transitional_idx])
    print('Size of pressure:', len(pressure_2))
    print(f'len(viscous regime): {viscous_window_length}')
    print(f'len(transitional regime): {transitional_window_length}')

    # Interpolate the pumping speed to the outgassing pressure
    pressure_ps = pumping_speed_df['Pressure (Torr)'].values
    pumping_speed = pumping_speed_df['Pumping Speed (L/s)'].values
    f = interp1d(x=pressure_ps, y=pumping_speed, kind='linear', bounds_error=False, fill_value='extrapolate')
    pumping_speed_interp = f(pressure_2)

    sg_mode = 'interp'
    sg_poly_order = 4
    pressure_2_smooth = np.empty_like(pressure_2)
    dPdt = np.empty_like(pressure_2)
    wl_t = int(transitional_window_length / 8)
    if wl_t % 2 == 0:
        wl_t += 1

    pressure_2_smooth = savgol_filter(
        pressure_2, window_length=wl_t, polyorder=6, delta=dt
    )
    pressure_2_smooth[transitional_idx] = savgol_filter(
        pressure_2[transitional_idx], window_length=wl_t, polyorder=6, delta=dt
    )
    # if viscous_window_length > 0:
    #     wl = int(viscous_window_length / 2)
    #     if wl % 2 == 0:
    #         wl -= 1
    #     pressure_2_smooth[viscous_idx] = savgol_filter(
    #         pressure_2[viscous_idx], window_length=wl, polyorder=sg_poly_order, delta=dt, mode=sg_mode
    #     )
    #
    #     dPdt[viscous_idx] = savgol_filter(pressure_2[viscous_idx], window_length=wl, polyorder=sg_poly_order, deriv=1,
    #                                       delta=dt)

    # dPdt[transitional_idx] = savgol_filter(
    #     pressure_2[transitional_idx], window_length=wl_t, polyorder=6, deriv=1, delta=dt, mode=sg_mode
    # )

    dPdt = savgol_filter(
        pressure_2, window_length=wl_t, polyorder=6, deriv=1, delta=dt, mode=sg_mode
    )

    dPdt_inst = np.gradient(pressure_2_smooth, measurement_time)

    # dPdt = savgol_filter(dPdt, window_length=41, polyorder=1)
    Vdp = -dPdt_inst * volume_extruder_chamber * 1E-3
    Sp = pumping_speed_interp * pressure_2
    q_tot = Vdp + Sp
    q_tot = savgol_filter(q_tot, window_length=11, polyorder=3)
    # q_tot[transitional_idx] = savgol_filter(q_tot[transitional_idx], window_length=51, polyorder=4)

    outgassing_df = pressure_df.iloc[idx_p0::, :].copy()
    outgassing_df.reset_index(inplace=True, drop=True)
    outgassing_df['Outgassing (Torr*L/m^2 s)'] = q_tot / surface_cm2 * 1E4
    outgassing_df.to_csv(
        os.path.join(base_dir, data_csv + '_outgassing.csv'),
        index=False
    )

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    xfmt1 = ticker.ScalarFormatter(useMathText=True)
    xfmt1.set_powerlimits((-2, 2))

    fig_p = plt.figure(tight_layout=True)
    fig_p.set_size_inches(4.5, 6.0)

    gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig_p)  # , height_ratios=[1.618, 1.618, 1])

    ax_1 = fig_p.add_subplot(gs[0])
    ax_2 = fig_p.add_subplot(gs[1])
    ax_3 = fig_p.add_subplot(gs[2])

    ax_1.set_xlim(0, measurement_time.max()/60.0)
    ax_2.set_xlim(0, measurement_time.max()/60.0)
    ax_3.set_xlim(0, measurement_time.max()/60.0)
    ax_1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax_1.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    ax_2.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax_2.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    ax_3.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax_3.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))

    ax_1.set_title(label)

    colors = ['C0', 'C1', 'C2']

    ax_1.plot(
        measurement_time / 60.0, pressure_2, ls='none', marker='o', color=colors[0], ms=6, mew=1.5, fillstyle='none'
    )

    ax_1.plot(
        measurement_time / 60.0, pressure_2_smooth, ls='--', lw=1.5, c=colors[0]
    )

    ax_1.set_yscale('log')
    ax_1.set_ylim(bottom=1E-3, top=1E-1)
    ax_1.set_ylabel('$p$ (Torr)', color=colors[0])
    ax_1.tick_params(axis='y', labelcolor=colors[0])


    ax_1_t = ax_1.twinx()
    ax_1_t.plot(
        measurement_time / 60.0, temperature_c, color='tab:red'
    )

    ax_1_t.set_ylabel('$T$ (°C)', color='tab:red')
    ax_1_t.tick_params(axis='y', labelcolor='tab:red')

    ax_2.plot(
        measurement_time / 60.0, Vdp,
        color='C1'
    )
    # ax_2.set_xlabel('Time (min)')
    xfmt2 = ticker.ScalarFormatter(useMathText=True)
    xfmt2.set_powerlimits((-2, 2))
    ax_2.set_ylabel('$V(dp/dt)$ (Torr$\\cdot$L/s)', color='C1')
    ax_2.tick_params(axis='y', labelcolor='C1')
    ax_2.yaxis.set_major_formatter(xfmt2)

    ax_2_t = ax_2.twinx()
    ax_2_t.plot(
        measurement_time / 60.0, Sp,
        color='C2'
    )
    ax_2_t.tick_params(axis='y', labelcolor='C2')
    # ax_2_t.set_yscale('log')
    xfmt3 = ticker.ScalarFormatter(useMathText=True)
    xfmt3.set_powerlimits((-2, 2))
    ax_2_t.yaxis.set_major_formatter(xfmt3)
    ax_2_t.set_ylabel('$S \\cdot p$ (Torr $\\cdot$ L / s)', color='C2')

    ax_3.plot(
        measurement_time / 60.0, q_tot / surface_cm2 * 1E4,
        color='C3'
    )

    txt_continuity_equation = r'$Q_{\mathrm{tot}} = V\left(\dfrac{dp}{dt}\right) + S \cdot p$'
    ax_3.text(
        0.95,
        0.95,
        txt_continuity_equation,
        fontsize=11,
        transform=ax_3.transAxes,
        va='top', ha='right',
        # bbox=props
    )


    ax_3.set_xlabel('Time (min)')
    ax_3.set_ylabel('$Q_{\mathrm{tot}}$ (Torr $\\cdot$ L / m${^2}$ s)')
    # ax_3.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    # ax_3.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    fig_diff = plt.figure(tight_layout=True)
    fig_diff.set_size_inches(4.5, 6.0)

    gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig_diff)  # , height_ratios=[1.618, 1.618, 1])

    ax1 = fig_diff.add_subplot(gs[0])
    ax2 = fig_diff.add_subplot(gs[1])
    ax3 = fig_diff.add_subplot(gs[2])

    ax1.plot(
        measurement_time / 60.0, temperature_c, color='tab:red'
    )

    ax2.plot(
        measurement_time / 60.0, pressure_1, color='C5', label='Heating Section'
    )

    ax2.plot(
        measurement_time / 60.0, pressure_2, color='C0',  label='Front surface'
    )

    ax2.set_yscale('log')

    ax2.legend(loc='best', frameon=False, prop={'size': 9})

    ax3.plot(
        measurement_time / 60.0, q_tot / surface_cm2 * 1E4,
        color='C3'
    )

    ax1.set_ylabel('$T$ (°C)')
    ax2.set_ylabel('$p$ (Torr)')
    ax3.set_ylabel('$Q_{\mathrm{tot}}$ (Torr $\\cdot$ L / m${^2}$ s)')
    ax1.set_title(label)

    ax3.set_xlabel('Time (min)')
    ax1.set_xlim(0, measurement_time.max() / 60.0)
    ax2.set_xlim(0, measurement_time.max() / 60.0)
    ax3.set_xlim(0, measurement_time.max() / 60.0)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))

    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1E-3, top=1E1)
    # ax2.yaxis.set_major_locator(ticker.LogLocator(base=10))
    # ax2.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))

    fig_p.savefig(
        os.path.join(base_dir, data_csv + '_outgassing_estimation.png'), dpi=600
    )

    fig_diff.savefig(
        os.path.join(base_dir, data_csv + '_outgassing_summary.png'), dpi=600
    )

    plt.show()
