import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
from matplotlib import ticker, gridspec
import matplotlib as mpl
from scipy.signal import savgol_filter
import json

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\pumping_speed'
data_csv = 'DEGASSING_EMPTY_CHAMBER_2022-05-10_1'

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

if __name__ == '__main__':
    pressure_df = pd.read_csv(os.path.join(base_dir, data_csv + '.csv'), comment="#").apply(pd.to_numeric)
    print(pressure_df)
    measurement_time = pressure_df['Measurement Time (s)'].values
    measurement_time = measurement_time[1:]
    pressure = pressure_df['Pressure (Torr)'].values
    pressure = pressure[1:]

    t1 = 0.6*3600.0
    t2 = 1.0*3600.0
    dt = t2 - t1
    idx_t1 = (np.abs(measurement_time - t1)).argmin()
    idx_t2 = (np.abs(measurement_time - t2)).argmin()
    p1, p2 = pressure[idx_t1], pressure[idx_t2]
    S = (volume_laser_chamber*1E-3/dt) * np.log(p1/p2)

    n = len(pressure)
    window_length = int(n / 25)
    if window_length % 2 == 0:
        window_length - 1

    pressure_smooth = savgol_filter(pressure, window_length=71, polyorder=3)
    dt = np.gradient(measurement_time).mean()
    print(np.gradient(measurement_time))

    dPdt = savgol_filter(pressure, window_length=71, polyorder=3, deriv=1, delta=dt)
    # dPdt = savgol_filter(dPdt, window_length=41, polyorder=1)
    pumping_speed = -dPdt * volume_laser_chamber * 1E-3 / pressure_smooth
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
        [t1/3600.0, t2/3600.0], [p1, p2], marker='o', c='tab:red', fillstyle='none', #ls='none'
    )

    ax_p.set_yscale('log')
    ax_s.set_xlabel('Time (h)')
    ax_p.set_ylabel('Pressure (Torr)')  # , color=colors[0])
    ax_p.set_xlim(left=0.0, right=1.0)
    ax_p.set_ylim(bottom=1E-2, top=1E3)
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
        f"S = {S:.2f} L/s",
        xy=(t1/3600.0, p1), xycoords='data',  # 'figure pixels', #data',
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
