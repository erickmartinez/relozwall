import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
from matplotlib import ticker, gridspec, patches
import matplotlib as mpl
import json



# graphite_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\surface_temperature\equilibrium'
# sample_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\IR_VS_POWER'
graphite_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\surface_temperature\equilibrium_redone'
sample_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\surface_temperature\equilibrium_redone\pebble_sample'
graphite_filetag = 'graphite_equilibrium_redone_files'
sample_filetag = 'pebble_sample_equilibrium_redone_files'
pebble_velocity_csv = 'R3N21_Pebble_Velocity.csv'
filetag = 'surface_temperature_plot'


if __name__ == '__main__':
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    graphite_temperature_df = pd.read_csv(os.path.join(graphite_path, graphite_filetag + '_surface_temperature.csv')).apply(pd.to_numeric)
    sample_temperature_df = pd.read_csv(os.path.join(sample_path, sample_filetag + '_surface_temperature.csv')).apply(pd.to_numeric)
    # pebble_velocity_df = pd.read_csv(os.path.join(sample_path, pebble_velocity_csv)).apply(pd.to_numeric)
    sample_outgassing_df = pd.read_csv(os.path.join(sample_path, sample_filetag + '_OUTGASSING.csv')).apply(pd.to_numeric)
    graphite_outgassing_df = pd.read_csv(os.path.join(graphite_path, graphite_filetag + '_OUTGASSING.csv')).apply(pd.to_numeric)

    laser_power_g = graphite_temperature_df['Laser power setpoint (%)'].values
    heat_flux_g = graphite_temperature_df['Heat flux (MW/m^2)'].values
    laser_power_s = sample_temperature_df['Laser power setpoint (%)'].values
    heat_flux_s = sample_temperature_df['Heat flux (MW/m^2)'].values
    graphite_surface_temperature = graphite_temperature_df['Max surface temperature (C)'].values
    graphite_time_to_equilibrium_temp = graphite_temperature_df['Time to equilibrium (s)'].values
    sample_surface_temperature = sample_temperature_df['Max surface temperature (C)'].values
    pressure_rise = sample_outgassing_df['Peak Pressure (mTorr)'].values
    pressure_rise_g = graphite_outgassing_df['Peak Pressure (mTorr)'].values
    sample_time_to_equilibrium_temp = sample_temperature_df['Time to equilibrium (s)'].values
    # pebble_velocity = pebble_velocity_df['Pebble Velocity (cm/s)'].values
    # pebble_velocity_std = pebble_velocity_df['Pebble Velocity Std (cm/s)'].values

    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(5.0, 10)

    gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)#, hspace=0.15)  # , height_ratios=[1.618, 1.618, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    ax1.plot(
        heat_flux_s, sample_surface_temperature, color='tab:red', label='Sample', marker='o', mec='k', lw=1.5
    )

    n_graphite = len(graphite_surface_temperature)
    n_sample = len(sample_surface_temperature)
    print(f'Sample data points: {n_sample}, Graphite data points: {n_graphite}')
    ax1.plot(
        heat_flux_g, graphite_surface_temperature, color='navy', label='Graphite', marker='s', mec='k', lw=1.5
    )

    ax1.set_ylabel('Surface Temperature (°C)')
    ax1.set_ylim(top=4000)


    ax1.legend(loc='best', frameon=False)

    ax2.plot(
        heat_flux_g, pressure_rise_g, color='navy', label='Graphite', marker='s', mec='k', lw=1.5, #color='tab:purple', marker='>', mec='k', lw=1.5
    )

    ax2.plot(
        heat_flux_s, pressure_rise, color='tab:red', label='Sample', marker='o', mec='k', lw=1.5
    )

    ax2.set_ylabel('Pressure rise (mTorr)')

    # ax3.plot(
    #     heat_flux_s, pebble_velocity, c='lightblue', mec='k', marker='^', lw=1.5
    # )

    ax3.set_xlabel('Heat flux (MW/cm$\\mathregular{^2}$)')
    ax3.set_ylabel('Pebble velocity (cm/s)')

    ax1.set_xlim(0, 30)
    ax2.set_xlim(0, 30)
    ax3.set_xlim(0, 30)

    ax1_xlim = ax1.get_xlim()
    # ax1_ylim = ax1.get_ylim()
    xy = (ax1_xlim[0], 3500)
    p_width = ax1_xlim[1]
    p_height = 500
    rect = patches.Rectangle(xy, p_width, p_height, linewidth=1, edgecolor='r', facecolor='orange', alpha=0.5)
    ax1.add_patch(rect)

    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(
        0.5,
        0.975,
        'Carbon sublimation?',
        fontsize=9,
        transform=ax1.transAxes,
        va='top', ha='center',
        color='tab:red'
        # bbox=props
    )

    fig.savefig(os.path.join(sample_path, filetag + '.svg'), dpi=600)
    fig.savefig(os.path.join(sample_path, filetag + '.png'), dpi=600)

    fig_t, ax_t = plt.subplots()
    fig_t.set_size_inches(4.5, 3.25)
    ax_t.plot(
        heat_flux_g, graphite_time_to_equilibrium_temp, c='navy', mec='k', marker='s', lw=1.5
    )

    ax_t.plot(
        heat_flux_s, sample_time_to_equilibrium_temp, color='tab:red', label='Sample', marker='o', mec='k', lw=1.5
    )

    ax_t.set_xlabel('Heat flux (MW/m$^2$)')
    ax_t.set_ylabel('Time to equilibrium temperature (s)')
    ax_t.set_xlim(0, 30)
    fig_t.tight_layout()

    fig_t.savefig(os.path.join(sample_path, filetag + '_time_to_equilibrium.svg'), dpi=600)
    fig_t.savefig(os.path.join(sample_path, filetag + '_time_to_equilibrium.png'), dpi=600)

    plt.show()
