import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
from matplotlib import ticker, gridspec, patches
import matplotlib as mpl
import json


graphite_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\IR_VS_POWER\graphite'
sample_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\IR_VS_POWER'
graphite_filetag = 'GT001688_firing_database'
sample_filetag = 'R3N21_firing_database'
pebble_velocity_csv = 'R3N21_Pebble_Velocity.csv'
filetag = 'surface_temperature_plot'

if __name__ == '__main__':
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    graphite_temperature_df = pd.read_csv(os.path.join(graphite_path, graphite_filetag + '_surface_temperature.csv')).apply(pd.to_numeric)
    sample_temperature_df = pd.read_csv(os.path.join(sample_path, sample_filetag + '_surface_temperature.csv')).apply(pd.to_numeric)
    pebble_velocity_df = pd.read_csv(os.path.join(sample_path, pebble_velocity_csv)).apply(pd.to_numeric)
    sample_outgassing_df = pd.read_csv(os.path.join(sample_path, sample_filetag + '_OUTGASSING.csv')).apply(pd.to_numeric)

    laser_power = graphite_temperature_df['Laser power setpoint (%)'].values
    heat_flux = graphite_temperature_df['Heat flux (MW/m^2)'].values
    graphite_surface_temperature = graphite_temperature_df['Max surface temperature (C)'].values
    sample_surface_temperature = sample_temperature_df['Max surface temperature (C)'].values
    pressure_rise = sample_outgassing_df['Peak Pressure (mTorr)'].values
    pebble_velocity = pebble_velocity_df['Pebble Velocity (cm/s)'].values
    pebble_velocity_std = pebble_velocity_df['Pebble Velocity Std (cm/s)'].values

    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(5.0, 10)

    gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)#, hspace=0.15)  # , height_ratios=[1.618, 1.618, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    ax1.plot(
        heat_flux, sample_surface_temperature, color='tab:red', label='Sample', marker='o', mec='k', lw=1.5
    )

    ax1.plot(
        heat_flux, graphite_surface_temperature, color='navy', label='Graphite', marker='s', mec='k', lw=1.5
    )

    ax1.set_ylabel('Surface Temperature (°C)')
    ax1.set_ylim(top=4000)
    ax1_xlim = ax1.get_xlim()
    ax1_ylim = ax1.get_ylim()
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

    ax1.legend(loc='best', frameon=False)

    ax2.plot(
        heat_flux, pressure_rise, color='tab:purple',  marker='>', mec='k', lw=1.5
    )

    ax2.set_ylabel('Pressure rise (mTorr)')

    ax3.plot(
        heat_flux, pebble_velocity, c='lightblue', mec='k', marker='^', lw=1.5
    )

    ax3.set_xlabel('Heat flux (MW/cm$\\mathregular{^2}$)')
    ax3.set_ylabel('Pebble velocity (cm/s)')

    fig.savefig(os.path.join(sample_path, filetag + '.svg'), dpi=600)
    fig.savefig(os.path.join(sample_path, filetag + '.png'), dpi=600)
    plt.show()
