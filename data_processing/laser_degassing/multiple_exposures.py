import sys

sys.path.append('../../data_processing')
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import re
from data_processing.utils import get_experiment_params

chamber_volume = 31.57  # L

data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\laser_degassing'
csv_filelist = 'filelist.csv'
sample_id = 'R4N11'


def get_file_list(csv_file, base_dir):
    file = os.path.join(base_dir, csv_file)
    df = pd.read_csv(filepath_or_buffer=file)
    return [f for f in df['file'].values]


def plot_pressure(
        base_dir: str, filelist: List, legends: List, output_filename: str, ax: plt.axes, plot_title=None,
        laser_warming=True
):
    base_pressures = np.empty_like(filelist, dtype=np.float64)
    peak_pressures = np.empty_like(filelist, dtype=np.float64)
    peak_dt = np.empty_like(filelist, dtype=np.float64)
    n = len(file_list)

    colors = plt.cm.cool(np.linspace(0, 1, n - 1)) if laser_warming else plt.cm.cool(np.linspace(0, 1, n - 1))
    if laser_warming:
        colors = [c for c in colors]
        colors.append('tab:red')

    for fn, leg, c, i in zip(filelist, legends, colors, range(n)):
        params = get_experiment_params(base_dir, fn)
        pressure_csv = f'{fn}.csv'
        print(pressure_csv)
        pressure_data = pd.read_csv(filepath_or_buffer=os.path.join(base_dir, pressure_csv), comment='#')
        pressure_data = pressure_data.apply(pd.to_numeric)
        time_s = pressure_data['Measurement Time (s)'].values
        time_s -= time_s.min()
        pressure = 1000 * pressure_data['Pressure (Torr)'].values
        base_pressures[i] = pressure[0]
        peak_pressures[i] = pressure.max()
        idx_peak = (np.abs(pressure - peak_pressures[i])).argmin()
        peak_dt[i] = time_s[idx_peak] - 0.5

        # title_str = 'Sample ' + params['Sample Name']['value'] + ', '
        # params_title = params
        # params_title.pop('Sample Name')
        #
        # for i, p in enumerate(params_title):
        #     title_str += f"{params_title[p]['value']}{params_title[p]['units']}"
        #     if i + 1 < len(params_title):
        #         title_str += ', '

        line = ax.plot(time_s, pressure, label=leg, color=c, lw=1.25)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pressure (mTorr)')
        # colors.append(line[0].get_color())

    ax.set_xlim(0, time_s.max())
    leg = ax.legend(frameon=True, loc='best', fontsize=8)
    for color, text, i in zip(colors, leg.get_texts(), range(n)):
        text.set_color(color)

    if plot_title is not None:
        ax.set_title(plot_title)

    outgassing_rate = chamber_volume * (peak_pressures - base_pressures) * 1E-3 / peak_dt

    outgas_df = pd.DataFrame(data={
        'Sample': legends,
        'Base Pressure (mTorr)': base_pressures,
        'Peak Pressure (mTorr)': peak_pressures,
        'Peak dt (s)': peak_dt,
        'Outgassing Rate (Torr L / s)': outgassing_rate
    })

    print(outgas_df)
    outgas_df.to_csv(os.path.join(base_dir, f'{output_filename}_OUTGASSING.csv'), index=False)


if __name__ == "__main__":
    # pressure_csv = f'{filename}_pressure.csv'
    # pressure_data = pd.read_csv(filepath_or_buffer=os.path.join(base_path, pressure_csv))
    # pressure_data = pressure_data.apply(pd.to_numeric)
    # # pressure_data['Pressure (Torr)'] = pressure_data['Pressure (Torr)']/1000
    # pressure_data.to_csv(os.path.join(base_path, pressure_csv), index=False)
    file_list = get_file_list(csv_file=csv_filelist, base_dir=data_path)
    n = len(file_list)
    legends = []
    for i, f in enumerate(file_list):
        params = get_experiment_params(relative_path=data_path, filename=f)
        laser_power = float(params['Laser Power Setpoint']['value'])
        legends.append(f'Shot {i + 1}/{n}, {laser_power:.0f} %')
        if i == 0:
            measurement_time = float(params['Measurement Time']['value'])
            emission_time = float(params['Emission Time']['value'])
            power_setting = float(params['Laser Power Setpoint']['value'])
            time_s = np.linspace(0, measurement_time, 500)
            laser_pulse1 = np.zeros_like(time_s, dtype=float)
            msk_pulse = (0.5 <= time_s) & (time_s <= emission_time+0.5)
            laser_pulse1[msk_pulse] = power_setting
        if i == (n - 1):
            laser_pulse2 = np.zeros_like(time_s, dtype=float)
            power_setting = float(params['Laser Power Setpoint']['value'])
            emission_time = float(params['Emission Time']['value'])
            msk_pulse = (0.5 <= time_s) & (time_s <= emission_time+0.5)
            laser_pulse2[msk_pulse] = power_setting

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(constrained_layout=True)
    fig.set_size_inches(4.5, 3.0)
    ax_pulse = ax.twinx()

    plot_pressure(base_dir=data_path, filelist=file_list, legends=legends, output_filename=sample_id, ax=ax)
    ax.set_ylim(1, 5.0)
    color_laser = 'tab:green'
    ax_pulse.plot(time_s, laser_pulse1, c=color_laser, ls='--', lw=1.25)
    ax_pulse.plot(time_s, laser_pulse2, c=color_laser, ls='-', lw=1.25)
    ax_pulse.tick_params(axis='y', labelcolor=color_laser)
    ax_pulse.set_ylabel('Laser pulse', color=color_laser)
    # fig.tight_layout()
    fig.savefig(os.path.join(data_path, f'{sample_id}_OUTGASSING.png'), dpi=600)
    # fig.savefig(os.path.join(base_path, f'{output_filename}_OUTGASSING.svg'), dpi=600)
    # fig.savefig(os.path.join(base_path, f'{output_filename}_OUTGASSING.eps'), dpi=600)
    plt.show()
