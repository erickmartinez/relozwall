import json
import os
import re
from typing import List
from utils import get_experiment_params
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

chamber_volume = 31.57 # L

data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\surface_temperature\equilibrium_redone\pebble_sample'

csv_database = 'pebble_sample_equilibrium_redone_files.csv'
filename = 'GC_GRAPHITE_POWER_SCAN'

add_time = 0.88




if __name__ == "__main__":
    database_df = pd.read_csv(
        os.path.join(data_path, csv_database), comment='#'
    )
    filelist = database_df['csv']
    database_df['sample diameter (cm)'] = database_df['sample diameter (cm)'].apply(pd.to_numeric)
    sample_diameters = database_df['sample diameter (cm)'].values

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    colors = plt.cm.jet(np.linspace(0, 1, len(filelist)))

    fig, ax = plt.subplots()
    fig.set_size_inches(5.0, 5.0)

    laser_power_setpoint = np.empty_like(filelist, dtype=float)
    base_pressures = np.empty_like(filelist, dtype=np.float64)
    peak_pressures = np.empty_like(filelist, dtype=np.float64)
    outgass_pressure = np.empty_like(filelist, dtype=np.float64)
    peak_dt = np.empty_like(filelist, dtype=np.float64)
    sample_id = []

    for fn, c, i in zip(filelist, colors, range(len(filelist))):
        params = get_experiment_params(data_path, fn)
        power_setting = params['Laser Power Setpoint']['value']
        laser_power_setpoint[i] = power_setting
        sample_id.append(params['Sample Name']['value'])
        leg = f'{float(power_setting):.1f} %'
        pressure_csv = f'{fn}_pressure.csv'
        print(pressure_csv)
        pressure_data = pd.read_csv(filepath_or_buffer=os.path.join(data_path, pressure_csv))
        pressure_data = pressure_data.apply(pd.to_numeric)
        time_s = pressure_data['Time (s)'].values
        time_s -= time_s.min() + 0.5
        pressure = 1000*pressure_data['Pressure (Torr)'].values
        base_pressures[i] = pressure[0]
        peak_pressures[i] = pressure.max()
        idx_peak = (np.abs(pressure - pressure.max())).argmin()
        t_peak = time_s[idx_peak]
        p_2 = pressure[idx_peak+1:]
        p_out = pressure[-1]

        idx_out = (np.abs(pressure - p_out)).argmin()
        outgass_pressure[i] = pressure[idx_out]
        t_out = time_s[-1]
        idx_peak = (np.abs(pressure - peak_pressures[i])).argmin()
        peak_dt[i] = t_out # time_s[idx_peak]


        # title_str = 'Sample ' + params['Sample Name']['value'] + ', '
        # params_title = params
        # params_title.pop('Sample Name')
        #
        # for i, p in enumerate(params_title):
        #     title_str += f"{params_title[p]['value']}{params_title[p]['units']}"
        #     if i + 1 < len(params_title):
        #         title_str += ', '

        line = ax.plot(time_s, pressure, label=leg, color=c)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pressure (mTorr)')
        # colors.append(line[0].get_color())

    # leg = ax.legend(frameon=True, loc='best', fontsize=8)
    leg = ax.legend(
        bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        ncol=2, mode="expand", borderaxespad=0., prop={'size': 8}
    )
    for color, text in zip(colors, leg.get_texts()):
        text.set_color(color)

    outgassing_rate = chamber_volume * (outgass_pressure - base_pressures) * 1E-3 / peak_dt
    outgassing_rate_normalized = outgassing_rate * 4.0E4 / (np.pi * sample_diameters[i] ** 2.0)

    outgas_df = pd.DataFrame(data={
        'Sample ID': sample_id,
        'Laser power setpoint (%)': laser_power_setpoint,
        'Base Pressure (mTorr)': base_pressures,
        'Peak Pressure (mTorr)': peak_pressures,
        'Outgass Pressure (mTorr)': outgass_pressure,
        'Peak dt (s)': peak_dt,
        'Outgassing rate (Torr L / s)': outgassing_rate,
        'Outgassing rate (Torr L / s / m^2)': outgassing_rate_normalized
    })

    print(outgas_df)
    
    outgas_df.to_csv(os.path.join(data_path, f'{filename}_OUTGASSING.csv'), index=False)

    fig.tight_layout()
    fig.savefig(os.path.join(data_path, f'{filename}_PRESSURE.png'), dpi=600)
    fig.savefig(os.path.join(data_path, f'{filename}_PRESSURE.svg'), dpi=600)
    fig.savefig(os.path.join(data_path, f'{filename}_PRESSURE.eps'), dpi=600)

    plt.show()
