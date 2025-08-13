from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from matplotlib.ticker import ScalarFormatter
import re

chamber_volume = 31.57 # L

# base_path = r"G:\Shared drives\ARPA-E Project\Lab\Data\Laser Tests\SAMPLES"
# base_path = r"G:\Shared drives\ARPA-E Project\Lab\Data\Laser Tests\STARTING_MATERIALS"
# base_path = r"G:\Shared drives\ARPA-E Project\Lab\Data\Laser Tests\SAMPLES\MULTIPLE EXPOSURES"
data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\GC_GRAPHITE'

# filename = 'BINDER_SCAN_PLOT'
# filename = 'STARTING_MATERIALS'
# filename = 'R3N18_MULTIPLE_FIRINGS'
filename = 'GC_GRAPHITE_DEGASSING'

# filelist = ['LT_R3N12_100PCT_40GAIN 2022-03-01_1', 'LT_R3N10_100PCT_40GAIN 2022-03-01_1',
#             'LT_R3N3_100PCT_40GAIN 2022-03-02_1', 'LT_R3N14_100PCT_40GAIN 2022-03-02_1']
# filelist = [
#     'LT_Tidal75_100PCT_40GAIN 2022-03-08_1',
#     'LT_GC_TYPE1_100PCT_40GAIN 2022-03-07_1',  #'LT_GC_TYPE2_100PCT_40GAIN 2022-03-07_1',
#      #'LT_Graphite_100PCT_40GAIN 2022-03-08_1',
#     'LT_GRAPHITE_POWDER_100PCT_40GAIN 2022-03-09_1',
#     "LT_hBN_100PCT_40GAIN 2022-03-21_1",
#     # 'LT_R3N10_100PCT_40GAIN 2022-03-01_1'
# ]
# filelist = [
#     'LT_R3N18_100PCT_40GAIN 2022-03-09_1',
#     'LT_R3N18_100PCT_40GAIN 2022-03-09_2',
#     'LT_R3N18_100PCT_40GAIN 2022-03-10_1',
#     'LT_R3N18_100PCT_40GAIN 2022-03-10_2',
#     # 'LT_R3N18_100PCT_50GAIN 2022-03-10_1',
#     # 'LT_R3N18_100PCT_40GAIN 2022-03-14_1',
#     # 'LT_R3N20_100PCT_40GAIN 2022-03-15_1',
#     # 'LT_R3N20_100PCT_40GAIN 2022-03-15_2'
# ]

filelist = [
    'LT_R3N40_100PCT_66.25h_2022-05-16_1',
    'LT_R3N20_100PCT_12.0h_2022-05-12_1',
    'LT_R3N40_100PCT_4.0h_2022-05-16_1',
    'LT_R3N20_100PCT_1.0h_2022-05-12_1',
    'LT_R3N40_NEGATIVE_2.5h_100PCT_2.5h_2022-05-17_1',
    'LT_R3N40_2_NEGATIVE_3.5h_100PCT_3.5h_2022-05-17_1',
    'LT_R3N40_100PCT_0.2h_2022-05-18_1',
    'LT_RN41_100PCT_1.0h_2022-06-30_1',
    'LT_R3N41_ROW_97_100PCT_1.0h_2022-07-05_1',
    'LT_R3N18_100PCT_40GAIN 2022-03-10_1',
    'LT_R3N18_100PCT_40GAIN 2022-03-10_2',
    'LT_R3N58_100PCT_0.0h_2022-07-11_1',
    'LT_R3N58_100PCT_0.0h_2022-07-11_2',
    'LT_R3N58_ROW100_100PCT_24.0h_2022-07-12_1',
    'LT_GT001688_100PCT_1.0h_2022-06-30_3'
]

# legends = [' 50 % Binder', ' 30 % Binder', ' 20 % Binder', '4:1 GC to BN']
# legends = ["Matrix Carbon",
#            'GC Type 1',
#            # 'GC Type 2',
#            # "Graphite Rod",
#            "Graphite Powder",
#            "hBN",
#            # "70% GC Type 2,\n15% Resin,\n15% Carbon Black"
#            ]
# legends = [
#     'First (Same day)',
#     'Second (Same day)',
#     'First (Overnight)',
#     'Second (Overnight)',
#     # 'Single Exposure',
#     # 'Second Exposure',
#     # 'Graphite (First)',
#     # 'Graphite (Second)'
# ]

legends = [
    '66 h', '12 h', '4 h', '1 h', '-2.5 h', '-3.5 h', '0.2 h', '1 h', '1 h',
    'First (Overnight)', 'Second (Overnight)',
    'Pre-baked GC (2 s, 30%)', 'Pre-baked GC (3 s, 30%)', 'Pre-baked GC (3 s 30%)',
    'GT001688'
]

colors = plt.cm.brg(np.linspace(0, 1, len(filelist)))
add_time = 0.88

def get_experiment_params(base_path: str, filename: str):
    # Read the experiment parameters
    results_csv = os.path.join(base_path, f'{filename}.csv')
    count = 0
    params = {}
    with open(results_csv) as f:
        for line in f:
            if line.startswith('#'):
                if count > 1:
                    l = line.strip()
                    print(l)
                    if l == '#Data:':
                        break
                    pattern1 = re.compile("\s+(.*?):\s(.*?)\s(.*?)$")
                    pattern2 = re.compile("\s+(.*?):\s(.*?)$")
                    matches1 = pattern1.findall(l)
                    matches2 = pattern2.findall(l)
                    if len(matches1) > 0:
                        params[matches1[0][0]] = {
                            'value': matches1[0][1],
                            'units': matches1[0][2]
                        }
                    elif len(matches2) > 0:
                        params[matches2[0][0]] = {
                            'value': matches2[0][1],
                            'units': ''
                        }
                count += 1
    return params


def plot_pressure(base_path: str, filelist: List, legends: List, output_filename:str, display=False, plot_title=None):
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots()
    fig.set_size_inches(5.0, 5.0)

    base_pressures = np.empty_like(filelist, dtype=np.float64)
    peak_pressures = np.empty_like(filelist, dtype=np.float64)
    outgass_pressure = np.empty_like(filelist, dtype=np.float64)
    peak_dt = np.empty_like(filelist, dtype=np.float64)

    for fn, leg, c, i in zip(filelist, legends, colors, range(len(filelist))):
        params = get_experiment_params(base_path, fn)
        pressure_csv = f'{fn}_pressure.csv'
        print(pressure_csv)
        pressure_data = pd.read_csv(filepath_or_buffer=os.path.join(base_path, pressure_csv))
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


    if plot_title is not None:
        ax.set_title(plot_title)

    outgassing_rate = chamber_volume * (outgass_pressure - base_pressures) * 1E-3 / peak_dt

    outgas_df = pd.DataFrame(data={
        'Sample': legends,
        'Base Pressure (mTorr)': base_pressures,
        'Peak Pressure (mTorr)': peak_pressures,
        'Outgass Pressure (mTorr)': outgass_pressure,
        'Peak dt (s)': peak_dt,
        'Outgassing Rate (Torr L / s)': outgassing_rate
    })

    print(outgas_df)
    outgas_df.to_csv(os.path.join(base_path, f'{output_filename}_OUTGASSING.csv'), index=False)

    fig.tight_layout()
    fig.savefig(os.path.join(base_path, f'{output_filename}_PRESSURE.png'), dpi=600)
    fig.savefig(os.path.join(base_path, f'{output_filename}_PRESSURE.svg'), dpi=600)
    fig.savefig(os.path.join(base_path, f'{output_filename}_PRESSURE.eps'), dpi=600)
    if display:
        fig.show()


if __name__ == "__main__":

    # pressure_csv = f'{filename}_pressure.csv'
    # pressure_data = pd.read_csv(filepath_or_buffer=os.path.join(base_path, pressure_csv))
    # pressure_data = pressure_data.apply(pd.to_numeric)
    # # pressure_data['Pressure (Torr)'] = pressure_data['Pressure (Torr)']/1000
    # pressure_data.to_csv(os.path.join(base_path, pressure_csv), index=False)
    plot_pressure(data_path, filelist, legends, filename)
