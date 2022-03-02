from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from matplotlib.ticker import ScalarFormatter
import re

base_path = r"G:\Shared drives\ARPA-E Project\Lab\Data\Laser Tests\SAMPLES"
filename = 'BINDER_SCAN_PLOT'
filelist = ['LT_R3N12_100PCT_40GAIN 2022-03-01_1', 'LT_R3N10_100PCT_40GAIN 2022-03-01_1', 'LT_R3N3_100PCT_40GAIN 2022-03-02_1']
legends = [' 50 % Binder', ' 30 % Binder', ' 20 % Binder']


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
    fig.set_size_inches(4.5, 3.0)

    colors = []
    for fn, leg in zip(filelist, legends):
        params = get_experiment_params(base_path, fn)
        pressure_csv = f'{fn}_pressure.csv'
        print(pressure_csv)
        pressure_data = pd.read_csv(filepath_or_buffer=os.path.join(base_path, pressure_csv))
        pressure_data = pressure_data.apply(pd.to_numeric)
        time_s = pressure_data['Time (s)'].values
        pressure = 1000*pressure_data['Pressure (Torr)'].values

        # title_str = 'Sample ' + params['Sample Name']['value'] + ', '
        # params_title = params
        # params_title.pop('Sample Name')
        #
        # for i, p in enumerate(params_title):
        #     title_str += f"{params_title[p]['value']}{params_title[p]['units']}"
        #     if i + 1 < len(params_title):
        #         title_str += ', '

        line = ax.plot(time_s, pressure, label=leg)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pressure (mTorr)')
        colors.append(line[0].get_color())

    leg = ax.legend(frameon=True, loc='best', fontsize=8)
    for color, text in zip(colors, leg.get_texts()):
        text.set_color(color)


    if plot_title is not None:
        ax.set_title(plot_title)



    fig.tight_layout()
    fig.savefig(os.path.join(base_path, f'{output_filename}_PRESSURE.png'), dpi=600)
    if display:
        fig.show()


if __name__ == "__main__":

    # pressure_csv = f'{filename}_pressure.csv'
    # pressure_data = pd.read_csv(filepath_or_buffer=os.path.join(base_path, pressure_csv))
    # pressure_data = pressure_data.apply(pd.to_numeric)
    # # pressure_data['Pressure (Torr)'] = pressure_data['Pressure (Torr)']/1000
    # pressure_data.to_csv(os.path.join(base_path, pressure_csv), index=False)
    plot_pressure(base_path, filelist, legends, filename)
