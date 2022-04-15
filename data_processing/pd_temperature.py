import ir_thermography.thermometry as irt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from matplotlib.ticker import ScalarFormatter
import os
import re

import numpy as np

base_path = r"C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests"
ir_file = 'LT_R3N21_100PCT_40GAIN 2022-03-17_1.csv'

color_voltage = 'tab:red'
color_temperature = 'tab:purple'


def get_experiment_params(base_path: str, filename: str, debug: bool = False):
    # Read the experiment parameters
    results_csv = os.path.join(base_path, f'{filename}.csv')
    count = 0
    params = {}
    with open(results_csv) as f:
        for line in f:
            if line.startswith('#'):
                if count > 1:
                    l = line.strip()
                    if debug:
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


if __name__ == "__main__":
    thermometry = irt.PDThermometer()
    file_tag = os.path.splitext(ir_file)[0]
    ir_df = pd.read_csv(os.path.join(base_path, ir_file), comment='#').apply(pd.to_numeric)
    ir_df = ir_df[ir_df['Measurement Time (s)'] <= 2.0]
    params = get_experiment_params(base_path=base_path, filename=file_tag)
    pd_gain = int(params['Photodiode Gain']['value'])
    print(pd_gain)
    thermometry.gain = pd_gain
    print(thermometry.calibration_factor)
    time_s = ir_df['Measurement Time (s)'].values
    voltage = ir_df['Photodiode Voltage (V)'].values
    temperature = thermometry.get_temperature(voltage=voltage) - 273.15

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(4.5, 3.0)

    ax1.tick_params(axis='y', labelcolor=color_voltage)

    ax1.plot(
        time_s,
        voltage, color=color_voltage
    )

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Photdiode Voltage (V)', color=color_voltage)

    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor=color_temperature)

    ax2.plot(
        time_s,
        temperature,
        color=color_temperature,
        # ls='none',
    )

    ax2.set_ylabel('Temperature (°C)', color=color_temperature)

    fig1.tight_layout()

    plt.show()
