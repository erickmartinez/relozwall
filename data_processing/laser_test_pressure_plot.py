import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from matplotlib.ticker import ScalarFormatter
import re

base_path = r"G:\Shared drives\ARPA-E Project\Lab\Data\Laser Tests\SAMPLES"
filename = 'LT_R3N12_100PCT_40GAIN 2022-03-01_1'

if __name__ == "__main__":
    # Read the experiment parameters
    results_csv = os.path.join(base_path,f'{filename}.csv')
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
    nparams = len(params)
    pressure_csv = f'{filename}_pressure.csv'
    pressure_data = pd.read_csv(filepath_or_buffer=os.path.join(base_path, pressure_csv))
    pressure_data = pressure_data.apply(pd.to_numeric)
    time_s = pressure_data['Time (s)'].values
    pressure = pressure_data['Pressure (Torr)'].values

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3.0)
    ax.plot(time_s, pressure, label='Pressure')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure (mTorr)')
    title_str = 'Sample ' + params['Sample Name']['value'] + ', '
    params_title = params
    params_title.pop('Sample Name')
    for i, p in enumerate(params_title):
        title_str += f"{params_title[p]['value']}{params_title[p]['units']}"
        if i+1 < len(params_title):
            title_str += ', '
    ax.set_title(title_str)

    fig.tight_layout()
    fig.savefig(os.path.join(base_path, f'{filename}_pressure.png'), dpi=600)
    plt.show()