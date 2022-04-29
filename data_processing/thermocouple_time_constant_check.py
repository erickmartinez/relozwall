import sys, os
from scipy.linalg import svd
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
import datetime
from instruments.esp32 import DualTCLogger
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from scipy import interpolate
import confidence as cf

csv = 'TC00120220427-164143.csv'


def model(temperature, t, tau):
    dTdt = np.gradient(temperature, t)
    return temperature + tau * dTdt


def correct_thermocouple_response(
        measured_time: np.ndarray, measured_temperature: np.ndarray, time_constant: float = 1.0,
):
    dts = np.diff(measured_time)
    dTs = np.diff(measured_temperature)
    T0 = measured_temperature[0]
    output_temperature = np.empty_like(measured_temperature, dtype=np.float64)
    output_temperature[0] = T0
    t_sum = T0
    for i in range(1, len(measured_temperature)):
        j = i - 1
        dT_correction = dTs[j] / (1.0 - np.exp(-(dts[j] - 1.339) / time_constant))
        t_sum += dT_correction
        output_temperature[i] = t_sum
    return output_temperature


if __name__ == '__main__':
    data_df = pd.read_csv(csv).apply(pd.to_numeric)
    tau = 2.1148
    t0 = 1.3399

    # Time (s),TC1 (C),TC2 (C)
    measured_time = data_df['Time (s)'].values
    tc1 = data_df['TC1 (C)'].values

    tc1_smooth = savgol_filter(tc1, 81, 3)

    msk = (0.5 <= measured_time) & (measured_time <= 100)

    T0 = tc1_smooth[0]
    dT = tc1_smooth[msk] - T0

    tc_ss = model(tc1_smooth, measured_time, tau)
    # tc_ss = correct_thermocouple_response(
    #     measured_time=measured_time, measured_temperature=tc1, time_constant=tau
    # )
    print(tc_ss)

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(4.5, 3.25)

    ax1.plot(measured_time, tc1)

    ax1.plot(measured_time, tc_ss)

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Steady State Temperature (°C)')

    fig.tight_layout()
    plt.show()
