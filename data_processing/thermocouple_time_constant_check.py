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


def correct_thermocouple_response(measured_temperature, measured_time, time_constant, order=1):
    n = len(measured_time)
    k = int(n / 10)
    k = k + 1 if k % 2 == 0 else k
    T = savgol_filter(measured_temperature, k, 3)
    dTdt = np.gradient(T, measured_time, edge_order=2)
    r = T + time_constant * dTdt

    if order >= 1:
        d2Tdt2 = np.gradient(dTdt, measured_time, edge_order=2)
        r += 0.5 * (time_constant ** 2.0) * d2Tdt2
    if order >= 2:
        d3Tdt3 = np.gradient(d2Tdt2, measured_time, edge_order=2)
        r += (1.0 / 6.0) * (time_constant ** 3.0) * d3Tdt3
    return savgol_filter(r, k, 3)


if __name__ == '__main__':
    data_df = pd.read_csv(csv).apply(pd.to_numeric)
    tau = 2.1148
    t0 = 1.3399

    # Time (s),TC1 (C),TC2 (C)
    measured_time = data_df['Time (s)'].values
    tc1 = data_df['TC1 (C)'].values
    measured_time -= t0

    idx_positive = (measured_time >= 0.0)  # & (measured_time <= 20.0)
    measured_time = measured_time[idx_positive]
    tc1 = tc1[idx_positive]

    # tc1_smooth = savgol_filter(tc1, 81, 3)

    msk = (0.0 <= measured_time) & (measured_time <= 10.0)

    # tc_ss = correct_thermocouple_response(tc1_smooth, measured_time, tau)
    # tc_ss = correct_thermocouple_response(
    #     measured_time=measured_time, measured_temperature=tc1, time_constant=tau
    # )
    # print(tc_ss)

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(4.5, 3.25)

    ax1.plot(measured_time[msk], tc1[msk], label='Data', marker='o', fillstyle='none', ls='none')

    for i in range(4):
        corrected_t = correct_thermocouple_response(
            measured_time=measured_time, measured_temperature=tc1, time_constant=tau+0.5,
            order=i
        )
        ax1.plot(
            measured_time[msk], corrected_t[msk], label=f'Corrected, order {i}'
        )

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Steady State Temperature (°C)')
    ax1.legend(loc='best')

    fig.tight_layout()
    plt.show()
