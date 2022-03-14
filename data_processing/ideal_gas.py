from instruments.esp32 import DualTCLogger
from instruments.mx200 import MX200
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import json
import time
import datetime

base_path = r'G:\Shared drives\ARPA-E Project\Lab\Data\Laser Tests\STARTING_MATERIALS\chamber_pressure'
log_time = 300  # s
chamber_volume_in = 14.0 * 15.0 * 11.0
chamber_volume_m = chamber_volume_in * (2.54E-2 ** 3)
log_file = 'IG_LOG_DATA_20220308_123650.csv'


def number_density(chamber_pressure, chamber_temperature):
    """
    kb = 1.380649e-23  # J/k = N * m / K
    pressure = 133.322 * pressure   # N / m^2
    n = pressure / (kb * (temperature + 273.15))  # (N / m^2) / (N * m ) = (#/m^3)
    n = n * 1E-6
    """
    p_k = 133.322 * chamber_pressure / 1.380649e-17
    return p_k / (chamber_temperature + 273.15)


if __name__ == '__main__':
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    if log_file is None:
        tc_logger = DualTCLogger(address='COM7')
        time.sleep(2.5)
        mx200 = MX200(address='COM3')
        time.sleep(2.5)
        mx200.delay = 0.020
        time.sleep(2.5)
        mx200.units = 'MT'
        interval = 0.1
        print('Stating data logging.')

        previous_time = 0.0
        total_time = 0.0

        elapsed_time = []
        pressure = []
        temperature = []
        start_time = time.time()
        while total_time <= log_time:
            current_time = time.time()
            if (current_time - previous_time) >= interval:
                p = mx200.pressure(2)
                temp = tc_logger.temperature[0]
                if p != '':
                    pressure.append(p)
                    elapsed_time.append(total_time)
                    temperature.append(temp)

                total_time = time.time() - start_time
                previous_time = current_time

        print('Done logging.')
        elapsed_time = np.array(elapsed_time, dtype=float)
        pressure = np.array(pressure, dtype=float)
        temperature = np.array(temperature, dtype=float)
        concentration = number_density(pressure, temperature)

        df = pd.DataFrame(
            data={
                'Time (s)': elapsed_time,
                'Pressure (Torr)': pressure,
                'Temperature (°C)': temperature,
                'Concentration (1/cm3)': concentration
            }
        )

        current_date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        print(current_date)
        filename = os.path.join(base_path, f'IG_LOG_DATA_{current_date}')

        df.to_csv(path_or_buf=filename + '.csv', index=False)
    else:
        df = pd.read_csv(os.path.join(base_path, log_file))
        elapsed_time = np.array(df['Time (s)'], dtype=float)
        pressure = np.array(df['Pressure (Torr)'], dtype=float)
        temperature = np.array(df['Temperature (°C)'], dtype=float)
        concentration = number_density(pressure, temperature)
        filename = os.path.splitext(log_file)[0]
        df['Concentration (1/cm3)'] = concentration
        df.to_csv(path_or_buf=os.path.join(base_path, filename) + '.csv', index=False)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(5.5, 3.0)

    color_1 = 'tab:blue'
    color_2 = 'tab:red'
    color_3 = 'tab:green'

    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.2))
    ax1.set_yscale('log')
    ax3.set_yscale('log')

    ax1.plot(elapsed_time, pressure, color=color_1, zorder=2)
    ax2.plot(elapsed_time, temperature, color=color_2, zorder=3)
    ax3.plot(elapsed_time, concentration, color=color_3, zorder=1)

    ax1.set_xlim(0, elapsed_time.max())
    ax2.set_xlim(0, elapsed_time.max())
    ax3.set_xlim(0, elapsed_time.max())

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pressure (Torr)', color=color_1)
    ax2.set_ylabel('Temperature (°C)', color=color_2)
    ax3.set_ylabel('Concentration (cm$^{-3}$)', color=color_3)

    ax1.tick_params(axis='y', labelcolor=color_1)
    ax2.tick_params(axis='y', labelcolor=color_2)
    ax3.tick_params(axis='y', labelcolor=color_3)

    fig.tight_layout()
    fig.savefig(os.path.join(base_path, filename + '.png'))
    plt.show()
