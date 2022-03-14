from instruments.tektronix import TBS2000
from instruments.esp32 import ESP32Trigger
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import time

TBS2000_RESOURCE_NAME = 'USB0::0x0699::0x03C7::C010461::INSTR'
ESP32_COM = 'COM10'
TC_LOGGER_COM = 'COM7'
MX200_COM = 'COM3'
TRIGGER_CHANNEL = 2
THERMOMETRY_CHANNEL = 1
SAMPLING_TIME = 3.0 # s
PULSE_WIDTH = 0.5

if __name__ == '__main__':
    scope = TBS2000(resource_name=TBS2000_RESOURCE_NAME)
    time.sleep(2)
    trigger = ESP32Trigger(address=ESP32_COM)
    trigger.pulse_duration = PULSE_WIDTH
    print(f'Trigger pulse width: {trigger.pulse_duration} s')
    scope.timeout = 1000 #(SAMPLING_TIME + PULSE_WIDTH) * 1000
    scope.display_channel_list((1, 1, 0, 0))
    scope.set_acquisition_time(SAMPLING_TIME+PULSE_WIDTH)
    scope.write(f'CH{THERMOMETRY_CHANNEL}:VOLTS 1.0')
    scope.write(f'CH{TRIGGER_CHANNEL}:VOLTS 1.0')
    print(scope.query('HORizontal?'))
    print(scope.query('ACQUIRE?'))
    sample_rate = float(scope.ask('HORizontal:SAMPLERATE?'))
    n_samples = int(sample_rate * (SAMPLING_TIME + PULSE_WIDTH))
    print(f'SAMPLE RATE: {sample_rate:8.4f}, N SAMPLES: {n_samples:5d}')
    scope.write('ACQUIRE:STATE RUN')
    previous_time = 0.0
    total_time = 0.0
    # time.sleep(0.5)
    trigger.fire()

    start_time = time.time()
    while total_time <= SAMPLING_TIME + PULSE_WIDTH:
        current_time = time.time()
        if (current_time - previous_time) >= 0.05:
            print(f'Running acquistion. Elapsed time {total_time:5.3f} s')
            total_time = time.time() - start_time
            previous_time = current_time

    print(scope.query('ACQuire:NUMACq?'))
    # scope.write('ACQUIRE:STATE STOP')

    # print(scope.query('*OPC?'))
    print('*** ACQUISITION OVER ***')
    print(scope.sesr)
    print(scope.all_events)

    print(scope.query('WFMPRE?'))
    ch1 = scope.get_curve(channel=TRIGGER_CHANNEL)
    print(ch1)
    print(f'Number of scope data points: {len(ch1)}')
    columns = ch1.dtype.names
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3.0)

    ax.plot(ch1[columns[0]], ch1[columns[1]])
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])

    trigger.close()
    del trigger

    fig.tight_layout()
    plt.show()
