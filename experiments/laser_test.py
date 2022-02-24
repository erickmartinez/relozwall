import logging
import sys, os

import numpy as np

sys.path.append('../')
sys.modules['cloudpickle'] = None

import threading
import time
from pymeasure.display.Qt import QtGui
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import IntegerParameter, FloatParameter, ListParameter
from pymeasure.experiment import unique_filename
from serial.serialutil import SerialException
from pyvisa.errors import VisaIOError
import datetime
from instruments.esp32 import ESP32Trigger
from instruments.tektronix import TBS2000
import matplotlib as mpl
import matplotlib.pyplot as plt
from instruments.inhibitor import WindowsInhibitor
import json

TBS2000_RESOURCE_NAME = 'USB0::0x0699::0x03C7::C010461::INSTR'
ESP32_COM = 'COM10'
TRIGGER_CHANNEL = 2
THERMOMETRY_CHANNEL = 1
TRIGGER_LEVEL = 12.0


class LaserProcedure(Procedure):
    emission_time = FloatParameter('Emission Time', units='s', default=0.5, minimum=0.001, maximum=3.0)
    measurement_time = FloatParameter('Measurement Time', units='s', default=3.0, minimum=1.0, maximum=3600.0)
    __oscilloscope: TBS2000 = None
    __esp32: ESP32Trigger = None
    __keep_alive: bool = False
    __on_sleep: WindowsInhibitor = None

    DATA_COLUMNS = ["Measurement Time (s)", "Photodiode Voltage (V)", "Trigger (V)"]

    def startup(self):
        self.__oscilloscope = TBS2000(resource_name=TBS2000_RESOURCE_NAME)
        # self.__oscilloscope.trigger_channel = TRIGGER_CHANNEL
        # self.__oscilloscope.trigger_level = TRIGGER_LEVEL
        # log.info(self.__oscilloscope.query('TRIGGER?'))
        self.__oscilloscope.display_channel_list((1, 1, 0, 0))

    def execute(self):
        log.info("Setting up Oscilloscope")
        self.__oscilloscope.horizontal_main_scale = self.measurement_time / 8
        self.__oscilloscope.write(f'CH{THERMOMETRY_CHANNEL}:VOLTS 1.0')
        self.__oscilloscope.write(f'CH{TRIGGER_CHANNEL}:VOLTS 1.0')
        log.info("Setting up Triggers")
        try:
            esp32 = ESP32Trigger(address=ESP32_COM)
        except SerialException as e:
            esp32 = ESP32Trigger(address=ESP32_COM)

        esp32.pulse_duration = float(self.emission_time)
        et = esp32.pulse_duration
        log.info(f'Pulse duration: {et:.2f} s.')

        # Prevent computer from going to sleep mode
        # self.inhibit_sleep()
        self.__oscilloscope.timeout = (self.measurement_time + 2.0) * 1000
        time.sleep(0.1)
        t1 = datetime.datetime.now()
        self.__oscilloscope.acquire_on()
        esp32.fire()

        time.sleep(self.measurement_time + et + 0.1)
        # self.__oscilloscope.write('TRIGger FORCe')
        self.__oscilloscope.acquire_off()
        t2 = datetime.datetime.now()
        # log.info(self.__oscilloscope.query('*OPC?'))
        # self.__oscilloscope.write('MEASU:IMMED:TYPE MEAN')
        # log.info(self.__oscilloscope.write('MEASU:IMMED:VALUE?'))
        dt = t2 - t1
        log.info(f"t1: {t1.strftime('%Y/%m/%d, %H:%M:%S')}, t2: {t2.strftime('%Y/%m/%d, %H:%M:%S')}, dt: {dt.total_seconds():.3f}")
        time.sleep(0.1)
        log.info(self.__oscilloscope.sesr)
        time.sleep(0.1)
        log.info(self.__oscilloscope.all_events)
        time.sleep(0.1)
        data = self.__oscilloscope.get_curve(channel=THERMOMETRY_CHANNEL)
        time.sleep(0.1)
        reference = self.__oscilloscope.get_curve(channel=TRIGGER_CHANNEL)
        # print(reference)
        columns = data.dtype.names
        npoints = len(data)
        log.info(f'Number of data points: {npoints}')
        data = data[data[columns[1]] >= 0]
        t_min = data[columns[0]].min()
        data = data[data[columns[0]] >= t_min]
        data[columns[0]] = data[columns[0]] - t_min
        npoints = len(data)
        log.info(f'Number of data points with positive voltage: {npoints}')
        for i in range(len(data)):
            log.info(f't = {data[columns[0]][i]:5.3f}, {data[columns[1]][i]:8.3E}')
        td = t2 - t1
        td_s = td.total_seconds()
        # time_s = np.linspace(0, td_s, npoints)
        columns_ref = reference.dtype.names
        log.info(f'Data Columns: {columns}')
        for i in range(len(data)):
            d = {
                "Measurement Time (s)": data[columns[0]][i],
                "Photodiode Voltage (V)": data[columns[1]][i],
                "Trigger (V)": reference[columns_ref[1]][i]
            }
            self.emit('results', d)
            self.emit('progress', (i + 1) * 100 / len(data))
            # print(f'{data[data.dtype.names[0]][i]:.4g}, {data[data.dtype.names[1]][i]:.4g}')

        # print(data)
        self.__oscilloscope.timeout = 1
        # # Plot data
        # with open('../plot_style.json', 'r') as style_file:
        #     mpl.rcParams.update(json.load(style_file)['defaultPlotStyle'])
        # fig, ax = plt.subplots()
        # ax.plot(time_s, reference[columns_ref[1]], label='Trigger')
        # ax.plot(time_s, data[columns[1]], label='Photodiode')
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Voltage (V)')
        # ax.set_xlim(0, round(td_s/10)*10)
        # ax.legend(loc='best', frameon=True)
        # fig.tight_layout()
        # fig.show()
        # self.unhinibit_sleep()

    def inhibit_sleep(self):
        if os.name == 'nt' and not self.__keep_alive:
            self.__on_sleep = WindowsInhibitor()
            self.__on_sleep.inhibit()
            self.__keep_alive = True

    def unhinibit_sleep(self):
        if os.name == 'nt' and self.__keep_alive:
            self.__on_sleep.unhinibit()
            self.__keep_alive = False


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=LaserProcedure,
            inputs=['emission_time', "measurement_time"],
            displays=['emission_time', "measurement_time"],
            x_axis="Measurement Time (s)",
            y_axis="Photodiode Voltage (V)",
            directory_input=True,
        )
        self.setWindowTitle('Laser data')

    def queue(self):
        directory = self.directory
        filename = unique_filename(directory, prefix='LASER_')
        log_file = os.path.splitext(filename)[0] + ' .log'
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)

        procedure = self.make_procedure()
        results = Results(procedure, filename)
        experiment = self.new_experiment(results)

        self.manager.queue(experiment)


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    log.addHandler(logging.NullHandler())

    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
