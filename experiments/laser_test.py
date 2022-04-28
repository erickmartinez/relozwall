import logging
import sys, os

import numpy as np
import pandas as pd

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
from pymeasure.display.Qt import QtGui
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import IntegerParameter, FloatParameter, ListParameter, Parameter
from pymeasure.experiment import unique_filename
from serial.serialutil import SerialException
from pyvisa.errors import VisaIOError
import datetime
from instruments.esp32 import ESP32Trigger
from instruments.esp32 import DualTCLogger
from instruments.tektronix import TBS2000
from instruments.mx200 import MX200
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from instruments.inhibitor import WindowsInhibitor
# import json
from scipy import interpolate

TBS2000_RESOURCE_NAME = 'USB0::0x0699::0x03C7::C010461::INSTR'
ESP32_COM = 'COM6'
TC_LOGGER_COM = 'COM10'
MX200_COM = 'COM3'
TRIGGER_CHANNEL = 2
THERMOMETRY_CHANNEL = 1
TRIGGER_LEVEL = 3.3
SAMPLING_INTERVAL = 0.005


class LaserProcedure(Procedure):
    emission_time = FloatParameter('Emission Time', units='s', default=0.5, minimum=0.001, maximum=3.0)
    measurement_time = FloatParameter('Measurement Time', units='s', default=3.0, minimum=1.0, maximum=3600.0)
    laser_power_setpoint = FloatParameter("Laser Power Setpoint", units='%', default=100, minimum=0.0, maximum=100.0)
    pd_gain = ListParameter('Photodiode Gain', choices=('0', '10', '20', '30', '40', '50', '60', '70'), units='dB',
                            default='30')
    sample_name = Parameter("Sample Name", default="UNKNOWN")
    __oscilloscope: TBS2000 = None
    __keep_alive: bool = False
    __on_sleep: WindowsInhibitor = None
    __tc_logger: DualTCLogger = None
    __mx200: MX200 = None
    pressure_data: pd.DataFrame = None
    __tc_data: pd.DataFrame = None
    __unique_filename: str = None

    DATA_COLUMNS = ["Measurement Time (s)", "Photodiode Voltage (V)", "Trigger (V)", "TC1 (C)", "TC2 (C)"]

    def startup(self):
        print('***  Startup ****')
        self.__oscilloscope = TBS2000(resource_name=TBS2000_RESOURCE_NAME)
        self.__mx200 = MX200(address=MX200_COM)
        self.__mx200.delay = 0.030
        self.__oscilloscope.display_channel_list((1, 1, 0, 0))
        print(self.__oscilloscope.sesr)
        print(self.__oscilloscope.all_events)

    @property
    def unique_filename(self) -> str:
        return self.__unique_filename

    @unique_filename.setter
    def unique_filename(self, val: str):
        print(f"Storing filepath: {val}")
        self.__unique_filename = val

    def save_pressure(self):
        if self.pressure_data is not None:
            filename = f'{os.path.splitext(self.__unique_filename)[0]}_pressure.csv'
            self.pressure_data.to_csv(filename, index=False)

    def execute(self):
        log.info("Setting up Oscilloscope")
        self.__oscilloscope.write(f'CH{THERMOMETRY_CHANNEL}:VOLTS 1.0')
        self.__oscilloscope.write(f'CH{TRIGGER_CHANNEL}:VOLTS 1.0')
        total_time = self.measurement_time + self.emission_time
        self.__oscilloscope.set_acquisition_time(total_time)
        self.__mx200.units = 'MT'
        time.sleep(1.0)

        log.info("Setting up Triggers")
        try:
            esp32 = ESP32Trigger(address=ESP32_COM)
        except SerialException as e:
            print("Error initializing ESP32 trigger")
            raise e

        tc_logger = DualTCLogger(address=TC_LOGGER_COM)
        time.sleep(2.0)

        esp32.pulse_duration = float(self.emission_time)
        time.sleep(0.5)
        et = esp32.pulse_duration
        log.info(f'Pulse duration: {et:.2f} s.')

        # Prevent computer from going to sleep mode
        # self.inhibit_sleep()
        self.__oscilloscope.timeout = 1000
        # time.sleep(0.05)
        t1 = time.time()
        tc_logger.log_time = self.measurement_time + self.emission_time
        time.sleep(0.1)
        tc_logger.start_logging()
        self.__oscilloscope.write('ACQUIRE:STATE 0')
        self.__oscilloscope.write('ACQUIRE:STOPAFTER SEQUENCE')
        self.__oscilloscope.write('ACQuire:MODe SAMple')
        self.__oscilloscope.write('ACQUIRE:STATE ON')
        esp32.fire()

        previous_time = 0.0
        total_time = 0.0

        elapsed_time = []
        pressure = []
        start_time = time.time()
        while total_time <= self.measurement_time + self.emission_time:
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break
            current_time = time.time()
            if (current_time - previous_time) >= 0.05:
                p = self.__mx200.pressure(2)
                if p != '':
                    pressure.append(p)
                    elapsed_time.append(total_time)
                total_time = time.time() - start_time
                previous_time = current_time

        # log.info("elapsed time:")
        # log.info(elapsed_time)
        elapsed_time = np.array(elapsed_time, dtype=float)
        pressure = np.array(pressure, dtype=float)
        self.pressure_data = pd.DataFrame(
            data={
                'Time (s)': elapsed_time,
                f'Pressure (Torr)': pressure
            }
        )

        self.save_pressure()
        t2 = time.time()

        dt = t2 - t1
        log.info(f"dt: {dt:.3f}")

        while True:
            busyA = self.__oscilloscope.ask("BUSY?")
            if busyA == '0':
                break

        print('*** ACQUISITION OVER ***')
        # print(self.__oscilloscope.sesr)
        # print(self.__oscilloscope.all_events)
        data = self.__oscilloscope.get_curve(channel=THERMOMETRY_CHANNEL)
        reference = self.__oscilloscope.get_curve(channel=TRIGGER_CHANNEL)
        tc_data: pd.DataFrame = tc_logger.read_temperature_log()
        time.sleep(5.0)
        print(tc_data)

        columns = data.dtype.names
        columns_ref = reference.dtype.names
        npoints = len(data)

        ir_df = pd.DataFrame(data={
            'Time (s)': data[columns[0]],
            'Voltage (V)': data[columns[1]]
        })

        filename = f'{os.path.splitext(self.__unique_filename)[0]}_irdata.csv'
        ir_df.to_csv(path_or_buf=filename, index=False)

        log.info(f'Number of osc data points: {npoints}')
        tc_data['Time (s)'] = tc_data['Time (s)'] - tc_data['Time (s)'].min()
        time_tc = tc_data['Time (s)'].values
        tc1 = tc_data['TC1 (C)'].values
        tc2 = tc_data['TC2 (C)'].values

        filename = f'{os.path.splitext(self.__unique_filename)[0]}_tcdata.csv'
        tc_data.to_csv(filename, index=False)

        data = data[data[columns[0]] <= time_tc.max()]
        time_osc = data[columns[0]]
        print('time_osc:')
        print(time_osc)
        print(f'len(time_osc): {len(time_osc)}, time_osc.min = {time_osc.min()}, time_osc.max = {time_osc.max()}')

        print('time_tc:')
        print(time_tc)
        print(f'len(time_tc): {len(time_tc)}, time_tc.min = {time_tc.min()}, time_tc.max = {time_tc.max()}')

        msk_tmax = time_osc <= time_tc.max()
        data = data[msk_tmax]
        reference = reference[msk_tmax]
        time_osc = data[columns[0]]

        msk_tmin = time_osc >= time_tc.min()
        data = data[msk_tmin]
        reference = reference[msk_tmin]
        time_osc = data[columns[0]]

        f1 = interpolate.interp1d(time_tc, tc1)
        f2 = interpolate.interp1d(time_tc, tc2)
        tc1_interp = f1(time_osc)
        tc2_interp = f2(time_osc)

        # npoints = len(data)
        # log.info(f'Number of data points with positive voltage: {npoints}')
        # for i in range(len(data)):
        #     log.info(f't = {data[columns[0]][i]:5.3f}, {data[columns[1]][i]:8.3E}')

        # log.info(f'Data Columns: {columns}')
        for i in range(len(data)):
            d = {
                "Measurement Time (s)": data[columns[0]][i],
                "Photodiode Voltage (V)": data[columns[1]][i],
                "Trigger (V)": reference[columns_ref[1]][i],
                "TC1 (C)": tc1_interp[i],
                "TC2 (C)": tc2_interp[i]
            }
            self.emit('results', d)
            self.emit('progress', (i + 1) * 100 / len(data))
            # print(f'{data[data.dtype.names[0]][i]:.4g}, {data[data.dtype.names[1]][i]:.4g}')

        # print(data)
        self.__oscilloscope.timeout = 1

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
            inputs=['emission_time', "measurement_time", "laser_power_setpoint", "pd_gain", "sample_name"],
            displays=['emission_time', "measurement_time", "laser_power_setpoint", "pd_gain", "sample_name"],
            x_axis="Measurement Time (s)",
            y_axis="Photodiode Voltage (V)",
            directory_input=True,
        )
        self.setWindowTitle('Laser Test')

    def queue(self):
        directory = self.directory

        procedure: LaserProcedure = self.make_procedure()
        sample_name = procedure.sample_name
        laser_setpoint = procedure.laser_power_setpoint
        photodiode_gain = procedure.pd_gain

        prefix = f'LT_{sample_name}_{laser_setpoint:03.0f}PCT_{photodiode_gain}GAIN '
        filename = unique_filename(directory, prefix=prefix)
        log_file = os.path.splitext(filename)[0] + ' .log'
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)

        results = Results(procedure, filename)
        experiment = self.new_experiment(results)
        procedure.unique_filename = filename
        self.manager.queue(experiment)


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    log.addHandler(logging.NullHandler())

    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
