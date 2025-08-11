import logging
import sys, os

import numpy as np

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
# from pymeasure.display.Qt import QtGui
from pymeasure.display.Qt import QtWidgets
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import FloatParameter, Parameter
from pymeasure.experiment import unique_filename, replace_placeholders
from pymeasure.log import console_log, file_log
from instruments.mx200 import MX200
from instruments.inhibitor import WindowsInhibitor
from instruments.esp32 import DualTCLoggerTCP
from serial import SerialException
from pymeasure.units import ureg

MX200_COM = 'COM3'
# TC_LOGGER_COM = 'COM10'
TC_LOGGER_IP = '192.168.4.3'

chamber_volume = 34.0  # L

ureg.load_definitions('./pint_units.txt')


def number_density(chamber_pressure, chamber_temperature):
    """
    kb = 1.380649e-23  # J/k = N * m / K
    pressure = 133.322 * pressure   # N / m^2
    n = pressure / (kb * (temperature + 273.15))  # (N / m^2) / (N * m ) = (#/m^3)
    n = n * 1E-6
    """
    p_k = 133.322 * chamber_pressure / 1.380649e-17
    return p_k / (chamber_temperature + 273.15)


class LaserChamberDegassing(Procedure):
    measurement_time = FloatParameter('Measurement Time', units='h', default=1.0, minimum=0.1667, maximum=48.0)
    interval = FloatParameter('Sampling Interval', units='s', default=0.1, minimum=0.01, maximum=60)
    test_name = Parameter("Test Name", default="PUMPDOWN")

    __mx200: MX200 = None
    __mx200_delay: float = 0.05
    __temperature_readout: DualTCLoggerTCP = None
    __keep_alive: bool = False
    __time_start = None
    __previous_reading: dict = None
    __degassing_time_start = None
    __is_degassing: bool = False

    DATA_COLUMNS = ["Measurement Time (s)", "Pressure (Torr)", "TC1 (C)", "TC2 (C)", "n (1/cm^3)", "Degassing Time (s)"]

    def startup(self):
        log.info("Setting up the experiment")

    def shutdown(self):
        log.info("Shutting down the experiment")
        # Remove file handlers from logger
        if "log" not in locals():
            return
        if len(log.handlers) > 0:
            for h in log.handlers:
                if isinstance(h, logging.FileHandler):
                    log.removeHandler(h)
                    h.close()

    def execute(self):
        print('***  Startup ****')
        # self.__mx200 = MX200(address=MX200_COM, keep_alive=True)
        self.__mx200 = MX200()
        time.sleep(2.0)
        log.info('Connection to pressure readout successful...')
        log.info('Connecting to the temperature readout...')
        self.__temperature_readout = DualTCLoggerTCP(ip_address=TC_LOGGER_IP)
        self.__mx200.units = 'MT'
        time.sleep(1.0)
        log.info("Connection to temperature readout successful...")
        pressure = self.__mx200.pressure(gauge_number=1, use_previous=True)
        time.sleep(0.1)

        if type(pressure) is str:
            print(pressure)
            raise SerialException(f'Could not read pressure from gauge 2.\nReturned: {pressure}')
        tc = self.__temperature_readout.temperature
        time.sleep(0.1)
        tc1 = tc[0]
        n = number_density(pressure, tc1)
        self.__previous_reading = {
            'Measurement Time (s)': 0.0,
            'Pressure (Torr)': pressure,
            "TC1 (C)": tc1,
            "TC2 (C)": tc[1],
            "n (1/cm^3)": n,
            "Degassing Time (s)": 0.0
        }

        previous_time = 0.0
        total_time = 0.0
        self.inhibit_sleep()
        self.__time_start = time.time()
        self.__is_degassing: bool = False
        while total_time <= self.measurement_time * 3600.0:
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break
            current_time = time.time()
            if (current_time - previous_time) >= self.interval:
                self.acquire_data(current_time)
                total_time = time.time() - self.__time_start
                if self.__previous_reading['Pressure (Torr)'] <= 10E-3:
                    if not self.__is_degassing:
                        self.__degassing_time_start = time.time()
                        self.__is_degassing = True
                previous_time = current_time

        self.unhinibit_sleep()
        self.__mx200.close()
        self.__temperature_readout.close()

    def acquire_data(self, current_time):
        if self.should_stop():
            log.warning("Caught the stop flag in the procedure")

        p = self.__mx200.pressure(gauge_number=1, use_previous=True)

        if p == '':
            p = self.__previous_reading['Pressure (Torr)']
        try:
            temperatures = self.__temperature_readout.temperature
            tc1, tc2 = temperatures[0], temperatures[1]
        except Exception as e:
            log.error('Error reading temperatures.')
            tc1, tc2 = self.__previous_reading['TC1 (C)'], self.__previous_reading['TC2 (C)']
        dt = current_time - self.__time_start
        n = number_density(p, tc1) if (not np.isnan(p)) and (not np.isnan(tc1)) else np.nan
        degassing_time = current_time - self.__degassing_time_start if self.__is_degassing else 0.0
        data = {
            "Measurement Time (s)": float(dt), # / 3600.0,
            "Pressure (Torr)": p,
            "TC1 (C)": tc1,
            "TC2 (C)": tc2,
            "n (1/cm^3)": n,
            "Degassing Time (s)": degassing_time #/ 3600.0
        }

        # log.info(f"Time: {dt:.3f}, Pressure: {p:6.3E}, TC1: {tc1:5.2f} °C, TC2: {tc2:5.2f} °C, n: {n:.3E} 1/cm^3")
        self.__previous_reading = data
        self.emit('results', data)
        self.emit('progress', float(dt) * 100.0 / self.measurement_time / 3600.0)
        # log.debug("Emitting results: {0}".format(data))

    def inhibit_sleep(self):
        if os.name == 'nt' and not self.__keep_alive:
            self.__on_sleep = WindowsInhibitor()
            self.__on_sleep.inhibit()
            self.__keep_alive = True

    def unhinibit_sleep(self):
        if os.name == 'nt' and self.__keep_alive:
            self.__on_sleep.uninhibit()
            self.__keep_alive = False


class MainWindow(ManagedWindow):
    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=LaserChamberDegassing,
            inputs=["measurement_time", "test_name", "interval"],
            displays=["measurement_time", "test_name", "interval"],
            x_axis="Measurement Time (s)",
            y_axis="Pressure (Torr)",
            # directory_input=True,
        )
        self.setWindowTitle('Degassing Test')
        self.filename = r'CHAMBER_{Test Name}_'
        self.file_input.extensions = ["csv"]

    def queue(self):
        directory = self.directory

        procedure: LaserChamberDegassing = self.make_procedure()

        file_path = unique_filename(
            directory=self.directory,
            prefix=replace_placeholders(procedure=procedure, string=self.filename)
        )

        # test_name = procedure.test_name

        # prefix = f'DEGASSING_{sample_name}_'
        # filename = unique_filename(directory, prefix=prefix)
        # log_file = os.path.splitext(filename)[0] + '.log'
        log_file = os.path.splitext(file_path)[0] + '.log'
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # fh = logging.FileHandler(log_file)
        # fh.setFormatter(formatter)
        # fh.setLevel(logging.DEBUG)
        # log.addHandler(fh)

        # Using pymeasure logging functionality which implements Queues for consistent multi-process logging.
        file_log(logger=log, log_filename=log_file, level=logging.DEBUG)
        log.info(f'Starting experiment')

        procedure.unique_filename = file_path
        procedure.directory = directory

        results = Results(procedure, file_path)
        experiment = self.new_experiment(results)

        self.manager.queue(experiment)


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    log.addHandler(logging.NullHandler())

    # create console handler and set level to debug
    has_console_handler = False
    if len(log.handlers) > 0:
        for handler in log.handlers:
            if isinstance(handler, logging.StreamHandler):
                has_console_handler = True

    if not has_console_handler:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        log.addHandler(ch)

    # app = QtGui.QApplication(sys.argv)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # sys.exit(app.exec_())
    sys.exit(app.exec())