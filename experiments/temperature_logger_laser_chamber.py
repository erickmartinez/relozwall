import logging
import sys, os

import numpy as np

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
from pymeasure.display.Qt import QtGui
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import FloatParameter, Parameter
from pymeasure.experiment import unique_filename
from instruments.inhibitor import WindowsInhibitor
from instruments.esp32 import DualTCLoggerTCP

TC_LOGGER_IP = '192.168.4.3'


class TemperatureLoggerLaserChamber(Procedure):
    measurement_time = FloatParameter('Measurement Time', units='min', default=1.0, minimum=0.1667, maximum=48.0)
    interval = FloatParameter('Sampling Interval', units='s', default=0.1, minimum=0.1, maximum=60)
    sample_name = Parameter("Sample Name", default="UNKNOWN")

    __temperature_readout: DualTCLoggerTCP = None
    __keep_alive: bool = False
    __time_start = None
    __previous_reading: dict = None

    DATA_COLUMNS = ["Measurement Time (min)", "TC1 (C)", "TC2 (C)"]

    def startup(self):
        log.info("Setting up the experiment")

    def shutdown(self):
        log.info("Shutting down the experiment")
        # Remove file handlers from logger
        if len(log.handlers) > 0:
            for h in log.handlers:
                if isinstance(h, logging.FileHandler):
                    log.removeHandler(h)

    def execute(self):
        print('***  Startup ****')
        time.sleep(1.0)
        log.info('Connecting to the temperature readout...')
        self.__temperature_readout = DualTCLoggerTCP(ip_address=TC_LOGGER_IP)
        log.info("Connection to temperature readout successful...")
        time.sleep(0.1)

        tc = self.__temperature_readout.temperature
        time.sleep(0.1)
        tc1 = tc[0]
        self.__previous_reading = {
            'Measurement Time (min)': 0.0,
            "TC1 (C)": tc1,
            "TC2 (C)": tc[1],
        }

        previous_time = 0.0
        total_time = 0.0
        self.inhibit_sleep()
        self.__time_start = time.time()
        while total_time <= self.measurement_time * 60:
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                self.shutdown()
                break
            current_time = time.time()
            if (current_time - previous_time) >= self.interval:
                self.acquire_data(current_time)
                total_time = current_time - self.__time_start
                previous_time = current_time

        self.unhinibit_sleep()
        self.__temperature_readout.close()

    def acquire_data(self, current_time):
        if self.should_stop():
            log.warning("Caught the stop flag in the procedure")
        try:
            temperatures = self.__temperature_readout.temperature
            tc1, tc2 = temperatures[0], temperatures[1]
        except Exception as e:
            log.error('Error reading temperatures.')
            tc1, tc2 = self.__previous_reading['TC1 (C)'], self.__previous_reading['TC2 (C)']
        dt = current_time - self.__time_start
        data = {
            "Measurement Time (min)": float(dt) / 60.0,
            "TC1 (C)": tc1,
            "TC2 (C)": tc2,
        }

        # log.info(f"Time: {dt:.3f}, Pressure: {p:6.3E}, TC1: {tc1:5.2f} °C, TC2: {tc2:5.2f} °C, n: {n:.3E} 1/cm^3")
        self.__previous_reading = data
        self.emit('results', data)
        self.emit('progress', float(dt) * 100.0 / self.measurement_time / 60.0)
        log.debug("Emitting results: {0}".format(data))

    def inhibit_sleep(self):
        if os.name == 'nt' and not self.__keep_alive:
            self.__on_sleep = WindowsInhibitor()
            self.__on_sleep.inhibit()
            self.__keep_alive = True

    def unhinibit_sleep(self):
        if os.name == 'nt' and self.__keep_alive:
            self.__on_sleep.uninhibit()
            self.__keep_alive = False

    def shutdown(self):
        self.unhinibit_sleep()
        # Remove file handlers from logger
        if len(log.handlers) > 0:
            for h in log.handlers:
                if isinstance(h, logging.FileHandler):
                    log.removeHandler(h)
                if isinstance(h, logging.NullHandler):
                    log.removeHandler(h)
                    log.addHandler(logging.NullHandler())


class MainWindow(ManagedWindow):
    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=TemperatureLoggerLaserChamber,
            inputs=["measurement_time", "sample_name", "interval"],
            displays=["measurement_time", "sample_name", "interval"],
            x_axis="Measurement Time (min)",
            y_axis="TC2 (C)",
            directory_input=True,
        )
        self.setWindowTitle('Temperature logging (laser chamber)')

    def queue(self):
        directory = self.directory

        procedure: TemperatureLoggerLaserChamber = self.make_procedure()
        sample_name = procedure.sample_name

        prefix = f'TLOGGER_{sample_name}_'
        filename = unique_filename(directory, prefix=prefix)
        log_file = os.path.splitext(filename)[0] + '.log'
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)

        results = Results(procedure, filename)
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

    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
