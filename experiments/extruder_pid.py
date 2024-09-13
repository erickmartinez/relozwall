"""
See:
https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method
"""

import logging
import os
import sys

import numpy as np

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
from pymeasure.display.Qt import QtWidgets
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import FloatParameter
from pymeasure.experiment.results import replace_placeholders, unique_filename
from pymeasure.log import file_log
from instruments.inhibitor import WindowsInhibitor
from instruments.esp32 import ExtruderReadout
from instruments.ametek import DCSource
from simple_pid import PID

DC_SOURCE_IP = '192.168.1.3'
EXT_READOUT_IP = '192.168.4.2'


class ExtruderPIDProcedure(Procedure):
    experiment_time = FloatParameter('Experiment Time', units='min', default=5, minimum=0.25, maximum=60)
    temperature_setpoint = FloatParameter('Temperature Setpoint', units='C', default=25.0, minimum=25.0, maximum=800.0)
    kp = FloatParameter('Kp', default=50, minimum=0.0)
    ki = FloatParameter('Ki', default=0.0, minimum=0.0)
    kd = FloatParameter('Kd', default=0.0, minimum=0.0)
    __on_sleep: WindowsInhibitor = None
    __dc_source: DCSource = None
    __keep_alive: bool = False
    DATA_COLUMNS = ["Time (s)", "Temperature (C)", "Voltage (V)", "PID Control (V)"]

    def startup(self):
        log.info("Setting up power supply")
        self.__dc_source = DCSource(ip_address=DC_SOURCE_IP)

    def execute(self):
        pid = PID(self.kp, self.ki, self.kd, setpoint=self.temperature_setpoint)
        pid.time_fn = time.perf_counter
        pid.differential_on_measurement = False
        pid.output_limits = (0, 200.0)
        extruder_readout = ExtruderReadout(ip_address=EXT_READOUT_IP)  # ExtruderReadout(address=EXT_READOUT_COM)
        time.sleep(1.0)
        # self.inhibit_sleep()

        log.info('Setting up DC voltage')
        # self.__dc_source.cls()
        # self.__dc_source.rst()
        self.__dc_source.current_setpoint = 8.0
        self.__dc_source.voltage_setpoint = 0.0
        self.__dc_source.output_on()
        # self.__dc_source.trigger_voltage()
        [TC1, _, _, _, _] = extruder_readout.reading
        # pid.set_auto_mode(True, last_output=TC1)
        tc1_readings = TC1 * np.ones(5, dtype=float)

        previous_time = 0.0
        total_time = 0.0
        dt = 0.05
        log.info(f'dt: {dt:.3f} s')
        n = int(self.experiment_time * 60 / dt) + 1
        log.info(f"Starting the loop of {n:d} datapoints.")

        pid.sample_time = 0.01
        prev_t_avg = 0.

        progress_factor = 100. / 60.0 / self.experiment_time

        start_time = time.perf_counter()
        while total_time <= (self.experiment_time * 60) + dt:
            current_time = time.perf_counter()
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                # self.__dc_source.output_off()
                break

            if (current_time - prev_t_avg) >= 0.01:
                prev_t_avg = current_time
                [temp, _, _, _, _] = extruder_readout.reading
                tc1_readings = np.roll(tc1_readings, -1)
                tc1_readings[-1] = temp

            if (current_time - previous_time) >= dt:
                previous_time = current_time
                total_time = current_time - start_time
                # [TC1, _, _, _, _] = extruder_readout.reading
                # [temp, _, _, _, _] = extruder_readout.reading
                # [temp, _, _, _, _] = extruder_readout.averaged_reading(n=2)
                # tc1_readings = np.roll(tc1_readings, -1)
                # tc1_readings[-1] = temp
                TC1 = tc1_readings.mean()

                control = pid(TC1)
                measured_v = control  # self.__dc_source.measured_voltage
                self.__dc_source.voltage_setpoint = control
                d = {
                    "Time (s)": total_time,
                    "Temperature (C)": TC1,
                    "Voltage (V)": measured_v,
                    "PID Control (V)": control,
                }
                self.emit('results', d)
                self.emit('progress', progress_factor * total_time)

        self.__dc_source.voltage_setpoint = 0.0
        self.__dc_source.output_off()
        # del extruder_readout
        # self.__dc_source.trigger_abort()

    def shutdown(self):
        # self.unhinibit_sleep()
        if self.__dc_source is not None:
            self.__dc_source.output_off()
            self.__dc_source.disconnect()
        # Remove file handlers from logger
        if len(log.handlers) > 0:
            for handler in log.handlers:
                if isinstance(handler, logging.FileHandler):
                    log.removeHandler(handler)

    def __del__(self):
        self.shutdown()

    def inhibit_sleep(self):
        if os.name == 'nt' and self.__keep_alive:
            self.__on_sleep = WindowsInhibitor()
            self.__on_sleep.inhibit()
            self.__keep_alive = True

    def unhinibit_sleep(self):
        if os.name == 'nt' and self.__keep_alive:
            self.__on_sleep.uninhibit()
            self.__keep_alive = False


def clear_log():
    if len(log.handlers) > 0:
        for h in log.handlers:
            if isinstance(h, logging.FileHandler):
                log.removeHandler(h)
                h.close()


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=ExtruderPIDProcedure,
            inputs=['experiment_time', 'temperature_setpoint', 'kp', 'ki', 'kd'],
            displays=['experiment_time', 'temperature_setpoint', 'kp', 'ki', 'kd'],
            x_axis="Time (s)",
            y_axis="Temperature (C)",
            # directory_input=True,
        )
        self.setWindowTitle('Extruder PID')
        self.filename = r'XTPID_{Temperature Setpoint}_KP{Kp:.2E}_KI{Ki:.2E}_KD{Kd:.2E}_'
        self.file_input.extensions = ["csv"]

    def queue(self):
        directory = self.directory

        procedure: ExtruderPIDProcedure = self.make_procedure()
        temperature_setpoint = procedure.temperature_setpoint
        kp = procedure.kp

        clear_log()

        file_path = unique_filename(
            directory=self.directory,
            prefix=replace_placeholders(procedure=procedure, string=self.filename)
        )

        log_file = os.path.splitext(file_path)[0] + '.log'

        # prefix = f'PID_{temperature_setpoint:03.0f}C_kp={kp:.2f}_'
        # filename = unique_filename(directory, prefix=prefix)
        # log_file = os.path.splitext(filename)[0] + '.log'
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # fh = logging.FileHandler(log_file)
        # fh.setFormatter(formatter)
        # fh.setLevel(logging.DEBUG)
        # log.addHandler(fh)
        file_log(logger=log, log_filename=log_file, level=logging.DEBUG)
        log.info(f'Starting experiment')

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

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
