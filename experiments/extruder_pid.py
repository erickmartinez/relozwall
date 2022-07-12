"""
See:
https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method
"""

import logging
import os
import sys

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
from pymeasure.display.Qt import QtGui
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import FloatParameter
from pymeasure.experiment import unique_filename
from instruments.inhibitor import WindowsInhibitor
from instruments.esp32 import ExtruderReadout
from instruments.ametek import DCSource
from simple_pid import PID

DC_SOURCE_IP = '192.168.1.3'
EXT_READOUT_IP = '192.168.4.2'



class ExtruderPIDProcedure(Procedure):
    experiment_time = FloatParameter('Experiment Time', units='min', default=5, minimum=0.25, maximum=60)
    temperature_setpoint = FloatParameter('Temperature Setpoint', units='C', default=25.0, minimum=25.0, maximum=800.0)
    kp = FloatParameter('Kp', default=5000, minimum=0.0)
    ki = FloatParameter('Ki', default=0.1, minimum=0.0)
    kd = FloatParameter('Kd', default=0.05, minimum=0.0)
    __on_sleep: WindowsInhibitor = None
    __dc_source: DCSource = None
    __keep_alive: bool = False
    DATA_COLUMNS = ["Time (min)", "Temperature (C)", "Voltage (V)", "PID Control (V)"]

    def startup(self):
        log.info("Setting up power supply")
        self.__dc_source = DCSource(ip_address=DC_SOURCE_IP)

    def execute(self):
        pid = PID(self.kp, self.ki, self.kd, setpoint=self.temperature_setpoint)
        pid.output_limits = (0, 200.0)
        extruder_readout = ExtruderReadout(ip_address=EXT_READOUT_IP)  # ExtruderReadout(address=EXT_READOUT_COM)
        time.sleep(3.0)
        self.inhibit_sleep()

        log.info('Setting up DC voltage')
        # self.__dc_source.cls()
        # self.__dc_source.rst()
        self.__dc_source.current_setpoint = 8.0
        self.__dc_source.voltage_setpoint = 0.0
        self.__dc_source.output_on()
        # self.__dc_source.trigger_voltage()
        [TC1, _, _, _, _] = extruder_readout.reading
        pid.set_auto_mode(True, last_output=TC1)

        current_time = 0.0
        previous_time = 0.0
        total_time = 0.0
        dt = 0.5
        n = int(self.experiment_time * 60 / dt) + 1
        log.info(f"Starting the loop of {n:d} datapoints.")
        counter = 0

        # pid.sample_time = dt
        set_real_dt = False
        start_time = time.time()
        while total_time <= (self.experiment_time * 60) + dt:
            current_time = time.time()
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                self.__dc_source.output_off()
                break
            if (current_time - previous_time) >= dt:
                [TC1, _, _, _, _] = extruder_readout.reading
                measured_v = self.__dc_source.measured_voltage
                control = pid(TC1)
                self.__dc_source.voltage_setpoint = control
                total_time = time.time() - start_time
                previous_time = current_time
                d = {
                    "Time (min)": total_time / 60,
                    "Temperature (C)": TC1,
                    "Voltage (V)": measured_v,
                    "PID Control (V)": control,
                }
                self.emit('results', d)
                self.emit('progress', 100 * total_time / 60.0 / self.experiment_time)
                counter += 1

        self.__dc_source.voltage_setpoint = 0.0
        self.__dc_source.output_off()
        del extruder_readout
        # self.__dc_source.trigger_abort()

    def shutdown(self):
        self.unhinibit_sleep()
        if self.__dc_source is not None:
            self.__dc_source.output_off()

    def __del__(self):
        self.shutdown()

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
            procedure_class=ExtruderPIDProcedure,
            inputs=['experiment_time', 'temperature_setpoint', 'kp', 'ki', 'kd'],
            displays=['experiment_time', 'temperature_setpoint', 'kp', 'ki', 'kd'],
            x_axis="Time (min)",
            y_axis="Temperature (C)",
            directory_input=True,
        )
        self.setWindowTitle('Extruder PID')

    def queue(self):
        directory = self.directory

        procedure: ExtruderPIDProcedure = self.make_procedure()
        temperature_setpoint = procedure.temperature_setpoint
        kp = procedure.kp

        prefix = f'PID_{temperature_setpoint:03.0f}C_kp={kp:.2f}_'
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

    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
