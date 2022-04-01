import logging
import sys, os

import numpy as np

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
from pymeasure.display.Qt import QtGui
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import IntegerParameter, FloatParameter, Parameter
from pymeasure.experiment import unique_filename
from instruments.mx200 import MX200
import datetime
from instruments.inhibitor import WindowsInhibitor
from instruments.esp32 import ExtruderReadout
from serial import SerialException

EXT_READOUT_COM = 'COM12'
MX200_COM = 'COM3'
NUMBER_OF_SAMPLES = 10000


class ExtrusionProcedure(Procedure):
    experiment_time = FloatParameter('Experiment Time', units='min', default=1, minimum=0.25, maximum=60)
    temperature_setpoint = FloatParameter('Temperature', units='C', default=350, minimum=25, maximum=1000.0)
    sample_name = Parameter("Sample Name", default="UNKNOWN")
    __mx200: MX200 = None
    __time_start: datetime.datetime = None
    __ndata_points: int = 0
    __on_sleep: WindowsInhibitor = None
    __mx200_delay: float = 0.05
    __keep_alive: bool = False
    __failed_readings = 0
    __max_attempts = 10
    __previous_reading: dict = None
    __previous_pressure: float = None

    DATA_COLUMNS = ["Time (s)", "Baking Pressure (Torr)", "Outgassing Pressure (Torr)", "Baking Temperature (C)",
                    "Outgassing Temperature (C)"]

    def startup(self):
        log.info("Setting up Televac MX200")
        self.__mx200 = MX200(address=MX200_COM)
        time.sleep(1.0)
        self.__mx200.units = 'MT'
        # time.sleep(3.0)
        # print(f"Pressures: {self.__mx200.pressures}")

    def execute(self):
        self.__mx200.units = 'MT'
        # Reset the counter for failed readings
        self.__failed_readings = 0
        dt = max(1.0, self.experiment_time * 60 / NUMBER_OF_SAMPLES)
        n = int(self.experiment_time * 60 / dt) + 1

        log.info(f"Starting the loop of {n:d} datapoints.")
        extruder_readout = ExtruderReadout(address=EXT_READOUT_COM)
        time.sleep(3.0)
        self.inhibit_sleep()

        previous_time = 0.0
        total_time = 0.0

        counter = 0
        start_time = time.time()
        while total_time <= (self.experiment_time * 60) + dt:
            current_time = time.time()
            if (current_time - previous_time) >= dt:
                pressures = self.__mx200.pressures
                for k, p in pressures.items():
                    if type(p) is str:
                        pressures[k] = np.nan
                [TC1, TC2, force, _] = extruder_readout.reading
                total_time = time.time() - start_time
                previous_time = current_time

                d = {
                    "Time (s)": total_time,
                    "Baking Pressure (Torr)": pressures[1],
                    "Outgassing Pressure (Torr)": pressures[2],
                    "Baking Temperature (C)": TC1,
                    "Outgassing Temperature (C)": TC2,
                }
                self.emit('results', d)
                self.emit('progress', (counter + 1) * 100 / n)
                counter += 1

    def shutdown(self):
        self.unhinibit_sleep()

    def __del__(self):
        self.shutdown()

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
            procedure_class=ExtrusionProcedure,
            inputs=['experiment_time', 'sample_name', 'temperature_setpoint'],
            displays=['experiment_time', 'sample_name', 'temperature_setpoint'],
            x_axis="Time (s)",
            y_axis="Outgassing Pressure (Torr)",
            directory_input=True,
        )
        self.setWindowTitle('Extrusion data')

    def queue(self):
        directory = self.directory

        procedure: ExtrusionProcedure = self.make_procedure()
        sample_name = procedure.sample_name
        temperature_setpoint = procedure.temperature_setpoint

        prefix = f'EXTRUSION_{sample_name}_{temperature_setpoint:03.0f}C'
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
