import logging
import os
import sys

import numpy as np
from serial import SerialException

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
from pymeasure.display.Qt import QtWidgets
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import FloatParameter, Parameter, ListParameter
from pymeasure.experiment import unique_filename
from instruments.imada import DST44A
import datetime
from instruments.inhibitor import WindowsInhibitor
from instruments.position_pot import DeflectionReader
from serial.tools import list_ports
import threading
from typing import Callable
from numpy.random import default_rng


"""
Check which ports are avilable:

Open Terminal and type:

> python -m serial.tools.list_ports

before connecting probe and after connecting it, to see which port is it attached to.
"""

ports = list_ports.comports(include_links=True)
string_ports = [p.name for p in ports]


def acquire_force(instrument:DST44A, value):
    value[0] = instrument.read()


def acquire_distance(instrument:DeflectionReader, conversion_callback:Callable, value):
    value[0], value[1] = conversion_callback(instrument.reading)


class SimulatedDST44A():
    __init_time: float = None
    __step_time: float = 0.025
    __current_time: float = 0.0
    def __init__(self, seed=None):
        rng = default_rng(seed)
        self.__init_time = rng.normal(loc=2., scale=1)
        self.__rng = rng
    def read(self):
        if self.__current_time < 10.:
            err_pct = self.__rng.normal(loc=1, scale=0.01)
            r = self.__current_time * 5. * err_pct
        else:
            r = 0.
        self.__current_time += self.__step_time
        return r

    def close(self):
        return True

class SimulatedDistanceReader():
    __init_time: float = None
    __step_time: float = 0.025
    __current_time: float = 0.0

    def __init__(self, seed=None):
        rng = default_rng(seed)
        self.__init_time = rng.normal(loc=2., scale=1)
        self.__rng = rng

    @property
    def reading(self):
        if self.__current_time < 10.:
            err_pct = self.__rng.normal(loc=1, scale=0.001)
            r = int(400 + self.__current_time * 1. * err_pct)
        else:
            r = 400.
        self.__current_time += self.__step_time
        return r

    def close(self):
        return True


class FlexuralStressProcedure(Procedure):
    experiment_time = FloatParameter('Experiment time', units='s', default=30)
    calibrated_thickness = FloatParameter('Calibrated thickness', units='mm', default=12.69)
    support_span = FloatParameter('Support span', units='mm', default=30)
    sample_diameter = FloatParameter('Sample diameter', units='mm', default=10)
    sample_name = Parameter('Sample name', default="TEST")
    force_gauge_port: ListParameter = ListParameter('Force gauge port', choices=string_ports)
    deflection_pot_port: ListParameter = ListParameter('Deflection pot port', choices=string_ports)

    __force_gauge: DST44A = None
    __potReadout: DeflectionReader = None
    __time_start: float = None
    __ndata_points: int = 0
    __on_sleep: WindowsInhibitor = None
    __keep_alive: bool = False
    __max_attempts = 10
    __calibration: dict = {
        'a0': 212, 'a1': 9E-3, 'b0': 2.91, 'b1': -60E-4, 'b2': 3.97E-6
    }
    __alpha_c = 400
    __simulation = True
    interval: float = 0.025

    DATA_COLUMNS = [
        'Time (s)', 'Force (N)', 'Flexural stress (Pa)',
        'Displacement (mm)', 'Displacement err (mm)'
    ]


    def startup(self):
        if self.__simulation:
            self.__force_gauge = SimulatedDST44A()
            self.__potReadout = SimulatedDistanceReader()
        else:
            log.info(f"Setting up DST44A on port {self.force_gauge_port}")
            self.__force_gauge = DST44A(address=self.force_gauge_port)
            self.__force_gauge.connect()
            time.sleep(0.2)
            self.__force_gauge.set_logger(log)
            self.__force_gauge.units('N')
            time.sleep(0.2)
            log.info(f"Setting up Pot readout on port {self.deflection_pot_port}")
            self.__potReadout = DeflectionReader(address=self.deflection_pot_port)
            time.sleep(0.2)
            self.__potReadout.set_logger(log)

            log.info(f"Current position: {self.adc_to_mm(self.__potReadout.reading)} mm")
            time.sleep(0.2)
            log.info(f"Current force: {self.__force_gauge.read():.0f} N")
            time.sleep(0.2)


    def execute(self):
        """
        Add user interaction to calibrate z0
        """
        input('Place reference sample with known thickness on the 3-point bend stand.')
        input('Move the lever down until the probe tip touches the reference sample.\n'
              'Do not apply more than 10 N of force. Keep the setup in that position for 30 seconds\n'
              'TO PROCEED PRESS ANY KEY')

        start_time = time.time()
        previous_time = 0.0
        total_time = 0.0

        reading_force = [self.__force_gauge.read()]
        reading_distance = [0, 0]
        adc_max = -1

        while total_time <= 30.:
            current_time = time.time()
            if (current_time - previous_time) >= self.interval:
                p_d = threading.Thread(target=acquire_distance,
                                       args=(self.__potReadout, self.adc_to_adc, reading_distance))
                p_d.start()
                time.sleep(0.005)
                p_d.join()
                adc_c, _ = reading_distance[0]
                adc_max = max(adc_c, adc_max)
                total_time = current_time - start_time

        self.__alpha_c = adc_max

        previous_time = 0.0
        total_time = 0.0
        start_time = time.time()
        self.__time_start = start_time

        round_factor = 0.001
        inv_round_factor = 1. / round_factor
        round_factor_d = 0.1
        inv_round_factor_d = 10.
        inv_experiment_time = 1. / self.experiment_time

        while total_time <= self.experiment_time:
            current_time = time.time()
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break
            if (current_time - previous_time) >= self.interval:
                # self.acquire_data(current_time)
                p_f = threading.Thread(target=acquire_force, args=(self.__force_gauge, reading_force))
                p_d = threading.Thread(target=acquire_distance,
                                       args=(self.__potReadout, self.adc_to_mm, reading_distance))
                p_f.start()
                p_d.start()
                time.sleep(0.005)
                p_f.join()
                p_d.join()
                force, displacement_mm, displacement_err_mm = reading_force[0], reading_distance[0], reading_distance[1]
                sigma_f = 8.0 * force * self.support_span / np.pi / (self.sample_diameter ** 3.0) * 1E6
                total_time = current_time - start_time

                data = {
                    "Time (s)": float(round_factor*round(total_time*inv_round_factor)),
                    "Force (N)": round_factor*round(force*inv_round_factor),
                    "Flexural stress (Pa)": sigma_f,
                    "Displacement (mm)": round_factor_d*round(displacement_mm*inv_round_factor_d),
                    'Displacement err (mm)': round_factor_d*round(displacement_err_mm*inv_round_factor_d)
                }
                # log.info(f"Time: {dt:.3f}, z: {displacement_mm:.2f} mm, Force: {force:.3E}, Flexural Stress: {sigma_f:.3E}.")
                self.emit('results', data)
                self.emit('progress', total_time * 100.0 * inv_experiment_time)
                previous_time = current_time

    def adc_to_mm(self, val):
        c = self.__calibration
        return (c['a1'] * (val - self.__alpha_c) - self.calibrated_thickness + self.sample_diameter,
                c['b0'] + c['b1'] * val + c['b2'] * val * val)


    def adc_to_adc(self, val):
        return val, val

    def shutdown(self):
        if self.__potReadout is not None:
            self.__potReadout.close()
        if self.__force_gauge is not None:
            self.__force_gauge.close()
        # Remove file handlers from logger
        if len(log.handlers) > 0:
            for handler in log.handlers:
                if isinstance(handler, logging.FileHandler):
                    log.removeHandler(handler)

    def inhibit_sleep(self):
        if os.name == 'nt' and not self.__keep_alive:
            self.__on_sleep = WindowsInhibitor()
            self.__on_sleep.inhibit()
            self.__keep_alive = True

    def unhinibit_sleep(self):
        if os.name == 'nt' and self.__keep_alive:
            self.__on_sleep.uninhibit()
            self.__keep_alive = False

    def __del__(self):
        pass


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=FlexuralStressProcedure,
            inputs=[
                'sample_name', 'experiment_time', 'support_span', 'sample_diameter',
                "force_gauge_port", 'calibrated_thickness', 'deflection_pot_port'
            ],
            displays=[
                'sample_name', 'experiment_time', 'support_span', 'sample_diameter',
                "force_gauge_port", 'calibrated_thickness', 'deflection_pot_port'
            ],
            x_axis="Time (s)",
            y_axis="Flexural stress (Pa)",
            directory_input=True,
        )
        self.setWindowTitle('3-Point Bend Test')

    def queue(self):
        procedure: FlexuralStressProcedure = self.make_procedure()
        sample_name = procedure.sample_name
        directory = self.directory
        prefix = f'3PBT_{sample_name}_'
        filename = unique_filename(directory, prefix=prefix)
        log_file = os.path.splitext(filename)[0] + '.log'
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
    log.setLevel(logging.DEBUG)

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
