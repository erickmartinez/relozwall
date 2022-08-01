import logging
import os
import sys

import numpy as np

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
from pymeasure.display.Qt import QtGui
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import FloatParameter, Parameter, ListParameter
from pymeasure.experiment import unique_filename
from instruments.imada import DST44A
import datetime
from instruments.inhibitor import WindowsInhibitor
from instruments.position_pot import DeflectionReader
from serial.tools import list_ports

"""
Check which ports are avilable:

Open Terminal and type:

> python -m serial.tools.list_ports

before connecting probe and after connecting it, to see which port is it attached to.
"""

ports = list_ports.comports(include_links=True)
string_ports = [p.name for p in ports]


class FlexuralStressProcedure(Procedure):
    experiment_time = FloatParameter('Experiment Time', units='s', default=30)
    # interval = FloatParameter('Sampling Interval', units='s', default=0.2)
    support_span = FloatParameter('Support Span', units='mm', default=40)
    beam_diameter = FloatParameter('Beam Diameter', units='mm', default=10)
    sample_name = Parameter("Sample Name", default="UNKNOWN")
    force_gauge_port: ListParameter = ListParameter('Force gauge port', choices=string_ports)
    deflection_pot_port: ListParameter = ListParameter('Deflection pot port', choices=string_ports)

    __force_gauge: DST44A = None
    __potReadout: DeflectionReader = None
    __time_start: datetime.datetime = None
    __ndata_points: int = 0
    __on_sleep: WindowsInhibitor = None
    __keep_alive: bool = False
    __max_attempts = 10
    __calibration: dict = {
        'a0': -9.0562, 'a1': 8.57E-3
    }
    interval: float = 0.025

    DATA_COLUMNS = ["Time (s)", "Force (N)", "Flexural Stress (Pa)", "Displacement (mm)"]

    def startup(self):
        log.info(f"Setting up DST44A on port {self.force_gauge_port}")
        self.__force_gauge = DST44A(address=self.force_gauge_port)
        self.__force_gauge.connect()
        time.sleep(0.1)
        self.__force_gauge.set_logger(log)
        self.__force_gauge.units('N')
        log.info(f"Setting up Pot readout on port {self.deflection_pot_port}")
        self.__potReadout = DeflectionReader(address=self.deflection_pot_port)
        time.sleep(0.1)
        self.__potReadout.set_logger(log)
        time.sleep(0.1)

    def execute(self):
        previous_time = 0.0
        total_time = 0.0
        start_time = time.time()
        self.__time_start = start_time
        while total_time <= self.experiment_time:
            current_time = time.time()
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break
            if (current_time - previous_time) >= self.interval:
                self.acquire_data(current_time)
                total_time = current_time - start_time
                previous_time = current_time

    def adc_to_mm(self, val):
        c = self.__calibration
        return c['a0'] + c['a1'] * val

    def shutdown(self):
        self.unhinibit_sleep()
        if self.__potReadout is not None:
            self.__potReadout.close()
        if self.__force_gauge is not None:
            self.__force_gauge.close()
        # Remove file handlers from logger
        if len(log.handlers) > 0:
            for handler in log.handlers:
                if isinstance(handler, logging.FileHandler):
                    log.removeHandler(handler)

    def acquire_data(self, current_time):
        if self.should_stop():
            log.warning("Caught the stop flag in the procedure")
        # force_data = self.__force_gauge.read()
        force = self.__force_gauge.read()
        time.sleep(0.0001)
        adc_pot = self.__potReadout.reading
        # force = force_data['reading']
        # units = force_data['units']
        # mode = force_data['mode']
        # judgment = force_data['judgement']
        # if force_data['judgement_code'] != 'O':
        #     logging.warning(f"Reading is {judgment.lower()}.")
        # https://en.wikipedia.org/wiki/Three-point_flexural_test
        sigma_f = 8.0 * force * self.support_span / np.pi / (self.beam_diameter ** 3.0) * 1E6
        displacement_mm = self.adc_to_mm(adc_pot)

        dt = current_time - self.__time_start
        data = {
            "Time (s)": float(dt),
            "Force (N)": force,
            "Flexural Stress (Pa)": sigma_f,
            "Displacement (mm)": displacement_mm
        }
        log.info(f"Time: {dt:.3f}, z: {displacement_mm:.2f} mm, Force: {force:.3E}, Flexural Stress: {sigma_f:.3E}.")
        self.emit('results', data)
        self.emit('progress', dt * 100.0 / self.experiment_time)

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
            inputs=["sample_name", 'experiment_time', 'support_span', 'beam_diameter', "force_gauge_port",
                    'deflection_pot_port'],
            displays=["sample_name", 'experiment_time', 'support_span', 'beam_diameter', "force_gauge_port",
                      'deflection_pot_port'],
            x_axis="Time (s)",
            y_axis="Flexural Stress (Pa)",
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
