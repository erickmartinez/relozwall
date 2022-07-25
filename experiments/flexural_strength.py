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
from pymeasure.experiment import IntegerParameter, FloatParameter, Parameter
from pymeasure.experiment import unique_filename
from instruments.imada import DST44A
import sched
import datetime
from instruments.inhibitor import WindowsInhibitor

"""
Check which ports are avilable:

Open Terminal and type:

> python -m serial.tools.list_ports

before connecting probe and after connecting it, to see which port is it attached to.
"""


class FlexuralStressProcedure(Procedure):
    experiment_time = FloatParameter('Experiment Time', units='s', default=30)
    interval = FloatParameter('Sampling Interval', units='s', default=0.2)
    support_span = FloatParameter('Support Span', units='mm', default=40)
    beam_diameter = FloatParameter('Beam Diameter', units='mm', default=10)
    sample_name = Parameter("Sample Name", default="UNKNOWN")

    __force_gauge: DST44A = None
    # __scheduler: sched.scheduler = None
    __time_start: datetime.datetime = None
    __ndata_points: int = 0
    # __thread: threading.Thread = None
    __on_sleep: WindowsInhibitor = None
    __delay: float = 0.001
    port = 'COM5'
    __keep_alive: bool = False
    __failed_readings = 0
    __max_attempts = 10
    __previous_reading: dict = None

    DATA_COLUMNS = ["Time (s)", "Force (N)", "Flexural Stress (Pa)"]

    def startup(self):
        log.info("Setting up DST44A")
        self.__force_gauge = DST44A(address=self.port)
        self.__force_gauge.units('N')

    def execute(self):
        self.__ndata_points = int(self.experiment_time / self.interval)
        # self.__scheduler = sched.scheduler(timefunc=time.time, delayfunc=time.sleep)
        # self.__time_start = datetime.datetime.now()

        self.__failed_readings = 0
        log.info("Starting the loop of {0:d} datapoints.".format(self.__ndata_points))
        log.info("Date time at start of measurement: {dt}.".format(dt=self.__time_start))
        delay = 0
        events = []
        n = 1
        previous_time = 0.0
        total_time = 0.0

        start_time = time.time()
        self.__time_start = start_time
        while total_time <= self.experiment_time:
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break
            current_time = time.time()
            if (current_time - previous_time) >= self.interval:
                self.acquire_data(n, current_time)
                n += 1
                total_time = time.time() - start_time
                previous_time = current_time

        # # log.info("elapsed time:")
        # # log.info(elapsed_time)
        # elapsed_time = np.array(elapsed_time, dtype=float)
        # pressure = np.array(pressure, dtype=float)
        #
        # while delay <= self.experiment_time:
        #     event_id = self.__scheduler.enterabs(
        #         time=self.__time_start.timestamp() + delay, priority=1,
        #         action=self.acquire_data, argument=(n,)
        #     )
        #     delay += self.interval
        #     events.append(event_id)
        #     n += 1
        #
        # self.__thread = threading.Thread(target=self.__scheduler.run)
        self.inhibit_sleep()
        # self.__thread.start()
        # self.__thread.join()

    def shutdown(self):
        self.unhinibit_sleep()
        # Remove file handlers from logger
        if len(log.handlers) > 0:
            for handler in log.handlers:
                if isinstance(handler, logging.FileHandler):
                    log.removeHandler(handler)

    def acquire_data(self, n, current_time):
        if self.should_stop():
            log.warning("Caught the stop flag in the procedure")

        force_data = self.__force_gauge.read()
        force = force_data['reading']
        units = force_data['units']
        mode = force_data['mode']
        judgment = force_data['judgement']
        if force_data['judgement_code'] != 'O':
            logging.warning(f"Reading is {judgment.lower()}.")
        # https://en.wikipedia.org/wiki/Three-point_flexural_test
        sigma_f = 8.0 * force * self.support_span / np.pi / (self.beam_diameter ** 3.0) * 1E6

        # dt = (datetime.datetime.now() - self.__time_start).total_seconds()
        dt = current_time - self.__time_start
        data = {
            "Time (s)": float(dt),
            "Force (N)": force,
            "Flexural Stress (Pa)": sigma_f
        }
        log.info(f"Time: {dt:.3f}, Force: {force:.3E}, Flexural Stress: {sigma_f:.3E}.")
        self.__failed_readings = 0
        self.__previous_reading = data
        self.emit('results', data)
        self.emit('progress', n * 100. / self.__ndata_points)
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

    def __del__(self):
        pass


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=FlexuralStressProcedure,
            inputs=['experiment_time', 'interval', 'support_span', 'beam_diameter', "sample_name"],
            displays=['experiment_time', 'interval', 'support_span', 'beam_diameter', "sample_name"],
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
