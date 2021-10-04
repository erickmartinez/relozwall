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
from pymeasure.experiment import IntegerParameter, FloatParameter
from pymeasure.experiment import unique_filename
from instruments.imada import DST44A
import sched
import datetime
from instruments.inhibitor import WindowsInhibitor


class FlexuralStressProcedure(Procedure):
    experiment_time = FloatParameter('Experiment Time', units='s', default=30)
    interval = FloatParameter('Sampling Interval', units='s', default=0.2)
    support_span = FloatParameter('Support Span', units='mm', default=40)
    beam_diameter = FloatParameter('Beam Diameter', units='mm', default=10)
    __force_gauge: DST44A = None
    __scheduler: sched.scheduler = None
    __time_start: datetime.datetime = None
    __ndata_points: int = 0
    __thread: threading.Thread = None
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
        self.__scheduler = sched.scheduler(timefunc=time.time, delayfunc=time.sleep)
        self.__time_start = datetime.datetime.now()

        self.__failed_readings = 0
        log.info("Starting the loop of {0:d} datapoints.".format(self.__ndata_points))
        log.info("Date time at start of measurement: {dt}.".format(dt=self.__time_start))
        delay = 0
        events = []
        n = 1
        while delay <= self.experiment_time:
            event_id = self.__scheduler.enterabs(
                time=self.__time_start.timestamp() + delay, priority=1,
                action=self.acquire_data, argument=(n,)
            )
            delay += self.interval
            events.append(event_id)
            n += 1

        self.__thread = threading.Thread(target=self.__scheduler.run)
        self.inhibit_sleep()
        self.__thread.start()
        self.__thread.join()

    def shutdown(self):
        self.kill_queue()
        self.unhinibit_sleep()

    def kill_queue(self):
        self.unhinibit_sleep()
        if self.__scheduler is not None:
            for e in self.__scheduler.queue:
                try:
                    self.__scheduler.cancel(e)
                except ValueError:
                    pass

    def acquire_data(self, n):
        if self.should_stop():
            log.warning("Caught the stop flag in the procedure")
            self.kill_queue()

        force_data = self.__force_gauge.read()
        force = force_data['reading']
        units = force_data['units']
        mode = force_data['mode']
        judgment = force_data['judgement']
        if force_data['judgement_code'] != 'O':
            logging.warning(f"Reading is {judgment.lower()}.")
        # https://en.wikipedia.org/wiki/Three-point_flexural_test
        sigma_f = 8.0 * force * self.support_span / np.pi / (self.beam_diameter ** 3.0) * 1E6

        dt = (datetime.datetime.now() - self.__time_start).total_seconds()
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
            self.__on_sleep.unhinibit()
            self.__keep_alive = False

    def __del__(self):
        self.kill_queue()


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=FlexuralStressProcedure,
            inputs=['experiment_time', 'interval', 'support_span', 'beam_diameter'],
            displays=['experiment_time', 'interval', 'support_span', 'beam_diameter'],
            x_axis="Time (s)",
            y_axis="Flexural Stress (Pa)",
            directory_input=True,
        )
        self.setWindowTitle('3-Point Bend Test')

    def queue(self):
        directory = self.directory
        filename = unique_filename(directory, prefix='3PBT_')
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