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
from pymeasure.experiment import Parameter, FloatParameter
from pymeasure.experiment import unique_filename
from instruments.mx200 import MX200
import sched
import datetime
from instruments.inhibitor import WindowsInhibitor

MX200_COM = 'COM3'


class BakingProcedure(Procedure):
    experiment_time = FloatParameter('Experiment Time', units='h', default=1)
    interval = FloatParameter('Sampling Interval', units='s', default=1)
    sample_name = Parameter("Sample Name", default="UNKNOWN")
    __mx200: MX200 = None
    __scheduler: sched.scheduler = None
    __time_start: datetime.datetime = None
    __ndata_points: int = 0
    __thread: threading.Thread = None
    __on_sleep: WindowsInhibitor = None
    __mx200_delay: float = 0.05
    __keep_alive: bool = False
    __failed_readings = 0
    __max_attempts = 10
    __previous_reading: dict = None
    __previous_pressure: float = None

    DATA_COLUMNS = ["Time (h)", "Pressure CH1 (Torr)"]

    def startup(self):
        log.info("Setting up Televac MX200")
        self.__mx200 = MX200(address=MX200_COM, keep_alive=True)
        # time.sleep(1.0)
        # self.__mx200_delay = self.__mx200.delay
        time.sleep(1.0)
        self.__mx200.units = 'MT'
        time.sleep(1.0)

    def execute(self):
        self.__ndata_points = int(self.experiment_time * 3600 / self.interval)
        self.__scheduler = sched.scheduler(timefunc=time.time, delayfunc=time.sleep)
        self.__time_start = datetime.datetime.now()
        # self.__mx200.units = 'MT'
        # Reset the counter for failed readings
        self.__failed_readings = 0
        log.info("Starting the loop of {0:d} datapoints.".format(self.__ndata_points))
        log.info("Date time at start of measurement: {dt}.".format(dt=self.__time_start))
        p_previous = self.__mx200.pressure(1)
        delay = 0
        events = []
        n = 1
        self.inhibit_sleep()
        previous_time = 0.0
        total_time = 0.0
        start_time = time.time()
        experiment_time_s = self.experiment_time * 3600.0

        while total_time <= experiment_time_s:
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break
            current_time = time.time()
            if (current_time - previous_time) >= self.interval:
                total_time = current_time - start_time
                pressure = self.__mx200.pressure(1)
                if type(pressure) == str:
                    log.warning("Could not read pressure at time: {0}. Message: {1}".format(
                        total_time, pressure
                    ))
                    pressure = p_previous

                data = {
                    "Time (h)": total_time / 3600.,
                    "Pressure CH1 (Torr)": pressure
                }
                self.emit('results', data)
                self.emit('progress', 100.0 * total_time / experiment_time_s)

                p_previous = pressure
                previous_time = current_time

        # while delay <= self.experiment_time * 3600:
        #     # event_id = self.__scheduler.enter(delay=delay, priority=1, action=self.get_bme_data, argument=(n,))
        #     bme_event_id = self.__scheduler.enterabs(
        #         time=self.__time_start.timestamp() + delay, priority=1,
        #         action=self.acquire_data, argument=(n,)
        #     )
        #     delay += self.interval
        #     events.append(bme_event_id)
        #     n += 1
        #
        # self.__thread = threading.Thread(target=self.__scheduler.run)
        # self.__thread.start()
        # self.__thread.join()

    def shutdown(self):
        self.kill_queue()
        self.unhinibit_sleep()
        if self.__mx200 is not None:
            self.__mx200.close()
        # Remove file handlers from logger
        if len(log.handlers) > 0:
            for h in log.handlers:
                if isinstance(h, logging.FileHandler):
                    log.removeHandler(h)
                if isinstance(h, logging.NullHandler):
                    log.removeHandler(h)
                    log.addHandler(logging.NullHandler())

    def kill_queue(self):
        self.unhinibit_sleep()
        if self.__scheduler is not None:
            for e in self.__scheduler.queue:
                try:
                    self.__scheduler.cancel(e)
                except ValueError:
                    pass

    def __del__(self):
        self.kill_queue()

    def acquire_data(self, n):
        if self.should_stop():
            log.warning("Caught the stop flag in the procedure")
            self.kill_queue()

        pressure = self.__mx200.pressure(1)
        # If the pressure gives a bad reading (e.g. OVERLOAD) try again
        if type(pressure) == str:
            if pressure == 'OVER':
                pressure = np.NaN
            else:
                if self.__failed_readings < self.__max_attempts:
                    self.__failed_readings += 1
                    log.warning("Could not read pressure at time: {0}. Message: {1}".format(
                        datetime.datetime.now().isoformat(), pressure
                    ))
                    time.sleep(0.1)
                    self.acquire_data(n)
                else:
                    log.warning('Error reading pressure. Read out: {0}'.format(pressure))
                    pressure = self.__previous_pressure if self.__previous_pressure is not None else np.NaN

        dt = (datetime.datetime.now() - self.__time_start).total_seconds()
        data = {
            "Time (h)": dt / 3600.,
            "Pressure CH1 (Torr)": pressure
        }

        self.__previous_pressure = pressure

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


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=BakingProcedure,
            inputs=['experiment_time', 'sample_name', 'interval'],
            displays=['experiment_time', 'sample_name', 'interval'],
            x_axis="Time (h)",
            y_axis="Pressure CH1 (Torr)",
            directory_input=True,
        )
        self.setWindowTitle('Backing data')

    def queue(self):
        directory = self.directory
        procedure: BakingProcedure = self.make_procedure()
        sample_name = procedure.sample_name
        prefix = f'BAKING_{sample_name}_'
        filename = unique_filename(directory, prefix=prefix)
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
