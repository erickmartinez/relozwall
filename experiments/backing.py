import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
import sys, os

sys.modules['cloudpickle'] = None
import threading
import time
from instruments.bme680 import BME
from pymeasure.display.Qt import QtGui
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import IntegerParameter, FloatParameter
from pymeasure.experiment import unique_filename
from instruments.mx200 import MX200
import sched
import datetime
from instruments.inhibitor import WindowsInhibitor


class BMEProcedure(Procedure):
    experiment_time = FloatParameter('Experiment Time', units='h', default=1)
    interval = FloatParameter('Sampling Interval', units='s', default=1)
    __bme: BME = None
    __mx200: MX200 = None
    __scheduler: sched.scheduler = None
    __time_start: datetime.datetime = None
    __ndata_points: int = 0
    __thread: threading.Thread = None
    __on_sleep: WindowsInhibitor = None
    port = 'COM3'

    DATA_COLUMNS = ["Time (s)", "Temperature (C)", r"Relative Humidity (percent)", "Pressure (Bar)",
                    "Gas Resistance (Ohm)", "Pressure CH1 (Torr)"]

    def startup(self):
        log.info("Creating BME680.")
        self.__bme = BME(uri='http://128.54.52.159', username='qwerty', password='12345')
        log.info("Setting up Televac MX200")
        self.__mx200 = MX200(address=self.port)
        self.__mx200_delay = self.__mx200.delay

    def execute(self):
        self.__ndata_points = int(self.experiment_time * 3600 / self.interval)
        self.__scheduler = sched.scheduler(timefunc=time.time, delayfunc=time.sleep)
        self.__time_start = datetime.datetime.now()
        log.info("Starting the loop of {0:d} datapoints.".format(self.__ndata_points))
        log.info("Date time at start of measurement: {dt}.".format(dt=self.__time_start))
        delay = 0
        events = []
        n = 1
        while delay <= self.experiment_time * 3600:
            # event_id = self.__scheduler.enter(delay=delay, priority=1, action=self.get_bme_data, argument=(n,))
            bme_event_id = self.__scheduler.enterabs(
                time=self.__time_start.timestamp() + delay, priority=1,
                action=self.acquire_data, argument=(n,)
            )
            delay += self.interval
            events.append(bme_event_id)
            n += 1

        self.__thread = threading.Thread(target=self.__scheduler.run)
        self.inhibit_sleep()
        self.__thread.start()
        self.__thread.join()

    def shutdown(self):
        self.unhinibit_sleep()

    def __del__(self):
        if self.__scheduler is not None:
            for e in self.__scheduler.queue:
                try:
                    self.__scheduler.cancel(e)
                except ValueError:
                    pass
        self.unhinibit_sleep()

    def acquire_data(self, n):
        if self.should_stop():
            log.warning("Caught the stop flag in the procedure")
            for e in self.__scheduler.queue:
                try:
                    self.__scheduler.cancel(e)
                except ValueError:
                    pass
            self.__scheduler = sched.scheduler(timefunc=time.time, delayfunc=time.sleep)
            self.unhinibit_sleep()
            return

        bme_data = self.__bme.read_env()
        pressure = self.__mx200.pressure(1)  / 1000.
        dt = (datetime.datetime.now() - self.__time_start).total_seconds()
        data = {
            "Time (s)": dt,
            "Pressure CH1 (Torr)": pressure
        }

        for row in bme_data:
            if row['type'] == 'temperature':
                data['Temperature (C)'] = row['value']
            elif row['type'] == 'humidity':
                data['Relative Humidity (percent)'] = row['value']
            elif row['type'] == 'pressure':
                data['Pressure (Bar)'] = row['value'] / 1000
            elif row['type'] == 'gas_resistance':
                data['Gas Resistance (Ohm)'] = row['value'] * 1000

        self.emit('results', data)
        self.emit('progress', n * 100. / self.__ndata_points)
        log.debug("Emitting results: {0}".format(data))

    def inhibit_sleep(self):
        if os.name == 'nt':
            self.__on_sleep = WindowsInhibitor()
            self.__on_sleep.inhibit()

    def unhinibit_sleep(self):
        if os.name == 'nt'and self.__on_sleep:
            self.__on_sleep.unhinibit()


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=BMEProcedure,
            inputs=['experiment_time', 'interval'],
            displays=['experiment_time', 'interval'],
            x_axis="Time (s)",
            y_axis="Pressure CH1 (Torr)",
            directory_input=True,
        )
        self.setWindowTitle('Backing data')

    def queue(self):
        directory = self.directory
        filename = unique_filename(directory, prefix='Environment')

        procedure = self.make_procedure()
        results = Results(procedure, filename)
        experiment = self.new_experiment(results)

        self.manager.queue(experiment)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
