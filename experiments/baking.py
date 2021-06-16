import logging
import sys, os

import requests.exceptions

sys.path.append('../')
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
    __mx200_delay: float = 0.001
    port = 'COM3'
    __keep_alive: bool = False
    __failed_readings = 0
    __max_attempts = 3

    DATA_COLUMNS = ["Time (h)", "Temperature (C)", r"Relative Humidity (percent)", "Pressure (Bar)",
                    "Gas Resistance (Ohm)", "Pressure CH1 (Torr)"]

    def startup(self):
        log.info("Creating BME680.")
        self.__bme = BME(uri='http://128.54.52.133', username='qwerty', password='12345')
        log.info("Setting up Televac MX200")
        self.__mx200 = MX200(address=self.port)
        self.__mx200_delay = self.__mx200.delay
        self.__mx200.units = 'mTorr'

    def execute(self):
        self.__ndata_points = int(self.experiment_time * 3600 / self.interval)
        self.__scheduler = sched.scheduler(timefunc=time.time, delayfunc=time.sleep)
        self.__time_start = datetime.datetime.now()
        self.__mx200.units = 'mTorr'
        # Reset the counter for failed readings
        self.__failed_readings = 0
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

    def __del__(self):
        self.kill_queue()

    def acquire_data(self, n):
        if self.should_stop():
            log.warning("Caught the stop flag in the procedure")
            self.kill_queue()

        try:
            bme_data = self.__bme.read_env()
        except requests.exceptions.ConnectionError:
            bme_data = [
                {"type": "temperature", "value": 0.0, "unit": "°C"},
                {"type": "humidity", "value": 0.0, "unit": "%"},
                {"type": "pressure", "value": 0.0, "unit": "mBar"},
                {"type": "gas_resistance", "value": 0.0, "unit": "kOhm"}
            ]
            log.warning('Could not access BME680.')
        pressure = self.__mx200.pressure(1)
        # If the pressure gives a bad reading (e.g. OVERLOAD) try again
        if type(pressure) == str:
            if self.__failed_readings < self.__failed_readings:
                self.__failed_readings += 1
                log.warning('Could not read pressure at time: {0}'.format(datetime.datetime.now().isoformat()))
                time.sleep(0.1)
                self.acquire_data(n)
            else:
                log.warning('Error reading pressure. Read out: {0}'.format(pressure))
        else:
            dt = (datetime.datetime.now() - self.__time_start).total_seconds()
            data = {
                "Time (h)": dt / 3600.,
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
            self.__failed_readings = 0

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
            procedure_class=BMEProcedure,
            inputs=['experiment_time', 'interval'],
            displays=['experiment_time', 'interval'],
            x_axis="Time (h)",
            y_axis="Pressure CH1 (Torr)",
            directory_input=True,
        )
        self.setWindowTitle('Backing data')

    def queue(self):
        directory = self.directory
        filename = unique_filename(directory, prefix='BAKING_')
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
