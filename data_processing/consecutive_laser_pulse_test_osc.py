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
from pymeasure.experiment import FloatParameter, Parameter, IntegerParameter, ListParameter
from pymeasure.experiment import unique_filename
from serial.serialutil import SerialException
from instruments.esp32 import ESP32Trigger
from instruments.esp32 import DualTCLoggerTCP
from instruments.mx200 import MX200
from instruments.IPG import YLR3000, LaserException
from instruments.inhibitor import WindowsInhibitor
from instruments.tektronix import TBS2000
from scipy.interpolate import interp1d

ESP32_COM = 'COM6'
MX200_COM = 'COM3'
IP_LASER = "192.168.3.230"
TBS2000_RESOURCE_NAME = 'USB0::0x0699::0x03C7::C010461::INSTR'

TRIGGER_CHANNEL = 2
THERMOMETRY_CHANNEL = 1


class LaserPulsesProcedure(Procedure):
    emission_time = FloatParameter('Pulse length', units='s', default=0.5, minimum=0.001, maximum=20.0)
    repetitions = IntegerParameter('Repetitions', default=10, minimum=1, maximum=20)
    pulse_delay = FloatParameter('Pulse delay', units='s', minimum=2, maximum=120, default=10)
    laser_power_setpoint = FloatParameter("Laser Power Setpoint", units="%", default=100, minimum=0.0, maximum=100.0)
    sample_name = Parameter("Sample Name", default="UNKNOWN")
    pd_gain = ListParameter('Photodiode Gain', choices=('0', '10', '20', '30', '40', '50', '60', '70'), units='dB',
                            default='40')

    __keep_alive: bool = False
    __on_sleep: WindowsInhibitor = None
    __tc_logger: DualTCLoggerTCP = None
    __oscilloscope: TBS2000 = None
    __mx200: MX200 = None
    __ylr: YLR3000 = None
    __previous_pressure = None

    DATA_COLUMNS = ["Measurement Time (s)", "Pressure (Torr)", "Photodiode Voltage (V)", "Trigger (V)"]

    def startup(self):
        print('***  Startup ****')
        self.__oscilloscope = TBS2000(resource_name=TBS2000_RESOURCE_NAME)
        self.__mx200 = MX200(address=MX200_COM)
        self.__mx200.delay = 0.030
        self.__mx200.units = 'MT'
        time.sleep(1.0)
        log.info("Setting up Lasers")
        self.__ylr = YLR3000(IP=IP_LASER)
        self.__previous_pressure = self.__mx200.pressure(2)
        log.info(self.__oscilloscope.sesr)
        log.info(self.__oscilloscope.all_events)

    def execute(self):
        log.info(f"Setting the laser to the current setpoint: {float(self.laser_power_setpoint):.2f} %")
        self.__ylr.current_setpoint = self.laser_power_setpoint
        time.sleep(0.1)
        log.info(f"Laser current setpoint: {float(self.__ylr.current_setpoint):.2f} %")

        pulse_interval = self.pulse_delay + self.emission_time + 0.5
        pulse_time = pulse_interval * self.repetitions  # seconds
        experiment_time = pulse_time + 120.0  # seconds

        log.info("Setting up Oscilloscope")
        self.__oscilloscope.write('ACQUIRE:STATE 0')
        self.__oscilloscope.write(f'CH{THERMOMETRY_CHANNEL}:VOLTS 1.0')
        self.__oscilloscope.write(f'CH{TRIGGER_CHANNEL}:VOLTS 1.0')
        self.__oscilloscope.set_acquisition_time(pulse_time)
        self.__oscilloscope.write('ACQUIRE:STOPAFTER SEQUENCE')
        self.__oscilloscope.write('ACQuire:MODe SAMple')

        try:
            self.__ylr.emission_on()
            emission_on = True
        except LaserException as e:
            log.warning(e)
            emission_on = False

        log.info("Setting up Triggers")
        try:
            esp32_trigger = ESP32Trigger(address=ESP32_COM)
        except SerialException as e:
            print("Error initializing ESP32 trigger")
            raise e

        esp32_trigger.pulse_duration = float(self.emission_time)
        time.sleep(1.0)
        et = esp32_trigger.pulse_duration
        log.info(f'Pulse duration: {et:.2f} s.')

        previous_time = 0.0
        previous_delay_time = 0.0
        total_time = 0.0
        pulse_counter = 0
        elapsed_time = []
        pressures = []

        log.info('Starting firing routine...')
        self.__oscilloscope.timeout = 1000
        self.__oscilloscope.write('ACQUIRE:STATE ON')
        start_time = time.time()
        while total_time <= experiment_time:
            current_time = time.time()
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break
            if (current_time - previous_delay_time) >= pulse_interval and pulse_counter < self.repetitions:
                esp32_trigger.fire()
                # time.sleep(0.05)
                print(f'Sent firing signal at {total_time:>7.3f} s')
                pulse_counter += 1
                previous_delay_time = current_time
            if (current_time - previous_time) >= 0.1:
                p = self.__mx200.pressure(2)
                if p == '':
                    p = self.__previous_pressure
                elapsed_time.append(total_time)
                pressures.append(p)
                total_time = current_time - start_time
                self.__previous_pressure = p
                # print(f'{total_time:>7.3f} s, {1000*p:>6.3f} mTorr')
                # time.sleep(0.005)
                previous_time = current_time

        elapsed_time = np.array(elapsed_time, dtype=float)
        pressures = np.array(pressures, dtype=float)

        if emission_on:
            try:
                self.__ylr.emission_off()
                emission_on = False
            except LaserException as e:
                log.warning(e)

        while True:
            busyA = self.__oscilloscope.ask("BUSY?")
            if busyA == '0':
                break
            if self.should_stop():
                break

        log.info('*** ACQUISITION OVER ***')
        log.info(self.__oscilloscope.sesr)
        log.info(self.__oscilloscope.all_events)
        data_pd = self.__oscilloscope.get_curve(channel=THERMOMETRY_CHANNEL)
        time.sleep(1.0)
        data_tr = self.__oscilloscope.get_curve(channel=TRIGGER_CHANNEL)
        time.sleep(1.0)

        columns_pd = data_pd.dtype.names
        columns_tr = data_tr.dtype.names

        voltage_pd = data_pd[data_pd[columns_pd[0]]]
        voltage_tr = data_tr[data_tr[columns_tr[0]]]
        time_osc = data_pd[columns_pd[0]]

        f = interp1d(elapsed_time, pressures)
        pressures_interp = f(time_osc)
        n_points = len(voltage_pd)

        for i in range(len(data_pd)):
            d = {
                "Measurement Time (s)": time_osc[i],
                "Pressure (Torr)": pressures_interp[i],
                "Photodiode Voltage (V)": voltage_pd[i],
                "Trigger (V)": voltage_tr[i]
            }
            self.emit('results', d)
            self.emit('progress', i * 100 / n_points)

    def inhibit_sleep(self):
        if os.name == 'nt' and not self.__keep_alive:
            self.__on_sleep = WindowsInhibitor()
            self.__on_sleep.inhibit()
            self.__keep_alive = True

    def unhinibit_sleep(self):
        if os.name == 'nt' and self.__keep_alive:
            self.__on_sleep.uninhibit()
            self.__keep_alive = False

    def shutdown(self):
        if self.__mx200 is not None:
            self.__mx200.close()
        if self.__ylr is not None:
            self.__ylr.disconnect()
        # Remove file handlers from logger
        if len(log.handlers) > 0:
            for handler in log.handlers:
                if isinstance(handler, logging.FileHandler):
                    log.removeHandler(handler)


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=LaserPulsesProcedure,
            inputs=['sample_name', "laser_power_setpoint", 'emission_time', "repetitions", "pulse_delay", 'pd_gain'],
            displays=['sample_name', "laser_power_setpoint", 'emission_time', "repetitions", "pulse_delay", 'pd_gain'],
            x_axis="Measurement Time (s)",
            y_axis="Pressure (Torr)",
            directory_input=True,
        )
        self.setWindowTitle('Laser Repetition Test')

    def queue(self):
        directory = self.directory

        procedure: LaserPulsesProcedure = self.make_procedure()
        sample_name = procedure.sample_name
        laser_setpoint = procedure.laser_power_setpoint
        repetitions = procedure.repetitions
        emission_time = procedure.emission_time

        prefix = f'{sample_name}_{laser_setpoint:03.0f}PCT_{repetitions}X_{emission_time}s_'
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
