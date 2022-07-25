import logging
import sys, os

import numpy as np

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
from pymeasure.display.Qt import QtGui
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import IntegerParameter, FloatParameter, Parameter, ListParameter, BooleanParameter
from pymeasure.experiment import unique_filename
from instruments.mx200 import MX200
from instruments.linear_translator import ISC08
import datetime
from instruments.inhibitor import WindowsInhibitor
from instruments.esp32 import ExtruderReadout
from instruments.ametek import DCSource
from simple_pid import PID
from serial import SerialException

EXT_READOUT_IP = '192.168.4.2'
DC_SOURCE_IP = '192.168.1.3'
MX200_COM = 'COM3'
ISC08_COM = 'COM4'
NUMBER_OF_SAMPLES = 200


class ExtrusionProcedure(Procedure):
    sample_name = Parameter("Sample name", default="BASELINE")
    speed_setting = ListParameter('Extruder speed', units='cm/s', default=0.57, choices=[0.11, 0.57, 1.1])
    start_position_in = FloatParameter('Start position', units='in', default=16.0, minimum=9.0, maximum=28.0)
    displacement_in = FloatParameter('Displacement', units='in', default=6.0, minimum=1.0, maximum=20.0)
    temperature_setpoint = FloatParameter('Setpoint temperature', units='°C', default=25, minimum=25, maximum=915.0)
    ramping_rate = FloatParameter('Ramping rate', units='°C/min', default=25.0, minimum=5.0, maximum=100.0)
    pid_stabilizing_time = FloatParameter('Temperature stabilization time', units='s', minimum=60, maximum=600,
                                          default=60)
    ku = FloatParameter('Ku', minimum=1., maximum=10000., default=5000.0)
    tu = FloatParameter('Tu', minimum=1., maximum=10000., default=15.5)
    is_baseline = BooleanParameter('Is baseline?', default=False)
    load_cell_range = FloatParameter('Load cell range', units='kg', default=30, minimum=10.0, maximum=50.0)
    load_cell_prediction_error_pct = FloatParameter('Load cell error %', units='%', default=9.8, minimum=0.001,
                                                    maximum=100.0)
    __allowable_force_threshold = 0.9
    # __mx200: MX200 = None
    __translator: ISC08 = None
    __on_sleep: WindowsInhibitor = None
    __readout: ExtruderReadout = None
    __max_attempts = 10
    __previous_reading: dict = None
    __previous_pressure: float = None
    __dc_source: DCSource = None
    __keep_alive: bool = False

    __pot_a0: float = 8.45
    __pot_a1: float = 0.0331
    __speed_setting_map = {0.11: 20, 0.57: 55, 1.1: 65}

    DATA_COLUMNS = ["Time (s)", "Position (cm)", 'Position (in)', "Force (N)", "Force error (N)",
                    "Temperature (C)"]

    def startup(self):
        log.info('Starting Experiment')
        # log.info("Setting up Televac MX200")
        # self.__mx200 = MX200(address=MX200_COM)
        # time.sleep(1.0)
        # self.__mx200.units = 'MT'
        # time.sleep(3.0)
        # log.info(f"Initial pressures: {self.__mx200.pressures}")

    def execute(self):
        log.info("Setting up linear translator")
        self.__translator = ISC08(address=ISC08_COM)
        time.sleep(1.0)
        log.info("Setting up extruder readout")
        self.__readout = ExtruderReadout(ip_address=EXT_READOUT_IP)
        time.sleep(1.0)
        log.info("Setting up DC source")
        self.__dc_source = DCSource(ip_address=DC_SOURCE_IP)

        [TC1, _, f, _, pot_adc] = self.readout.reading
        d0 = self.adc_to_cm(pot_adc)
        log.info(f"TC1: {TC1:6.2f} °C, F: {f:4.1f} N, x0: {d0:.1f} cm")
        initial_displacement = self.start_position_in * 2.54 - d0
        initial_moving_time = abs(initial_displacement / 0.57)
        if abs(initial_displacement) > 0.5:
            log.info(
                f'The current position ({d0 / 2.54} in) does not match the inital position ({self.start_position_in} in).')
            log.info('Translating the sample to the starting point...')
            self.__translator.move_by_cm(distance=initial_displacement, speed=0.57)
            time.sleep(initial_moving_time)

        [TC1, _, f, _, pot_adc] = self.readout.reading

        log.info("Setting the PID controller")
        kp = 0.2 * self.ku
        ki = 0.4 * self.ku / self.tu
        kd = 2.0 * self.ku * self.tu / 30.0

        pid = PID(kp, ki, kd, setpoint=TC1)
        pid.output_limits = (0, 200.0)
        force_threshold = self.__allowable_force_threshold * self.load_cell_range * 9.82

        pid.set_auto_mode(True, last_output=TC1)
        d0 = self.adc_to_cm(pot_adc)
        log.info("Taring the load cell...")
        self.readout.zero()
        time.sleep(2.0)

        log.info('Setting up DC voltage')
        # self.__dc_source.cls()
        # self.__dc_source.rst()
        self.__dc_source.current_setpoint = 8.0
        self.__dc_source.voltage_setpoint = 0.0
        self.__dc_source.output_on()

        adc_speed_setting = self.__speed_setting_map[self.speed_setting]
        log.info(f"Speed setting: {adc_speed_setting}")
        max_distance = self.displacement_in * 2.54  # cm
        moving_time = max_distance / self.speed_setting

        dt = max(moving_time / NUMBER_OF_SAMPLES, 0.4)
        log.info(f'dt = {dt:.3f} s')

        current_time = 0.0
        previous_time = 0.0
        previous_time_reading = 0.0
        total_time = 0.0
        current_ramping_time = 0.0
        initial_temperature = TC1
        ramping_time = 60.0 * (self.temperature_setpoint - initial_temperature) / self.ramping_rate
        log.info(f'Ramping time: {ramping_time:.3f} s')
        ramping_t0 = time.time()
        t0 = current_time
        moving = False

        run_pid = True
        ramping = True
        stabilizing = False
        displacement = 0
        LINE_CLEAR = '\x1b[2K'  # <-- ANSI sequence

        while run_pid:
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break
            current_time = time.time()
            if (current_time - previous_time_reading) >= 0.1:
                [TC1, _, fi, _, pot_adc] = self.readout.reading
                previous_time_reading = current_time
                control = pid(TC1)
                self.__dc_source.voltage_setpoint = control

            if ramping and current_ramping_time <= ramping_time + 0.1:
                temperature_setpoint = initial_temperature + self.ramping_rate * current_ramping_time / 60.0
                if temperature_setpoint > self.temperature_setpoint:
                    temperature_setpoint = self.temperature_setpoint
                    ramping = False
                print(
                    f"T = {TC1:>6.2f} °C, (Setpoint: {temperature_setpoint:>6.1f} °C), Ramping Time: {current_ramping_time:>5.2f} s",
                    end='\r', flush=True)
                pid.setpoint = temperature_setpoint
                current_ramping_time = current_time - ramping_t0
                # time.sleep(0.01)
            if current_ramping_time >= ramping_time and ramping:
                ramping = False
                stabilizing = True
                current_ramping_time = 0
                ramping_t0 = current_time
                print("")
                # time.sleep(0.01)
            if stabilizing and current_ramping_time <= self.pid_stabilizing_time:
                print(f"T = {TC1:>6.2f} °C, Stabilizing Time: {current_ramping_time:>5.2f} s", end='\r', flush=True)
                current_ramping_time = current_time - ramping_t0
                if current_ramping_time > self.pid_stabilizing_time:
                    stabilizing = False
                    t0 = current_time
                    total_time = 0
                # time.sleep(0.01)

            if (not ramping) and (not stabilizing) and not moving:
                print("")
                self.__translator.move_by_time(moving_time=moving_time, speed_setting=adc_speed_setting)
                print("")
                moving = True

            if (not ramping) and (not stabilizing) and displacement <= self.displacement_in:
                if (current_time - previous_time) >= dt:
                    d = self.adc_to_cm(pot_adc)
                    displacement = (d - d0) / 2.54
                    total_time = current_time - t0
                    fi_err = fi * self.load_cell_prediction_error_pct * 1E-2
                    if fi >= force_threshold:
                        self.translator.stop()
                        msg = f'The force on the sample ({fi} N) is larger than the allowable limit: ' \
                              f'{force_threshold} (N) '
                        print(msg)
                        raise ValueError(msg)
                        break
                    print(f"{total_time:8.3f} s, {d:>5.1f} cm, {fi:>5.1f} ± {fi_err:>5.1f}N, {TC1:>5.2f} °C",
                          end='\r', flush=True)
                    """
                    DATA_COLUMNS = ["Time (s)", "Position (cm)",'Position (in)', "Force (N)", "Force error (N)",
                    "Temperature (C)"]
                    """
                    d = {
                        "Time (s)": total_time,
                        "Position (cm)": d,
                        'Position (in)': d / 2.54,
                        "Force (N)": fi,
                        "Force error (N)": fi_err,
                        "Temperature (C)": TC1,
                    }
                    self.emit('results', d)
                    self.emit('progress', 100.0 * displacement / self.displacement_in)
                    control = pid(TC1)
                    self.__dc_source.voltage_setpoint = control
                    previous_time = current_time

            if displacement >= self.displacement_in or total_time > moving_time:
                run_pid = False

        self.__dc_source.voltage_setpoint = 0.0
        self.__dc_source.output_off()
        print("")
        avg_speed = displacement * 2.54 / moving_time
        log.info(f'Average speed: {avg_speed:.2f} cm/s')
        self.__translator.disconnect()
        self.__dc_source.disconnect()
        self.__readout.disconnect()

    @property
    def readout(self) -> ExtruderReadout:
        return self.__readout

    @property
    def translator(self) -> ISC08:
        return self.__translator

    def get_pot_adc_avg(self, averages: int = 10):
        points = int(averages)
        x = np.empty(points)
        for j in range(points):
            time.sleep(0.01)
            [_, _, _, _, x[j]] = self.readout.reading
        return x.mean(), x.std()

    def adc_to_cm(self, x: np.ndarray):
        return self.__pot_a0 + self.__pot_a1 * x

    @property
    def current_position_cm(self, averages: int = 10):
        x = np.array(self.get_pot_adc_avg(averages))
        return self.adc_to_cm(x)

    def shutdown(self):
        self.unhinibit_sleep()
        # Remove file handlers from logger
        if len(log.handlers) > 0:
            for handler in log.handlers:
                if isinstance(handler, logging.FileHandler):
                    log.removeHandler(handler)

    def __del__(self):
        self.shutdown()

    def inhibit_sleep(self):
        if os.name == 'nt' and not self.__keep_alive:
            self.__on_sleep = WindowsInhibitor()
            self.__on_sleep.inhibit()

    def unhinibit_sleep(self):
        if os.name == 'nt' and self.__keep_alive:
            self.__on_sleep.uninhibit()
            self.__keep_alive = False


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=ExtrusionProcedure,
            inputs=['sample_name', 'is_baseline', 'speed_setting', 'start_position_in', 'displacement_in',
                    'temperature_setpoint', 'ramping_rate', 'pid_stabilizing_time', 'ku', 'tu', 'load_cell_range',
                    'load_cell_prediction_error_pct'],
            displays=['sample_name', 'is_baseline', 'speed_setting', 'start_position_in', 'displacement_in',
                      'temperature_setpoint', 'ramping_rate', 'pid_stabilizing_time', 'ku', 'tu', 'load_cell_range',
                      'load_cell_prediction_error_pct'],
            x_axis="Time (s)",
            y_axis="Force (N)",
            directory_input=True,
        )
        self.setWindowTitle('Friction data')

    def queue(self):
        directory = self.directory

        procedure: ExtrusionProcedure = self.make_procedure()
        sample_name = procedure.sample_name
        temperature_setpoint = procedure.temperature_setpoint
        speed_setting = procedure.speed_setting

        prefix = f'FRICTION_{sample_name}_{temperature_setpoint:03.0f}C_{speed_setting}CMPS_'
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
