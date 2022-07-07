import logging
import sys, os

import numpy as np

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
from pymeasure.display.Qt import QtGui
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import IntegerParameter, FloatParameter, Parameter, BooleanParameter
from pymeasure.experiment import unique_filename
from instruments.mx200 import MX200
import datetime
from instruments.inhibitor import WindowsInhibitor
from instruments.esp32 import ExtruderReadout
from instruments.ametek import DCSource
from simple_pid import PID
import instruments.linear_translator as lnt
from serial import SerialException

# EXT_READOUT_COM = 'COM12'
EXT_READOUT_IP = '192.168.4.2'
DC_SOURCE_IP = '192.168.1.3'
MX200_COM = 'COM3'
ISC08_COM = 'COM4'
NUMBER_OF_SAMPLES = 10000


class ExtrusionProcedure(Procedure):
    bake_time = FloatParameter('Bake Time', units='min', default=1, minimum=1.0, maximum=60.0)
    temperature_setpoint = FloatParameter('Temperature setpoint', units='C', default=20.0, minimum=0.0, maximum=800.0)
    temperature_ramp_rate = FloatParameter('Ramping rate', units='C/min', default=25, minimum=10.0, maximum=100.0)
    temperature_stabilization_time = FloatParameter('Stabilization Time', units='min', default=10.0, minimum=1.0,
                                                    maximum=60.0)
    baking_position = FloatParameter('Baking Position', units='in', default=18.0, minimum=16, maximum=22.5)
    sample_length = FloatParameter('Sample Length', units='cm', default=5.0, minimum=1.0, maximum=15.0)
    sample_name = Parameter("Sample Name", default="UNKNOWN")
    __mx200: MX200 = None
    __time_start: datetime.datetime = None
    __ndata_points: int = 0
    __on_sleep: WindowsInhibitor = None
    __mx200_delay: float = 0.05
    __keep_alive: bool = False
    __max_attempts = 10
    __previous_pressure: dict = None
    __dc_source: DCSource = None
    __translator: lnt.ISC08 = None

    __pot_a0: float = 8.45
    __pot_a1: float = 0.0331
    __pid_ku = 800.0
    __pid_tu = 65.0

    __translator_velocity = 0.57
    __velocity_adc_setpoint = 55

    DATA_COLUMNS = ["Time (s)", "Baking Pressure (Torr)", "Outgassing Pressure (Torr)", "Baking Temperature (C)",
                    "Outgassing Temperature (C)", "DC Voltage (V)", "Is Baking?"]



    def startup(self):
        log.info("Setting up Televac MX200")
        self.__mx200 = MX200(address=MX200_COM)
        time.sleep(1.0)
        self.__mx200.units = 'MT'
        time.sleep(1.0)
        log.info("Setting up power supply")
        self.__dc_source = DCSource(ip_address=DC_SOURCE_IP)
        self.__previous_pressure = self.__mx200.pressures
        time.sleep(0.5)
        log.info("Setting up linear translator")
        self.__translator = lnt.ISC08(address=ISC08_COM)
        time.sleep(1.0)
        log.info(f"Initial pressures = {self.__previous_pressure} Torr")


    def execute(self):
        # log.info('Ramping up the voltage')
        # self.__dc_source.setup_ramp_voltage(output_voltage=self.temperature_setpoint, time_s=self.temperature_ramp_rate)
        # self.__dc_source.run_voltage_ramp()
        # Reset the counter for failed readings

        # extruder_readout = ExtruderReadout(address=EXT_READOUT_COM)
        log.info("Initializing Extruder Readout...")
        extruder_readout = ExtruderReadout(ip_address=EXT_READOUT_IP)
        self.inhibit_sleep()
        log.info("Taring the load cell...")
        extruder_readout.zero()
        log.info("Getting the initial position and initial temperature")
        [initial_temperature, _, _, _, pot_adc] = extruder_readout.reading
        x0_cm = self.adc_to_cm(pot_adc)
        log.info(f"Initial position: {x0_cm:.1f} cm")
        log.info(f"Initial temperature: {initial_temperature:.2f} °C")
        maximum_expected_position_in = self.baking_position + self.sample_length / 2.54
        if x0_cm > maximum_expected_position_in * 2.54:
            raise ValueError(
                f'The sample is at position ({x0_cm / 2.54:.1f} in) greater than the expected position ({maximum_expected_position_in:.1f} in')

        delta_x_cm = self.baking_position * 2.54 - x0_cm
        moving_time = delta_x_cm / self.__translator_velocity
        # self.__translator.move_by_time(moving_time=moving_time, speed_setting=speed_setting)

        temperature_delta = self.temperature_setpoint - initial_temperature
        ramping_time = np.ceil(60.0 * temperature_delta / self.temperature_ramp_rate)
        experiment_time = ramping_time*1.5 + self.temperature_stabilization_time * 60 + self.bake_time * 60 + moving_time

        dt = max(1.0, experiment_time / NUMBER_OF_SAMPLES)

        log.info(f"Ramping rate: {self.temperature_ramp_rate:.1f} °C/min")
        log.info(f"Ramping time: {ramping_time / 60.0:.1f} min")


        log.info("Setting up PID controller")
        kp = 0.2 * self.__pid_ku
        ki = 0.4 * self.__pid_ku / self.__pid_tu
        kd = 2.0 * self.__pid_ku * self.__pid_tu / 30.0
        pid = PID(kp, ki, kd, setpoint=initial_temperature)
        pid.output_limits = (0, 200.0)

        log.info('Setting up DC voltage')
        # self.__dc_source.cls()
        # self.__dc_source.rst()
        self.__dc_source.current_setpoint = 8.0
        self.__dc_source.voltage_setpoint = 0.0
        self.__dc_source.output_on()

        run_pid = True
        ramping = True
        stabilizing = False
        baking = False
        moving = False

        previous_time = 0.0
        total_time = 0.0
        start_time = time.time()
        current_ramping_time = 0
        current_stabilizing_time = 0
        moving_t0 = 0
        baking_t0 = 0
        temperature_setpoint = initial_temperature

        print("")
        while run_pid:
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break
            current_time = time.time()
            [TC1, TC2, force, load_cell_adc, pot_adc] = extruder_readout.reading
            pressures = self.__mx200.pressures
            for k, p in pressures.items():
                if type(p) is str:
                    pressures[k] = self.__previous_pressure[k]
            self.__previous_pressure = pressures
            control = pid(TC1)
            self.__dc_source.voltage_setpoint = control
            if ramping and temperature_setpoint < self.temperature_setpoint:
                temperature_setpoint = initial_temperature + self.temperature_ramp_rate * current_ramping_time / 60.0
                if temperature_setpoint >= self.temperature_setpoint:
                    temperature_setpoint = self.temperature_setpoint
                    ramping = False
                    stabilizing = True
                    stabilizing_time_t0 = time.time()
                pid.setpoint = temperature_setpoint
                current_ramping_time = current_time - start_time
                print(
                    f"T = {TC1:>6.2f} °C, (Setpoint: {temperature_setpoint:>6.1f} °C), Ramping Time: {current_ramping_time:>5.2f} s",
                    end='\r', flush=True)
            if ramping and current_ramping_time >= ramping_time:
                print("")
                time.sleep(0.01)
            if total_time >= ramping_time and not stabilizing:
                ramping = False
                stabilizing = True
                stabilizing_time_t0 = time.time()
            if stabilizing and current_stabilizing_time <= self.temperature_stabilization_time * 60.0:
                # Line Cleaning: https://stackoverflow.com/questions/5419389/how-to-overwrite-the-previous-print-to-stdout
                print(f"T = {TC1:>6.2f} °C, Stabilizing Time: {current_time - stabilizing_time_t0:>5.2f} s", end='\x1b[1K\r', flush=True)
                current_stabilizing_time = current_time - stabilizing_time_t0
            if stabilizing and current_stabilizing_time > self.temperature_stabilization_time * 60.0:
                stabilizing = False
                print("")

            if (not ramping) and (not stabilizing) and (not baking) and (not moving):
                self.__translator.move_by_time(moving_time=moving_time, speed_setting=self.__velocity_adc_setpoint)
                moving = True
                moving_t0 = current_time

            if (not ramping) and (not stabilizing) and (not baking) and moving and current_time - moving_t0 > moving_time and not baking:
                baking = True
                baking_t0 = current_time

            if baking and (not ramping) and (not stabilizing) and (not baking) and moving and current_time - baking_t0 > self.bake_time * 60.0:
                baking = False
                temperature_setpoint = initial_temperature

            if current_time - start_time > experiment_time:
                run_pid = False

            if (current_time - previous_time) >= dt:
                total_time = current_time - start_time
                d = {
                    "Time (s)": total_time,
                    "Baking Pressure (Torr)": pressures[1],
                    "Outgassing Pressure (Torr)": pressures[2],
                    "Baking Temperature (C)": TC1,
                    "Outgassing Temperature (C)": TC2,
                    "DC Voltage (V)": control,
                    "Is Baking?": float(baking)
                }
                self.emit('results', d)
                self.emit('progress', 100.0 * total_time / experiment_time)
                print(f"T = {TC1:>6.2f} °C, Total Time: {total_time:>5.2f} s", end='\x1b[1K\r',
                      flush=True)
                previous_time = current_time

        log.info("Turning off the ouput of the DC power supply.")
        self.__dc_source.voltage_setpoint = 0.0
        self.__dc_source.output_off()

    def adc_to_cm(self, x: np.ndarray):
        return self.__pot_a0 + self.__pot_a1 * x

    def shutdown(self):
        if self.__translator is not None:
            self.__translator.disconnect()
        if self.__mx200 is not None:
            self.__mx200.close()
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
            self.__on_sleep.uninhibit()
            self.__keep_alive = False


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=ExtrusionProcedure,
            inputs=['sample_name', 'bake_time', 'temperature_setpoint', 'temperature_ramp_rate', 'temperature_stabilization_time', 'baking_position', 'sample_length'],
            displays=['sample_name', 'bake_time', 'temperature_setpoint', 'temperature_ramp_rate', 'temperature_stabilization_time', 'baking_position', 'sample_length'],
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

        prefix = f'EXTRUSION_{sample_name}_{temperature_setpoint:03.0f}V'
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
