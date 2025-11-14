import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
# from pymeasure.display.Qt import QtGui
from pymeasure.log import console_log, file_log
from pymeasure.display.Qt import QtWidgets
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import FloatParameter, ListParameter, Parameter, BooleanParameter
# from pymeasure.experiment import unique_filename
from pymeasure.experiment.results import replace_placeholders, unique_filename
from serial.serialutil import SerialException
from instruments.esp32 import ESP32Trigger
from instruments.esp32 import DualTCLoggerTCP
from instruments.mx200 import MX200
from instruments.IPG import YLR3000, LaserException
from instruments.tektronix import TBS2000
from instruments.inhibitor import WindowsInhibitor
from instruments.flir42 import Camera, TriggerType
import PySpin
from scipy import interpolate
import threading
from pymeasure.units import ureg

# from pint import UnitRegistry

ESP32_COM = 'COM6'
TC_LOGGER_IP = '192.168.4.3'
# MX200_COM = 'COM3'
TRIGGER_CHANNEL = 2
THERMOMETRY_CHANNEL = 1
TRIGGER_LEVEL = 3.3
SAMPLING_INTERVAL = 0.005
IP_LASER = "192.168.3.230"
TBS2000_RESOURCE_NAME = 'USB0::0x0699::0x03C7::C010461::INSTR'

# ureg = UnitRegistry()
ureg.load_definitions('./pint_units.txt')


def trigger_camera(cam: Camera):
    try:
        cam.acquire_images()
    except PySpin.SpinnakerException as ex:
        log.error(ex)
        cam.shutdown()
        raise ex
    return


def get_duty_cycle_params(duty_cycle: float, period_ms: float = 1.0) -> tuple:
    frequency = 1.0E3 / period_ms
    pulse_width = duty_cycle * period_ms
    return frequency, pulse_width


class LaserProcedure(Procedure):
    emission_time = FloatParameter('Emission Time', units='s', default=0.5, minimum=0.001, maximum=20.0)
    measurement_time = FloatParameter('Measurement Time', units='s', default=10.0, minimum=1.0, maximum=3600.0)
    laser_power_setpoint = FloatParameter("Laser Power Setpoint", units="%", default=100, minimum=0.0, maximum=100.0)
    camera_exposure_time = FloatParameter("Camera exposure time", minimum=1, units='us', default=4)
    camera_acquisition_time = FloatParameter("Camera acquisition time", units="s", minimum=0.01, maximum=600,
                                             default=1.0)
    camera_frame_rate = FloatParameter("Frame rate", units="Hz", minimum=1, maximum=200, default=200)
    camera_gain = FloatParameter("Gain", units="dB", minimum=0, maximum=47.994, default=5)
    acquisition_mode = ListParameter("Acquisition mode", choices=('Continuous', 'Single', 'Multi frame'),
                                     default='Multi frame')
    # trigger_delay = FloatParameter("Trigger delay", units="us", minimum=9, maximum=10E6)
    sample_name = Parameter("Sample Name", default="UNKNOWN")
    __camera: Camera = None
    # __oscilloscope: TBS2000 = None
    __keep_alive: bool = False
    __on_sleep: WindowsInhibitor = None
    __tc_logger: DualTCLoggerTCP = None
    __mx200: MX200 = None
    __ylr: YLR3000 = None
    pressure_data: pd.DataFrame = None
    __tc_data: pd.DataFrame = None
    __unique_filename: str = None
    __old_ylr_pulse_width: float = 0.05
    __old_ylr_frequency: float = 1.0
    __directory: str = None

    DATA_COLUMNS = ['Measurement Time (s)', 'Pressure (Torr)', 'TC1 (C)', 'TC2 (C)', 'Trigger (V)',
                    'Laser output power (W)', 'Laser output peak power (W)']

    def startup(self):
        log.info('***  Startup ****')
        self.__camera: Camera = Camera()
        self.__mx200 = MX200()  # address=MX200_COM, keep_alive=True)
        log.info("Setting up Lasers")
        self.__ylr = YLR3000(IP=IP_LASER)

    @property
    def unique_filename(self) -> str:
        return self.__unique_filename

    @unique_filename.setter
    def unique_filename(self, val: str):
        print(f"Storing filepath: {val}")
        self.__unique_filename = val

    @property
    def directory(self) -> str:
        return self.__directory

    @directory.setter
    def directory(self, value: str):
        self.__directory = value

    def save_pressure(self):
        if self.pressure_data is not None:
            filename = f'{os.path.splitext(self.__unique_filename)[0]}_pressure.csv'
            self.pressure_data.to_csv(filename, index=False)

    def get_image_path(self, acquisition_mode):
        if acquisition_mode == 'Single':
            path_to_images = self.directory
        else:
            base_name = os.path.splitext(self.__unique_filename)[0]
            path_to_images = os.path.join(self.directory, base_name + '_images')
            log.info(f'Image captures will be stored at:\n{path_to_images}')
        if not os.path.exists(path_to_images):
            os.makedirs(path_to_images)
        return path_to_images

    def execute(self):
        self.__camera.path_to_images = self.get_image_path(acquisition_mode=self.acquisition_mode)
        self.__camera.image_prefix = self.sample_name + '_IMG'
        # self.__camera.print_device_info()
        self.__camera.gain = float(self.camera_gain)
        self.__camera.exposure = float(self.camera_exposure_time)
        # self.__camera.trigger_delay = int(self.trigger_delay)
        self.__camera.frame_rate = float(self.camera_frame_rate)
        self.__camera.number_of_images = 1
        if self.acquisition_mode == 'Continuous':
            self.__camera.acquisition_mode = PySpin.AcquisitionMode_Continuous
            log.info(f'Acquisition mode set to continuous.')
            self.__camera.acquisition_time = self.camera_acquisition_time
            self.__camera.chosen_trigger = TriggerType.SOFTWARE
            # self.__camera.chosen_trigger = TriggerType.HARDWARE
            self.__camera.configure_trigger(trigger_type=PySpin.TriggerSelector_AcquisitionStart)
        elif self.acquisition_mode == 'Multi frame':
            self.__camera.acquisition_mode = PySpin.AcquisitionMode_MultiFrame
            log.info(f'Acquisition mode set to multi frame.')
            self.__camera.acquisition_time = self.camera_acquisition_time
            self.__camera.chosen_trigger = TriggerType.HARDWARE
            self.__camera.configure_trigger(trigger_type=PySpin.TriggerSelector_AcquisitionStart)
        else:
            self.__camera.acquisition_mode = PySpin.AcquisitionMode_SingleFrame
            log.info(f'Acquisition mode set to single frame.')
            self.__camera.chosen_trigger = TriggerType.HARDWARE
            self.__camera.configure_trigger(trigger_type=PySpin.TriggerSelector_AcquisitionStart)

        log.info(f'Current Gain: {self.__camera.gain}')
        log.info(f'The exposure read from the camera: {self.__camera.exposure}')
        log.info(f'The frame rate read from the camera is: {self.__camera.frame_rate} Hz')
        log.info(f'The number of images to take: {self.__camera.number_of_images}')
        log.info(f'The acquisition time is: {self.__camera.acquisition_time} s')
        # log.info(f'The trigger delay is: {self.__camera.trigger_delay} us.')
        self.camera_frame_rate = self.__camera.frame_rate

        flir_trigger = threading.Thread(target=trigger_camera, args=(self.__camera,))

        log.info(f"Setting the laser to the current setpoint: {float(self.laser_power_setpoint):.2f} %")
        emission_on = False
        gate_mode = False
        if self.laser_power_setpoint >= 10.0:
            self.__ylr.current_setpoint = self.laser_power_setpoint
        else:
            period_ms = 1.0
            frequency = 1.0E3 / period_ms
            duty_cycle = self.laser_power_setpoint / 20.0
            log.info(
                f'Power setpoint {self.laser_power_setpoint} %, using duty cycle of {duty_cycle * 100:.2f} % on 20% '
                f'setting, with a 1 ms period.')
            pulse_width = period_ms * duty_cycle
            self.__ylr.current_setpoint = 20.0
            self.__ylr.disable_modulation()
            self.__ylr.enable_gate_mode()
            gate_mode = True
            self.__ylr.pulse_repetition_rate = frequency
            self.__ylr.pulse_width = pulse_width

        time.sleep(0.1)
        log.info(f"Laser current setpoint: {float(self.__ylr.current_setpoint):.2f} %")
        try:
            self.__ylr.emission_on()
            emission_on = True
        except LaserException as e:
            log.warning(e)
            emission_on = False
            self.__ylr.aiming_beam_on = True

        self.__mx200.units = 'MT'
        time.sleep(5.0)

        log.info("Setting up Triggers")
        try:
            esp32 = ESP32Trigger()
        except SerialException as e:
            log.error("Error initializing ESP32 trigger")
            raise e

        tc_logger = DualTCLoggerTCP(ip_address=TC_LOGGER_IP)
        log.info('Successfully initialized thermocouple readout...')
        tc_logger.set_logger(log)

        esp32.pulse_duration = float(self.emission_time)
        time.sleep(0.1)
        et = esp32.pulse_duration
        # log.info(f'Pulse duration: {et:.2f} s.')

        t1 = time.time()
        tc_logger.log_time = self.measurement_time
        # self.__oscilloscope.write('ACQUIRE:STATE ON')
        tc_logger.start_logging()
        # Start firing sequence
        elapsed_time = []
        pressure = []

        # esp32.fire()

        previous_time = 0.0
        total_time = 0.0

        laser_output_power = []
        laser_output_peak_power = []
        start_time = time.time()

        esp32.fire()
        flir_trigger.start()
        started_acquisition = False
        # trigger_fired = False

        while total_time <= self.measurement_time + 0.015:
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break
            current_time = time.time()
            # if not trigger_fired:
            #     esp32.fire()
            #     trigger_fired = True
            # if not started_acquisition:
            #     if (current_time - start_time) >= 0.35:
            #         flir_trigger.start()
            #         started_acquisition = True
            if (current_time - previous_time) >= 0.015:
                total_time = current_time - start_time
                p = self.__mx200.pressure(gauge_number=1, use_previous=True)
                pressure.append(p)
                power_value = self.__ylr.output_power
                power_peak_value = self.__ylr.output_peak_power
                # if type(power_peak_value) == str:
                #     power_peak_value = 0.0
                # if type(power_value) == str:
                #     power_value = 0.0
                laser_output_power.append(power_value)
                laser_output_peak_power.append(power_peak_value)
                elapsed_time.append(total_time)
                # if emission_on:
                #     if round(power_peak_value) == 0. and (total_time >= (float(self.emission_time) + 0.5)):
                #         try:
                #             self.__ylr.emission_off()
                #             emission_on = False
                #         except LaserException as e:
                #             self.__ylr.aiming_beam_on = False
                #             log.warning(e)
                previous_time = current_time

        while self.__camera.busy:
            time.sleep(0.1)

        # self.__camera.shutdown()
        # self.__camera = None

        if emission_on:
            try:
                self.__ylr.emission_off()
                emission_on = False
            except LaserException as e:
                self.__ylr.aiming_beam_on = False
                log.warning(e)

        if gate_mode:
            self.__ylr.disable_gate_mode()
            self.__ylr.enable_modulation()
            gate_mode = False

        # Remove strings from laser_output_power and laser_output_peak_power
        laser_output_power = [0. if type(val) is str else val for val in laser_output_power]
        laser_output_peak_power = [0. if type(val) is str else val for val in laser_output_peak_power]

        elapsed_time = np.array(elapsed_time, dtype=float)
        elapsed_time -= elapsed_time.min()
        pressure = np.array(pressure, dtype=float)
        laser_output_power = np.array(laser_output_power)
        laser_output_peak_power = np.array(laser_output_peak_power)
        msk_on = laser_output_peak_power > 0.
        trigger_voltage = np.zeros_like(laser_output_peak_power)
        trigger_voltage[msk_on] = 3.
        # log.info(f'YLR output power: {laser_output_power.mean()}')
        # log.info(f'YLR output peak power: {laser_output_peak_power.max()}')
        laser_output_power_full = laser_output_power  # np.zeros_like(elapsed_time)
        laser_output_peak_power_full = laser_output_peak_power  # np.zeros_like(elapsed_time)
        # msk_power = elapsed_time <= (self.emission_time + 0.5)
        # laser_output_power_full[msk_power] = laser_output_power
        # laser_output_peak_power_full[msk_power] = laser_output_peak_power
        # elapsed_time -= elapsed_time.min()

        self.pressure_data = pd.DataFrame(
            data={
                'Time (s)': elapsed_time,
                'Pressure (Torr)': pressure,
                'Laser output power (W)': laser_output_power_full,
                'Laser peak power (W)': laser_output_peak_power
            }
        )

        flir_trigger.join()

        # self.save_pressure()
        # t2 = time.time()

        # dt = t2 - t1
        # log.info(f"dt: {dt:.3f}")

        log.info('*** ACQUISITION OVER ***')

        try:
            tc_data: pd.DataFrame = tc_logger.read_temperature_log()
        except (SerialException, ValueError) as e:
            log.error(e)
            raise ValueError(e)

        tc_logger.close()
        tc_time = tc_data['Time (s)'].values
        idx_mt = (np.abs(tc_time - self.measurement_time)).argmin()
        tc_data = tc_data.iloc[0:idx_mt + 1, :]

        # tc_data = tc_data[tc_data['Time (s)'] <= self.measurement_time]
        print(tc_data)

        # tc_data['Time (s)'] = tc_data['Time (s)'] - tc_data['Time (s)'].min()
        time_tc = tc_data['Time (s)'].values
        tc1 = tc_data['TC1 (C)'].values
        tc2 = tc_data['TC2 (C)'].values
        dt_tc = np.round(np.gradient(time_tc).mean(), 3)

        # print('time_tc:')
        # print(time_tc)
        print(f'len(time_tc): {len(time_tc)}, time_tc.min = {time_tc.min():.3f}, time_tc.max = {time_tc.max():.3f}, '
              f'dt: {time_tc[1] - time_tc[0]:.3f}')
        # print('elapsed_time:')
        # print(elapsed_time)
        print(f'len(elapsed_time): {len(elapsed_time)}, elapsed_time.min = {elapsed_time.min():.3f}, '
              f'elapsed_time.max = {elapsed_time.max():.3f}, dt: {elapsed_time[1] - elapsed_time[0]}')

        t_min = max(round(time_tc.min(), 3), round(elapsed_time.min(), 3))
        t_max = min(round(time_tc.max(), 3), round(elapsed_time.max(), 3))
        t_max = round(t_max*200)/200

        f0 = interpolate.interp1d(time_tc, tc1, fill_value="extrapolate")
        f1 = interpolate.interp1d(time_tc, tc2, fill_value='extrapolate')
        pressure_is_zero = False
        if all(v <= 0. for v in pressure):
            pressure_is_zero = True
        else:
            f2 = interpolate.interp1d(elapsed_time, pressure, fill_value="extrapolate")
        f3 = interpolate.interp1d(elapsed_time, trigger_voltage, fill_value="extrapolate")
        f4 = interpolate.interp1d(elapsed_time, laser_output_power_full, fill_value="extrapolate")
        f5 = interpolate.interp1d(elapsed_time, laser_output_peak_power_full, fill_value="extrapolate")

        n_data_points = int(t_max / 0.005) + 1
        time_interp = 0.0 + 0.005 * np.arange(0, n_data_points)
        msk_interp = time_interp <= t_max
        time_interp = time_interp[msk_interp]
        n_data_points = len(time_interp)

        try:
            tc1_interp = f0(time_interp)
            tc2_interp = f1(time_interp)
            if pressure_is_zero:
                pressure_interp = np.zeros_like(time_interp)
            else:
                pressure_interp = f2(time_interp)
            trigger_interp = f3(time_interp)
            power_interp = f4(time_interp)
            peak_power_interp = f5(time_interp)
        except Exception as ex:
            log.exception(ex)
            raise ex

        filename = f'{os.path.splitext(self.__unique_filename)[0]}_tcdata.csv'
        # tc_data.to_csv(filename, index=False)

        """
        DATA_COLUMNS = ["Measurement Time (s)", "Pressure (Torr)", "TC1 (C)", "TC2 (C)", "Trigger (V)",
                    "Laser output power (W)", "Laser output peak power (W)"]
        """

        data_interp = pd.DataFrame(data={
            "Measurement Time (s)": time_interp,
            "Pressure (Torr)": pressure_interp,
            "TC1 (C)": tc1_interp,
            "TC2 (C)": tc2_interp,
            "Trigger (V)": trigger_interp,
            "Laser output power (W)": power_interp,
            "Laser output peak power (W)": peak_power_interp
        })

        # data_interp.to_csv(f'{os.path.splitext(self.__unique_filename)[0]}_dinterp.csv', index=False)

        pct_f = 100. / n_data_points
        esp32.close()

        time.sleep(5.0)

        for ii in range(n_data_points):
            dd = {
                'Measurement Time (s)': np.round(time_interp[ii], 4),
                'Pressure (Torr)': np.round(pressure_interp[ii], 3),
                'TC1 (C)': np.round(tc1_interp[ii], 2),
                'TC2 (C)': np.round(tc2_interp[ii], 2),
                'Trigger (V)': np.round(trigger_interp[ii], 3),
                'Laser output power (W)': np.round(power_interp[ii], 1),
                'Laser output peak power (W)': np.round(peak_power_interp[ii], 1)
            }
            self.emit('results', dd)
            # log.debug("Emitting results: %s" % dd)
            if (ii % 10 == 0) or (ii + 1 == n_data_points):
                self.emit('progress', (ii + 1) * pct_f)
            # time.sleep(0.005)
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break

    def shutdown(self):
        if self.__mx200 is not None:
            self.__mx200.close()
        if self.__ylr is not None:
            self.__ylr.disconnect()

        if self.__camera is not None:
            if not self.__camera.busy:
                try:
                    self.__camera.shutdown()
                except PySpin.SpinnakerException as ex:
                    log.error(ex)

        # Remove file handlers from logger
        self.clear_log_handlers()

    @staticmethod
    def clear_log_handlers():
        if "log" not in locals():
            return
        if len(log.handlers) > 0:
            for h in log.handlers:
                if isinstance(h, logging.FileHandler):
                    log.removeHandler(h)
                    h.close()
                # if isinstance(h, logging.NullHandler):
                #     log.removeHandler(h)
                #     log.addHandler(logging.NullHandler())


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=LaserProcedure,
            inputs=['emission_time', 'measurement_time', 'laser_power_setpoint', 'camera_acquisition_time',
                    'camera_exposure_time',
                    'camera_frame_rate', 'camera_gain', 'acquisition_mode', 'sample_name'],
            displays=['emission_time', 'measurement_time', 'laser_power_setpoint', 'camera_acquisition_time',
                      'camera_exposure_time',
                      'camera_frame_rate', 'camera_gain', 'acquisition_mode', 'sample_name'],
            x_axis='Measurement Time (s)',
            y_axis='Pressure (Torr)',
            # directory_input=True,
        )
        self.setWindowTitle('Laser test (camera)')
        self.filename = r'LCT_{Sample Name}_{Laser Power Setpoint:03.0f}PCT_'
        self.file_input.extensions = ["csv"]

    def clear_log(self):
        if len(log.handlers) > 0:
            for h in log.handlers:
                if isinstance(h, logging.NullHandler):
                    log.removeHandler(h)
                    log.addHandler(logging.NullHandler())
                if isinstance(h, logging.FileHandler):
                    log.removeHandler(h)
                    h.close()

    def queue(self):
        directory = self.directory

        procedure: LaserProcedure = self.make_procedure()
        file_path = unique_filename(
            directory=self.directory,
            prefix=replace_placeholders(procedure=procedure, string=self.filename)
        )

        self.clear_log()

        # prefix = f'LCT_{sample_name}_{laser_setpoint:03.0f}PCT_'
        # filename = unique_filename(directory, prefix=prefix)
        filename = os.path.basename(file_path)
        log_file = os.path.splitext(file_path)[0] + '.log'
        # formatter = logging.Formatter('%(actime)s - %(levelname)s - %(message)s')
        # fh = logging.FileHandler(log_file)
        # fh.setFormatter(formatter)
        # fh.setLevel(logging.DEBUG)
        # log.addHandler(fh)
        # Using pymeasure logging functionality which implements Queues for consistent multi-process logging.
        file_log(logger=log, log_filename=log_file, level=logging.DEBUG)
        log.info(f'Starting experiment')

        procedure.unique_filename = file_path
        procedure.directory = directory

        results = Results(procedure, file_path)
        # results.CHUNK_SIZE = 1000  # increased it from 1000 to check if it resolves emit result issues scrambling rows
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

    app = QtWidgets.QApplication(sys.argv)
    # app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # sys.exit(app.exec_())
    sys.exit(app.exec())
