import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
from pymeasure.display.Qt import QtGui
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import FloatParameter, ListParameter, Parameter, BooleanParameter
from pymeasure.experiment import unique_filename
from serial.serialutil import SerialException
from instruments.esp32 import ESP32Trigger
from instruments.esp32 import DualTCLoggerTCP
from instruments.tektronix import TBS2000
from instruments.mx200 import MX200
from instruments.IPG import YLR3000, LaserException
from instruments.inhibitor import WindowsInhibitor
from scipy import interpolate

TBS2000_RESOURCE_NAME = 'USB0::0x0699::0x03C7::C010461::INSTR'
ESP32_COM = 'COM6'
# TC_LOGGER_COM = 'COM10'
TC_LOGGER_IP = '192.168.4.3'
MX200_COM = 'COM3'
TRIGGER_CHANNEL = 2
THERMOMETRY_CHANNEL = 1
TRIGGER_LEVEL = 3.3
SAMPLING_INTERVAL = 0.005
IP_LASER = "192.168.3.230"


def get_duty_cycle_params(duty_cycle: float, period_ms: float = 1.0) -> tuple:
    frequency = 1.0E3 / period_ms
    pulse_width = duty_cycle * period_ms


class LaserProcedure(Procedure):
    emission_time = FloatParameter('Emission Time', units='s', default=0.5, minimum=0.001, maximum=20.0)
    measurement_time = FloatParameter('Measurement Time', units='s', default=3.0, minimum=1.0, maximum=3600.0)
    laser_power_setpoint = FloatParameter("Laser Power Setpoint", units="%", default=100, minimum=0.0, maximum=100.0)
    pd_gain = ListParameter('Photodiode Gain', choices=('0', '10', '20', '30', '40', '50', '60', '70'), units='dB',
                            default='30')
    sample_name = Parameter("Sample Name", default="UNKNOWN")
    __oscilloscope: TBS2000 = None
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

    DATA_COLUMNS = ["Measurement Time (s)", "Photodiode Voltage (V)", "Trigger (V)", "TC1 (C)", "TC2 (C)"]

    def startup(self):
        print('***  Startup ****')
        self.__oscilloscope = TBS2000(resource_name=TBS2000_RESOURCE_NAME)
        self.__mx200 = MX200(address=MX200_COM, keep_alive=True)
        self.__oscilloscope.display_channel_list((1, 1, 0, 0))
        log.debug(self.__oscilloscope.sesr)
        log.debug(self.__oscilloscope.all_events)
        self.__mx200.set_logger(log)
        log.info("Setting up Lasers")
        self.__ylr = YLR3000(IP=IP_LASER)
        self.__ylr.set_logger(log)

    @property
    def unique_filename(self) -> str:
        return self.__unique_filename

    @unique_filename.setter
    def unique_filename(self, val: str):
        print(f"Storing filepath: {val}")
        self.__unique_filename = val

    def save_pressure(self):
        if self.pressure_data is not None:
            filename = f'{os.path.splitext(self.__unique_filename)[0]}_pressure.csv'
            self.pressure_data.to_csv(filename, index=False)

    def execute(self):
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
        log.info("Setting up Oscilloscope")
        self.__oscilloscope.write('ACQUIRE:STATE 0')
        self.__oscilloscope.write(f'CH{THERMOMETRY_CHANNEL}:VOLTS 1.0')
        self.__oscilloscope.write(f'CH{TRIGGER_CHANNEL}:VOLTS 1.0')
        self.__oscilloscope.set_acquisition_time(self.measurement_time)
        self.__oscilloscope.write('ACQUIRE:STOPAFTER SEQUENCE')
        self.__oscilloscope.write('ACQuire:MODe SAMple')
        self.__mx200.units = 'MT'
        time.sleep(1.0)

        log.info("Setting up Triggers")
        try:
            esp32 = ESP32Trigger(address=ESP32_COM)
        except SerialException as e:
            log.error("Error initializing ESP32 trigger")
            raise e

        tc_logger = DualTCLoggerTCP(ip_address=TC_LOGGER_IP)
        log.info('Successfully initialized thermocouple readout...')
        tc_logger.set_logger(log)

        esp32.pulse_duration = float(self.emission_time)
        time.sleep(0.1)
        et = esp32.pulse_duration
        log.info(f'Pulse duration: {et:.2f} s.')

        # Prevent computer from going to sleep mode
        self.__oscilloscope.timeout = 1000
        t1 = time.time()
        tc_logger.log_time = self.measurement_time
        tc_logger.start_logging()
        self.__oscilloscope.write('ACQUIRE:STATE ON')
        # Start firing sequence
        elapsed_time = []
        pressure = []
        p_previous = self.__mx200.pressure(2)

        esp32.fire()

        previous_time = 0.0
        total_time = 0.0
        start_time = time.time()
        laser_power_value = 0.0
        laser_power_peak_value = 0.0

        while total_time <= self.measurement_time:
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break
            current_time = time.time()
            if (current_time - previous_time) >= 0.010:
                total_time = current_time - start_time
                p = self.__mx200.pressure(2)
                if type(p) == str:
                    p = p_previous
                pressure.append(p)
                elapsed_time.append(total_time)
                p_previous = p
                if total_time < self.emission_time:
                    laser_power_value = self.__ylr.output_power
                    power_peak_value = self.__ylr.output_peak_power
                    if type(power_peak_value) == float:
                        laser_power_peak_value = max(power_peak_value, laser_power_peak_value)
                # if total_time >= self.emission_time + 1.0 and emission_on:
                #     self.__ylr.emission_off()
                #     emission_on = False
                previous_time = current_time


        log.info(f'YLR output power: {laser_power_value}')
        log.info(f'YLR output peak power: {laser_power_peak_value}')

        if emission_on:
            try:
                self.__ylr.emission_off()
                emission_on = False
            except LaserException as e:
                log.warning(e)

        if gate_mode:
            self.__ylr.disable_gate_mode()
            self.__ylr.enable_modulation()
            gate_mode = False

        # log.info("elapsed time:")
        # log.info(elapsed_time)
        elapsed_time = np.array(elapsed_time, dtype=float)
        pressure = np.array(pressure, dtype=float)
        self.pressure_data = pd.DataFrame(
            data={
                'Time (s)': elapsed_time,
                f'Pressure (Torr)': pressure
            }
        )

        self.save_pressure()
        t2 = time.time()

        dt = t2 - t1
        log.info(f"dt: {dt:.3f}")

        while True:
            busyA = self.__oscilloscope.ask("BUSY?")
            if busyA == '0':
                break

        log.info('*** ACQUISITION OVER ***')

        data = self.__oscilloscope.get_curve(channel=THERMOMETRY_CHANNEL)
        time.sleep(1.0)
        reference = self.__oscilloscope.get_curve(channel=TRIGGER_CHANNEL)
        time.sleep(1.0)
        try:
            tc_data: pd.DataFrame = tc_logger.read_temperature_log()
        except (SerialException, ValueError) as e:
            log.error(e)
            raise ValueError(e)

        tc_time = tc_data['Time (s)']
        tc_time_max_idx = tc_time.idxmax()
        tc_data = tc_data.iloc[0:tc_time_max_idx + 1, :]
        print(tc_data)

        columns = data.dtype.names
        columns_ref = reference.dtype.names
        npoints = len(data)

        ir_df = pd.DataFrame(data={
            'Time (s)': data[columns[0]],
            'Voltage (V)': data[columns[1]]
        })

        filename = f'{os.path.splitext(self.__unique_filename)[0]}_irdata.csv'
        ir_df.to_csv(path_or_buf=filename, index=False)

        log.info(f'Number of osc data points: {npoints}')
        tc_data['Time (s)'] = tc_data['Time (s)'] - tc_data['Time (s)'].min()
        time_tc = tc_data['Time (s)'].values
        tc1 = tc_data['TC1 (C)'].values
        tc2 = tc_data['TC2 (C)'].values

        filename = f'{os.path.splitext(self.__unique_filename)[0]}_tcdata.csv'
        tc_data.to_csv(filename, index=False)

        data = data[data[columns[0]] <= time_tc.max()]
        reference = reference[reference[columns_ref[0]] <= time_tc.max()]
        time_osc = data[columns[0]]
        print('time_osc:')
        print(time_osc)
        print(f'len(time_osc): {len(time_osc)}, time_osc.min = {time_osc.min()}, time_osc.max = {time_osc.max()}')

        print('time_tc:')
        print(time_tc)
        print(f'len(time_tc): {len(time_tc)}, time_tc.min = {time_tc.min()}, time_tc.max = {time_tc.max()}')

        # msk_tmin = time_osc >= time_tc.min()
        # data = data[msk_tmin]
        # reference = reference[msk_tmin]
        time_osc = data[columns[0]]

        f1 = interpolate.interp1d(time_tc, tc1)
        f2 = interpolate.interp1d(time_tc, tc2)
        tc1_interp = f1(time_osc)
        tc2_interp = f2(time_osc)

        for i in range(len(data)):
            d = {
                "Measurement Time (s)": data[columns[0]][i],
                "Photodiode Voltage (V)": data[columns[1]][i],
                "Trigger (V)": reference[columns_ref[1]][i],
                "TC1 (C)": tc1_interp[i],
                "TC2 (C)": tc2_interp[i]
            }
            self.emit('results', d)
            self.emit('progress', (i + 1) * 100 / len(data))
            time.sleep(0.0005)

        self.__oscilloscope.timeout = 1

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
        if self.__oscilloscope is not None:
            self.__oscilloscope.close()
        if self.__ylr is not None:
            self.__ylr.disconnect()
        # Remove file handlers from logger
        if len(log.handlers) > 0:
            for h in log.handlers:
                if isinstance(h, logging.FileHandler):
                    log.removeHandler(h)


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=LaserProcedure,
            inputs=['emission_time', "measurement_time", "laser_power_setpoint", "pd_gain", "sample_name"],
            displays=['emission_time', "measurement_time", "laser_power_setpoint", "pd_gain", "sample_name"],
            x_axis="Measurement Time (s)",
            y_axis="Photodiode Voltage (V)",
            directory_input=True,
        )
        self.setWindowTitle('Laser Test')

    def queue(self):
        directory = self.directory

        procedure: LaserProcedure = self.make_procedure()
        sample_name = procedure.sample_name
        laser_setpoint = procedure.laser_power_setpoint
        # photodiode_gain = procedure.pd_gain

        prefix = f'LT_{sample_name}_{laser_setpoint:03.0f}PCT_'
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
