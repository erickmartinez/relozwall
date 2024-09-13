import logging
import time
from instruments.BaseSerial import BaseSerial
import numpy as np
import serial
from time import sleep
import re

from serial import SerialException

PATTERN = re.compile(r'(\d{2})\=(.*)')


class MX200(BaseSerial):
    """
    Represents the Televac MX200 Controller
    """

    __timeout = 0.01
    __delay = 0.002
    _log: logging.Logger = None

    units_mapping = {
        'PA': 'Pascal',
        'TR': 'Torr',
        'MB': 'millibar',
        'TD': 'Torr decimal',
        'MT': 'mTorr'
    }

    def __init__(self):
        super().__init__(name='MX200')
        self._serial_settings = {
            "baudrate": 115200,
            "bytesize": serial.EIGHTBITS,
            "parity": serial.PARITY_NONE,
            "stopbits": serial.STOPBITS_ONE,
            "xonxoff": True,
            "rtscts": False,
            "dsrdtr": False,
            "exclusive": None,
            "timeout": self.__timeout,
            "write_timeout": 0.05,
        }


        self.set_id_validation_query(
            id_validation_query=self.id_validation_query,
            valid_id_specific='406714'
        )

        self.auto_connect()

        self._log = logging.getLogger(__name__)
        self._log.addHandler(logging.NullHandler())
        # create console handler and set level to debug
        has_console_handler = False
        if len(self._log.handlers) > 0:
            for h in self._log.handlers:
                if isinstance(h, logging.StreamHandler):
                    has_console_handler = True
        if not has_console_handler:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            self._log.addHandler(ch)

        self._ppsee_pattern = re.compile(r"\d{5}")
        self._previous_pressures = ['', '']
        for i in range(2):
            while type(self._previous_pressures[i]) != float:
                self._previous_pressures[i] = self.pressure(i + 1)

    def id_validation_query(self) -> str:
        response = self.query('SN')
        return response

    def set_logger(self, log: logging.Logger):
        self._log = log

    @property
    def timeout(self):
        return self._serial['timeout']

    @timeout.setter
    def timeout(self, value: float):
        value = abs(float(value))
        self.__timeout = value
        self._serial_settings['timeout'] = value
        if self._serial is not None:
            self._serial.timeout = value

    def check_id(self, attempt: int = 0) -> bool:
        time.sleep(0.1)
        old_delay = self.delay
        old_timeout = self.timeout
        self.delay = 0.1
        self.timeout = 0.1
        check_id = self.query('SN')
        self.delay = old_delay
        self.delay = old_timeout
        if check_id != '406714':
            if attempt <= 3:
                attempt += 1
                return self.check_id(attempt=attempt)
            else:
                return False
        else:
            return True

    @property
    def pressures(self) -> dict:
        response: str = self.query("S1")
        time.sleep(self.__delay)
        pressures_str = response.split()
        if len(pressures_str) == 0:
            return None
        pressures = {}
        for i, p in enumerate(pressures_str):
            match = PATTERN.match(p)
            if match is not None:
                groups = match.groups()
                gauge_number = int(groups[0])
                reading = groups[1]
                if self._ppsee_pattern.match(reading) is not None:
                    pressures[gauge_number] = self.ppsee(reading)
                else:
                    pressures[gauge_number] = reading
            else:
                pressures[i] = None
        return pressures

    def pressure(self, gauge_number: int, use_previous = True):
        if 1 <= gauge_number <= 2:
            q = 'S1{0:02d}'.format(gauge_number)
            # pressure = self.query(q)
            self._serial.write(f"{q}\r".encode('utf-8'))
            time.sleep(self.__delay)
            pressure = self._serial.read(7).decode('utf-8').rstrip("\r\n")
            if self._ppsee_pattern.match(pressure) is not None:
                pressure = self.ppsee(pressure)
                self._previous_pressures[gauge_number] = pressure
            elif use_previous:
                pressure = self._previous_pressures[gauge_number]
            return pressure
        else:
            msg = "Invalid gauge number ({0:d}). Valid gauges are 1-2.".format(gauge_number)
            raise ValueError(msg)

    @property
    def serial_number(self) -> str:
        return self.query("SN")

    @property
    def units(self):
        return self.query('R1')

    @units.setter
    def units(self, value: str):
        self.set_units(value)

    def set_units(self, value: str, attempts=0):
        if value in self.units_mapping:
            q = f"W1{value.upper()}"
            self._serial.write(f'{q}\r'.encode('utf-8'))
            time.sleep(2.0)
            # r = self._serial.read(4).decode('utf-8').rstrip('\r\n')
            # time.sleep(self.__delay)
            r = self.query('R1')
            if r != value:
                self._log.warning(f'Units {value} could not be set. Query \'{q}\' returned \'{r}\'')
                if attempts < 3:
                    self.set_units(value, attempts + 1)

    @property
    def sensor_types(self) -> dict:
        response = self.query('S4')
        if response is None:
            return {}
        sensors_str = response.split()
        sensors = {}
        for i, s in enumerate(sensors_str):
            match = PATTERN.match(s)
            if match is not None:
                groups = match.groups()
                gauge_number = int(groups[0])
                sensor_type = groups[1]
                sensors[gauge_number] = sensor_type

        return sensors

    def read_calibration(self, channel: int, adjustment_point: int) -> int:
        adjustment_point = int(adjustment_point)
        channel = int(channel)
        if 2 < channel < 1:
            raise Warning(f"Channel '{channel}' is not available.")
        if 1 <= adjustment_point <= 5:
            query: str = f"RC{adjustment_point}{str(channel).zfill(2)}"
            result = self.query(q=query)
            self._log.debug(result)
            return self.baa(result)
        else:
            raise Warning(f"Invalid adjustment point: {adjustment_point}.")

    def set_calibration(self, channel: int, adjustment_point: int, set_point: int):
        adjustment_point = int(adjustment_point)
        channel = int(channel)
        if 2 < channel < 1:
            raise Warning(f"Channel '{channel}' is not available.")
        if 4 < adjustment_point < 1:
            raise Warning(f"Invalid adjustment point: {adjustment_point}.")
        baa = self.integer2baa(set_point)
        query = f"WC{adjustment_point}{str(channel).zfill(2)}{baa}"
        self._log.debug(query)
        self.write(q=query)
        # r = self.query(q=query)
        # if re.match(r"\d{5}", r):
        #     r = self.ppsee(r)
        # return r
        # time.sleep(self.__delay)
        # q = 'S1{0:02d}'.format(channel)
        # pressure = self.query(q)
        # pressure = float(pressure)
        # print(f"Pressure: {pressure:.1e}")
        # return pressure

    @staticmethod
    def ppsee(string_value: str) -> float:
        if string_value is None:
            return -1
        mantissa = float(string_value[0:2]) / 10
        s = -1 if string_value[2] == '0' else 1
        exponent = float(string_value[3:5])
        return mantissa * 10.0 ** (s * exponent)

    @staticmethod
    def baa(string_value: str):
        s = -1 if string_value[0] == '0' else 1
        aa = int(string_value[1::])
        return s * aa

    @staticmethod
    def integer2baa(value: int):
        if abs(value) > 99:
            raise Warning(f"Invalid value: {value}.\nValid range is -99 to 99.")
        b = '0' if np.sign(value) == -1 else 1
        aa = str(abs(value)).zfill(2)
        return f"{b}{aa}"

    @property
    def delay(self) -> float:
        return self.__delay

    @delay.setter
    def delay(self, value):
        value = float(value)
        if value > 0:
            self.__delay = value

    def write(self, q: str):
        self._serial.write("{0}\r".format(q).encode('utf-8'))
        sleep(self.__delay)

    def query(self, q: str) -> str:
        self._serial.write("{0}\r".format(q).encode('utf-8'))
        sleep(self.__delay)
        line = self._serial.readline()
        sleep(self.__delay)
        return line.decode('utf-8').rstrip("\r\n").rstrip(" ")
