import logging
import time
from instruments.BaseSerial import BaseSerial
import numpy as np
import serial
from time import sleep
import re


class DST44A(BaseSerial):
    """
    Represents the IMADA DST-44A Force Gauge

    Attributes
    ----------
    __address: str
        The address at which the gauge is located
    """

    __baud_rate = 256000
    __byte_size = serial.EIGHTBITS
    __timeout = 0.005
    __parity = serial.PARITY_NONE
    __stopbits = serial.STOPBITS_ONE
    __xonxoff = 1
    __delay = 0.005

    __previous: float = 0.0
    __previous_json: dict = {}

    __D_PATTERN = re.compile(r"(\+|\-)(\d+\.?\d*)([OKN])([TP])([LOHE])")

    __units = {'O': 'lbf', 'K': 'kgf', 'N': 'N'}
    __units_r = {'lbf': 'O', 'kgf': 'K', 'N': 'N'}
    __modes = {'T': 'Real Time', 'P': 'Peak'}
    __judgments = {
        'L': 'Below low setpoint',
        'O': 'Between high and low setpoints',
        'H': 'Above high setpoint',
        'E': 'Overload'
    }

    _serial: serial.Serial = None
    __log: logging.Logger = None

    def __init__(self):
        super().__init__(name='DST44A')
        self._serial_settings = {
            "baudrate": 256000,
            "bytesize": serial.EIGHTBITS,
            "parity": serial.PARITY_NONE,
            "stopbits": serial.STOPBITS_ONE,
            "xonxoff": True,
            "rtscts": False,
            "dsrdtr": False,
            "exclusive": None,
            "timeout": 0.005,
            "write_timeout": 0.005,
        }
        
        self.__log = logging.getLogger(__name__)
        self.__log.addHandler(logging.NullHandler())

        # create console handler and set level to debug
        has_console_handler = False
        if len(self.__log.handlers) > 0:
            for handler in self.__log.handlers:
                if isinstance(handler, logging.StreamHandler):
                    has_console_handler = True

        if not has_console_handler:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            self.__log.addHandler(ch)

        self.set_id_validation_query(
            id_validation_query=self.id_validation_query,
            valid_id_specific='DST44A'
        )

        self.auto_connect()

        self.zero()

    def id_validation_query(self) -> str:
        old_delay = self.delay
        old_timeout = self.timeout
        self.delay = 0.5
        self.timeout = 2.
        self._serial.write("D\r".encode('utf-8'))
        time.sleep(self.__delay)
        data_str = self._serial.read(10).decode('utf-8').rstrip("\r").rstrip(" ")
        self.delay = old_delay
        self.timeout = old_timeout
        match = self.__D_PATTERN.match(data_str)
        if match is not None:
            return 'DST44A'
        return False

    def logger(self) -> logging.Logger:
        return self.__log

    def set_logger(self, log: logging.Logger):
        self.__log = log

    def read(self, json=False, attempts=0):
        self._serial.write("D\r".encode('utf-8'))
        # self._serial.flush()
        time.sleep(self.__delay)
        data_str = self._serial.read(10).decode('utf-8').rstrip("\r").rstrip(" ")
        # self._serial.flush()
        match = self.__D_PATTERN.match(data_str)
        if match is None:
            logging.warning(f'Received gauge reponse: {data_str}')
            attempts += 1
            if attempts < 3:
                self.__log.warning(f'Failed to read the force. Trying again. (Attempt {attempts+1}/3)')
                return self.read(attempts=attempts)
            else:
                self.__log.warning(f'I tried reading the force {attempts} times and failed. Returning previous value.')
                if json:
                    return self.__previous_json
                return self.__previous
        groups = match.groups()
        s = 1 if groups[0] == '+' else -1
        reading = s * float(groups[1])
        self.__previous = reading
        if json:
            units = self.__units[groups[2]]
            mode = self.__modes[groups[3]]
            judgment = self.__judgments[groups[4]]
            r = {
                'reading': reading,
                'units': units,
                'mode': mode,
                'judgement': judgment,
                'judgement_code': groups[4],
            }
            self.__previous_json = r
            return r
        return reading

    def zero(self):
        self.query("Z")

    def real_time_mode(self):
        self.query("T")

    def peak_mode(self):
        self.query("P")

    def units(self, value: str):
        if value in self.__units_r:
            u = self.__units_r[value]
            r = self.query(u)
            self.__log.info(f'Response to units query (\'{u}\'): {r}')
        elif value in self.__units:
            r = self.query(value)
            self.__log.info(f'Response to units query (\'{value}\'): {r}')
        else:
            raise ValueError(f"Invalid unit: {value}. Available units: {self.__units.keys()} ({self.__units_r.keys()}).")

    @property
    def high_low_setpoints(self) -> dict:
        r = self.query("E")
        h = r[1:4]
        l = r[5:8]
        return {
            'high': int(h),
            'low': int(l)
        }

    def set_high_low_setpoints(self, high:int, low:int):
        q = f"E{high:04d}{low:04d}"
        self.query(q)

    def write(self, q: str):
        self._serial.write(f'{q}\r'.encode('utf-8'))
        time.sleep(self.__delay)

    def query(self, q: str) -> str:
        self.write(q)
        line = self._serial.readline()
        return line.decode('utf-8').rstrip("\r").rstrip(" ")