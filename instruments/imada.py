import logging
import time

import numpy as np
import serial
from time import sleep
import re


class DST44A:
    """
    Represents the IMADA DST-44A Force Gauge

    Attributes
    ----------
    __address: str
        The address at which the gauge is located
    """

    __address = 'COM5'
    __baud_rate = 256000
    __byte_size = serial.EIGHTBITS
    __timeout = 0.005
    __parity = serial.PARITY_NONE
    __stopbits = serial.STOPBITS_ONE
    __xonxoff = 1
    __delay = 0.005

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

    __serial: serial.Serial = None
    __log: logging.Logger = None

    def __init__(self, address: str):
        self.__address = address
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
        self.zero()

    def logger(self) -> logging.Logger:
        return self.__log

    def set_logger(self, log: logging.Logger):
        self.__log = log

    def connect(self):
        self.__serial = serial.Serial(
            port=self.__address,
            baudrate=self.__baud_rate,
            bytesize=self.__byte_size,
            timeout=self.__timeout,
            parity=self.__parity,
            stopbits=self.__stopbits,
            xonxoff=self.__xonxoff
        )
        sleep(self.__delay)

    def close(self):
        if self.__serial is not None:
            self.__serial.close()

    def __del__(self):
        self.close()

    def read(self, json=False, attempts=0):
        if self.__serial is None:
            data_str = self.query("D")
        else:
            self.__serial.flush()
            self.__serial.write("D\r".encode('utf-8'))
            time.sleep(self.__delay)
            data_str = self.__serial.read(10).decode('utf-8').rstrip("\r").rstrip(" ")
        match = self.__D_PATTERN.match(data_str)
        if match is None:
            logging.warning(f'Received gauge reponse: {data_str}')
            attempts += 1
            if attempts < 10:
                self.__log.warning('Failed to read the force. Trying again...')
                return self.read(attempts=attempts)
            else:
                self.__log.info(f'I tried reading the force {attempts} times and failed. Sorry...')
                return {}
        groups = match.groups()
        s = 1 if groups[0] == '+' else -1
        reading = s * float(groups[1])
        if json:
            units = self.__units[groups[2]]
            mode = self.__modes[groups[3]]
            judgment = self.__judgments[groups[4]]
            return {
                'reading': reading,
                'units': units,
                'mode': mode,
                'judgement': judgment,
                'judgement_code': groups[4],
            }
        return reading

    def zero(self):
        self.write("Z")

    def real_time_mode(self):
        self.write("T")

    def peak_mode(self):
        self.write("P")

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
        self.write(q)

    def write(self, q: str):
        if self.__serial is None:
            with serial.Serial(
                    port=self.__address,
                    baudrate=self.__baud_rate,
                    bytesize=self.__byte_size,
                    timeout=self.__timeout,
                    parity=self.__parity,
                    stopbits=self.__stopbits,
                    xonxoff=self.__xonxoff
            ) as ser:
                sleep(self.__delay)
                ser.write("{0}\r".format(q).encode('utf-8'))
                sleep(self.__delay)
        else:
            self.__serial.write(f'{q}\r'.encode('utf-8'))
            sleep(self.__delay)

    def query(self, q: str) -> str:
        if self.__serial is None:
            with serial.Serial(
                    port=self.__address,
                    baudrate=self.__baud_rate,
                    bytesize=self.__byte_size,
                    timeout=self.__timeout,
                    parity=self.__parity,
                    stopbits=self.__stopbits,
                    xonxoff=self.__xonxoff
            ) as ser:
                ser.write("{0}\r".format(q).encode('utf-8'))
                sleep(self.__delay)
                line = ser.readline()
                return line.decode('utf-8').rstrip("\r").rstrip(" ")
        else:
            self.write(q)
            line = self.__serial.readline()
            sleep(self.__delay)
            return line.decode('utf-8').rstrip("\r").rstrip(" ")