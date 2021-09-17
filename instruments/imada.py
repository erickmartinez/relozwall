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
    __timeout = 0
    __parity = serial.PARITY_NONE
    __stopbits = serial.STOPBITS_ONE
    __xonxoff = 1
    __delay = 0.1

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

    def __init__(self, address: str):
        self.__address = address

    def read(self):
        data_str = self.query("D")
        match = self.__D_PATTERN.match(data_str)
        if match is not None:
            groups = match.groups()
            s = 1 if groups[0] == '+' else -1
            reading = s * float(groups[1])
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
        return {}

    def zero(self):
        self.write("Z")

    def real_time_mode(self):
        self.write("T")

    def peak_mode(self):
        self.write("P")

    def units(self, value: str):
        if value in self.__units_r:
            u = self.__units_r[value]
            self.write(u)
        if value in self.__units:
            self.write(value)
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

    def query(self, q: str) -> str:
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
            ser.write("{0}".format(q).encode('utf-8'))
            sleep(self.__delay)
            line = ser.readline()
            return line.decode('utf-8').rstrip("\r").rstrip(" ")