import numpy as np
import serial
from time import sleep
import re

PATTERN = re.compile(r'(\d{2})\=(.*)')


class MX200:
    """
    Represents the Televac MX200 Controller
    """

    __address = 'COM3'
    __baud_rate = 57600
    __byte_size = serial.EIGHTBITS
    __timeout = 0
    __parity = serial.PARITY_NONE
    __stopbits = serial.STOPBITS_ONE
    __xonxoff = 1
    __delay = 0.1

    __units_mapping = {
        'PA': 'Pascal',
        'TR': 'Torr',
        'MB': 'millibar',
        'TD': 'Torr decimal',
        'MT': 'mTorr'
    }

    def __init__(self, address: str):
        self.__address = address

    @property
    def pressures(self) -> dict:
        response: str = self.query("S1")
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
                if re.match(r"\d{5}", reading):
                    pressures[gauge_number] = self.ppsee(reading)
                else:
                    pressures[gauge_number] = reading
            else:
                pressures[i] = None
        return pressures

    def pressure(self, gauge_number: int):
        if 1 <= gauge_number <= 2:
            q = 'S1{0:02d}'.format(gauge_number)
            pressure = self.query(q)
            if re.match(r"\d{5}", pressure):
                pressure = self.ppsee(pressure)
            return pressure
        else:
            msg = "Invalid gauge number ({0:d}). Valid gauges are 1-2.".format(gauge_number)
            raise Exception(msg)

    @property
    def serial_number(self) -> str:
        return self.query("SN")

    @property
    def units(self):
        return self.query('R1')

    @units.setter
    def units(self, value):
        if value in self.__units_mapping:
            q = "W1{0}\r".format(value)
            r = self.query(q)
            if r != value:
                raise Warning("Units could not be set.")

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

    @property
    def delay(self):
        return self.__delay

    @delay.setter
    def delay(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if value > 0:
                self.__delay = value


    @staticmethod
    def ppsee(string_value: str):
        mantissa = float(string_value[0:2]) #/ 10
        sign = -1 if string_value[2] == 0 else 1
        exponent = float(string_value[3:5])
        return mantissa * 10.0 ** (sign * exponent) / 10

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
            ser.write("{0}\r".format(q).encode('utf-8'))
            sleep(self.__delay)
            line = ser.readline()
            return line.decode('utf-8').rstrip("\r\n").rstrip(" ")
