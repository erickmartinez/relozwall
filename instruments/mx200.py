import time

import numpy as np
import serial
from time import sleep
import re

from serial import SerialException

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
    __delay = 0.05
    __keep_alive: bool = False
    __serial: serial.Serial = None

    units_mapping = {
        'PA': 'Pascal',
        'TR': 'Torr',
        'MB': 'millibar',
        'TD': 'Torr decimal',
        'MT': 'mTorr'
    }

    def __init__(self, address: str, keep_alive: bool = False):
        self.__address = address
        self.__keep_alive = bool(keep_alive)
        if self.__keep_alive:
            self.__serial = serial.Serial(
                port=self.__address,
                baudrate=self.__baud_rate,
                bytesize=self.__byte_size,
                timeout=self.__timeout,
                parity=self.__parity,
                stopbits=self.__stopbits,
                xonxoff=self.__xonxoff
            )
        check_connection = self.check_id()
        if not check_connection:
            msg = f"MX200 not found in port {self.__address}"
            raise SerialException(msg)

    def check_id(self, attempt: int = 0) -> bool:
        check_id = self.query('SN')
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
    def units(self, value: str):
        if value in self.units_mapping:
            q = f"W1{value.upper()}"
            # print(q)
            r = self.query(q)
            # print(r)
            # # if r != value:
            # #     print(f'Units {value} could not be set.')

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
        if 1 <= adjustment_point <= 4:
            query: str = f"RC{adjustment_point}{str(channel).zfill(2)}"
            result = self.query(q=query)
            print(result)
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
        print(query)
        self.write(q=query)
        # time.sleep(self.__delay)
        # q = 'S1{0:02d}'.format(channel)
        # pressure = self.query(q)
        # pressure = float(pressure)
        # print(f"Pressure: {pressure:.1e}")
        # return pressure

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

    def write(self, q: str):
        if self.__keep_alive:
            self.__serial.write("{0}\r".format(q).encode('utf-8'))
            sleep(self.__delay)
        else:
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
        if self.__keep_alive:
            self.__serial.write("{0}\r".format(q).encode('utf-8'))
            sleep(self.__delay)
            line = self.__serial.readline()
        else:
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
