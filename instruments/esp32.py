import io
import time
import numpy as np

import serial
from time import sleep
from io import StringIO
import pandas as pd
from serial import SerialException


class ArduinoSerial:
    """
    Represents an Arduino or ESP32 Serial device
    """

    __address = None
    __baud_rate = 115200
    __byte_size = serial.EIGHTBITS
    __timeout = 0.5
    __parity = serial.PARITY_NONE
    __stopbits = serial.STOPBITS_ONE
    __xonxoff = 1
    __delay = 0.1

    def __init__(self, address: str):
        self.__address = address
        self.__serial = serial.Serial(
            port=self.__address,
            baudrate=self.__baud_rate,
            bytesize=self.__byte_size,
            timeout=self.__timeout,
            parity=self.__parity,
            stopbits=self.__stopbits,
            xonxoff=self.__xonxoff
        )

    def close(self):
        try:
            print(f'Closing serial connection to ESP32 at {self.__address}.')
            self.__serial.close()
        except AttributeError as e:
            print('Connection already closed')

    def __del__(self):
        try:
            print(f'Closing serial connection to ESP32 at {self.__address}.')
            self.__serial.close()
        except AttributeError as e:
            print('Connection already closed')

    def write(self, q: str):
        # with serial.Serial(
        #         port=self.__address,
        #         baudrate=self.__baud_rate,
        #         bytesize=self.__byte_size,
        #         timeout=self.__timeout,
        #         parity=self.__parity,
        #         stopbits=self.__stopbits,
        #         xonxoff=self.__xonxoff
        # ) as ser:
        # sleep(self.__delay)
        self.__serial.write(f'{q}\r'.encode('utf-8'))
        sleep(self.__delay)

    def query(self, q: str) -> str:
        # with serial.Serial(
        #         port=self.__address,
        #         baudrate=self.__baud_rate,
        #         bytesize=self.__byte_size,
        #         timeout=self.__timeout,
        #         parity=self.__parity,
        #         stopbits=self.__stopbits,
        #         xonxoff=self.__xonxoff
        # ) as ser:
        # sleep(self.__delay)
        self.write(f"{q}")
        sleep(self.__delay)
        line = self.__serial.readline()
        sleep(self.__delay)
        return line.decode('utf-8').rstrip("\n").rstrip(" ")


class ESP32Trigger(ArduinoSerial):
    __address = 'COM10'
    __pulse_duration_min = 40E-6
    __pulse_duration_max = 3

    def __init__(self, address: str):
        super().__init__(address=address)

    @property
    def pulse_duration(self) -> float:
        """
        Queries the pulse length in the microcontroller

        Returns
        -------
        float:
            The pulse duration in seconds
        """
        try:
            res = self.query('t?')
            pulse_duration = float(res) / 1000.0
        except AttributeError as e:
            print(res, e)
            raise AttributeError(e)
        except ValueError as e:
            print(res, e)
            raise ValueError(e)
        return pulse_duration

    @pulse_duration.setter
    def pulse_duration(self, value_in_seconds):
        value_in_seconds = float(value_in_seconds)
        if self.__pulse_duration_min > value_in_seconds or value_in_seconds > self.__pulse_duration_max:
            msg = f'Cannot set the pulse duration to {value_in_seconds}. Value is outside valid range:'
            msg += f'[{self.__pulse_duration_min:.4g}, {self.__pulse_duration_max:.4g}] s.'
            raise Warning(msg)
        else:
            interval_ms = value_in_seconds * 1000.0
            q = f't {interval_ms:.0f}'
            self.query(q=q)

    def fire(self):
        self.write('f')


class DualTCLogger(ArduinoSerial):
    __address = 'COM7'

    def __init__(self, address: str):
        super().__init__(address=address)

    @property
    def temperature(self):
        try:
            res = self.query('r')
            temp = [float(x) for x in res.split(',')]
        except AttributeError as e:
            print(res, e)
            raise AttributeError(e)
        except ValueError as e:
            print(res, e)
            raise ValueError(e)
        return temp

    def start_logging(self):
        self.write('l')

    @property
    def log_time(self):
        try:
            res = self.query('t?')
            log_time = float(res) / 1000.0
        except AttributeError as e:
            print(res, e)
            raise AttributeError(e)
        except ValueError as e:
            print(res, e)
            raise ValueError(e)
        return log_time

    @log_time.setter
    def log_time(self, value_in_seconds: float):
        value_in_seconds = float(value_in_seconds)
        if 0.0 > value_in_seconds or value_in_seconds > 20:
            msg = f'Cannot set the log duration to {value_in_seconds}. Value is outside valid range:'
            msg += f'[0, 20] s.'
            raise Warning(msg)
        else:
            interval_ms = value_in_seconds * 1000.0
            q = f't {interval_ms:.0f}'
            self.query(q=q)

    def read_temperature_log(self):
        header_list = ["Time (s)", "TC1 (C)", "TC2 (C)"]
        try:
            res = self.query('r')
            df = pd.read_csv(io.StringIO(res), sep=',', lineterminator=";", names=header_list)
            df = df.apply(pd.to_numeric, errors='coerce')
            df.dropna(inplace=True)
        except AttributeError as e:
            print(res, e)
            raise AttributeError(e)
        except ValueError as e:
            print(res, e)
            raise ValueError(e)

        return df


class ExtruderReadout(ArduinoSerial):
    __address = 'COM12'

    def __init__(self, address: str):
        super().__init__(address=address)
        check_connection = self.check_id()
        if not check_connection:
            msg = f"EXTRUDER_READOUT not found in port {self.address}"
            raise SerialException(msg)

    def check_id(self, attempt: int = 0) -> bool:
        check_id = self.query('i')
        if check_id != 'EXTRUDER_READOUT':
            if attempt <= 3:
                attempt += 1
                return self.check_id(attempt=attempt)
            else:
                return False
        else:
            return True

    @property
    def reading(self):
        try:
            res = self.query('r')
            reading = [float(x) for x in res.split(',')]
        except AttributeError as e:
            print(res, e)
            raise AttributeError(e)
        except ValueError as e:
            print(res, e)
            raise ValueError(e)
        return reading

    def zero(self):
        r = self.query('z')
        return r

    @property
    def calibration_factor(self):
        try:
            res = self.query('c?')
            cf = float(res)
        except AttributeError as e:
            print(res, e)
            raise AttributeError(e)
        except ValueError as e:
            print(res, e)
            raise ValueError(e)
        return cf

    @calibration_factor.setter
    def calibration_factor(self, value: float):
        value = float(value)
        if value == 0.0:
            msg = f'Cannot set the log duration to {value}.'
            raise Warning(msg)
        else:
            q = f'c {value:10.3E}'
            self.write(q=q)
