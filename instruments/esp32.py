import io
import time
import numpy as np

import serial
from time import sleep
from io import StringIO
import pandas as pd
from serial import SerialException
import socket
import struct


class ArduinoTCP:
    """
    Represents an Arduino or ESP32 device through TCP/IP
    """
    __connection: socket.socket = None
    __ip_address: str = None
    __port: int = 3001

    def __init__(self, ip_address: str, port: int = 3001):
        self.__ip_address = ip_address
        self.__port = port
        self.connect()

    def connect(self):
        self.__connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__connection.connect((self.__ip_address, self.__port))

    def close(self):
        self.disconnect()

    def disconnect(self):
        if self.__connection is not None:
            self.__connection.close()
            self.__connection = None

    def query(self, q: str, attempts=1) -> str:
        try:
            self.__connection.sendall(f"{q}\r".encode('utf-8'))
        except (ConnectionAbortedError, ConnectionResetError) as e:
            self.disconnect()
            self.connect()
            attempts += 1
            if attempts < 5:
                print(e)
                return self.query(q=q, attempts=attempts)
        buffer = b''
        while b'\n' not in buffer:
            data = self.__connection.recv(1024)
            if not data:
                return ''
            buffer += data
        line, sep, buffer = buffer.partition(b'\n')
        return line.decode('utf-8').rstrip("\n").rstrip(" ")

    def query_binary(self, q, attempts: int = 1):
        try:
            self.__connection.sendall(f"{q}\r".encode('utf-8'))
            raw_msg_len = self.__connection.recv(4)
            n = struct.unpack('<I', raw_msg_len)[0]
            raw_msg_cols = self.__connection.recv(4)
            cols = struct.unpack('<I', raw_msg_cols)[0]
            data = bytearray()
            while len(data) < n:
                packet = self.__connection.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            rows = int(n / 4 / cols)
        except (ConnectionError, ConnectionResetError) as e:
            self.disconnect()
            self.connect()
            attempts += 1
            while attempts <= 5:
                print(e)
                return self.query_binary(q, attempts=attempts)
        return rows, cols, data

    def write(self, q: str, attempts: int = 1):
        try:
            self.__connection.sendall(f"{q}\r".encode('utf-8'))
        except (ConnectionAbortedError, ConnectionResetError) as e:
            self.disconnect()
            self.connect()
            attempts += 1
            if attempts <= 5:
                print(e)
                self.write(q=q, attempts=attempts)

    def __del__(self):
        self.disconnect()


class ArduinoTCPLoose:
    """
        Represents an Arduino or ESP32 device through TCP/IP
        """
    __connection: socket.socket = None
    __ip_address: str = None
    __port: int = 3001

    def __init__(self, ip_address: str, port: int = 3001):
        self.__ip_address = ip_address
        self.__port = port

    def connect(self):
        print('Connecting ArduinoTCP will be deprecated.')

    def close(self):
        self.disconnect()

    def disconnect(self):
        print('Disconnecting ArduinoTCP will be deprecated.')

    def query(self, q: str, attempts=1) -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.__ip_address, self.__port))
                s.sendall(f"{q}\r".encode('utf-8'))
                buffer = b''
                while b'\n' not in buffer:
                    data = s.recv(1024)
                    if not data:
                        return ''
                    buffer += data
        except (ConnectionAbortedError, ConnectionResetError) as e:
            attempts += 1
            if attempts < 5:
                print(e)
                return self.query(q=q, attempts=attempts)
        line, sep, buffer = buffer.partition(b'\n')
        return line.decode('utf-8').rstrip("\n").rstrip(" ")

    def query_binary(self, q, attempts: int = 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.__ip_address, self.__port))
                s.sendall(f"{q}\r".encode('utf-8'))
                raw_msg_len = s.recv(4)
                n = struct.unpack('<I', raw_msg_len)[0]
                raw_msg_cols = s.recv(4)
                cols = struct.unpack('<I', raw_msg_cols)[0]
                data = bytearray()
                while len(data) < n:
                    packet = s.recv(n - len(data))
                    if not packet:
                        return None
                    data.extend(packet)
                rows = int(n / 4 / cols) if n > 4 else 1
        except (ConnectionError, ConnectionResetError, struct.error) as e:
            attempts += 1
            if attempts <= 5:
                print(e)
                return self.query_binary(q, attempts=attempts)
            else:
                raise e
        return rows, cols, data

    def write(self, q: str, attempts: int = 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.__ip_address, self.__port))
                s.sendall(f"{q}\r".encode('utf-8'))
        except (ConnectionAbortedError, ConnectionResetError) as e:
            attempts += 1
            if attempts <= 5:
                print(e)
                self.write(q=q, attempts=attempts)



class ArduinoSerial:
    """
    Represents an Arduino or ESP32 Serial device
    """

    __address = None
    __baud_rate = 115200
    __byte_size = serial.EIGHTBITS
    __timeout = 0.1
    __parity = serial.PARITY_NONE
    __stopbits = serial.STOPBITS_ONE
    __xonxoff = 1
    __delay = 0.1

    def __init__(self, address: str):
        self.__address = address
        self.connect()

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

    @property
    def timeout(self):
        return self.__timeout

    @timeout.setter
    def timeout(self, value: float):
        value = abs(float(value))
        self.__timeout = value
        self.__serial.timeout = value

    @property
    def delay(self) -> float:
        return self.__delay

    @delay.setter
    def delay(self, value):
        value = float(value)
        if value > 0:
            self.__delay = value

    def close(self):
        try:
            print(f'Closing serial connection to ESP32 at {self.__address}.')
            self.__serial.close()
        except AttributeError as e:
            print('Connection already closed')

    def __del__(self):
        self.close()

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

    def flush_output(self):
        self.__serial.reset_output_buffer()

    def flush_input(self):
        self.__serial.reset_input_buffer()


class ESP32Trigger(ArduinoSerial):
    __address = 'COM10'
    __pulse_duration_min: float = 40E-6
    __pulse_duration_max: float = 20.0

    def __init__(self, address: str):
        super().__init__(address=address)
        check_connection = self.check_id()
        if not check_connection:
            msg = f"TRIGGER not found in port {self.address}"
            raise SerialException(msg)

    def check_id(self, attempt: int = 0) -> bool:
        time.sleep(0.5)
        old_delay = self.delay
        old_timeout = self.timeout
        self.delay = 0.5
        self.timeout = 0.5
        check_id = self.query('i')
        self.delay = old_delay
        self.timeout = old_timeout
        if check_id != 'TRIGGER':
            if attempt <= 3:
                attempt += 1
                return self.check_id(attempt=attempt)
            else:
                return False
        else:
            return True

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
    __address = 'COM10'

    def __init__(self, address: str):
        super().__init__(address=address)
        check_connection = self.check_id()
        if not check_connection:
            msg = f"EXTRUDER_READOUT not found in port {self.address}"
            raise SerialException(msg)

    def check_id(self, attempt: int = 0) -> bool:
        time.sleep(0.5)
        old_delay = self.delay
        old_timeout = self.timeout
        self.delay = 0.5
        self.timeout = 0.5
        check_id = self.query('i')
        self.delay = old_delay
        self.timeout = old_timeout
        if check_id != 'TCLOGGER':
            if attempt <= 3:
                attempt += 1
                return self.check_id(attempt=attempt)
            else:
                return False
        else:
            return True

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
        return np.array(temp, dtype=np.float64)

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
        if 0.0 > value_in_seconds or value_in_seconds > 120:
            msg = f'Cannot set the log duration to {value_in_seconds}. Value is outside valid range:'
            msg += f'[0, 120] s.'
            raise Warning(msg)
        else:
            interval_ms = value_in_seconds * 1000.0
            q = f't {interval_ms:.0f}'
            self.write(q=q)

    def read_temperature_log(self, attempts=0):
        header_list = ["Time (s)", "TC1 (C)", "TC2 (C)"]
        error_empty = False
        try:
            old_delay = self.delay
            old_timeout = self.timeout
            self.delay = 5.0
            self.timeout = 5.0
            res = self.query('r')
            print(res)
            if (len(res) == 0) or (';' not in res) or res == '':
                print('Error reading the temperatre log. Response:')
                print(res)
                print('Trying again...')
                attempts += 1
                if attempts < 10:
                    return self.read_temperature_log(attempts=attempts)
                else:
                    error_empty = True
            df = pd.read_csv(io.StringIO(res), sep=',', lineterminator=";", names=header_list)
            df = df.apply(pd.to_numeric, errors='coerce')
            df.dropna(inplace=True)
        except ValueError as e:
            print(res, e)
            raise ValueError(e)
        finally:
            self.delay = old_delay
            self.timeout = old_timeout
        if error_empty:
            msg = 'Could not retrieve the temperature log or the response was incomplete:\n'
            msg += res
            raise SerialException(msg)
        return df


class DualTCLoggerTCP(ArduinoTCP):
    __ip_address = '192.168.4.3'

    def __init__(self, ip_address: str = '192.168.4.3'):
        super().__init__(ip_address=ip_address)
        check_connection = self.check_id()
        if not check_connection:
            msg = f"TCLOGGER not found on IP {ip_address}"
            raise ConnectionError(msg)

    def check_id(self, attempt: int = 0) -> bool:
        check_id = self.query('i')
        # self.delay = old_delay
        # self.timeout = old_timeout
        if check_id != 'TCLOGGER':
            if attempt <= 3:
                attempt += 1
                return self.check_id(attempt=attempt)
            else:
                return False
        else:
            return True

    @property
    def temperature(self):
        try:
            _, _, res = self.query_binary('r')
            temp = struct.unpack('<2f', res)
        except AttributeError as e:
            print(res, e)
            raise AttributeError(e)
        except ValueError as e:
            print(res, e)
            raise ValueError(e)
        return np.array(temp, dtype=np.float64)

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
        if 0.0 > value_in_seconds or value_in_seconds > 120:
            msg = f'Cannot set the log duration to {value_in_seconds}. Value is outside valid range:'
            msg += f'[0, 120] s.'
            raise Warning(msg)
        else:
            interval_ms = value_in_seconds * 1000.0
            q = f't {interval_ms:.0f}'
            self.write(q=q)

    def read_temperature_log(self, attempts=0):
        error_empty = False
        try:
            rows, cols, res = self.query_binary('r')
            # print('rows:', rows, 'cols:', cols, 'len(res):', len(res))
            if rows == 0:
                print('Error reading the temperatre log. Response:')
                print(res)
                print('Trying again...')
                attempts += 1
                if attempts < 0:
                    return self.read_temperature_log(attempts=attempts)
                else:
                    error_empty = True
            data = np.frombuffer(res, dtype=np.dtype([('Time (s)', 'f'), ('TC1 (C)', 'f'), ('TC2 (C)', 'f')]))
            df = pd.DataFrame(data=data).apply(pd.to_numeric)
            # df.reset_index(drop=True, inplace=True)
            # df = df.apply(pd.to_numeric, errors='coerce')
        except ValueError as e:
            print(res, e)
            raise ValueError(e)

        if error_empty:
            msg = 'Could not retrieve the temperature log or the response was incomplete:\n'
            msg += res
            self.disconnect()
            raise ConnectionError(msg)
        return df


class ExtruderReadout(ArduinoTCP):
    __ip_address = '192.168.4.2'
    __port = 3001

    def __init__(self, ip_address: str = '192.168.4.2'):
        super().__init__(ip_address=ip_address)
        # self.delay = 0.2
        # self.timeout = 0.2
        # check_connection = self.check_id()
        #
        # if not check_connection:
        #     msg = f"EXTRUDER_READOUT not found in port {self.__address}"
        #     raise SerialException(msg)

    def check_id(self, attempt: int = 0) -> bool:
        # time.sleep(0.5)
        # old_delay = self.delay
        # old_timeout = self.timeout
        # self.delay = 0.5
        # self.timeout = 0.5
        check_id = self.query('i')
        # self.delay = old_delay
        # self.timeout = old_timeout
        if check_id != 'EXTRUDER_READOUT':
            print(f"Error checking id at {self.__ip_address}. Response: '{check_id}'")
            if attempt <= 3:
                attempt += 1
                return self.check_id(attempt=attempt)
            else:
                return False
        else:
            return True

    def get_reading(self, attempts=1):
        try:
            _, _, res = self.query_binary('r')
            # reading = [float(x) for x in res.split(',')]
            result = np.array(list(struct.unpack('<ffflH', res)))
            if np.isnan(result).any():
                attempts += 1
                if attempts <= 5:
                    return self.get_reading(attempts=attempts)
        except AttributeError as e:
            print(res, e)
            raise AttributeError(e)
        except ValueError as e:
            print(res, e)
            raise ValueError(e)
        # time.sleep(0.01)
        return result

    @property
    def reading(self):
        return self.get_reading()

    def zero(self):
        # old_delay = self.delay
        # old_timeout = self.timeout
        # self.delay = 1.0
        # self.timeout = 1.0
        r = self.query('z')
        # sleep(1.0)
        if r == '':
            print('Error taring')
            return self.zero()
        # self.timeout = old_timeout
        # self.delay = old_delay
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
