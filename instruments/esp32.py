import io
import os.path
import socket
import struct
import sys
import time
from time import sleep
import logging
from typing import Callable, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import serial
from serial import SerialException
import serial.tools.list_ports


class ArduinoTCP:
    """
    Represents an Arduino or ESP32 device through TCP/IP
    """
    __connection: socket.socket = None
    __ip_address: str = None
    __port: int = 3001
    _log: logging.Logger = None

    def __init__(self, ip_address: str, port: int = 3001):
        self.__ip_address = ip_address
        self.__port = port
        self.connect()
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

    def connect(self):
        self.__connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__connection.connect((self.__ip_address, self.__port))

    def close(self):
        self.disconnect()

    def disconnect(self):
        if self.__connection is not None:
            self.__connection.close()
            self.__connection = None

    def set_logger(self, log: logging.Logger):
        self._log = log

    def query(self, q: str, attempts=1) -> str:
        try:
            self.__connection.sendall(f"{q}\r".encode('utf-8'))
        except (ConnectionAbortedError, ConnectionResetError) as e:
            self.disconnect()
            self.connect()
            attempts += 1
            if attempts < 5:
                logging.warning(e)
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
                logging.warning(e)
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
                logging.warning(e)
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
        # https://stackoverflow.com/questions/17667903/python-socket-receive-large-amount-of-data
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

    _baud_rate = 115200
    _byte_size = serial.EIGHTBITS
    _timeout = 0.1
    _parity = serial.PARITY_NONE
    _stopbits = serial.STOPBITS_ONE
    _xonxoff = 1

    def __init__(self, name='DEV_1'):
        self.name = name
        self._serial: Optional[serial.Serial] = None
        self._logger: Optional[logging.Logger] = None
        self._serial_settings = {
            "baudrate": 115200,
            "bytesize": serial.EIGHTBITS,
            "parity": serial.PARITY_NONE,
            "stopbits": serial.STOPBITS_ONE,
            "xonxoff": False,
            "rtscts": False,
            "dsrdtr": False,
            # "exclusive": None,
            "timeout": 1.,
            "write_timeout": 1.,
        }
        self._id_validation_query = None
        self._valid_id_specific: Optional[str] = None
        self.path: Path = Path(__file__).parent / 'config' /f'{self.name}_port.txt'
        self._delay = 0.1

    def set_logger(self, logger: logging.Logger):
        self._logger = logger

    def log(self, msg: str, level: int = logging.INFO):
        if self._logger is not None:
            self._logger.log(level=level, msg=msg)
        else:
            print(f"[{level}] {msg}")

    def set_id_validation_query(
            self, id_validation_query: Callable[[], str], valid_id_specific: object
    ):
        self._id_validation_query: object = id_validation_query
        self._valid_id_specific: object = valid_id_specific

    def connect_at_port(self, port: str, verbose=False):
        if verbose:
            self.log(f"Connecting to '{self.name}' at port '{port}'.", level=logging.DEBUG)
        try:
            # Check if already open and close it to be safe
            if self._serial and self._serial.is_open:
                self._serial.close()

            self._serial = serial.Serial(
                port=port,
                **self._serial_settings
            )
            time.sleep(self._delay) # Wait for DTR reset/bootloader
        except serial.SerialException as e:
            self.log(f"Could not open port {port}.", level=logging.ERROR)
            return False
        except Exception as err:
            self.log(f"Unexpected error opening port: {err}", level=logging.ERROR)
            raise err
        # If no validation required, we are done
        if self._id_validation_query is None:
            self.log(f"Serial connection success!", level=logging.INFO)
            return True

        # Validation Logic
        try:
            # Flush buffers before validation to ensure clean state
            self.flush_input()
            reply = self._id_validation_query()
            if reply == self._valid_id_specific:
                self.log(f"Found '{self._valid_id_specific}' at port '{port}'", level=logging.INFO)
                return True
            else:
                self.log(f"Device at {port} replied '{reply}', expected '{self._valid_id_specific}'",
                         level=logging.WARNING)
        except Exception as e:
            self.log(f"Validation failed at '{port}': {e}", level=logging.ERROR)

        self.log("Wrong device or validation failed.", level=logging.ERROR)
        self.close()
        return False

    def scan_ports(self, verbose: bool = False) -> bool:
        if verbose:
            self.log(f"Scanning ports for '{self.name}'", level=logging.INFO)
        ports = serial.tools.list_ports.comports()
        for p in ports:
            # p.device contains the COM port string (e.g., 'COM3' or '/dev/ttyUSB0')
            if self.connect_at_port(p.device):
                return True

        self.log(f"Error: device '{self.name}' not found in any port.", level=logging.ERROR)
        return False

    def auto_connect(self):
        # 1. Try Last Known Port
        last_port = self._get_last_known_port()
        if last_port:
            self.log(f"Attempting last known port: {last_port}", level=logging.DEBUG)
            if self.connect_at_port(last_port):
                # Re-save to ensure timestamp/file validity if needed
                self._store_last_known_port(self._serial.port)
                return True
        # 2. Scan all ports if last known failed
        if self.scan_ports():
            self._store_last_known_port(self._serial.port)
            return True

        return False

    def _get_last_known_port(self):
        if self.path.is_file():
            try:
                return self.path.read_text().strip()
            except Exception as e:
                self.log(f"Failed to read config file: {e}", level=logging.WARNING)
        return None

    def _store_last_known_port(self, port_str):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(port_str)
            return True
        except Exception as e:
            self.log(f"Failed to write config file: {e}", level=logging.WARNING)
            return False

    @property
    def timeout(self):
        return self._serial_settings['timeout']

    @timeout.setter
    def timeout(self, value: float):
        value = abs(float(value))
        self._serial_settings['timeout'] = value
        if self._serial:
            self._serial.timeout = value

    @property
    def delay(self) -> float:
        return self._delay

    @delay.setter
    def delay(self, value):
        self._delay = max(0.0, float(value))

    def close(self):
        if self._serial and self._serial.is_open:
            try:
                self._serial.cancel_read()
                self._serial.cancel_write()
                self._serial.close()
                self.log(f"Closed serial connection to {self.name}.", level=logging.INFO)
            except Exception as e:
                self.log(f"Error closing serial: {e}", level=logging.ERROR)

    def __del__(self):
        self.close()

    def write(self, q: str):
        if not self._serial or not self._serial.is_open:
            raise serial.SerialException("Attempted write on closed port")
        try:
            self._serial.write(f'{q}\r'.encode('utf-8'))
        except serial.SerialTimeoutException:
            self.log("Write timeout", level=logging.ERROR)

    def query(self, q: str) -> str:
        self.write(q)
        try:
            line = self._serial.readline()
            return line.decode('utf-8').strip()
        except UnicodeDecodeError:
            self.log("Decode error in query response", level=logging.WARNING)
            return ""

    def flush_output(self):
        self._serial.reset_output_buffer()

    def flush_input(self):
        self._serial.reset_input_buffer()


class ESP32Trigger(ArduinoSerial):
    __pulse_duration_min: float = 40E-6
    __pulse_duration_max: float = 20.0

    def __init__(self, logger: logging.Logger = None):
        super().__init__(name='ARD_TRIGGER')

        # Configure validation
        self.set_id_validation_query(
            id_validation_query=self.id_validation_query,
            valid_id_specific='TRIGGER'
        )

        if logger:
            self.set_logger(logger)

        # Attempt connection immediately
        if not self.auto_connect():
            self.log("ESP32Trigger failed to auto-connect on initialization.", level=logging.WARNING)

    def id_validation_query(self) -> str:
        # We can temporarily tighten timeout for validation to speed up scanning
        old_timeout = self.timeout
        self.timeout = 0.5
        try:
            response = self.query('i')
        finally:
            self.timeout = old_timeout  # Always restore timeout
        return response

    def check_id(self, max_attempts: int = 3) -> bool:
        """Verifies the device ID with retries."""
        for attempt in range(max_attempts + 1):
            time.sleep(0.5)
            # Temporarily adjust settings for the check
            old_delay = self.delay
            old_timeout = self.timeout
            self.delay = 0.5
            self.timeout = 0.5

            try:
                check_id = self.query('i')
            except Exception:
                check_id = None
            finally:
                self.delay = old_delay
                self.timeout = old_timeout

            if check_id == 'TRIGGER':
                return True

            if attempt < max_attempts:
                self.log(f"Check ID failed (got '{check_id}'). Retrying...", level=logging.DEBUG)

        return False

    @property
    def pulse_duration(self) -> float:
        """
        Queries the pulse length in the microcontroller

        Returns
        -------
        float:
            The pulse duration in seconds
        """
        res = self.query('t?')
        try:
            # Assuming response is milliseconds string
            return float(res) / 1000.0
        except ValueError as e:
            self.log(f"Invalid pulse duration response: '{res}'", level=logging.ERROR)
            raise e

    @pulse_duration.setter
    def pulse_duration(self, value_in_seconds):
        value_in_seconds = float(value_in_seconds)
        if not (self.__pulse_duration_min <= value_in_seconds <= self.__pulse_duration_max):
            msg = (f"Pulse duration {value_in_seconds:.4g}s outside valid range "
                   f"[{self.__pulse_duration_min:.4g}, {self.__pulse_duration_max:.4g}]s.")
            raise Warning(msg)
        else:
            interval_ms = value_in_seconds * 1000.0
            q = f't {interval_ms:.0f}'
            self.query(q=q)

    def fire(self):
        self.write('f')


class DualTCLogger(ArduinoSerial):
    def __init__(self):
        super().__init__(name='DEV_TCLOGGER')
        self.set_id_validation_query(
            id_validation_query=self.id_validation_query,
            valid_id_specific='TRIGGER'
        )
        self.auto_connect()
        # check_connection = self.check_id()
        # if not check_connection:
        #     msg = f"EXTRUDER_READOUT not found in port {self.address}"
        #     raise SerialException(msg)

    def id_validation_query(self) -> str:
        # time.sleep(0.5)
        # old_delay = self.delay
        # old_timeout = self.timeout
        # self.delay = 0.5
        # self.timeout = 0.5
        response = self.query('i')
        # self.delay = old_delay
        # self.timeout = old_timeout
        return response

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
        if 0.0 > value_in_seconds or value_in_seconds > 350.0:
            msg = f'Cannot set the log duration to {value_in_seconds}. Value is outside valid range:'
            msg += f'[0, 350] s.'
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
            self._log.error(res, e)
            raise AttributeError(e)
        except ValueError as e:
            self._log.error(res, e)
            raise ValueError(e)
        return log_time

    @log_time.setter
    def log_time(self, value_in_seconds: float):
        value_in_seconds = float(value_in_seconds)
        if 0.0 > value_in_seconds or value_in_seconds > 350.:
            msg = f'Cannot set the log duration to {value_in_seconds}. Value is outside valid range:'
            msg += f'[0, 350] s.'
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
                self._log.warning('Error reading the temperatre log. Response:')
                self._log.warning(res)
                self._log.warning('Trying again...')
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
            self._log.error(res, e)
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
            self._log.warning(f"Error checking id at {self.__ip_address}. Response: '{check_id}'")
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
                print(f'Value error received in extrusion readout.')
                attempts += 1
                if attempts <= 5:
                    return self.get_reading(attempts=attempts)
        except AttributeError as e:
            self._log.error(res, e)
            raise AttributeError(e)
        except ValueError as e:
            self._log.error(res, e)
            raise ValueError(e)
        # time.sleep(0.01)
        return result

    @property
    def reading(self):
        return self.get_reading()

    def averaged_reading(self, n=3):
        if n < 1:
            return self.reading
        sum_r = np.zeros(5, dtype=float)
        for i in range(n):
            r = self.reading
            sum_r += r
        inv_n = 1. / n
        return inv_n * sum_r

    def zero(self):
        # old_delay = self.delay
        # old_timeout = self.timeout
        # self.delay = 1.0
        # self.timeout = 1.0
        r = self.query('z')
        # sleep(1.0)
        if r == '':
            self._log.warning('Error taring. Trying again..')
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
            self._log.error(res, e)
            raise AttributeError(e)
        except ValueError as e:
            self._log.error(res, e)
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
