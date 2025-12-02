import os.path
import sys
import time
from pathlib import Path
from typing import Callable, Optional
import logging
import serial
import serial.tools.list_ports


class BaseSerial:
    """
    Represents Serial device
    """

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
            "timeout": 0.1,
            "write_timeout": 0.1,
        }

        self._id_validation_query = None
        self._valid_id_specific: Optional[str] = None
        self.path: Path = Path(__file__).parent / 'config' / f'{self.name}_port.txt'
        self._delay = 0.1

    def set_logger(self, logger: logging.Logger):
        self._logger = logger

    def set_id_validation_query(
            self, id_validation_query: Callable[[], str], valid_id_specific: object
    ):
        self._id_validation_query = id_validation_query
        self._valid_id_specific = valid_id_specific

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
            time.sleep(self._delay)  # Wait for DTR reset/bootloader
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


    def log(self, msg: str, level: int = logging.INFO):
        if self._logger is not None:
            self._logger.log(level=level, msg=msg)
        else:
            print(f"[{level}] {msg}")
