import os.path
import sys
import time
from pathlib import Path
from typing import Callable

import serial
import serial.tools.list_ports


class BaseSerial:
    """
    Represents Serial device
    """
    _delay = 0.1

    def __init__(self, name='DEV_1'):
        self.name = name
        self._serial: serial.Serial = None
        self._id_validation_query = None
        self._valid_id_specific = None
        self.path: Path = Path(os.path.join(os.path.dirname(__file__), 'config', self.name + '_port.txt'))
        self._delay = 0.1
        self._serial_settings = {
            "baudrate": 115200,
            "bytesize": serial.EIGHTBITS,
            "parity": serial.PARITY_NONE,
            "stopbits": serial.STOPBITS_ONE,
            "xonxoff": False,
            "rtscts": False,
            "dsrdtr": False,
            "exclusive": None,
            "timeout": 0.2,
            "write_timeout": 0.1,
        }

    def set_id_validation_query(
            self, id_validation_query: Callable[[], str], valid_id_specific: object
    ):
        self._id_validation_query = id_validation_query
        self._valid_id_specific = valid_id_specific

    def connect_at_port(self, port: str, verbose=False):
        if verbose:
            print(f"Connecting to '{self.name}' at port '{port}'.")
        try:
            self._serial = serial.Serial(
                port=port,
                **self._serial_settings
            )
            time.sleep(self._delay)
        except serial.SerialException as e:
            print(f"Could not open port {port}.")
            print(f"\t{e}")
            return False
        except Exception as err:
            print(err)
            sys.exit(0)
        if self._id_validation_query is None:
            print(f"Serial connection success!")
            return True

        try:
            reply = self._id_validation_query()
            if reply == self._valid_id_specific:
                print(f"Found '{self.name}' at port '{port}'")
                return True
            else:
                print(f"'{self.name}' not found in port '{port}'")
        except Exception as e:
            print(f"'{self.name}' not found in port '{port}'")

        print("Wrong device.")
        self.close()
        return False

    def scan_ports(self, verbose: bool = False) -> bool:
        if verbose:
            print(f"Scanning ports for '{self.name}'")
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            port = p[0]
            if self.connect_at_port(port):
                return True
            else:
                continue

        print(f"  Error: device '{self.name}' not found.")
        return False

    def auto_connect(self):
        port = self._get_last_known_port()
        if port is None:
            if self.scan_ports():
                self._store_last_known_port(port_str=self._serial.portstr)
                return True
            return False
        if self.connect_at_port(port):
            self._store_last_known_port(self._serial.portstr)
            return True
        # if self.scan_ports():
        #     self._store_last_known_port(self._serial.portstr)
        #     return True

        return False

    def _get_last_known_port(self):
        if isinstance(self.path, Path):
            if self.path.is_file():
                try:
                    with self.path.open() as f:
                        port = f.readline().strip()
                    return port
                except Exception as e:
                    print(e)
                    pass  # Do not panic and remain silent
        return None

    def _store_last_known_port(self, port_str):
        if isinstance(self.path, Path):
            if not self.path.parent.is_dir():
                try:
                    self.path.parent.mkdir()
                except Exception:
                    pass  # Do not panic and remain silent

            try:
                # Write the cnonfig file
                self.path.write_text(port_str)
            except Exception:
                pass  # Do not panic and remain silent

            return True
        return False

    @property
    def timeout(self):
        return self._serial_settings['timeout']

    @timeout.setter
    def timeout(self, value: float):
        value = abs(float(value))
        self._serial_settings['timeout'] = value
        self._serial.timeout = value

    @property
    def delay(self) -> float:
        return self._delay

    @delay.setter
    def delay(self, value):
        value = float(value)
        if value > 0:
            self._delay = value

    def close(self):
        if self._serial is not None:
            try:
                self._serial.cancel_read()
            except Exception:
                pass
            try:
                self._serial.cancel_write()
            except Exception:
                pass

            try:
                self._serial.close()
                print(f'Closed serial connection to \'{self.name}\'.')
            except AttributeError as e:
                print(e)
            except Exception as e:
                print(e)

    def __del__(self):
        self.close()

    def write(self, q: str):
        self._serial.write(f'{q}\r'.encode('utf-8'))
        # time.sleep(self._delay)

    def query(self, q: str) -> str:
        self.write(f"{q}")
        time.sleep(self._delay)
        line = self._serial.readline()
        time.sleep(self._delay)
        return line.decode('utf-8').rstrip("\n").rstrip(" ")

    def flush_output(self):
        self._serial.reset_output_buffer()

    def flush_input(self):
        self._serial.reset_input_buffer()
