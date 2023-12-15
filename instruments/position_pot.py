import logging
import struct
import numpy as np
import serial
import time

from serial import SerialException


class ArduinoSerial:
    """
    Represents an Arduino or ESP32 Serial device
    """

    __address = None
    __baud_rate = 115200 # 57600 # 19200  # 38400
    __byte_size = serial.EIGHTBITS
    __timeout = 0.02
    __parity = serial.PARITY_NONE
    __stopbits = serial.STOPBITS_ONE
    __xonxoff = 0
    __delay = 0.01
    __serial: serial.Serial = None
    _log: logging.Logger = None
    _previous_val: int = 0

    def __init__(self, address: str):
        self.__address = address
        self.connect()
        self._log = logging.getLogger(__name__)
        self._log.addHandler(logging.NullHandler())
        # create console handler and set level to debug
        has_console_handler = False
        if len(self._log.handlers) > 0:
            for handler in self._log.handlers:
                if isinstance(handler, logging.StreamHandler):
                    has_console_handler = True

        if not has_console_handler:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            self._log.addHandler(ch)

    def logger(self) -> logging.Logger:
        return self._log

    def set_logger(self, log: logging.Logger):
        self._log = log

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
        time.sleep(0.5)

    def reset_serial(self):
        self.__serial.close()
        self.__serial = serial.Serial(
            port=self.__address,
            baudrate=self.__baud_rate,
            bytesize=self.__byte_size,
            timeout=self.__timeout,
            parity=self.__parity,
            stopbits=self.__stopbits,
            xonxoff=self.__xonxoff
        )


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
            self._log.info(f'Closing serial connection to Arduino at {self.__address}.')
            self.__serial.close()
        except AttributeError as e:
            if self._log is not None:
                self._log.warning('Connection already closed')

    def write(self, q: str):
        self.__serial.write(f'{q}\r'.encode('utf-8'))
        # time.sleep(self.__delay)

    def reset_input_buffer(self):
        self.__serial.reset_input_buffer()

    def reset_output_buffer(self):
        self.__serial.reset_output_buffer()


    def flush(self):
        self.__serial.flush()

    def query(self, q: str) -> str:
        self.write(f"{q}")
        # time.sleep(self.__delay)
        line = self.__serial.readline()
        # time.sleep(self.__delay)
        return line.decode('utf-8').rstrip("\n").rstrip(" ")

    def query_binary(self, q, packets: bool = False, size: int = 2):
        data = bytearray()
        self.write(f"{q}")
        if packets:
            raw_msg_len = self.__serial.read(4)
            n = struct.unpack('<I', raw_msg_len)[0]
            while len(data) < n:
                packet = self.__serial.read(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
        else:
            data = self.__serial.read(size)
            # self.__serial.flush()
            # If there is unread data in the input buffer, discard it
            # in_waiting = self.__serial.in_waiting
            # if in_waiting > 0:
            #     self.reset_input_buffer()

        return data

    def __del__(self):
        try:
            self.close()
        except SerialException as e:
            self._log.error(e)


class DeflectionReader(ArduinoSerial):
    def __init__(self, address: str):
        super().__init__(address=address)
        time.sleep(1.0)
        if not self.check_id():
            msg = f"Could not find deflection pot in {address}."
            self.close()
            raise SerialException(msg)
        time.sleep(0.5)
        self._log.info(f'Start reading: {self.reading}')

    def check_id(self, attempt: int = 0) -> bool:
        # old_delay = self.delay
        # old_timeout = self.timeout
        # self.delay = 0.1
        # self.timeout = 0.3
        # time.sleep(0.3)
        check_id = self.query('i')
        # self.delay = old_delay
        # self.timeout = old_timeout
        if not 'DEFLECTION_POT' in check_id:
            self._log.error(f'Error checking id. Response: \'{check_id}\'')
            if attempt < 3:
                attempt += 1
                # self.reset_input_buffer()
                return self.check_id(attempt=attempt)
            else:
                return False
        else:
            return True

    def get_reading(self, attempts=0) -> int:
        res = self.query_binary('r', size=2)
        if len(res) == 0:
            self.reset_serial()
            self._log.warning(f'Failed reading position. Returning previous value...')
            return self._previous_val
        if len(res) < 2:
            self._log.warning(
                f'Received: \"{res}\".')
            adc = int.from_bytes(res, byteorder="little", signed=False)
        else:
            adc = struct.unpack('<H', res)[0]
        self._previous_val = adc
        return adc

    @property
    def reading(self):
        return self.get_reading()
