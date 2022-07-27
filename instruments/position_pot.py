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
    __baud_rate = 57600
    __byte_size = serial.EIGHTBITS
    __timeout = 0.1
    __parity = serial.PARITY_NONE
    __stopbits = serial.STOPBITS_ONE
    __xonxoff = 1
    __delay = 0.1
    __serial: serial.Serial = None

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
        time.sleep(self.__delay)

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
            self.__serial.flush()
            self.__serial.close()
        except AttributeError as e:
            print('Connection already closed')

    def write(self, q: str):
        self.__serial.write(f'{q}\r'.encode('utf-8'))
        time.sleep(self.__delay)

    def query(self, q: str) -> str:
        self.write(f"{q}")
        time.sleep(self.__delay)
        line = self.__serial.readline()
        time.sleep(self.__delay)
        return line.decode('utf-8').rstrip("\n").rstrip(" ")

    def query_binary(self, q, attempts: int = 1):
        data = bytearray()
        try:
            self.write(f"{q}")
            raw_msg_len = self.__serial.read(4)
            n = struct.unpack('<I', raw_msg_len)[0]
            while len(data) < n:
                packet = self.__serial.read(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
        except (ConnectionError, ConnectionResetError) as e:
            self.disconnect()
            self.connect()
            attempts += 1
            while attempts <= 5:
                print(e)
                return self.query_binary(q, attempts=attempts)
        return data

    def __del__(self):
        self.close()


class DeflectionReader(ArduinoSerial):
    def __init__(self, address: str):
        super().__init__(address=address)
        check_connection = self.check_id()
        if not check_connection:
            msg = f"DEFLECTION_POT not found in port {self.__address}"
            raise SerialException(msg)

    def check_id(self, attempt: int = 0) -> bool:
        time.sleep(0.25)
        old_delay = self.delay
        old_timeout = self.timeout
        self.delay = 0.25
        self.timeout = 0.25
        check_id = self.query('i')
        self.delay = old_delay
        self.timeout = old_timeout
        if check_id != 'DEFLECTION_POT':
            if attempt <= 3:
                attempt += 1
                return self.check_id(attempt=attempt)
            else:
                return False
        else:
            return True

    def reset_clock(self):
        self.write('z')

    @property
    def reading(self):
        res = self.query_binary('r')
        (dt_ms, adc) = np.array(list(struct.unpack('<LH', res)))
        return np.round(dt_ms*0.001,decimals=3), adc

    def __del__(self):
        super(DeflectionReader, self).__del__()
