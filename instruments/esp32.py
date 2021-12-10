import serial
from time import sleep


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
        # sleep(self.__delay)

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
        self.write(f'{q}')
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
            pd = float(res) / 1000
        except AttributeError as e:
            print(res, e)
            raise AttributeError(e)
        except ValueError as e:
            print(res, e)
            raise ValueError(e)
        return pd

    @pulse_duration.setter
    def pulse_duration(self, value_in_seconds):
        value_in_seconds = float(value_in_seconds)
        if self.__pulse_duration_min > value_in_seconds or value_in_seconds > self.__pulse_duration_max:
            msg = f'Cannot set the pulse duration to {value_in_seconds}. Value is outside valid range:'
            msg += f'[{self.__pulse_duration_min:.4g}, {self.__pulse_duration_max:.4g}] s.'
            raise Warning(msg)
        else:
            interval_ms = value_in_seconds * 1000.0
            q = f't {interval_ms:.1f}'
            self.query(q=q)

    def fire(self):
        self.write('f')
