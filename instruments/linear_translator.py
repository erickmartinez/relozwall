import serial
from time import sleep


class LinearTranslator:
    """
    Represents the Linear Translator used in the Extrusion system

    Attributes
    ----------
    __address: str
        The address at which the gauge is located
    """

    __address = 'COM6'
    __baud_rate = 57600
    __byte_size = serial.EIGHTBITS
    __timeout = 5
    __parity = serial.PARITY_NONE
    __stopbits = serial.STOPBITS_ONE
    __xonxoff = 1
    __delay = 0.05
    __serial: serial.Serial = None

    def __init__(self, address: str):
        self.__address = address
        self.connect()

    @property
    def address(self) -> str:
        return self.__address

    def move_forward(self):
        msg = self.query('f')
        print(msg)
        return msg

    def move_backward(self):
        msg = self.query('r')
        print(msg)
        return msg

    @property
    def status(self):
        msg = self.query('s')
        print(msg)
        return msg

    def move_steps(self, steps: int):
        steps = int(steps)
        msg = self.query('m')
        #print(msg)
        msg = self.query(f'{steps}')
        print(msg)
        return msg

    def stop(self):
        msg = self.query(' ')
        print(msg)
        return msg

    def write(self, q: str):
        self.__serial.write(bytes(q, 'utf-8'))
        sleep(self.__delay)

    def write_eol(self, q: str):
        self.__serial.write("{0}\r".format(q).encode('utf-8'))
        sleep(self.__delay)

    def read(self) -> str:
        line = self.__serial.readline()
        # ser.flush()
        return line.decode('utf-8').rstrip("\n").rstrip(" ")

    def query(self, q: str) -> str:
        self.__serial.write(bytes(q, 'utf-8'))
        sleep(self.__delay)
        return self.read()

    def query_eol(self, q: str) -> str:
        self.__serial.write("{0}\r".format(q).encode('utf-8'))
        sleep(self.__delay)
        return self.read()

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
        self.__serial.flush()

    def disconnect(self):
        if self.__serial is not None:
            self.__serial.close()
            self.__serial = None

    def __del__(self):
        self.disconnect()
