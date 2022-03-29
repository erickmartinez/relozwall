import time

import numpy as np
import serial
from time import sleep
import re

class DCSource:
    """
        Represents the Televac MX200 Controller
        """

    __address = 'COM13'
    __baud_rate = 115200
    __byte_size = serial.EIGHTBITS
    __timeout = 0.1
    __parity = serial.PARITY_NONE
    __stopbits = serial.STOPBITS_ONE
    __xonxoff = 1
    __delay = 0.1

    def __init__(self, address: str):
        self.__address = address

    def write(self, q: str):
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
            ser.write(f"{q}\r\n".encode('utf-8'))
            sleep(self.__delay)

    def query(self, q: str) -> str:
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
            ser.write("{0}\r\n".format(q).encode('utf-8'))
            sleep(self.__delay)
            line = ser.readline()
            return line.decode('utf-8').rstrip("\r\n").rstrip(" ")