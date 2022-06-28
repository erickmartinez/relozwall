import time
import logging

import numpy as np
import socket
from time import sleep
import re

LASER_IP = "192.168.3.230"
LASER_PORT = 10001

class LaserException(Exception):
    def ___init__(self, message):
        super().___init__(message)


class YLR3000:
    __connection: socket.socket = None
    __ip_address: str = LASER_IP
    __log: logging.Logger = None

    def __init__(self, IP: str = LASER_IP):
        self.__ip_address = IP
        self.__log = logging.getLogger(__name__)
        self.__log.addHandler(logging.NullHandler())
        self.__connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect()
        self.disable_analog_control()
        self.disable_gate_mode()
        self.disable_external_guide_control()
        self.disable_external_control()
        self.enable_modulation()

    @property
    def current_setpoint(self) -> float:
        r = self.query("RCS")
        p = re.compile(r'RCS\:\s*(\d+\.?\d*)')
        m = re.findall(p, r)
        if len(m) > 0:
            return float(r[0])
        msg = rf'Error reading current setpoint. Response: \'{r}\'.'
        raise ValueError(msg)

    @current_setpoint.setter
    def current_setpoint(self, value):
        value = float(value)
        if 0.0 < value < 100.0:
            q = f"SDC {value:.1f}"
            r = self.query(q)
            if r[0:] == 'ERR':
                raise ValueError(r)

    def disable_analog_control(self):
        r = self.query("DEC")
        if r == "DEC":
            self.__log.info("Disabled external control.")
        else:
            msg = "Emission is on!"
            self.__log.error(msg)
            raise LaserException(msg)

    def disable_external_guide_control(self):
        r = self.query("DEABC")
        if r == "DEABC":
            self.__log.info("Disabled external aiming beam control.")
        else:
            msg = "Error disabling external aiming beam control."
            self.__log.error(msg)
            raise LaserException(msg)

    def disable_gate_mode(self):
        r = self.query("DGM")
        if r == "DGM":
            self.__log.info("Disabled gate mode.")
        else:
            msg = "Emission is on!"
            self.__log.error(msg)
            raise LaserException(msg)

    def disable_external_control(self):
        r = self.query("DLE")
        if r == "DLE":
            self.__log.info("Disabled hardware emission control.")
        else:
            msg = "Emission is on!"
            self.__log.error(msg)
            raise LaserException(msg)

    def enable_modulation(self):
        r = self.query("EMOD")
        if r == "EMOD":
            self.__log.info("Enabled modulation mode.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.__log.error(msg)
            raise LaserException(msg)
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise LaserException(msg)

    def emission_off(self, attempts=1):
        r = self.query("EMOFF")
        if r == "EMOFF":
            self.__log.info("Stopped emission.")
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise LaserException(msg)

    def emission_on(self):
        r = self.query("EMON")
        if r == "EMON":
            self.__log.info("Started emission.")
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise LaserException(msg)

    def read_extended_device_status(self):
        r = self.query("EPM")

        if r == "EPM":
            self.__log.info("Enabled PULSE Mode.")
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def connect(self):
        self.__connection.connect((self.__ip_address, LASER_PORT))

    def disconnect(self):
        if self.__connection is not None:
            self.__connection.close()

    def query(self, q: str, attempts=1) -> str:
        try:
            self.__connection.sendall(f"{q}\r".encode('utf-8'))
        except ConnectionAbortedError as e:
            self.__connection.close()
            self.__connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__connection.connect((self.__ip_address, LASER_PORT))
            attempts += 1
            if attempts <= 5:
                print(e)
                return self.query(q=q, attempts=attempts)

        line = self.__connection.recv(1024)
        return line.decode('utf-8').rstrip("\r").rstrip(" ")

    def __del__(self):
        self.disconnect()
