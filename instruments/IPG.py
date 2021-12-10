import time
import logging


import numpy as np
import serial
from time import sleep
import re

EPM_PATTERN = re.compile(r'EPM:(\d+),((\d{2})\=(.*)')


class YLR3000:

    __address: str = 'COM10'
    __baud_rate = 57600
    __byte_size = serial.EIGHTBITS
    __timeout = 0
    __parity = serial.PARITY_NONE
    __stopbits = serial.STOPBITS_ONE
    __xonxoff = 1
    __delay = 0.1
    __log: logging.Logger = None

    def __init__(self, address: str):
        self.__address = address
        self.__log = logging.getLogger(__name__)
        self.__log.addHandler(logging.NullHandler())

    def aiming_beam_on(self):
        r = self.query("ABN")
        if r == "ABN":
            self.__log.info("Enabled laser guide.")
        elif r == "ERR":
            msg = "Cannot enable guide beam because external guide control is enabled."
            self.__log.error(msg)
            raise ResourceWarning(msg)
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def aiming_beam_off(self):
        r = self.query("ABF")
        if r == "ABN":
            self.__log.info("Disabled laser guide.")
        elif r == "ERR":
            msg = "Cannot disable  guide beam because external guide control is enabled."
            self.__log.error(msg)
            raise ResourceWarning(msg)
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def disable_external_aiming_beam_control(self):
        r = self.query("DEABC")
        if r == "DEABC":
            self.__log.info("Disabled External Aiming Beam Control – Disabled hardware aiming "
                            "beam control.")
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def disable_external_control(self):
        r = self.query("DEC")
        if r == "DEC":
            self.__log.info("Disabled External Control – Disabled the analog current control "
                            "input.")
            self.__log.info("Disables Dynamic Scaling in Waveform mode.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.__log.error(msg)
            raise ResourceWarning(msg)
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def disable_gate_mode(self):
        r = self.query("DGM")
        if r == "DGM":
            self.__log.info("Disabled gate mode - Disabled internal pulse generator.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.__log.error(msg)
            raise ResourceWarning(msg)
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def disable_hardware_emission_control(self):
        r = self.query("DLE")
        if r == "DLE":
            self.__log.info("Disabled Hardware Emission Control - Disabled hardware emission"
                            "control.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.__log.error(msg)
            raise ResourceWarning(msg)
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def disable_modulation(self):
        r = self.query("DMOD")
        if r == "DMOD":
            self.__log.info("Disabled Modulation – Disabled modulation mode.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.__log.error(msg)
            raise ResourceWarning(msg)
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def disable_pulse_mode(self):
        r = self.query("DPM")
        if r == "DPM":
            self.__log.info("Disabled PULSE Mode – Disabled PULSE mode.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.__log.error(msg)
            raise ResourceWarning(msg)
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def enable_external_aiming_beam_control(self):
        r = self.query("EEABC")
        if r == "EEABC":
            self.__log.info("Enabled External Aiming Beam Control – Enabled hardware aiming "
                            "beam control.")
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def enable_external_control(self):
        r = self.query("EEC")
        if r == "EEC":
            self.__log.info("Enabled External Control – Enabled the analog current control input.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.__log.error(msg)
            raise ResourceWarning(msg)
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def enable_gate_mode(self):
        r = self.query("EGM")
        if r == "EGM":
            self.__log.info("Enabled Gate Mode – Enabled internal pulse generator gated by "
                            "signal applied to modulation input.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.__log.error(msg)
            raise ResourceWarning(msg)
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def enable_hardware_emission_control(self):
        r = self.query("ELE")
        if r == "ELE":
            self.__log.info("Enabled Hardware Emission Control – Enabled hardware emission "
                            "control.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.__log.error(msg)
            raise ResourceWarning(msg)
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def enable_modulation(self):
        r = self.query("EMOD")
        if r == "EMOD":
            self.__log.info("Enabled modulation mode.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.__log.error(msg)
            raise ResourceWarning(msg)
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def emission_off(self):
        r = self.query("EMOFF")
        if r == "EMOFF":
            self.__log.info("Stopped emission.")
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def emission_off(self):
        r = self.query("EMON")
        if r == "EMON":
            self.__log.info("Started emission.")
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def enable_pulse_mode(self):
        r = self.query("EPM")
        if r == "EPM":
            self.__log.info("Enabled PULSE Mode.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.__log.error(msg)
            raise ResourceWarning(msg)
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

    def read_extended_device_status(self):
        r = self.query("EPM")

        if r == "EPM":
            self.__log.info("Enabled PULSE Mode.")
        else:
            msg = f"Unknown error: {r}"
            self.__log.error(msg)
            raise ResourceWarning(msg)

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
            ser.write("{0}\r".format(q).encode('utf-8'))
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
            ser.write("{0}\r".format(q).encode('utf-8'))
            sleep(self.__delay)
            line = ser.readline()
            return line.decode('utf-8').rstrip("\r").rstrip(" ")