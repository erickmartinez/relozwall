import logging
import re
import socket

LASER_IP = "192.168.3.230"
LASER_PORT = 10001


class LaserException(Exception):
    def ___init__(self, message):
        super().___init__(message)


class YLR3000:
    __connection: socket.socket = None
    __ip_address: str = LASER_IP
    _log: logging.Logger = None
    _aiming_beam_on: bool = False

    def __init__(self, IP: str = LASER_IP):
        self.__ip_address = IP
        # self._log = logging.getLogger(__name__)
        # self._log.addHandler(logging.NullHandler())
        # create console handler and set level to debug
        # has_console_handler = False
        # if len(self._log.handlers) > 0:
        #     for h in self._log.handlers:
        #         if isinstance(h, logging.StreamHandler):
        #             has_console_handler = True
        # if not has_console_handler:
        #     ch = logging.StreamHandler()
        #     ch.setLevel(logging.DEBUG)
        #     self._log.addHandler(ch)

        self.__connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect()
        self.disable_analog_control()
        self.disable_gate_mode()
        self.disable_external_guide_control()
        self.disable_external_control()
        self.enable_modulation()

    def set_logger(self, log: logging.Logger):
        self._log = log

    @property
    def aiming_beam_on(self) -> bool:
        return self._aiming_beam_on

    @aiming_beam_on.setter
    def aiming_beam_on(self, setting:bool):
        setting = bool(setting)
        self._aiming_beam_on = setting
        if setting:
            r = self.query("ABN")
            if r == "ABN":
                self.log("Aiming beam is on.")
            elif r == "ERR":
                msg = "Cannot enable guide beam because external guide control is enabled."
                self.log(msg=msg, level=logging.ERROR)
                raise LaserException(msg)
            else:
                msg = f"Unknown error: {r}"
                self.log(msg=msg, level=logging.ERROR)
                raise LaserException(msg)
        else:
            r = self.query("ABF")
            if r == "ABF":
                self.log("Aiming beam is off.")
            elif r == "ERR":
                msg = "Cannot disable guide beam because external guide control is enabled."
                self.log.error(msg, logging.ERROR)
                raise LaserException(msg)
            else:
                msg = f"Unknown error: {r}"
                self.log.error(msg, logging.ERROR)
                raise LaserException(msg)

    @property
    def current_setpoint(self) -> float:
        r = self.query("RCS")
        p = re.compile(r'RCS\:\s*(\d+\.?\d*)')
        m = re.findall(p, r)
        if len(m) > 0:
            return float(m[0])
        msg = rf'Error reading current setpoint. Response: \'{r}\'.'
        raise ValueError(msg)

    @current_setpoint.setter
    def current_setpoint(self, value):
        value = float(value)
        if 0.0 < value <= 100.0:
            q = f"SDC {value:.1f}"
            r = self.query(q)
            if r[0:] == 'ERR':
                raise ValueError(r)
            self.log(f"Set the laser diode current to: {value:.2f} %")

    @property
    def output_power(self):
        r = self.query("ROP")
        p = re.compile(r'ROP\:\s*(\d+\.?\d*)')
        m = re.findall(p, r)
        if len(m) > 0:
            return float(m[0])
        return r[4:].strip(" ")

    @property
    def output_peak_power(self) -> float:
        r = self.query("RPP")
        p = re.compile(r'RPP\:\s*(\d+\.?\d*)')
        m = re.findall(p, r)
        if len(m) > 0:
            return float(m[0])
        return r[4:].strip(" ")

    @property
    def pulse_repetition_rate(self):
        r = self.query("RPRR")
        p = re.compile(r'RPRR\:\s*(\d+\.?\d*)')
        m = re.findall(p, r)
        if len(m) > 0:
            return float(m[0])
        return r[5:].strip(" ")

    @pulse_repetition_rate.setter
    def pulse_repetition_rate(self, value: float):
        """
        Sets the repetition rate of the pulse
        Parameters
        ----------
        value: float
            The repetition rate in Hz
        """
        value = float(value)
        q = f"SPRR {value:.1f}"
        r = self.query(q)
        if r[0:] == 'ERR':
            self.log(msg=r[4:].strip(" "), level=logging.ERROR)
            raise ValueError(r)
        self.log(f"Set the pulse repetition rate to: {value:.2f} %")

    @property
    def pulse_width(self):
        r = self.query("RPW")
        p = re.compile(r'RPW\:\s*(\d+\.?\d*)?')
        m = re.findall(p, r)
        if len(m) > 0:
            return float(m[0])
        return r[4:].strip(" ")

    @pulse_width.setter
    def pulse_width(self, value):
        value = float(value)
        q = f"SPW {value:.1f}"
        r = self.query(q)
        if r[0:] == 'ERR':
            self.log(msg=r[4:].strip(" "), level=logging.ERROR)
            raise ValueError(r)
        self.log(f"Set the pulse width to: {value:.2f} %")

    def disable_analog_control(self):
        r = self.query("DEC")
        if r == "DEC":
            self.log("Disabled external control.")
        else:
            msg = "Emission is on!"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)

    def disable_external_guide_control(self):
        r = self.query("DEABC")
        if r == "DEABC":
            self.log("Disabled external aiming beam control.")
        else:
            msg = "Error disabling external aiming beam control."
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)

    def disable_external_control(self):
        r = self.query("DLE")
        if r == "DLE":
            self.log("Disabled hardware emission control.")
        else:
            msg = "Emission is on!"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)

    def enable_modulation(self):
        r = self.query("EMOD")
        if r == "EMOD":
            self.log("Enabled modulation mode.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)
        else:
            msg = f"Unknown error: {r}"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)

    def disable_modulation(self):
        r = self.query("DMOD")
        if r == "DMOD":
            self.log("Disabled modulation mode.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)
        else:
            msg = f"Unknown error: {r}"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)

    def enable_gate_mode(self):
        r = self.query("EGM")
        if r == "EGM":
            self.log("Enabled gate mode.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)
        else:
            msg = f"Unknown error: {r}"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)

    def disable_gate_mode(self):
        r = self.query("DGM")
        if r == "DGM":
            self.log("Disabled gate mode.")
        else:
            msg = "Emission is on!"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)

    def enable_pulse_mode(self):
        r = self.query("EPM")
        if r == "EPM":
            self.log("Enabled pulse mode.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)
        else:
            msg = f"Unknown error: {r}"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)

    def disable_pulse_mode(self):
        r = self.query("DPM")
        if r == "DPM":
            self.log("Disabled pulse mode.")
        elif r == "ERR":
            msg = "Emission is on!"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)
        else:
            msg = f"Unknown error: {r}"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)

    def emission_off(self):
        r = self.query("EMOFF")
        if r == "EMOFF":
            self.log("Stopped emission.")
        else:
            msg = f"Unknown error: {r}"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)

    def emission_on(self):
        r = self.query("EMON")
        if r == "EMON":
            self.log("Started emission.")
        else:
            msg = f"Unknown error: {r}"
            self.log(msg=msg, level=logging.ERROR)
            raise LaserException(msg)

    def read_extended_device_status(self):
        r = self.query("ESTA")
        return r[5:].strip(" ")

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

    def log(self, msg, level=logging.INFO):
        if self._log is None:
            print(msg)
            return
        self._log.log(level=level, msg=msg)
        return

    def __del__(self):
        self.disconnect()
