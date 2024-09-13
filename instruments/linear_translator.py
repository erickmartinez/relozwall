import time
from instruments.BaseSerial import BaseSerial
import serial
from time import sleep
from serial import SerialException


class ISC08(BaseSerial):
    """
    Represents the ISC08 (Integrated Stepper Controller 8 A)
    used by the linear translator in the extrusion system

    Attributes
    ----------

    """
    _delay = 0.5
    __serial: serial.Serial = None
    __speed: int = 60
    __direction: str = 'forward'
    """
    VALUES BEFORE 2024/09/10
    __calibration: dict = {'a0': 0.5377032793, 'a1': 191.9863223, 'b0': 44.47470398, 'b1': 18.90660644}
    __calibration_intercept = 0.2538541300601601
    """
    __calibration: dict = {'a0': -4.7, 'a1': 228., 'b0': 45.0, 'b1': 19.0}
    __calibration_intercept = 0.24
    """
    __address = 'COM6'
    __baud_rate = 57600
    __byte_size = serial.EIGHTBITS
    __timeout = 0.2
    __parity = serial.PARITY_NONE
    __stopbits = serial.STOPBITS_ONE
    __xonxoff = 1
    __delay = 0.2
    __serial: serial.Serial = None
    """

    def __init__(self):
        super().__init__(name='TRANSLATOR')
        self._delay = 0.2
        self._serial_settings = {
            "baudrate": 57600,
            "bytesize": serial.EIGHTBITS,
            "parity": serial.PARITY_NONE,
            "stopbits": serial.STOPBITS_ONE,
            "xonxoff": 1,
            # "rtscts": False,
            # "dsrdtr": False,
            # "exclusive": None,
            "timeout": 0.5,
            # "write_timeout": 0.5,
        }

        self.set_id_validation_query(
            id_validation_query=self.id_validation_query,
            valid_id_specific='TRANSLATOR'
        )

        self.auto_connect()

    def id_validation_query(self) -> str:
        # old_delay = self.delay
        # old_timeout = self.timeout
        # self.delay = 0.5
        # self.timeout = 2.
        # self._serial.write('i\r'.encode('utf-8'))
        # time.sleep(self._delay)
        # res = self._serial.read(11).decode('utf-8').rstrip("\n").rstrip(" ")
        # time.sleep(self._delay)
        # print(res)
        response = self.query('i')
        response = self.query('i')
        # print(response)
        # self.delay = old_delay
        # self.timeout = old_timeout
        return response

    def check_id(self, attempt: int = 0) -> bool:
        check_id = self.query('i')
        if check_id != 'TRANSLATOR':
            if attempt <= 3:
                attempt += 1
                return self.check_id(attempt=attempt)
            else:
                return False
        else:
            return True

    @property
    def calibration(self) -> dict:
        return self.__calibration

    def set_calibration(self, c):
        for e in self.__calibration:
            if not e in c:
                raise KeyError(f"The calibration value {e}, was not found in the provided calibration.")
        for e in c:
            self.__calibration[e] = float(c[e])
        self.__calibration_intercept = (c['b0'] - c['a0']) / (c['a1'] - c['b1'])

    def set_speed_cms(self, value):
        value = abs(value)
        c = self.__calibration
        if value <= self.__calibration_intercept:
            voltage_setting = c['a0'] + c['a1'] * value
        else:
            voltage_setting = c['b0'] + c['b1'] * value
        self.speed = voltage_setting

        # if (5.0 < voltage_setting) and (voltage_setting < 90.0):
        #     self.speed = voltage_setting
        # else:
        #     self.speed = 50.0
        print(f"Input Speed: {value:.2f} cm/s, Voltage Setting: {self.speed:02.0f}")

    def move_by_cm(self, distance: float, speed: float = 2.0):
        speed = abs(speed)
        self.set_speed_cms(speed)
        translation_time = abs(distance / speed)
        direction = 'f' if distance >= 0 else 'r'
        query = f"{direction}{self.__speed:02d}{translation_time * 10:.0f}"
        print(query)
        self.write(q=query)

    def move_by_in(self, distance: float, speed: float = 0.5):
        speed_cm = speed * 2.54
        distance_cm = distance * 2.54
        self.move_by_cm(distance=distance_cm, speed=speed_cm)

    @property
    def speed(self) -> int:
        return self.__speed

    @speed.setter
    def speed(self, value: int):
        self.set_speed(value)

    def set_speed(self, value: int):
        value = int(value)
        self.__direction = 'forward' if value >= 0 else 'reverse'
        value = abs(value)
        if value <= 100:
            self.__speed = value

    def stop(self):
        self.write('s')

    def move_by_time(self, moving_time: float, **kwargs):
        speed = kwargs.get('speed_setting', 55)
        self.set_speed(value=speed)
        direction = 'f' if self.__direction == 'forward' else 'r'
        moving_time = abs(moving_time)
        query = f"{direction}{self.__speed:02d}{moving_time * 10:.0f}"
        print(query)
        response = self.query(q=query)
        if response == 'ERROR_MOVE_IN':
            raise Exception('Cannot move forward. Reached limit')
        if response == 'ERROR_MOVE_OUT':
            raise Exception('Cannot move back. Reached limit')

    def quick_out(self):
        self.write(q='q')

    def write(self, q: str):
        self._serial.write(bytes(f"{q}\r", 'utf-8'))
        sleep(self._delay)

    def read(self) -> str:
        line = self._serial.readline()
        time.sleep(self._delay)
        return line.decode('utf-8').rstrip("\n").rstrip(" ").rstrip("\r")

    def query(self, q: str) -> str:
        self._serial.write(bytes(f"{q}\r", 'utf-8'))
        time.sleep(self._delay)
        return self.read()


class ISC08_old:
    """
    Represents the ISC08 (Integrated Stepper Controller 8 A)
    used by the linear translator in the extrusion system

    Attributes
    ----------
    __address: str
        The physical address of the motor driver

    """
    __address = 'COM6'
    __baud_rate = 57600
    __byte_size = serial.EIGHTBITS
    __timeout = 0.2
    __parity = serial.PARITY_NONE
    __stopbits = serial.STOPBITS_ONE
    __xonxoff = 1
    __delay = 0.2
    __serial: serial.Serial = None
    __speed: int = 60
    __direction: str = 'forward'
    __calibration: dict = {'a0': 0.5377032793, 'a1': 191.9863223, 'b0': 44.47470398, 'b1': 18.90660644}
    __calibration_intercept = 0.2538541300601601

    def __init__(self, address: str):
        self.__address = address
        self.connect()
        check_connection = self.check_id()
        if not check_connection:
            msg = f"ISC08 not found in port {self.address}"
            raise SerialException(msg)

    def check_id(self, attempt: int = 0) -> bool:
        check_id = self.query('i')
        if check_id != 'TRANSLATOR':
            if attempt <= 3:
                attempt += 1
                return self.check_id(attempt=attempt)
            else:
                return False
        else:
            return True

    @property
    def calibration(self) -> dict:
        return self.__calibration

    def set_calibration(self, c):
        for e in self.__calibration:
            if not e in c:
                raise KeyError(f"The calibration value {e}, was not found in the provided calibration.")
        for e in c:
            self.__calibration[e] = float(c[e])
        self.__calibration_intercept = (c['b0'] - c['a0']) / (c['a1'] - c['b1'])

    def set_speed_cms(self, value):
        value = abs(value)
        c = self.__calibration
        if value <= self.__calibration_intercept:
            voltage_setting = c['a0'] + c['a1'] * value
        else:
            voltage_setting = c['b0'] + c['b1'] * value
        self.speed = voltage_setting

        # if (5.0 < voltage_setting) and (voltage_setting < 90.0):
        #     self.speed = voltage_setting
        # else:
        #     self.speed = 50.0
        print(f"Input Speed: {value:.2f} cm/s, Voltage Setting: {self.speed:02.0f}")

    def move_by_cm(self, distance: float, speed: float = 2.0):
        speed = abs(speed)
        self.set_speed_cms(speed)
        translation_time = abs(distance / speed)
        direction = 'f' if distance >= 0 else 'r'
        query = f"{direction}{self.__speed:02d}{translation_time * 10:.0f}"
        print(query)
        self.write(q=query)

    def move_by_in(self, distance: float, speed: float = 0.5):
        speed_cm = speed * 2.54
        distance_cm = distance * 2.54
        self.move_by_cm(distance=distance_cm, speed=speed_cm)

    @property
    def speed(self) -> int:
        return self.__speed

    @property
    def address(self) -> str:
        return self.__address

    @address.setter
    def address(self, value):
        self.__address = value
        self.connect()
        check_id = self.query('i')
        if check_id != 'TRANSLATOR':
            msg = f"ISC08 not found in port {self.address}"
            raise SerialException(msg)

    @speed.setter
    def speed(self, value: int):
        self.set_speed(value)

    def set_speed(self, value: int):
        value = int(value)
        self.__direction = 'forward' if value >= 0 else 'reverse'
        value = abs(value)
        if value <= 100:
            self.__speed = value

    def stop(self):
        self.write('s')

    def move_by_time(self, moving_time: float, **kwargs):
        speed = kwargs.get('speed_setting', 55)
        self.set_speed(value=speed)
        direction = 'f' if self.__direction == 'forward' else 'r'
        moving_time = abs(moving_time)
        query = f"{direction}{self.__speed:02d}{moving_time * 10:.0f}"
        print(query)
        response = self.query(q=query)
        if response == 'ERROR_MOVE_IN':
            raise Exception('Cannot move forward. Reached limit')
        if response == 'ERROR_MOVE_OUT':
            raise Exception('Cannot move back. Reached limit')

    def quick_out(self):
        self.write(q='q')

    def write(self, q: str):
        self.__serial.write(bytes(f"{q}\r", 'utf-8'))
        sleep(self.__delay)

    def read(self) -> str:
        line = self.__serial.readline()
        # ser.flush()
        return line.decode('utf-8').rstrip("\n").rstrip(" ")

    def query(self, q: str) -> str:
        self.__serial.write(bytes(f"{q}\r", 'utf-8'))
        time.sleep(self.__delay)
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
        sleep(self.__delay)
        self.__serial.flush()

    def disconnect(self):
        if self.__serial is not None:
            self.__serial.close()
            self.__serial = None

    def __del__(self):
        self.disconnect()


class L6470:
    """
    Represents the L6470 step motor driver used by the Linear Translator in the Extrusion system

    Attributes
    ----------
    __address: str
        The address at which the gauge is located
    """

    __address = 'COM6'
    __baud_rate = 57600
    __byte_size = serial.EIGHTBITS
    __timeout = 0.01
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

    @address.setter
    def address(self, value):
        self.__address = value

    def move_forward(self):
        msg = self.query('f')
        print(msg)
        return msg

    def move_backward(self):
        msg = self.query('r')
        print(msg)
        return msg

    def decode_status(self, register_hex: str) -> dict:
        mot_status_flags = {
            '00': 'stopped', '01': 'acceleration', '10': 'deceleration', '11': 'constant speed'
        }
        status_register = bin(int(register_hex, 16))[2:].zfill(16)[::-1]
        hiz_bit = int(status_register[0])
        busy_bit = int(status_register[1])
        sw_f_bit = int(status_register[2])
        sw_evn_bit = int(status_register[3])
        dir_bit = int(status_register[4])
        mot_status_bit = status_register[5:7]
        notperf_cmd_bit = int(status_register[7])
        wrong_cmd_bit = int(status_register[8])
        uvlo_bit = int(status_register[9])
        th_wrn_bit = int(status_register[10])
        th_sd_bit = int(status_register[10])
        ocd_bit = int(status_register[12])
        step_loss_a_bit = int(status_register[13])
        step_loss_b_bit = int(status_register[14])
        sck_mod_bit = int(status_register[15])
        status = {
            'high impedance': bool(hiz_bit),
            'undervoltage': not bool(uvlo_bit),
            'thermal warning': not bool(th_wrn_bit),
            'thermal shutdown': not bool(th_sd_bit),
            'overcurrent': not bool(ocd_bit),
            'step loss a': not bool(step_loss_a_bit),
            'step loss b': not bool(step_loss_b_bit),
            'cmd not performed': bool(notperf_cmd_bit),
            'wrong cmd': bool(wrong_cmd_bit),
            'switch status': bool(sw_f_bit),
            'switch event': bool(sw_evn_bit),
            'direction': 'forward' if dir_bit == 1 else 'reverse',
            'busy': bool(busy_bit),
            'motor status': mot_status_flags[mot_status_bit],
            'step clock mode': bool(sck_mod_bit)
        }
        return status

    @property
    def status(self):
        self.__serial.write(bytes("s\r", 'utf-8'))
        sleep(self.__delay)
        line = self.__serial.readline()
        register_hex = line.decode('utf-8').rstrip("\n")
        print(line)
        if register_hex == '':
            return {}
        status = self.decode_status(register_hex=register_hex)
        status['hex_string'] = line  # hex(int(register_hex, 2))
        return status

    @property
    def position(self) -> int:
        x = self.query('p')
        return int(x)

    @property
    def run_speed(self) -> int:
        rs = self.query('v')
        print(rs)
        return int(rs)

    @run_speed.setter
    def run_speed(self, value) -> int:
        value = abs(int(value))
        rs = self.query(f"v {value}")
        print(rs)
        return int(rs)

    def move_steps(self, steps: int):
        steps = int(steps)
        msg = self.query(f"m {steps}")
        print(msg)
        return msg

    def stop(self):
        self.write(' ')

    def write(self, q: str):
        self.__serial.write(bytes(f"{q}\r", 'utf-8'))
        sleep(self.__delay)

    def read(self) -> str:
        line = self.__serial.readline()
        # ser.flush()
        return line.decode('utf-8').rstrip("\n").rstrip(" ")

    def query(self, q: str) -> str:
        self.__serial.write(bytes(f"{q}\r", 'utf-8'))
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
