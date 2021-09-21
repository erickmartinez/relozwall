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
    __timeout = 0.1
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
