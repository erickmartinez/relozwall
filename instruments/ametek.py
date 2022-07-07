import socket


class DCSource:
    """
    Represents the amtek DC Source
    """
    __voltage_ramp_configured = False
    __socket: socket.socket = None
    __ip_address: str = 'ex'
    __port: int = 52000

    protection_conditon_esr = {
        0: 'Constant Voltage Operation',
        1: 'Constant Current Operation',
        2: 'Constant Power Operation',
        3: 'Overpower Protection Fault',
        4: 'Over Temperature Fault',
        5: 'External Shutdown',
        6: 'Foldback Mode Operation',
        7: 'Remote Programming Error',
        8: 'Fan fault',
        9: 'Line Drop Fault',
        10: 'DC Mode Fault',
        11: 'PFC Fault',
        12: 'OCP Fault',
        13: 'AUX Supply Fault',
        14: 'Line Status Changed',
        15: 'Parallel Cable Fault',
        16: 'Salve System Fault',
        18: 'Remote Sense Fault',
        19: 'Regulation Fault',
        20: 'Current Feedback Fault'
    }

    __standard_event_status_register = {
        0: 'Operation Complete',
        1: 'Request Control - not used',
        2: 'Query Error',
        3: 'Device Dependent Error',
        4: 'Execution Error',
        5: 'Command Error',
        6: 'User Request - not used',
        7: 'Power On'
    }

    def __init__(self, ip_address: str = '192.168.1.3'):
        self.__ip_address = ip_address
        self.connect()
        self.cls()
        self.rst()
        self.current_limit = 8.0

    def connect(self):
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__socket.connect((self.__ip_address, self.__port))

    def disconnect(self):
        if self.__socket is not None:
            self.__socket.close()
            self.__socket = None

    def output_on(self):
        self.write('OUTPut:STATe 1')

    def output_off(self):
        self.write('OUTPut:STATe 0')

    @property
    def current_setpoint(self) -> float:
        r = self.query('SOUR:CURR?')
        return float(r)

    @current_setpoint.setter
    def current_setpoint(self, value):
        value = float(value)
        if 0.0 <= value <= 9.0:
            self.write(f'SOUR:CURR {value:.3f}')

    @property
    def current_limit(self) -> float:
        r = self.query('SOURce:CURRent:LIMit?')
        return float(r)

    @current_limit.setter
    def current_limit(self, value: float):
        value = float(value)
        if 0.0 <= value <= 9.0:
            self.write(f'SOURce:CURRent:LIMit {value:.3f}')

    def setup_ramp_voltage(self, output_voltage: float, time_s: float = 60.0):
        output_voltage, time_s = float(output_voltage), float(time_s)
        if abs(output_voltage) < 200.0 and time_s > 1.0:
            self.write(f"SOURce:VOLTage:RAMP:TRIGger {output_voltage:.3f} {time_s:.3f}")
            self.__voltage_ramp_configured = True

    def run_voltage_ramp(self):
        if self.__voltage_ramp_configured:
            self.write('TRIG:RAMP')

    def trigger_voltage(self):
        self.write('TRIGger:TYPe 3')

    def trigger_abort(self):
        self.write('TRIGger:ABORt')

    @property
    def voltage_setpoint(self) -> float:
        r = self.query('SOURce:VOLTage?')
        return float(r)

    @voltage_setpoint.setter
    def voltage_setpoint(self, value):
        value = float(value)
        if abs(value) <= 200.0:
            self.write(f'SOURce:VOLTage:LEVel:IMMediate {value:.3f}')

    @property
    def measured_voltage(self) -> float:
        r = self.query('MEASure:VOLTage?')
        return float(r)

    @property
    def measured_current(self) -> float:
        r = self.query('MEASure:CURRent?')
        return float(r)

    @property
    def is_ramping(self) -> bool:
        q = 'SOURce:VOLTage:RAMP?'
        return self.query(q) == '1'

    def abort_ramp_voltage(self):
        if self.is_ramping:
            self.write('SOURce:VOLTage:RAMP:ABORt')

    def cls(self):
        self.write('*CLS')

    def rst(self):
        self.write('*RST')

    @property
    def standard_event_status_register(self) -> list:
        q = '*ESR?'
        r = self.query(q)
        esr_binary = "{0:07b}".format(int(r[2::], 16))
        return self.decode_vent_status_register(esr_binary, self.__standard_event_status_register)

    @property
    def status_protection_condition_register(self) -> list:
        q = 'STAT:PROT:COND?'
        r = self.query(q)
        esr_binary = "{0:032b}".format(int(r[4::], 16))
        events = self.decode_vent_status_register(esr_binary, self.protection_conditon_esr)
        return events

    @staticmethod
    def decode_vent_status_register(esr_binary: str, esr_table: dict) -> list:
        events = []
        for i, c in enumerate(esr_binary[::-1]):
            if c == '1':
                events.append(esr_table[i])
        return events

    def query(self, q: str, attempts=1) -> str:
        try:
            self.__socket.sendall(f"{q}\r".encode('utf-8'))
        except ConnectionAbortedError as e:
            self.__socket.close()
            self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__socket.connect((self.__ip_address, self.__port))
            attempts += 1
            if attempts < 5:
                print(e)
                return self.query(q=q, attempts=attempts)
        buffer = b''
        while b'\n' not in buffer:
            data = self.__socket.recv(1024)
            if not data:
                return ''
            buffer += data
        line, sep, buffer = buffer.partition(b'\n')
        return line.decode('utf-8').rstrip("\n").rstrip(" ")

    def write(self, q: str, attempts: int = 1):
        try:
            self.__socket.sendall(f"{q}\r".encode('utf-8'))
        except ConnectionAbortedError as e:
            self.__socket.close()
            self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__socket.connect((self.__ip_address, self.__port))
            attempts += 1
            if attempts <= 5:
                print(e)
                self.write(q=q, attempts=attempts)

    def __del__(self):
        self.disconnect()

