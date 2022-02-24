import time
from typing import Any, Tuple, Union, Dict
from typing_extensions import TypedDict
import numpy as np
import pyvisa
from pyvisa.errors import VisaIOError
from time import sleep
import re

TBS2000_RESOURCE_NAME = 'USB0::0x0699::0x03C7::C010461::INSTR'


class TraceNotReady(Exception):
    pass


class OutputDict(TypedDict):
    no_of_bytes: int
    no_of_bits: int
    encoding: str
    binary_format: str
    byte_order: str
    no_of_points: int
    waveform_ID: str
    point_format: str
    x_incr: float
    x_zero: float
    x_unit: str
    y_multiplier: float
    y_zero: float
    y_offset: float
    y_unit: str


class TBS2000:
    __resource_name: str = TBS2000_RESOURCE_NAME
    __instrument: pyvisa.Resource = None
    __rm: pyvisa.ResourceManager = None
    __delay: float = 0.1

    def __init__(self, resource_name: str = TBS2000_RESOURCE_NAME):
        self.__resource_name = resource_name
        self.__rm = pyvisa.ResourceManager()
        self.__instrument: pyvisa.Resource = self.__rm.open_resource(self.__resource_name)
        self.__instrument.read_termination = '\n'
        self.__instrument.write_termination = '\r\n'
        self.reset()

        for i in range(1, 4):
            self.write(f'CH{i:d}:PRObe:GAIN 1.0')
        # time.sleep(self.__delay)
        # self.horizontal_main_scale = 2.0
        # self.trigger_channel = 2
        # self.trigger_level = 24.0

    def reset(self):
        # REM = Remark
        self.write('REM "Check for any messages, and clear them from the queue."')
        time.sleep(self.__delay)
        print('*ESR?')
        print(self.sesr)
        print('ALLEV?')
        print(self.all_events)
        self.write('REM "Set the instrument to the default state."')
        time.sleep(self.__delay)
        self.write('FACTORY')
        self.write('HORizontal:RESOlution 2000')
        self.write('HORizontal:POSition 0')
        # self.write('ACQUIRE:STOPAFTER RUNStop')
        # self.write('ACQuire:MODe SAMple')

    @property
    def timeout(self) -> int:
        return self.__instrument.timeout

    @timeout.setter
    def timeout(self, value_ms):
        self.__instrument.timeout = value_ms

    def acquire_on(self):
        self.write('ACQUIRE:STOPAFTER RUNStop')
        self.write('ACQuire:MODe SAMPLE')
        # self.write('ACQUIRE:STOPAFTER SEQUENCE')
        # time.sleep(self.__delay)
        self.write('ACQUIRE:STATE RUN')
        # time.sleep(self.__delay)
        self.write('REM "Wait for the acquisition to complete."')
        # time.sleep(self.__delay)
        self.write('REM "Note: your controller program time-out must be set long enough to'
                   'handle the wait."')

    def acquire_off(self):
        self.write('ACQUIRE:STATE STOP')
        time.sleep(self.__delay)
        opc = self.query('*OPC?')
        # print(f'OPC: {opc}')

    def get_curve(self, channel: int):
        self.write('REM "Use the instrument built-in measurements to measure the waveform you acquired."')
        esr = self.sesr
        if esr != 0:
            all_events = self.all_events
            msg = '\n'.join([f'{e["code"]}: {e["event"]}' for e in all_events])
            # raise VisaIOError(msg)
            print(msg)
        self.write('REM "Query out the waveform points, for later analysis on your controller computer."')
        # time.sleep(self.__delay)
        time.sleep(self.__delay)
        # time.sleep(self.__delay)
        self.write('DATaINIT')
        time.sleep(self.__delay)
        self.write('WFMINPRE:NR_PT 2000')
        self.write('DATa:WIDTh 2')
        # time.sleep(self.__delay)

        time.sleep(self.__delay)
        self.write(f'DATa:SOUrce CH{channel}')
        time.sleep(self.__delay)
        self.write('DATA:SNAp')
        time.sleep(self.__delay)
        # self.write('DATA:START 1')
        # time.sleep(self.__delay)
        # self.write('DATA:STOP 10000')
        # time.sleep(self.__delay)
        self.write('DATa:ENCdg ASCII')
        time.sleep(self.__delay)
        print(self.query('DATA?'))
        time.sleep(self.__delay)
        preamble_str = self.query('WFMPre?')[8::]
        time.sleep(self.__delay)
        # preamble_str = self.query('WFMOutpre?')[11::]
        print(preamble_str)
        self.write("Query out the waveform points, for later analysis on your controller computer.")
        res = self.query('CURVE?')
        # print(res)
        res = res[7::]
        time.sleep(self.__delay)
        # print(res)
        curve = np.array(res.split(',')).astype(float)

        preamble = self._preamble_parser(preamble_str)

        xstart = preamble['x_zero']
        xinc = preamble['x_incr']
        no_of_points = preamble['no_of_points']
        y_zero = preamble['y_zero']
        y_mult = preamble['y_multiplier']
        y_off = preamble['y_offset']
        x_unit = preamble['x_unit'].strip('"')
        y_unit = preamble['y_unit'].strip('"')
        point_off = preamble['point_off']

        # xdata = np.linspace(xstart, no_of_points * xinc + xstart, no_of_points)
        # xdata = xstart + xinc * (np.arange(no_of_points) - point_off)
        xdata = xinc * (np.arange(no_of_points) - point_off)
        ydata = y_zero + y_mult * (curve - y_off)

        data = np.empty(
            ydata.size, dtype=np.dtype([
                (f'x ({x_unit})', 'd'), (f'y ({y_unit})', 'd')
            ])
        )
        data[f'x ({x_unit})'] = xdata
        data[f'y ({y_unit})'] = ydata

        return data

    @property
    def horizontal_main_scale(self) -> float:
        return float(self.query('HOR:MAIN:SCALE?'))

    @horizontal_main_scale.setter
    def horizontal_main_scale(self, value: float):
        self.write(f'HOR:MAIN:SCALE {value:.4g}')

    @property
    def trigger_channel(self) -> str:
        return self.query('TRIGger:A:EDGE:SOUrce?')

    @trigger_channel.setter
    def trigger_channel(self, channel: int = 2):
        if isinstance(channel, str):
            m = re.findall(r'CH(\d)', channel)
            print(m)
            if len(m) == 0:
                try:
                    channel = int(channel)
                except ValueError:
                    channel = 2
            else:
                channel = int(m[0])
        self.write(f'TRIGger:A:EDGE:SOUrce CH{channel}')
        time.sleep(self.__delay)

    @property
    def trigger_level(self) -> float:
        return float(self.query('TRIG:MAIN:LEVEL?'))

    @trigger_level.setter
    def trigger_level(self, level: float):
        self.write(f'TRIG:MAIN:LEVEL {level:.4g}')
        time.sleep(self.__delay)

    def display_channel_list(self, channel_list: tuple):
        display_channels = np.array(channel_list).astype(bool)[0:3]
        display_channels = ['ON' if d else 'OFF' for d in display_channels]
        for i, d in enumerate(display_channels):
            ch = i + 1
            self.write(f'SELect:CH{ch} {d}')
        time.sleep(self.__delay)

    def calculate_set_points(self) -> Tuple[np.ndarray, int]:
        message = self.query('WFMPre')
        preamble = self._preamble_parser(message)
        xstart = preamble['x_zero']
        xinc = preamble['x_incr']
        no_of_points = preamble['no_of_points']
        xdata = np.linspace(xstart, no_of_points * xinc + xstart, no_of_points)
        return xdata, no_of_points

    @staticmethod
    def _preamble_parser(response: str) -> dict:
        """
        Parser function for the curve preamble

        Args:
            response: The response of WFMPre?

        Returns:
            A dictionary containing the following keys:
              no_of_bytes, no_of_bits, encoding, binary_format,
              byte_order, no_of_points, waveform_ID, point_format,
              x_incr, x_zero, x_unit, y_multiplier, y_zero, y_offset, y_unit
        """
        response_str_list = response.split(';')
        response_dict = {}
        p = re.compile(r'(^[A-Z_]+)\s(.*)')
        for r in response_str_list:
            m = p.findall(r)[0]
            response_dict[m[0]] = m[1]

        # for p in response_dict:
        #     print(p, response_dict[p])

        # outdict: OutputDict = {
        #     'no_of_bytes': int(response_list[0]),
        #     'no_of_bits': int(response_list[1]),
        #     'encoding': response_list[2],
        #     'binary_format': response_list[3],
        #     'byte_order': response_list[4],
        #     'waveform_ID': response_list[5],
        #     'no_of_points': int(response_list[6]),
        #     'point_format': response_list[7],
        #     'x_unit': response_list[8],
        #     'x_incr': float(response_list[9]),
        #     'x_zero': float(response_list[10]),
        #     'point_off': int(response_list[11]),
        #     'y_unit': response_list[12],
        #     'y_multiplier': float(response_list[13]),
        #     'y_offset': float(response_list[14]),
        #     'y_zero': float(response_list[15])
        # }

        outdict: OutputDict = {
            'no_of_bytes': int(response_dict['BYT_NR']),
            'no_of_bits': int(response_dict['BIT_NR']),
            'encoding': response_dict['ENCDG'],
            'binary_format': response_dict['BN_FMT'],
            'byte_order': response_dict['BYT_OR'],
            'waveform_ID': response_dict['WFID'],
            'no_of_points': int(response_dict['NR_PT']),
            'point_format': response_dict['PT_FMT'],
            'x_unit': response_dict['XUNIT'],
            'x_incr': float(response_dict['XINCR']),
            'x_zero': float(response_dict['XZERO']),
            'point_off': int(response_dict['PT_OFF']),
            'y_unit': response_dict['YUNIT'],
            'y_multiplier': float(response_dict['YMULT']),
            'y_offset': float(response_dict['YOFF']),
            'y_zero': float(response_dict['YZERO']),
            # 'v_scale': float(response_dict['VSCALE'])
        }

        return outdict

    def write(self, q: str):
        self.__instrument.write(q)

    def query(self, q: str) -> str:
        r = self.__instrument.query(q)
        time.sleep(self.__delay)
        return r

    @property
    def sesr_bit_functions(self) -> dict:
        return {
            'PON': 'Power On. Shows that the instrument was powered on. On completion, the diagnostic self tests also '
                   'set this bit.',
            'URQ': 'User Request. Indicates that an application event has occurred.',
            'CME': 'Command Error. Shows that an error occurred while the instrument was parsing a command or query.',
            'EXE': 'Execution Error. Shows that an error executing a command or query.',
            'DDE': 'Device Error. Shows that a device error occurred.',
            'QYE': 'Query Error. Either an attempt was made to read the Output Queue when no data was present or '
                   'pending, or that data in the Output Queue was lost.',
            'RQC': 'Request Control. This is not used.',
            'OPC': 'Operation Complete. Shows that the operation is complete.'
                   'This bit is set when all pending operations complete following an *OPC command.'
        }

    @property
    def sesr(self) -> dict:
        r = self.query('*ESR?')
        if int(r) == 0:
            return 0
        sesr_register = "{0:07b}".format(int(r))
        sesr_keys = self.sesr_bit_functions
        sesr = {}
        for i, key in zip(range(len(sesr_register)), sesr_keys.keys()):
            sesr[key] = bool(int(sesr_register[i]))
        return sesr

    @property
    def all_events(self) -> str:
        r = self.query('ALLEV?')
        events = re.findall(r'(\d+,".*?")', r)
        messages = []
        for e in events:
            m = re.findall(r'(\d+),"(.*?)"', e)[0]
            msg = {'code': m[0], 'event': m[1]}
        return messages

    @property
    def instrument(self) -> pyvisa.Resource:
        return self.__instrument

    def __del__(self):
        if self.__instrument is not None:
            self.__instrument.close()
            self.__instrument = None
