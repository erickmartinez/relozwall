import numpy as np
import logging
import sys, os

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
from pymeasure.experiment import unique_filename
from instruments.mx200 import MX200
from instruments.linear_translator import ISC08
import datetime
from instruments.inhibitor import WindowsInhibitor
from instruments.esp32 import ExtruderReadout
from serial import SerialException

SPEED_CM = 0.2  # cm/s
DISTANCE_IN = 4.0  # inches
STATIC_MEASUREMENT_TIME = 2.0 # s
data_path = r"G:\Shared drives\ARPA-E Project\Lab\Data\Extruder\Friction"

EXT_READOUT_COM = 'COM12'
MX200_COM = 'COM3'
ISC08_COM = 'COM4'
NUMBER_OF_SAMPLES = 100

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    prefix = f'EXTRUSION_'
    filename = unique_filename(data_path, prefix=prefix)
    log_file = os.path.splitext(filename)[0] + '.log'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # add formatter to ch
    ch.setFormatter(formatter)

    log.addHandler(fh)
    log.addHandler(ch)

    distance_cm = DISTANCE_IN * 2.54
    experiment_time_s = distance_cm / SPEED_CM

    log.info(f"Connecting to MX200 readout at port {MX200_COM}")
    mx200 = MX200(address=MX200_COM, keep_alive=True)
    time.sleep(1.0)
    # pressures = mx200.pressures
    # for k, p in pressures.items():
    #     if type(p) is str:
    #         pressures[k] = np.nan
    # print(f"P1: {pressures[1]}")

    log.info(f"Connecting to linear translator at port {ISC08_COM}")
    translator = ISC08(address=ISC08_COM)
    time.sleep(3.0)

    log.info(f"Connecting to extruder readout at port {EXT_READOUT_COM}")
    extruder_readout = ExtruderReadout(address=EXT_READOUT_COM)
    time.sleep(1.0)
    log.info("Taring the load cell")
    extruder_readout.zero()
    time.sleep(3.0)

    print(extruder_readout.reading)

    hardware_dt = extruder_readout.delay

    dt = max(hardware_dt, experiment_time_s / NUMBER_OF_SAMPLES)
    n = int(experiment_time_s / dt) + 1
    log.info(f"dt : {dt:7.3f}")

    previous_time = 0.0
    total_time = 0.0

    counter = 0
    failed = False
    moving = False
    start_time = time.time()
    while total_time <= experiment_time_s + STATIC_MEASUREMENT_TIME+ dt:
        current_time = time.time()
        if total_time > STATIC_MEASUREMENT_TIME and not moving:
            translator.move_by_cm(distance=distance_cm, speed=SPEED_CM)
            moving = True
        if (current_time - previous_time) >= dt:
            pressures = mx200.pressures
            for k, p in pressures.items():
                if type(p) is str:
                    pressures[k] = np.nan
            try:
                [TC1, TC2, force, load_cell_adc, pot_adc] = extruder_readout.reading
            except ValueError as e:
                log.error(e)
                translator.stop()
                failed = True
                break
            total_time = time.time() - start_time
            previous_time = current_time
            d_in_t = SPEED_CM * total_time / 2.54
            print(f"{total_time:8.3f} s, {pressures[1]:5.3E} Torr, {force:8.1f} N, T1: {TC1} °C, distance: {d_in_t:4.1f} in")

    time.sleep(1.0)
    if not failed:
        translator.move_by_cm(distance=-distance_cm, speed=SPEED_CM)
    time.sleep(experiment_time_s)
