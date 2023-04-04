import time

import numpy as np
from datetime import datetime
import os
from instruments.mx200 import MX200
import numpy as np
import pandas as pd
from time import sleep

MX200_COM = 'COM3'
reference_gauge: int = 2
target_gauge: int = 1

calibration_point = 1
calibration_point_map = {
    1: 0, 2: 1, 3: 70, 4: 760
}
kp = 0.1

save_path = r'G:\Shared drives\ARPA-E Project\Lab\Data\MX200 Calibration'


def clamp(value: int):
    if value > 99:
        return 99
    if value < -99:
        return -99
    return value

def read_pressures(mx):
    r = mx.pressures
    sleep(0.5)
    if r is None:
        return read_pressures(mx)
    return r


def main():
    mx = MX200(address=MX200_COM, keep_alive=True)
    print(f'Calibration on point {calibration_point} ({calibration_point_map[calibration_point]})')
    # input('Press any key to continue')
    print('Storing old calibration values...')
    old_calibration = np.empty(
        len(calibration_point_map),
        dtype=np.dtype([('Point', 'i'), ('Pressure (Torr)', 'i'), ('Value', 'i')])
    )
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    file_tag = f'SAVED_CALIBRTION_GAUGE{target_gauge}_{date_str}'
    for i in range(len(calibration_point_map)):
        cp = i + 1
        cv = mx.read_calibration(channel=target_gauge, adjustment_point=cp)
        old_calibration[i] = (cp, calibration_point_map[cp], cv)
    df = pd.DataFrame(data=old_calibration)
    df.to_csv(
        path_or_buf=os.path.join(save_path, file_tag + '.csv'),
        index=False
    )
    print(df)
    del mx

    delta_pct = np.inf
    tol_pct = 5.  # percent
    while abs(delta_pct) > tol_pct:
        mx = MX200(address=MX200_COM, keep_alive=False)
        sleep(0.5)
        pressures = mx.pressures
        sleep(0.5)
        print(f'pressures: {pressures}')
        p = pressures[1]
        p_ref = pressures[2]
        if p == 'OVER':
            p = 2E3

        e = p - p_ref
        delta_pct = 100.*e / p_ref
        sign = -1 if e < 0. else 1
        current = mx.read_calibration(channel=target_gauge, adjustment_point=calibration_point)
        sleep(0.5)
        new = clamp(current + e * kp)
        print(f'P_CAL: {p:.3E} Torr, P_REF{p_ref:.3E}, current setpoint: {current}, new setpoint: {new}')
        mx.set_calibration(channel=target_gauge, adjustment_point=calibration_point, set_point=new)
        time.sleep(0.5)
        mx.close()
        del mx
        # print(f'response: {r}')
    mx.close()


if __name__ == '__main__':
    main()
