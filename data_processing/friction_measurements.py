import logging
import numpy as np
import instruments.esp32 as esp32
import instruments.linear_translator as lnt
import time
import datetime
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import json
import matplotlib.ticker as ticker
from instruments.ametek import DCSource
from simple_pid import PID
from scipy.interpolate import interp1d

# EXT_READOUT_COM = 'COM12'
# DC_SOURCE_COM = 'COM13'
DC_SOURCE_IP = '192.168.1.3'
EXT_READOUT_IP = '192.168.4.2'

ISC08_COM = 'COM4'
base_path = r"G:\Shared drives\ARPA-E Project\Lab\Data\Extruder\Friction"
pid_stabilizing_time = 600.0  # seconds
SPEED_CMS = 0.11
speed_setting_map = {0.11: 20, 0.57: 55, 1.1: 65}
temperature = 715
sample = 'R3N54'
baseline = False
# MOVING_LENGTH = 12.0 # in
MOVING_LENGTH = 6.0  # 5.0 / 2.54  # in <--- During heating move only 5 cm (or the length of the coil)
if baseline:
    MOVING_LENGTH = 20.0  # <------- FOR BASELINE 20 INCHES STARTING FROM POSITION = 9.0 IN

plot_csv = True
csv_file = 'FRICTION_R3N54_715C_0.11CMPS_2022-07-11_1.csv'
# ********************** BASELINE ******************************************
baseline_csv = 'FRICTION_BASELINE_SACRIFICIAL_715C_0.11CMPS_2022-07-11_1.csv'
# **************************************************************************
RAMPING_RATE = 25.0 # °C/min
load_cell_prediction_error_pct = 9.8  # %
load_cell_range = 30.0  # kg
allowable_force_threshold = 90  # percentage of the nominal range of the load cell
ku = 5000.0
Tu = 15.5
pid_params = {'ku': 5000.0, 'kp': 0.2 * ku, 'ki': 0.4 * ku / Tu, 'kd': 2.0 * ku * Tu / 30.0}
N_POINTS = 100
X_MAX = 55.0

"""
def move_forward_by_distance(distance_cm):
    import instruments.linear_translator as lnt
    m_time = distance_cm / 0.57
    if '__translator' not in locals():
        __translator = lnt.ISC08(address=ISC08_COM)
    __translator.move_by_time(moving_time=m_time, speed_setting=55)
    del __translator


def move_back_by_distance(distance_cm):
    import instruments.linear_translator as lnt
    m_time = distance_cm / 0.57
    if '__translator' not in locals():
        __translator = lnt.ISC08(address=ISC08_COM)
    __translator.move_by_time(moving_time=m_time, speed_setting=-55)
    del __translator
"""


class FrictionExperiment:
    __translator: lnt.ISC08 = None
    __readout: esp32.ExtruderReadout = None
    __dc_power_supply: DCSource = None
    __address_translator: str = ISC08_COM
    __address_readout: str = None
    __x0: float = None
    __isc08_calibration_m: float = 0.034
    __isc08_calibration_b: float = -1.0
    __pot_a0: float = 8.45
    __pot_a1: float = 0.0331

    def __init__(self, address_translator, address_readout):
        self.__address_translator = address_translator
        self.__address_readout = address_readout
        self.__translator = lnt.ISC08(address=address_translator)
        time.sleep(1.0)
        self.__readout = esp32.ExtruderReadout(ip_address=EXT_READOUT_IP)
        self.__x0 = self.current_position_cm[0]

    @property
    def readout(self) -> esp32.ExtruderReadout:
        return self.__readout

    @property
    def translator(self) -> lnt.ISC08:
        return self.__translator

    def get_pot_adc_avg(self, averages: int = 10):
        points = int(averages)
        x = np.empty(points)
        for j in range(points):
            time.sleep(0.01)
            [_, _, _, _, x[j]] = self.readout.reading
        return x.mean(), x.std()

    def adc_to_cm(self, x: np.ndarray):
        return self.__pot_a0 + self.__pot_a1 * x

    @property
    def current_position_cm(self, averages: int = 10):
        x = np.array(self.get_pot_adc_avg(averages))
        return self.adc_to_cm(x)

    @property
    def get_displacement_cm(self, x0: float = None):
        if x0 is None:
            x0 = self.__x0
        x = self.current_position_cm
        return x[0] - x0[0]

    def cmps_to_speed_setting(self, cmps) -> int:
        return (cmps - self.__isc08_calibration_b) / self.__isc08_calibration_m


def cm2in(value):
    return value / 2.54


if __name__ == "__main__":
    if SPEED_CMS not in speed_setting_map:
        msg = f"Speed {SPEED_CMS} not defined! Valid values are: {[k for k in speed_setting_map.keys()]}"
        raise ValueError(msg)
    SPEED_SETTING = speed_setting_map[SPEED_CMS]
    # print(f"Speed setting: {SPEED_SETTING}")
    if plot_csv:
        file_tag = os.path.splitext(csv_file)[0]
        friction_df = pd.read_csv(os.path.join(base_path, csv_file), comment='#').apply(pd.to_numeric)
        friction_df.sort_values(by=['Position (cm)'], ascending=True, inplace=True)
        friction_df.reset_index(drop=True, inplace=True)
        if X_MAX is not None and not baseline:
            friction_df = friction_df[friction_df['Position (cm)'] <= X_MAX]
        background_df = pd.read_csv(os.path.join(base_path, baseline_csv), comment='#').apply(pd.to_numeric)
        x_b = background_df['Position (cm)'].values
        f_b = background_df['Force (N)'].values
        f = interp1d(x_b, f_b, kind='linear')

        #     pd.DataFrame(data={
        #     'Time (s)': elapsed_time,
        #     'Position (cm)': position,
        #     'Force (N)': force
        # })
        elapsed_time = friction_df['Time (s)'].values
        position = friction_df['Position (cm)'].values
        force = friction_df['Force (N)'].values
        if not baseline:
            try:
                force -= f(position)
            except ValueError as ve:
                print(f'Sample Force: min={force:.1F} N, max={}')
        force_err = force * (2.0 ** 0.5) * load_cell_prediction_error_pct * 1E-2

        if not baseline:
            friction_baselined_df = friction_df.copy()
            friction_baselined_df['Background Force (N)'] = f(position)
            friction_baselined_df['Baselined Force (N)'] = force
            friction_baselined_df['Baselined Force Error (N)'] = force_err
            print(friction_baselined_df)
            friction_baselined_df.to_csv(path_or_buf=os.path.join(base_path, file_tag + '_baselined.csv'), index=False)

        # elapsed_time = elapsed_time[:-1]
        # position = position[:-1]
        # force = force[:-1]

        avg_speed = (position.max() - position.min()) / elapsed_time.max()

        with open('plot_style.json', 'r') as file:
            json_file = json.load(file)
            plot_style = json_file['defaultPlotStyle']
        mpl.rcParams.update(plot_style)

        fig, ax1 = plt.subplots()
        fig.set_size_inches(4.5, 3.25)

        ax2 = ax1.twiny()

        ax1.errorbar(
            position, force, yerr=force_err,
            capsize=2.75, mew=1.25, marker='o', ms=8, elinewidth=1.25,
            color='C0', fillstyle='none',
            ls='-',
            label='Data',
            zorder=1
        )

        # ax1.plot(
        #     position, force,
        #     color='C0', fillstyle='none', marker='o',
        #     ls='-',
        #     label='Data',
        #     zorder=1
        # )

        xmin, xmax = ax1.get_xlim()
        ax2.set_xlim(cm2in(xmin), cm2in(xmax))

        leg = ax1.legend(
            loc='lower right', frameon=True, ncol=1,
            # fontsize=8, bbox_to_anchor=(1.05, 1),
            # borderaxespad=0.,
            prop={'size': 10}
        )

        # info_txt = f"$F_{{\mathrm{{bgd}}}} = {f0.mean():3.1f}±{f0.std():4.2f}$ N" + "\n"
        info_txt = rf"$\langle F_{{\mathrm{{ {SPEED_CMS:.2f} }} }}\rangle  = {force.mean():3.1f}±{force.std():4.2f}$ N"  # + "\n"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(
            0.05,
            0.95,
            info_txt,
            fontsize=12,
            # color='tab:green',
            transform=ax1.transAxes,
            va='top', ha='left',
            bbox=props
        )

        ax1.set_xlabel('Position (cm)')
        ax1.set_ylabel('Force (N)')
        ax2.set_xlabel('Position (in)')
        ax1.set_title(f"{sample}, {temperature:>3.0f} °C, {avg_speed:3.2} cm/s")

        ax1.ticklabel_format(useMathText=True)
        ax1.xaxis.set_minor_locator(ticker.MaxNLocator(6))
        #
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(6))
        # ax1.yaxis.set_minor_locator(ticker.MaxNLocator(12))

        fig.tight_layout()
        fig.savefig(os.path.join(base_path, file_tag + '.png'), dpi=600)
        plt.show()
    else:
        log = logging.getLogger(__name__)
        log.addHandler(logging.NullHandler())
        log.setLevel(logging.DEBUG)
        today = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        sample = sample.upper()
        file_tag = f"FRICTION_{sample}_{temperature:>3.0f}C_{SPEED_CMS:3.2f}CMPS_{today}"
        log_file = os.path.join(base_path, file_tag + '.csv')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        log.addHandler(ch)

        experiment = FrictionExperiment(address_translator=ISC08_COM, address_readout=EXT_READOUT_IP)
        dc_source = DCSource(ip_address=DC_SOURCE_IP)

        log.info("Setting the PID controller")
        [TC1, _, _, _, pot_adc] = experiment.readout.reading
        pid = PID(pid_params['kp'], pid_params['ki'], pid_params['kd'], setpoint=TC1)
        pid.output_limits = (0, 200.0)
        allowable_force_threshold_n = allowable_force_threshold * load_cell_range * 9.82E-2

        log.info('Setting up DC voltage')
        # self.__dc_source.cls()
        # self.__dc_source.rst()
        dc_source.current_setpoint = 8.0
        dc_source.voltage_setpoint = 0.0
        dc_source.output_on()

        # read the force in the absence of torque:
        # f0 = np.empty(n)

        log.info("Taring the load cell...")
        experiment.readout.zero()
        time.sleep(2.0)
        # self.__dc_source.trigger_voltage()
        [TC1, _, f, _, pot_adc] = experiment.readout.reading
        d0 = experiment.adc_to_cm(pot_adc)
        log.info(f"TC1: {TC1:6.2f} °C, F: {f:4.1f} N, x0: {d0:.1f} cm")
        pid.set_auto_mode(True, last_output=TC1)

        speed_setting = SPEED_SETTING  # experiment.cmps_to_speed_setting(cmps=SPEED_CMS)
        print(f"Speed setting: {speed_setting}")
        max_distance_in = MOVING_LENGTH
        max_distance = max_distance_in * 2.54  # cm
        moving_time = max_distance / SPEED_CMS

        initial_position, _ = experiment.current_position_cm

        dt = max(moving_time / N_POINTS, 0.4)
        print(f'dt = {dt:.3f} s')

        elapsed_time = []
        force = []
        position = []
        baking_temperature = []
        current_time = 0.0
        previous_time = 0.0
        total_time = 0.0
        current_ramping_time = 0.0
        initial_temperature = TC1
        ramping_time = 60.0 * (temperature - initial_temperature) / RAMPING_RATE
        ramping_t0 = time.time()
        moving = False

        run_pid = True
        ramping = True
        stabilizing = False
        displacement = 0
        # experiment.readout.zero()
        # time.sleep(5.0)
        while run_pid:
            current_time = time.time()
            [TC1, _, fi, _, pot_adc] = experiment.readout.reading
            control = pid(TC1)
            dc_source.voltage_setpoint = control
            if ramping and current_ramping_time <= ramping_time:
                temperature_setpoint = initial_temperature + RAMPING_RATE * current_ramping_time / 60.0
                if temperature_setpoint > temperature:
                    temperature_setpoint = temperature
                    ramping = False
                print(f"T = {TC1:>6.2f} °C, (Setpoint: {temperature_setpoint:>6.1f} °C), Ramping Time: {current_ramping_time:>5.2f} s", end='\r', flush=True)
                current_ramping_time = time.time() - ramping_t0
                pid.setpoint = temperature_setpoint
                time.sleep(0.01)
            if temperature_setpoint >= temperature and ramping:
                ramping = False
                stabilizing = True
                current_ramping_time = 0
                ramping_t0 = current_time
                print("")
                time.sleep(0.01)
            if stabilizing and current_ramping_time <= pid_stabilizing_time:
                print(f"T = {TC1:>6.2f} °C, Stabilizing Time: {current_ramping_time:>5.2f} s", end='\r', flush=True)
                current_ramping_time = time.time() - ramping_t0
                if current_ramping_time > pid_stabilizing_time:
                    stabilizing = False
                    t0 = current_time
                    total_time = 0
                time.sleep(0.1)

            if (not ramping) and (not stabilizing) and not moving:
                experiment.translator.move_by_time(moving_time=moving_time, speed_setting=speed_setting)
                moving = True

            if (not ramping) and (not stabilizing) and total_time <= moving_time:
                # current_time = time.time()
                if (current_time - previous_time) >= dt:
                    # [TC1, _, fi, _, pot_adc] = experiment.readout.reading
                    d = experiment.adc_to_cm(pot_adc)
                    displacement = (d - d0) / 2.54
                    total_time = current_time - t0
                    elapsed_time.append(total_time)
                    position.append(d)
                    fi_err = fi * load_cell_prediction_error_pct * 1E-2
                    force.append(fi)
                    baking_temperature.append(TC1)
                    if fi >= allowable_force_threshold_n:
                        msg = f'The force on the sample ({fi} N) is larger than the allowable limit: ' \
                              f'{allowable_force_threshold_n} (N) '
                        print(msg)
                        raise ValueError(msg)
                        break
                    print(f"{total_time:8.3f} s, ADC: {pot_adc:5.0f} -> {d:>5.1f} cm, {fi:>5.1f} ± {fi_err:>5.1f}N")
                    previous_time = current_time
            if displacement >= MOVING_LENGTH or total_time > moving_time:
                run_pid = False

        dc_source.voltage_setpoint = 0.0
        dc_source.output_off()

        elapsed_time = np.array(elapsed_time)
        position = np.array(position)
        force = np.array(force)
        current_position = experiment.current_position_cm
        displacement = current_position - initial_position
        baking_temperature = np.array(baking_temperature)

        avg_speed = displacement / moving_time

        print(f'Displacement: {displacement[0]:4.1f}±{displacement[1]:5.2f} cm')
        print(f'Moving time: {moving_time:5.3f} s')
        print(f'Average Speed: {avg_speed[0]:4.2f}±{avg_speed[1]:4.2f}cm/s')

        friction_df = pd.DataFrame(data={
            'Time (s)': elapsed_time,
            'Position (cm)': position,
            'Force (N)': force,
            'TC1 (C)': baking_temperature
        })

        print(friction_df)
        friction_df.to_csv(path_or_buf=os.path.join(base_path, file_tag + '.csv'), index=False)
        print('Path of the results file:')
        print(os.path.join(base_path, file_tag + '.csv'))
