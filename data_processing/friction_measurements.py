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

EXT_READOUT_COM = 'COM12'
DC_SOURCE_COM = 'COM13'
ISC08_COM = 'COM4'
voltage_setpoint = 40.0
voltage_ramp_time = 60.0
SPEED_CMS = 1.1
# MOVING_LENGTH = 12.0 # in
MOVING_LENGTH = 5.0 / 2.54  # in <--- During heating move only 5 cm (or the length of the coil)
speed_setting_map = {0.11: 20, 0.57: 55, 1.1: 65}


base_path = r"G:\Shared drives\ARPA-E Project\Lab\Data\Extruder\Friction"
sample = 'R3N50_3_350C'
plot_csv = True
csv_file = 'FRICTION_R3N50_3_350C_1.10CMPS_20220608-185244.csv'
calibration_factor = 14651.0
load_cell_prediction_error_pct = 15.7  # %
load_cell_range = 20.0 # kg
allowable_force_threshold = 90  # percentage of the nominal range of the load cell


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


class FrictionExperiment:
    __translator: lnt.ISC08 = None
    __readout: esp32.ExtruderReadout = None
    __dc_power_supply: DCSource = None
    __address_translator: str = 'COM4'
    __address_readout: str = 'COM12'
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
        self.__readout = esp32.ExtruderReadout(address=address_readout)
        time.sleep(1.0)
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
    allowable_force_threshold_n = allowable_force_threshold * load_cell_range * 9.82E-2
    if SPEED_CMS not in speed_setting_map:
        msg = f"Speed {SPEED_CMS} not defined! Valid values are: {[k for k in speed_setting_map.keys()]}"
        raise ValueError(msg)
    SPEED_SETTING = speed_setting_map[SPEED_CMS]
    if plot_csv:
        file_tag = os.path.splitext(csv_file)[0]

        friction_df = pd.read_csv(os.path.join(base_path, csv_file)).apply(pd.to_numeric)
        #     pd.DataFrame(data={
        #     'Time (s)': elapsed_time,
        #     'Position (cm)': position,
        #     'Force (N)': force
        # })
        elapsed_time = friction_df['Time (s)'].values
        position = friction_df['Position (cm)'].values
        force = friction_df['Force (N)'].values

        elapsed_time = elapsed_time[:-1]
        position = position[:-1]
        force = force[:-1]

        print(friction_df)

        avg_speed = (position.max() - position.min()) / elapsed_time.max()

        with open('plot_style.json', 'r') as file:
            json_file = json.load(file)
            plot_style = json_file['defaultPlotStyle']
        mpl.rcParams.update(plot_style)

        fig, ax1 = plt.subplots()
        fig.set_size_inches(4.5, 3.25)

        ax2 = ax1.twiny()

        force_err = force * load_cell_prediction_error_pct * 1E-2

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
        ax1.set_title(f"{sample}, {avg_speed:3.2} cm/s")

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
        today = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        sample = sample.upper()
        file_tag = f"FRICTION_{sample}_{SPEED_CMS:3.2f}CMPS_{today}"
        log_file = os.path.join(base_path, file_tag + '.csv')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)

        experiment = FrictionExperiment(address_translator=ISC08_COM, address_readout=EXT_READOUT_COM)
        dc_source = DCSource(address=DC_SOURCE_COM)
        dc_source.setup_ramp_voltage(output_voltage=voltage_setpoint, time_s=voltage_ramp_time)
        dc_source.run_voltage_ramp()
        # read the force in the absence of torque:
        n = 20
        f0 = np.empty(n)

        experiment.readout.zero()
        time.sleep(2.0)
        for i in range(n):
            [_, _, f0[i], _, _] = experiment.readout.reading
            # print(f"{f0[i]:3.1f} N")
            time.sleep(0.01)

        background_reading = f0.mean()
        background_std = f0.std()
        print(f"Background Force Reading: {background_reading:3.1f}±{background_std:4.2f}")

        speed_setting = SPEED_SETTING  # experiment.cmps_to_speed_setting(cmps=SPEED_CMS)
        print(f"Speed setting: {speed_setting}")
        max_distance_in = MOVING_LENGTH
        max_distance = max_distance_in * 2.54  # cm
        moving_time = max_distance / SPEED_CMS

        initial_position, _ = experiment.current_position_cm

        dt = moving_time / n

        elapsed_time = []
        force = []
        position = []
        current_time = 0.0
        previous_time = 0.0
        total_time = 0.0
        moving = False
        t0 = time.time()
        # experiment.readout.zero()
        # time.sleep(5.0)

        while total_time <= moving_time:
            current_time = time.time()
            if not moving:
                experiment.translator.move_by_time(moving_time=moving_time, speed_setting=speed_setting)
                moving = True
            if (current_time - previous_time) >= dt:
                [_, _, fi, _, pot_adc] = experiment.readout.reading
                d = experiment.adc_to_cm(pot_adc)
                total_time = current_time - t0
                elapsed_time.append(total_time)
                position.append(d)
                fi_err = fi * load_cell_prediction_error_pct * 1E-2
                force.append(fi)
                if fi >= allowable_force_threshold_n:
                    msg = f'The force on the sample ({fi} N) is larger than the allowable limit: ' \
                          f'{allowable_force_threshold_n} (N) '
                    print(msg)
                    raise ValueError(msg)
                    break
                print(f"{total_time:8.3f} s, ADC: {pot_adc:5.0f} -> {d:>5.1f} cm, {fi:>5.1f} ± {fi_err:>5.1f}N")
                previous_time = current_time

        elapsed_time = np.array(elapsed_time)
        position = np.array(position)
        force = np.array(force)
        current_position = experiment.current_position_cm
        displacement = current_position - initial_position

        avg_speed = displacement / moving_time

        print(f'Displacement: {displacement[0]:4.1f}±{displacement[1]:5.2f} cm')
        print(f'Moving time: {moving_time:5.3f} s')
        print(f'Average Speed: {avg_speed[0]:4.2f}±{avg_speed[1]:4.2f}cm/s')


        friction_df = pd.DataFrame(data={
            'Time (s)': elapsed_time,
            'Position (cm)': position,
            'Force (N)': force
        })

        print(friction_df)
        friction_df.to_csv(path_or_buf=os.path.join(base_path, file_tag + '.csv'), index=False)
        print('Path of the results file:')
        print(os.path.join(base_path, file_tag + '.csv'))
