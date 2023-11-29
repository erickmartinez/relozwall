"""
This code plots the individual sphere temperature vs time for a pebble sample subject to a laser heat load
"""
import pandas as pd
import numpy as np
from data_processing.utils import get_experiment_params, latex_float, lighten_color
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tkinter.filedialog import askdirectory
import re

import platform

drive_path = ''
platform_system = platform.system()
if platform_system == 'Windows':
    drive_path = r'C:\Users\erick\OneDrive'
elif platform_system == 'Darwin':
    drive_path = '/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'

data_path = os.path.join(drive_path, r'Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC')
base_path = os.path.join(drive_path, r'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/')
tracking_list_csv = r'Firing Tests - Mass Loss - Tracking.csv'
deposition_rates_csv = os.path.join(drive_path,
                                    r'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/deposition_rates_20231128.csv')
calibration_path = os.path.join(drive_path,
                                r'Documents\ucsd\Postdoc\research\thermal camera\calibration\CALIBRATION_20231010')  # \calibration_20231010_4us.csv'

laser_power_dir = os.path.join(drive_path,
                               r'Documents\ucsd\Postdoc\research\data\firing_tests\MATERIAL_SCAN\laser_output')

"""
Parameters for the deposition cone
"""
h0 = 26.67  # cm
nn = 7.4
r_viewport = 12.5

"""
Absorption coefficient to correct signal
"""
absorption_coefficient = 5.740E-3  # /nm

"""
Parameters to estimate the laser power
"""
beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707

"""
Parameters to estimate the graphite sublimation rates
E_a: 8.2321 eV
r_0: 2.7424E+18 C/s/nm^2
"""
e_a = 8.231
r_0_c = 2.7424E+18
torrL2atoms = 3.219E19
r_0 = 2.7424/32.19*1E18 # (Torr-L/s/m^2)
"""
Parameters for the sample coordinates
"""
pixel_size = 20.4215  # pixels/mm
px2mm = 1. / pixel_size
px2cm = 0.1 * px2mm
center_mm = np.array([12.05, 30.78])

sensor_pixel_size_cm = 3.45E-4

all_tol = np.finfo(np.float64).eps
frame_rate = 200.


def load_plot_style():
    with open('../../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def load_calibration(calibration_csv):
    df = pd.read_csv(calibration_csv).apply(pd.to_numeric)
    return df['Temperature [K]'].values


def convert_to_temperature(adc, cal):
    if (type(adc) == list) or (type(adc) == np.ndarray):
        r = []
        for a in adc:
            r.append(cal[int(a)])
        return np.array(r)
    return cal[int(adc)]


def get_sublimation_rate(temperature_k, r0=r_0, ea=e_a):
    return r0 * np.exp(-ea / (8.617333262e-05 * temperature_k))

def map_laser_power_settings():
    global laser_power_dir, drive_path, beam_radius
    if platform_system != 'Windows':
        laser_power_dir = laser_power_dir.replace('\\', '/')
    rdir = laser_power_dir

    file_list = os.listdir(rdir)
    mapping = {}
    for i, f in enumerate(file_list):
        if f.endswith('.csv'):
            params = get_experiment_params(relative_path=rdir, filename=os.path.splitext(f)[0])
            laser_setpoint = int(params['Laser power setpoint']['value'])
            df = pd.read_csv(os.path.join(rdir, f), comment='#').apply(pd.to_numeric)
            laser_power = df['Laser output peak power (W)'].values
            laser_power = laser_power[laser_power > 0.0]
            mapping[laser_setpoint] = laser_power.mean()

    keys = list(mapping.keys())
    keys.sort()
    return {i: mapping[i] for i in keys}


def get_deposition_rates_df():
    global deposition_rates_csv, platform_system
    if platform_system != 'Windows':
        deposition_rates_csv = deposition_rates_csv.replace('\\', '/')
    df: pd.DataFrame = pd.read_csv(deposition_rates_csv)
    columns = df.columns
    df[columns[2::]] = df[columns[2::]].apply(pd.to_numeric)
    df[['Big spheres', 'Small spheres']] = df['Pebble material'].str.extract(
        r'\[(\d+)\,(\d+)\]').apply(pd.to_numeric)
    return df


def get_time_from_timestamps(images_path):
    file_list = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
    # The regexp pattern that scans for sample id, row id, frame id and timestamp
    p = re.compile(r'(\w+\d+\d+)\-?(\d+)?\_ROW(\d+)\_IMG\-(\d+)\-(\d+)\.jpg')
    columns = ['sample_id', 'row_id', 'frame_id', 'timestamp', 'time (s)']
    df = pd.DataFrame(columns=columns)
    df.set_index(['row_id', 'frame_id'], inplace=True)
    # df['timestamp'] = df['timestamp'].astype('u8')
    t0 = np.inf
    for i, f in enumerate(file_list):
        m = p.match(f)
        sample_id = m.group(1)
        row_id = int(m.group(3))
        frame_id = int(m.group(4))
        timestamp = int(m.group(5))
        # if i == 0:
        t0 = min(t0, timestamp)
        # t = (timestamp - t0) * 1E-9
        # print(f"SAMPLE ID: {sample_id:>7}, ROW ID: {row_id:>4d}, FRAME ID: {frame_id:>4d}, t: {t0:>4.3f} (s)")
        row_df = pd.DataFrame(data={
            'sample_id': [sample_id],
            'row_id': [row_id],
            'frame_id': [frame_id],
            'timestamp': [timestamp],
            'time (s)': [timestamp],
            # 'filename': [f]
        })
        row_df.set_index(['row_id', 'frame_id'], inplace=True)
        # row_df['timestamp'] = row_df['timestamp'].astype('str')
        # df = df.append(row_df)
        df = pd.concat([df, row_df])
    # df[['row_id', 'frame_id', 'time (s)']] = df[['row_id', 'frame_id', 'time (s)']].apply(pd.to_numeric)
    df['time (s)'] = (df['timestamp'].astype(float) - df['timestamp'].astype(float).min()) * 1E-9
    df.sort_values(by=['row_id', 'frame_id'], inplace=True)
    return df


def modified_knudsen(r_, h0_=h0, n_=nn):
    cos_q = h0_ / np.sqrt(r_ ** 2. + h0_ ** 2.)
    return np.power(cos_q, n_ + 2.)


def normalize_path(the_path):
    global platform_system
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    return the_path


def main():
    global data_path, base_path, deposition_rates_csv, calibration_path
    data_path = normalize_path(data_path)
    base_path = normalize_path(base_path)
    deposition_rates_csv = normalize_path(deposition_rates_csv)
    calibration_path = normalize_path(calibration_path)

    tracking_list_df = pd.read_csv(os.path.join(base_path, tracking_list_csv))
    samples_files = tracking_list_df['Test'].unique()

    deposition_rates_df = pd.read_csv(deposition_rates_csv)
    deposition_rates_df = deposition_rates_df.loc[:, ~deposition_rates_df.columns.str.contains('Unnamed')]
    print(deposition_rates_df)

    load_plot_style()
    laser_power_map = map_laser_power_settings()
    p_sid = re.compile(r'(R\d+N\d+)\-?(\d+)?_ROW(\d+)$')
    baking_times = {
        1: 15, 2: 30, 3: 60
    }

    # Add a column to the tracking_list_df which will contain the max temperature of each file
    tracking_list_df['Max temperature (K)'] = np.nan
    tracking_list_df['Peak pressure (Torr)'] = np.nan

    cone_factor = modified_knudsen(r_=r_viewport)

    for i, sf in enumerate(samples_files):
        fig, axes = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(4., 3.)
        print(f'Processing {sf}')
        path_to_images = os.path.join(data_path, sf + '_images')
        times_df = get_time_from_timestamps(path_to_images)
        params = get_experiment_params(data_path, sf)
        exposure_time = float(params['Camera exposure time']['value'])
        # print(f'Exposure time: {exposure_time}')
        calibration_csv = f'calibration_20231010_{exposure_time:.0f}_us.csv'
        temperature_calibration = load_calibration(os.path.join(calibration_path, calibration_csv))
        sample_id = params['Sample Name']['value']
        m_sid = p_sid.match(sample_id)
        m_groups = m_sid.groups()
        sample_id = m_groups[0]
        bk_id = int(m_groups[1]) if not m_groups[1] is None else 1
        row_id = int(m_groups[2])
        # print(f'sample_id: {sample_id}')
        baking_time = baking_times[bk_id]
        times_list = times_df['time (s)'].tolist()
        laser_power_setpoint = int(params['Laser Power Setpoint']['value'])
        laser_power_w = laser_power_map[laser_power_setpoint]
        particles_df = tracking_list_df[tracking_list_df['Test'] == sf]
        dr_sid = f'{sample_id}'
        # Read peak pressure
        experiment_csv = os.path.join(data_path, sf + '.csv')
        experiment_data_df = pd.read_csv(experiment_csv, comment='#').apply(pd.to_numeric)
        peak_pressure = experiment_data_df['Pressure (Torr)'].max()

        deposition_rate_sid_df = deposition_rates_df[deposition_rates_df['ROW'] == row_id]
        deposition_rate = deposition_rate_sid_df['Deposition rate (nm/s)'].values[0]
        print(f'Deposition rate (nm/s): {deposition_rate:.1f}')
        if len(deposition_rate_sid_df) == 0:
            print(f'ROW: {row_id} returned empty deposition df')
            raise KeyError(f'ROW: {row_id} returned empty deposition df')

        for j, row in particles_df.iterrows():
            t_csv = os.path.join(path_to_images, 'tracking', row['Analysis file'])
            # print(f'path: {t_csv}')
            adc_df: pd.DataFrame = pd.read_csv(t_csv).apply(pd.to_numeric)
            adc_df.rename(columns={' ': 'Frame', 'StdDev': 'Std', 'Unnamed: 0': 'Frame'}, inplace=True)
            first_frame = int(row['Start frame'])
            adc_df['Time (s)'] = np.array([times_list[first_frame + fi] for fi in adc_df['Frame'].tolist()])
            adc_df['Film thickness (nm)'] = deposition_rate * adc_df['Time (s)']
            adc_df['Film thickness corrected (nm)'] = cone_factor * adc_df['Film thickness (nm)']
            adc_df['Corrected mean'] = adc_df['Mean'] * np.power(10., absorption_coefficient*adc_df['Film thickness corrected (nm)'])
            adc_df['Temperature (K)'] = np.array([temperature_calibration[int(adc_mc)] for adc_mc in adc_df['Corrected mean'].tolist()])
            peak_temperature = adc_df['Temperature (K)'].max()
            # Check if you can add values to the tracking_list_df by index
            # ff = tracking_list_df.loc[j, "Analysis file"]
            # print(f'row.index: {j}, tracking_list[Analysis file]: {ff}, {row["Analysis file"]}')
            tracking_list_df.loc[j, 'Max temperature (K)'] = peak_temperature
            tracking_list_df.loc[j, 'Peak pressure (Torr)'] = peak_pressure
            adc_out_path = os.path.splitext(t_csv)[0] + '_temperature.csv'
            with open(adc_out_path, 'w') as aout:
                aout.write(f'# Sample ID: {sample_id}\n')
                aout.write(f'# ROW: {row_id}\n')
                aout.write(f'# Camera exposure time (us): {exposure_time:.0f}\n')
                aout.write(f'# Deposition rate (nm/s): {deposition_rate:.1f}\n')
                adc_df.to_csv(aout, index=False)


    tracking_list_df['Graphite sublimation rate (Torr-L/s/m^2)'] = get_sublimation_rate(temperature_k=tracking_list_df['Max temperature (K)'])
    print(tracking_list_df[['Analysis file', 'Max temperature (K)', 'Graphite sublimation rate (Torr-L/s/m^2)', 'Peak pressure (Torr)']])
    output_tracking_csv = os.path.join(base_path, tracking_list_csv)
    output_tracking_csv = os.path.splitext(output_tracking_csv)[0] + '_gsr_20231128.csv'
    tracking_list_df.to_csv(output_tracking_csv, index=False)



if __name__ == '__main__':
    load_plot_style()
    main()
