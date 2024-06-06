import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import json
import platform
from data_processing.utils import get_experiment_params
import re
from scipy.stats.distributions import t

base_path = 'Documents/ucsd/Postdoc/research/data/firing_tests/SS_TUBE/GC'
csv_db = './data/pebble_ir_list.csv'
output_path = './data/tracking'

beam_radius = 0.5 * 0.8165  # cm
sample_diameter = 1.015  # cm
coating_rate = 125.  # nm/s

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

laser_power_dir = r'Documents/ucsd\Postdoc/research/data/firing_tests/MATERIAL_SCAN/laser_output'
ir_calibration_path = 'Documents/ucsd/Postdoc/research/thermal camera/calibration/CALIBRATION_20231010'

platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'


def load_calibration(calibration_csv):
    df = pd.read_csv(calibration_csv).apply(pd.to_numeric)
    return df['Temperature [K]'].values

def modified_knudsen(r_, h0_=h0, n_=nn):
    cos_q = h0_ / np.sqrt(r_ ** 2. + h0_ ** 2.)
    return np.power(cos_q, n_ + 2.)

def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def frame_number_to_time(path_to_img_folder, file_ext='.tiff'):
    file_list = [fn for fn in os.listdir(path_to_img_folder) if fn.endswith(file_ext)]
    pattern_frame_n = re.compile(rf".*?IMG-(\d+)-(\d+){file_ext}")
    mapping = {}
    for fn in file_list:
        match = pattern_frame_n.match(fn)
        mapping[int(match.group(1))] = 1E-9*float(match.group(2))

    keys = list(mapping.keys())
    keys.sort()
    time_s = np.array([mapping[k] for k in keys])
    time_s -= time_s[0]
    return time_s

def normalize_path(the_path):
    global platform_system, drive_path
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    else:
        the_path = the_path.replace('/', '\\')
    return os.path.join(drive_path, the_path)


def map_laser_power_settings():
    global laser_power_dir#, drive_path, beam_radius
    rdir = normalize_path(laser_power_dir)

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


def gaussian_beam_aperture_factor(beam_r, sample_r):
    return 1.0 - np.exp(-2.0 * (sample_r / beam_r) ** 2.0)

def main():
    global base_path, beam_radius, sample_diameter, csv_db, ir_calibration_path, coating_rate
    base_path = normalize_path(base_path)
    ir_calibration_path = normalize_path(ir_calibration_path)
    files_df: pd.DataFrame = pd.read_csv(csv_db, comment='#')
    num_cols = ['Particle ID','Start frame', 'Start x (px)', 'Start y (px)']
    files_df[num_cols] = files_df[num_cols].apply(pd.to_numeric)
    laser_test = files_df.loc[0, 'Test']
    pattern_test_id = re.compile(r'LCT.(R\d+N\d+).ROW(\d+).\d+PCT.*')
    match_test_id = pattern_test_id.match(laser_test)
    sample_id = match_test_id.group(1)
    row_id = match_test_id.group(2)
    # Get the path to the folder containing the IR images for the test
    path_images = os.path.join(base_path, laser_test + '_images')
    # Load the experiment params from the test csv
    test_params = get_experiment_params(relative_path=base_path, filename=laser_test)
    # Estimate the heat load based on the laser power setting, sample diameter, beam radius
    # and power output readout
    laser_setpoint = int(test_params['Laser Power Setpoint']['value'])
    laser_power_mapping = map_laser_power_settings()
    laser_power = round(laser_power_mapping[laser_setpoint])*1.
    av = gaussian_beam_aperture_factor(beam_r=beam_radius, sample_r=0.5*sample_diameter)
    heat_load = 1E-2 * 4. * av * laser_power / (np.pi * sample_diameter**2.)
    heat_load_rounded = round(heat_load/5)*5.
    """
    Write the sample diameter, beam diameter, heat load, sample id, and laser test row id as comments
    in the db file
    """
    with open(csv_db, 'w') as fout:
        fout.write(f'# Sample diameter: {sample_diameter:.3f} cm\n')
        fout.write(f'# Beam diameter: {beam_radius * 2.:.4f} cm\n')
        fout.write(f'# Heat load: {heat_load:.0f} MW/m^2\n')
        fout.write(f'# Sample ID: {sample_id}\n')
        fout.write(f'# Laser test: ROW{row_id}\n')
        files_df.to_csv(fout, index=False)

    # Get the IR calibration table
    exposure_time = float(test_params['Camera exposure time']['value'])
    calibration_csv = f'calibration_20231010_{exposure_time:.0f}_us.csv'
    temperature_calibration = load_calibration(os.path.join(ir_calibration_path, calibration_csv))

    """
    Before converting ADC value to temperature adjust for carbon coating rate on the view port
    """
    cone_factor = modified_knudsen(r_=r_viewport)
    coating_rate = cone_factor * coating_rate
    # adc_df['Film thickness (nm)'] = deposition_rate * adc_df['Time (s)']
    # adc_df['Film thickness corrected (nm)'] = cone_factor * adc_df['Film thickness (nm)']

    """
    Create a mapping that takes the image frame number and converts it to elapsed time
    """
    time_mapping = frame_number_to_time(path_images)

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 4.0)

    n_files = len(files_df)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    norm = mpl.colors.Normalize(vmin=0, vmax=(n_files-1))
    cmap = mpl.colormaps.get_cmap('brg')
    lcolor = [cmap(norm(i)) for i in range(n_files)]
    markers = ["o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11
]
    for i, row in files_df.iterrows():
        particle_id = i + 1
        analysis_csv = row['Analysis file']
        file_tag = os.path.splitext(analysis_csv)[0]
        csv = os.path.join(base_path, laser_test + '_images', 'tracking', analysis_csv)
        temp_df: pd.DataFrame = pd.read_csv(csv).apply(pd.to_numeric)
        temp_df.rename(columns={temp_df.columns[0]:'FID'}, inplace=True)
        """
        Estimate the standard error sigma = STDEV / SQRT(N)
        where N is the area (number of pixels over which the ADC value is averaged)
        """
        n_px = temp_df['Area'].values
        se = temp_df['StdDev'].values / np.sqrt(n_px)
        confidence = 0.95
        alpha = 1.0 - confidence
        dof = n_px - 1
        tval = np.array([t.ppf(1.0 - alpha / 2.0, x) for x in dof])
        delta = np.round(se * tval, decimals=3)
        lpb_raw, upb_raw = temp_df['Mean'].values - delta, temp_df['Mean'].values + delta

        temp_df['Time (s)'] = np.array([round(time_mapping[i]*1E5)/1E5 for i in temp_df.index])
        temp_df['Viewport coating (nm)'] = np.round(coating_rate *temp_df['Time (s)'], decimals=3)
        correction_factor = np.power(10., absorption_coefficient * temp_df['Viewport coating (nm)'].values)

        temp_df['Corrected mean'] = np.round(temp_df['Mean'] * correction_factor, decimals=3)
        temp_df['Temperature (K)'] = np.array(
            [round(temperature_calibration[int(adc_mc)]) * 1. for adc_mc in temp_df['Corrected mean'].tolist()]
        )

        lpb = np.round(lpb_raw * correction_factor, decimals=3)
        upb = np.round(upb_raw * correction_factor, decimals=3)
        temp_df['Temperature lpb (K)'] = np.array(
            [round(temperature_calibration[int(x)]) * 1. for x in lpb]
        )

        temp_df['Temperature upb (K)'] = np.array(
            [round(temperature_calibration[int(x)]) * 1. for x in upb]
        )

        out_csv = os.path.join(output_path, file_tag + '_temperature.csv')
        with open(out_csv, 'w') as fout:
            fout.write(f'# Sample diameter: {sample_diameter:.3f} cm\n')
            fout.write(f'# Beam diameter: {beam_radius*2.:.4f} cm\n')
            fout.write(f'# Heat load: {heat_load:.0f} MW/m^2\n')
            fout.write(f'# Sample ID: {sample_id}\n')
            fout.write(f'# Laser test: ROW{row_id}\n')
            temp_df.to_csv(fout, index=False)
        lbl = f'Particle {particle_id:d}'
        if particle_id == n_files:
            lbl = 'Matrix'
        ax.plot(
            temp_df['Time (s)'].values, temp_df['Temperature (K)'].values, label=analysis_csv, c=lcolor[i],
            marker=markers[i], fillstyle='none'
        )


    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (K)')
    ax.legend(loc='lower center', frameon=True, fontsize=9)

    plt.show()




if __name__ == '__main__':
    main()

