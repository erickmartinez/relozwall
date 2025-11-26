import pandas as pd
import numpy as np
import tifffile
import os
from tkinter.filedialog import askopenfile
from pathlib import Path
import json
from data_processing.utils import get_experiment_params
import re
import cv2
from time import sleep

CALIBRATION_PATH = '../calibration/CALIBRATION_20231010_boron'
DEPOSITION_RATE = 0.  # nm/s
ABSORPTION_COEFFICIENT = 1E-3
INFO_CSV = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/MOLTEN_BORON/LCT_POLYBORON-16_040PCT_2025-11-24_1.csv'
# INFO_CSV = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/BORON_PHENOLIC/LCT_R5N15_0602_020PCT_2025-06-03_2'
LN10 = np.log(10.)
def correct_for_window_deposit_intensity(img: np.ndarray, t, deposition_rate=DEPOSITION_RATE, absorption_coefficient=ABSORPTION_COEFFICIENT):
    global LN10
    return np.exp(LN10 * absorption_coefficient * t * deposition_rate) * img

def load_times_from_json(json_file):
    with open(json_file) as json_file:
        data = json.load(json_file)
        time_s = np.array(data['t (s)'])
    return time_s

def convert_to_temperature(signal, cali: np.ndarray, dt, emission_time) -> np.ndarray:
    n, m = signal.shape
    temp_img = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            if signal[i, j] == 0:
                temp_img[i, j] = 300.
                continue
            # s = correct_for_window_deposit_intensity(signal[i, j], dt)
            # if dt <= emission_time + 0.005:
            #     s = correct_for_window_deposit_intensity(signal[i, j], emission_time)
            s = min(signal[i, j], 255)
            temp_img[i, j] = cali[int(s)] if s > 0 else 300.
    return temp_img

def load_calibration(calibration_csv):
    df = pd.read_csv(calibration_csv).apply(pd.to_numeric)
    return df['Temperature [K]'].values

def main(calibration_path, info_csv=None):
    if info_csv is None:
        file = askopenfile(title="Select laser experiment file", filetypes=[("CSV files", ".csv")])
        info_csv = file.name
    info_csv = Path(info_csv)
    base_path = info_csv.parent
    top_path = Path(base_path).parent
    base_name = Path(info_csv).name
    file_tag = info_csv.stem
    save_img_dir = top_path / 'thermal_images'
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_dir = save_img_dir / file_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    images_path = base_path / f'{file_tag}_images'

    # images_path.mkdir(parents=True, exist_ok=True)
    # image_save_path = save_dir / 'processed_images'
    # image_save_path.mkdir(parents=True, exist_ok=True)
    output_tiff_path = save_dir / f'{file_tag}_temperature.tiff'
    # output_grayscale_tif = save_dir / f'{file_tag}_grayscale_stack.tif'

    params = get_experiment_params(relative_path=base_path, filename=file_tag)
    pulse_length = float(params['Emission Time']['value'])
    sample_name = params['Sample Name']['value']
    exposure_time = float(params['Camera exposure time']['value'])
    calibration_csv = f'calibration_20231010_{exposure_time:.0f}_us.csv'
    input_tiff_path = images_path / f'{sample_name}_IMG.tiff'

    temperature_calibration = load_calibration(os.path.join(calibration_path, calibration_csv))
    time_s = load_times_from_json(images_path / f'{input_tiff_path.stem}_metadata.json')
    with tifffile.TiffFile(str(input_tiff_path)) as tif:
        img0 = tif.pages[0].asarray()

    with tifffile.TiffWriter(str(output_tiff_path), bigtiff=False, imagej=False) as output_tif, tifffile.TiffFile(str(input_tiff_path)) as input_tiff:
        for i, t, page in zip(range(len(time_s)), time_s, input_tiff.pages):
            print(f'Processing frame {i+1:>3d}/{len(input_tiff.pages)}')
            img = page.asarray()
            # img_subtracted = cv2.subtract(img, img0)
            temp_im = convert_to_temperature(img, temperature_calibration, t, pulse_length)
            output_tif.write(
                temp_im,
                compression=('zlib', 9),
                # contiguous=True
                # photometric='minisblack'
            )
            sleep(0.01)

        # # Write ImageJ metadata with timing info
        # tif.imagej_metadata = {
        #     'frames': len(list_of_files),
        #     'finterval': np.mean(np.diff(time_s)),  # Frame interval in seconds
        #     'unit': 'sec',
        # }

    metadata_file = str(output_tiff_path).replace('.tiff', '_metadata.json')
    print(f'Writing metadata to {metadata_file}')
    with open(metadata_file, 'w') as f:
        json.dump({
            't (s)': time_s.tolist(),
        }, f, indent=2)


if __name__ == '__main__':
    main(calibration_path=CALIBRATION_PATH, info_csv=INFO_CSV)
