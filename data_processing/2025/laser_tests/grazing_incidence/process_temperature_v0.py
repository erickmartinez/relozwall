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

CALIBRATION_PATH = '../calibration/CALIBRATION_20231010_boron'
DEPOSITION_RATE = 0.  # nm/s
ABSORPTION_COEFFICIENT = 1E-3
INFO_CSV = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/GRAZING_INCIDENCE/LCT_R5N16-0912_100PCT_2025-09-15_1'

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
            s = correct_for_window_deposit_intensity(signal[i, j], dt)
            if dt <= emission_time + 0.005:
                s = correct_for_window_deposit_intensity(signal[i, j], emission_time)
            s = min(s, 255)
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
    output_tif = save_dir / f'{file_tag}_temperature_stack.tif'
    output_grayscale_tif = save_dir / f'{file_tag}_grayscale_stack.tif'

    params = get_experiment_params(relative_path=base_path, filename=file_tag)
    pulse_length = float(params['Emission Time']['value'])
    sample_name = params['Sample Name']['value']
    exposure_time = float(params['Camera exposure time']['value'])
    calibration_csv = f'calibration_20231010_{exposure_time:.0f}_us.csv'

    temperature_calibration = load_calibration(os.path.join(calibration_path, calibration_csv))
    list_of_files = [filename for filename in os.listdir(str(images_path)) if filename.endswith('.tiff')]
    files_dict = {}
    p2 = re.compile(r'.*?-(\d+)-(\d+)\.tiff')
    time_unsorted = {}
    for i, f in enumerate(list_of_files):
        m2 = p2.match(f)
        fn = int(m2.group(1))
        ts = float(m2.group(2))
        time_unsorted[fn] = ts
        files_dict[fn] = f

    frame_keys = list(files_dict.keys())
    frame_keys.sort()
    list_of_files = [files_dict[i] for i in frame_keys]
    time_s = np.array([time_unsorted[i]*1E-9 for i in frame_keys])
    time_s -= time_s.min()

    img0 = cv2.imread(str(images_path / list_of_files[0]), 0)

    with tifffile.TiffWriter(str(output_tif), bigtiff=True, imagej=False) as tif, tifffile.TiffWriter(str(output_grayscale_tif), bigtiff=True, imagej=False) as grayscale_tif:
        for i, t, fn in zip(range(len(time_s)), time_s, list_of_files):
            print(f'Processing frame {i+1:>3d}/{len(list_of_files)}')
            img = cv2.imread(str(images_path / fn), 0)
            img_subtracted = cv2.subtract(img, img0)
            temp_im = convert_to_temperature(img_subtracted, temperature_calibration, t, pulse_length)
            tif.write(
                temp_im,
                compression='zlib',
                # contiguous=True
            )
            grayscale_tif.write(
                img, compression='zlib', photometric='minisblack'
            )
        # # Write ImageJ metadata with timing info
        # tif.imagej_metadata = {
        #     'frames': len(list_of_files),
        #     'finterval': np.mean(np.diff(time_s)),  # Frame interval in seconds
        #     'unit': 'sec',
        # }

    metadata_file = str(output_tif).replace('.tif', '_metadata.json')
    print(f'Writing metadata to {metadata_file}')
    with open(metadata_file, 'w') as f:
        json.dump({
            't (s)': time_s.tolist(),
        }, f, indent=2)


if __name__ == '__main__':
    main(calibration_path=CALIBRATION_PATH, info_csv=INFO_CSV)
