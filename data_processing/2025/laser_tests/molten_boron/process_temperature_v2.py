from typing import List

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
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

CALIBRATION_PATH = '../calibration/CALIBRATION_20231010_boron'
DEPOSITION_RATE = 0.  # nm/s
ABSORPTION_COEFFICIENT = 1E-3
INFO_CSV = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/MOLTEN_BORON/LCT_POLYBORON-16_040PCT_2025-11-24_1.csv'
LN10 = np.log(10.)

# Set number of workers (use None for auto-detect, or specify a number)
NUM_WORKERS = None  # Will use cpu_count() - 1


def correct_for_window_deposit_intensity(img: np.ndarray, t, deposition_rate=DEPOSITION_RATE,
                                         absorption_coefficient=ABSORPTION_COEFFICIENT):
    global LN10
    return np.exp(LN10 * absorption_coefficient * t * deposition_rate) * img


def load_times_from_json(json_file):
    with open(json_file) as json_file:
        data = json.load(json_file)
        time_s = np.array(data['t (s)'])
    return time_s


def convert_to_temperature_vectorized(signal, cali: np.ndarray, dt, emission_time) -> np.ndarray:
    """
    Vectorized version of temperature conversion - much faster than nested loops.
    """
    # Clip signal values to valid range [0, 255]
    signal_clipped = np.clip(signal, 0, 255).astype(int)

    # Use advanced indexing to convert all pixels at once
    temp_img = cali[signal_clipped]

    # Set zero-signal pixels to 300K
    temp_img[signal == 0] = 300.

    return temp_img


def process_single_frame(args):
    """
    Process a single frame - designed for parallel processing.
    Returns (frame_index, temperature_array)
    """
    frame_idx, img_data, time_val, temperature_calibration, pulse_length = args
    temp_im = convert_to_temperature_vectorized(img_data, temperature_calibration, time_val, pulse_length)
    return frame_idx, temp_im


def load_calibration(calibration_csv):
    df = pd.read_csv(calibration_csv).apply(pd.to_numeric)
    return df['Temperature [K]'].values



def main(calibration_path, info_csv=None, use_parallel=True, compression_level=6):
    """
    Main processing function with optional parallel processing.

    Parameters:
    -----------
    calibration_path : str
        Path to calibration directory
    info_csv : str, optional
        Path to experiment CSV file
    use_parallel : bool, default=True
        Whether to use parallel processing
    compression_level : int, default=6
        ZLIB compression level (0-9). Higher = better compression but slower.
        Use 6 for good balance, 9 for maximum compression, or None for no compression.
    """
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

    output_tiff_path = save_dir / f'{file_tag}_temperature.tiff'

    params = get_experiment_params(relative_path=base_path, filename=file_tag)
    pulse_length = float(params['Emission Time']['value'])
    sample_name = params['Sample Name']['value']
    exposure_time = float(params['Camera exposure time']['value'])
    calibration_csv = f'calibration_20231010_{exposure_time:.0f}_us.csv'
    input_tiff_path = images_path / f'{sample_name}_IMG.tiff'

    print(f"Loading calibration from {calibration_csv}...")
    temperature_calibration = load_calibration(os.path.join(calibration_path, calibration_csv))

    print(f"Loading time metadata...")
    time_s = load_times_from_json(images_path / f'{input_tiff_path.stem}_metadata.json')

    print(f"Loading input TIFF stack from {input_tiff_path}...")
    with tifffile.TiffFile(str(input_tiff_path)) as tif:
        total_frames = len(tif.pages)

    # Determine compression settings
    if compression_level is None:
        compression = None
        comp_str = "no compression"
    else:
        compression = ('zlib', compression_level)
        comp_str = f"zlib level {compression_level}"

    print(f"\nProcessing {total_frames} frames...")
    print(f"Parallel processing: {'ON' if use_parallel else 'OFF'}")
    print(f"Compression: {comp_str}")

    if use_parallel:
        # Parallel processing mode
        n_workers = NUM_WORKERS if NUM_WORKERS is not None else max(1, cpu_count() - 1)
        print(f"Using {n_workers} worker processes\n")

        # Load all frames into memory (if you have enough RAM)
        # For very large stacks, you might need to process in chunks
        print("Loading frames into memory...")
        with tifffile.TiffFile(str(input_tiff_path)) as input_tiff:
            frames = [page.asarray() for page in tqdm(input_tiff.pages, desc="Loading")]

        # with tifffile.TiffFile(str(input_tiff_path)) as input_tiff:
        #     time_metadata = [page.tags()]

        # Prepare arguments for parallel processing
        process_args = [
            (i, frames[i], time_s[i], temperature_calibration, pulse_length)
            for i in range(len(frames))
        ]

        # Process frames in parallel
        print("\nConverting to temperature...")
        with Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_frame, process_args),
                total=len(process_args),
                desc="Processing"
            ))

        # Sort results by frame index (in case they come back out of order)
        results.sort(key=lambda x: x[0])

        # Write results to output TIFF
        print("\nWriting output TIFF...")
        with tifffile.TiffWriter(str(output_tiff_path), bigtiff=False, imagej=False) as output_tif:
            for frame_idx, temp_im in tqdm(results, desc="Writing"):
                output_tif.write(temp_im, compression=compression)

    else:
        # Sequential processing mode (lower memory usage)
        with tifffile.TiffWriter(str(output_tiff_path), bigtiff=False, imagej=False) as output_tif, \
                tifffile.TiffFile(str(input_tiff_path)) as input_tiff:

            for i, t, page in tqdm(
                    zip(range(len(time_s)), time_s, input_tiff.pages),
                    total=len(time_s),
                    desc="Processing"
            ):
                img = page.asarray()
                temp_im = convert_to_temperature_vectorized(img, temperature_calibration, t, pulse_length)
                output_tif.write(temp_im, compression=compression)

    # Write metadata
    metadata_file = str(output_tiff_path).replace('.tiff', '_metadata.json')
    print(f'\nWriting metadata to {metadata_file}')
    with open(metadata_file, 'w') as f:
        json.dump({
            't (s)': time_s.tolist(),
        }, f, indent=2)

    print(f"\n✓ Done! Output saved to {output_tiff_path}")


if __name__ == '__main__':
    # Configuration options:
    # - use_parallel=True: Use multi-core processing (faster, uses more RAM)
    # - use_parallel=False: Sequential processing (slower, uses less RAM)
    # - compression_level: 0-9 (higher = smaller file, slower) or None (no compression, fastest)

    main(
        calibration_path=CALIBRATION_PATH,
        info_csv=INFO_CSV,
        use_parallel=True,  # Set to False if you run into memory issues
        compression_level=6  # Good balance of speed and compression
    )