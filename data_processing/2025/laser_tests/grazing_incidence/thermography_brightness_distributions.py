import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tkinter.filedialog import askopenfile
import matplotlib as mpl
import matplotlib.ticker as ticker
import pandas as pd
import json
from data_processing.misc_utils.plot_style import load_plot_style
from pathlib import Path
import tifffile
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.animation import FFMpegWriter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.patches as patches
from scipy.interpolate import CubicSpline
import h5py
from data_processing.utils import get_experiment_params
import cv2


"""
Estimate the temperature distribution in the IR thermography images from laser heat tests. 
Sample temperature from an area corresponding to the DiMES head.
Using preprocessed images from 'thermography_make_video.py', which removed spurious reflections.
"""
# STACK_FILE =  r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/thermal_images/LCT_R5N15_0602_020PCT_2025-06-03_2/LCT_R5N15_0602_020PCT_2025-06-03_2_grayscale_stack.tif'
# METADATA_FILE = 'LCT_R5N15_0602_020PCT_2025-06-03_2_temperature_stack_metadata.json'
STACK_FILE =  r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/thermal_images/LCT_R5N16-0914_060PCT_2025-09-15_1/LCT_R5N16-0914_060PCT_2025-09-15_1_grayscale_stack.tif'
METADATA_FILE = 'LCT_R5N16-0914_060PCT_2025-09-15_1_temperature_stack_metadata.json'
# STACK_FILE =  r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/thermal_images/LCT_R5N16-0903_040PCT_2025-09-11_1/LCT_R5N16-0903_040PCT_2025-09-11_1_grayscale_stack.tif'
# METADATA_FILE = 'LCT_R5N16-0903_040PCT_2025-09-11_1_temperature_stack_metadata.json'

DIMES_DIAMETER = 3.95 # cm
COLOR_MAP = 'viridis'
SHOW_FRAME = 95 # The number of the frame over which to plot the sampling area (though all frames will be processed)
CAMERA_CALIBRATION_PATH = '../calibration/CALIBRATION_20231010_boron'


""" Uncomment for grazing incidence tests """
REFERENCE_ROD_DIAMETER = 1.27 # cm
MEASURED_ELIPSE_RADII = [85.37, 155.85] # minor and major radius in pixels The major radius should be vertical
BEAM_CENTER = [1006, 436] # The beam center in pixels (measured in imageJ), used to draw a circular mask around it
BEAM_DIAMETER = 448 # The beam diameter in pixels.
REFLECTIONS_FRAME_REF = 95 # The number of the frame containing strong spurious reflections but no particles
FLARE_CENTER = [625, 608] # The center of the 468 px diameter circle around which we want to avoid the flare
ROI_DIAMETER_PX = 196
EXPERIMENT_DATA_PATH = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/GRAZING_INCIDENCE/'


""" Uncomment for normal incidence """
# REFERENCE_ROD_DIAMETER = 1.0 # cm
# MEASURED_ELIPSE_RADII = [184*0.5, 202*0.5] # minor and major radius in pixels The major radius should be vertical
# BEAM_CENTER = [446, 432] # The beam center in pixels (measured in imageJ), used to draw a circular mask around it
# BEAM_DIAMETER = 600 # The beam diameter in pixels.
# REFLECTIONS_FRAME_REF = 175 # The number of the frame containing strong spurious reflections but no particles
# FLARE_CENTER = [889, 628] # The center of the 468 px diameter circle around which we want to avoid the flare
# ROI_DIAMETER_PX = 184
# EXPERIMENT_DATA_PATH = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/BORON_PHENOLIC'



def get_distance_calibration(reference_diameter, ellipse_radii):
    radius_minor, radius_major = ellipse_radii # the ellipse radii in pixels
    pixels_per_cm = 2 * radius_major / reference_diameter
    angle = np.arccos(radius_minor / radius_major)
    return pixels_per_cm, np.degrees(angle)

def load_times_from_json(json_file):
    with open(json_file) as json_file:
        data = json.load(json_file)
        time_s = np.array(data['t (s)'], dtype=float)
    return time_s

def convert_to_temperature(signal, cali: np.ndarray) -> np.ndarray:
    n, m = signal.shape
    temp_img = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            if signal[i, j] == 0:
                temp_img[i, j] = 300.
                continue
            s = signal[i, j]
            s = min(s, 255)
            temp_img[i, j] = cali[int(s)] if s > 0 else 300.
    return temp_img

def plot_sampling_area(
    img, t, pixels_per_cm, beam_center_pixels, dimes_diameter_cm, roi_diameter, mask_dimes, temp_calibration, color_map=COLOR_MAP, debug=False
):
    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(4., 2.6)
    cmap = mpl.colormaps.get_cmap(color_map)
    # norm = plt.Normalize(vmin=0, vmax=255)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    dimes_radius_px = dimes_diameter_cm * pixels_per_cm
    temp_img = convert_to_temperature(signal=img, cali=temp_calibration)
    temp_max = temp_img.flatten().max()
    norm = plt.Normalize(vmin=1000, vmax=np.ceil(temp_max / 200) * 200)
    cs = ax.imshow(temp_img, interpolation='gaussian', norm=norm, cmap=cmap, rasterized=False)  # ,
    circle_dimes = patches.Circle(
        (beam_center_pixels[0], beam_center_pixels[1]), dimes_radius_px / 2, fill=False, edgecolor='red', linewidth=1
    )
    circle_roi = patches.Circle(
        (beam_center_pixels[0], beam_center_pixels[1]), roi_diameter / 2, fill=False, edgecolor='red', linewidth=1
    )
    ax.add_patch(circle_dimes)
    ax.add_patch(circle_roi)

    if debug:
        marker = ax.scatter(beam_center_pixels[0], beam_center_pixels[1], s=300, marker='x', c='r')
        img_copy = img.copy()
        img_copy[~mask_dimes] = 0
        img_copy[mask_dimes] = 255
        overlay_cmap = mpl.colormaps.get_cmap('binary')
        circle_dimes_msk = ax.imshow(img_copy, interpolation='gaussian', norm=plt.Normalize(vmin=0, vmax=255), cmap=overlay_cmap, rasterized=False, alpha=0.25)


    cbar = fig.colorbar(cs, cax=cax)#, extend='min')
    cbar.set_label('Temperature', size=9)  # , labelpad=9)
    # cbar.ax.set_ylim(1000, np.ceil(temp_max/200)*200)
    cbar.ax.tick_params(labelsize=9)
    scalebar = ScaleBar(1 / pixels_per_cm, 'cm', frameon=False, color='w', scale_loc='top', location='lower left')
    ax.add_artist(scalebar)
    ax.set_xticks([])
    ax.set_yticks([])

    clock_txt = ax.text(
        0.035, 0.95, f'{t:>4.3f} s',
        horizontalalignment='left',
        verticalalignment='top',
        color='w',
        transform=ax.transAxes,
        fontsize=10
    )

    ax.text(
        beam_center_pixels[0], (dimes_radius_px - beam_center_pixels[1])*0.5, r'DiMES head',
        ha='center', va='center',
        # transform=ax.transAxes,
        color='red',
        fontsize=10
    )

    fig.tight_layout()
    plt.show()
    return fig


def extend_for_circle(arr, i_c, j_c, radius, fill_value=0, debug=False):
    old_h, old_w = arr.shape

    # Calculate overflow on each side
    top_overflow = max(0, radius - i_c)
    bottom_overflow = max(0, (i_c + radius) - (old_h - 1))
    left_overflow = max(0, radius - j_c)
    right_overflow = max(0, (j_c + radius) - (old_w - 1))

    if debug:
        print(f"Debug info:")
        print(f"  Original shape: {arr.shape}")
        print(f"  Original center: ({i_c}, {j_c})")
        print(f"  Radius: {radius}")
        print(f"  Overflows - top: {top_overflow}, bottom: {bottom_overflow}")
        print(f"  Overflows - left: {left_overflow}, right: {right_overflow}")

    # Create extended array
    new_h = old_h + top_overflow + bottom_overflow
    new_w = old_w + left_overflow + right_overflow
    extended = np.full((new_h, new_w), fill_value)

    # Place original array
    extended[top_overflow:top_overflow + old_h,
    left_overflow:left_overflow + old_w] = arr

    # New center coordinates
    new_i_c = i_c + top_overflow
    new_j_c = j_c + left_overflow

    if debug:
        print(f"  New shape: {extended.shape}")
        print(f"  New center: ({new_i_c}, {new_j_c})")

        # Verification
        print(f"  Verification:")
        print(f"    Distance to top: {new_i_c} (need >= {radius})")
        print(f"    Distance to bottom: {new_h - 1 - new_i_c} (need >= {radius})")
        print(f"    Distance to left: {new_j_c} (need >= {radius})")
        print(f"    Distance to right: {new_w - 1 - new_j_c} (need >= {radius})")

    return extended, new_i_c, new_j_c


def find_pebble_temperature(counts, bin_centers):
    msk_roi = bin_centers >= 10
    bins_roi = bin_centers[msk_roi]
    counts_roi = counts[msk_roi]
    eps = float(np.finfo(float).eps)
    log_pdf = np.log(counts_roi + eps)
    log_pdf = log_pdf - np.min(log_pdf)

    peaks, properties = find_peaks(log_pdf, prominence=0.01)


    if len(peaks) == 0:
        print(f"No peaks found")
        return 0, counts[0]

    peak_centers = bins_roi[peaks]
    peak_values = [counts_roi[p] for p in peaks]
    # peak_brightness = bin_centers[peaks[-1]]
    # peak_pdf = peak_values[-1]

    counts_max = -1
    peak_brightness = -1
    for i, p in enumerate(peak_values):
        if p > counts_max:
            counts_max = p
            peak_brightness = peak_centers[i]

    counts_max = np.max(counts_roi[peaks])


    return peak_brightness, counts_max

def load_calibration(file_tag, base_path=EXPERIMENT_DATA_PATH, calibration_path=CAMERA_CALIBRATION_PATH):
    params = get_experiment_params(relative_path=base_path, filename=file_tag)
    exposure_time = float(params['Camera exposure time']['value'])
    calibration_csv = Path(calibration_path) / f'calibration_20231010_{exposure_time:.0f}_us.csv'
    df = pd.read_csv(calibration_csv).apply(pd.to_numeric)
    return df['Temperature [K]'].values, exposure_time



def update_frame(
    frame, line1, clock_text, mean_brightness_handle, time_s, histograms, mean_brightness,
    temperature_cal, temperature_calibration_interp
):
    counts = histograms[frame]
    for bar, height in zip(line1, counts):
        bar.set_height(height)
    time_txt = f'{time_s[frame]:04.3f} s'
    clock_text.set_text(time_txt)
    mean_brightness_txt = rf'$\langle B \rangle = {mean_brightness[frame]:.1f} ({temperature_calibration_interp(mean_brightness[frame]):.0f}~\mathrm{{K}})$'
    mean_brightness_handle.set_text(mean_brightness_txt)
    return line1, clock_text

def main(
    stack_file=None, metadata_file=None, reference_diameter=REFERENCE_ROD_DIAMETER, ellipse_radii=MEASURED_ELIPSE_RADII,
    color_map=COLOR_MAP, show_frame=SHOW_FRAME, beam_center=BEAM_CENTER, roi_diameter=ROI_DIAMETER_PX, dimes_diameter=DIMES_DIAMETER,
):
    if stack_file is None:
        file = askopenfile(title="Select tif file", filetypes=[("TIF files", ".tif")])
        stack_file = file.name

    pixels_per_cm, angle = get_distance_calibration(reference_diameter, ellipse_radii)

    path_to_stack_file = Path(stack_file)
    file_tag = path_to_stack_file.stem

    # Get the experiment file tag from the parent directory
    experiment_tag = path_to_stack_file.parent.name
    temperature_calibration, exposure_time = load_calibration(file_tag=experiment_tag)

    # Load the experiment time from the json metadata file for the original tiff stack
    if metadata_file is None:
        file = askopenfile(title="Select metadata file", filetypes=[("JSON files", ".json")])
        metadata_file = Path(file.name).name
    path_to_json = path_to_stack_file.parent / metadata_file
    time_s = load_times_from_json(path_to_json)
    time_at_frame = time_s[show_frame]

    brightness_max = -1
    with tifffile.TiffFile(str(path_to_stack_file)) as tif:
        img = tif.pages[show_frame].asarray() - tif.pages[0].asarray()
        for page in tif.pages:
            b_max = page.asarray().flatten().max()
            if b_max > brightness_max:
                brightness_max = b_max
        total_frames = len(tif.pages)

    # print(f'Image shape: {img.shape}')

    dimes_radius_pixels = 0.5 * dimes_diameter * pixels_per_cm

    extended_img, new_jc, new_ic = extend_for_circle(
        img, i_c=beam_center[1], j_c=beam_center[0], radius=int(dimes_radius_pixels), fill_value=0
    )
    new_beam_center = [new_ic, new_jc]
    # Calculate the distance in pixels from the center of the beam to every pixel
    y_pixels, x_pixels = extended_img.shape
    distance_to_center = np.empty_like(extended_img)
    for i in range(y_pixels):
        for j in range(x_pixels):
            distance_to_center[i, j] = np.linalg.norm([i - new_beam_center[1], j - new_beam_center[0]])
    # Create a mask around the center of the beam with a diameter corresponding the DiMES head diameter
    msk_dimes = distance_to_center <= dimes_radius_pixels
    msk_rod = distance_to_center <= roi_diameter / 2

    fig = plot_sampling_area(
        extended_img, time_at_frame, pixels_per_cm, new_beam_center,
        dimes_diameter,
        roi_diameter,
        msk_dimes,
        temp_calibration=temperature_calibration,
        color_map=color_map,
        debug=False
    )

    # Now get all the histograms
    temp_min, temp_max = 300, np.ceil(temperature_calibration[int(brightness_max)]/200)*200
    temperature_resolution = 20 #
    n_bins = int((temp_max - temp_min) // temperature_resolution)

    histrogram_matrix = np.zeros((total_frames, n_bins))
    pdf_matrix = np.zeros((total_frames, n_bins))
    mean_brightness = np.zeros(total_frames)
    max_counts = 0

    with tifffile.TiffFile(str(path_to_stack_file)) as tif:
        frame0 = tif.pages[0].asarray()
        for i, page in enumerate(tif.pages):
            frame = page.asarray()
            # frame = cv2.subtract(page.asarray(), frame0)
            frame_flat = frame.flatten()

            # Extend the frame (if the DiMES head area in pixels is larger than the image) to get the statistics over
            # cool pixels (assume 300 K)
            extended_img, new_jc, new_ic = extend_for_circle(
                frame, i_c=beam_center[1], j_c=beam_center[0], radius=int(dimes_radius_pixels),fill_value=0
            )
            extended_img[~msk_rod] = 0
            brightnesses = extended_img[msk_dimes]
            # Get the mean over the brightness
            mean_brightness[i] = np.mean(brightnesses.flatten())
            # Get the histogram over the temperatures
            temperature_rod = convert_to_temperature(signal=extended_img, cali=temperature_calibration)
            temperature_rod_flat = temperature_rod[msk_rod].flatten()
            msk_gt_300 = temperature_rod_flat >= 300

            counts, bin_edges = np.histogram(temperature_rod_flat[msk_gt_300], bins=n_bins, density=False, range=(temp_min, temp_max))
            pdf, _ = np.histogram(temperature_rod_flat[msk_gt_300], bins=n_bins, density=True, range=(temp_min, temp_max))
            histrogram_matrix[i, :] = counts
            pdf_matrix[i, :] = pdf
            max_counts = max(np.max(counts), max_counts)
            if i == 0:
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 #- 0.5


    # Define function for change from brightness to temperature and viceversa
    brightness_cal = np.arange(0, 256)
    temperature_cal = temperature_calibration[brightness_cal]
    temperature_cal[0] = 300
    slope_around_zero = (temperature_cal[2] - temperature_cal[0]) / 2
    temperature_cal[1] = temperature_cal[0] + slope_around_zero
    cs_b2t = CubicSpline(np.arange(0,256), temperature_cal)
    cs_t2b = CubicSpline(temperature_cal, np.arange(0,256))

    path_to_output_data = Path(r'./thermography/data')
    path_to_output_data.mkdir(parents=True, exist_ok=True)
    new_file_tag = file_tag.replace('_grayscale_stack', '')
    path_to_h5 = path_to_output_data / f"{new_file_tag}_histogram.h5"
    with h5py.File(str(path_to_h5), "w") as hf:
        time_ds = hf.create_dataset('time', data=time_s, compression="gzip")
        time_ds.attrs['units'] = 's'
        histogram_ds = hf.create_dataset('histogram', data=histrogram_matrix, compression="gzip")
        pdf_ds = hf.create_dataset('pdf', data=pdf_matrix, compression="gzip")
        histogram_ds.attrs['type'] = 'counts'
        bin_centers_ds = hf.create_dataset('bin_centers', data=bin_centers)
        bin_centers_ds.attrs['units'] = 'Temperature (K)'
        calibration_ds = hf.create_dataset('calibration', data=temperature_cal, compression="gzip")
        calibration_ds.attrs['units'] = 'K/(unit of brightness)'
        calibration_ds.attrs['exposure_time'] = exposure_time
        mean_brightness_ds = hf.create_dataset('mean_brightness', data=mean_brightness, compression="gzip")
        mean_temperature_ds = hf.create_dataset('mean_temperature', data=cs_b2t(mean_brightness), compression="gzip")
        mean_temperature_ds.attrs['units'] = 'K'




    load_plot_style()
    fig_histogram, ax_histogram = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig_histogram.set_size_inches(4.5, 3.)



    line1 = ax_histogram.bar(bin_centers, histrogram_matrix[0,:], width=np.diff(bin_edges))#, edgecolor='k', alpha=0.7)
    # ax_histogram.fill_between(bin_centers[mask_brightness], 0, histrogram_matrix[show_frame,mask_brightness], color='tab:red', alpha=0.9, label='ROI')

    ax_histogram.set_xlabel('Temperature (K)')
    ax_histogram.set_ylabel('# Pixels')
    ax_histogram.set_yscale('log')
    mean_brightness_txt = rf'$\langle B \rangle = {mean_brightness[0]:.0f} ({cs_b2t(mean_brightness[0]):.0f}~\mathrm{{K}})$'
    mean_text_handle = ax_histogram.text(
        0.975, 0.975, mean_brightness_txt,
        transform=ax_histogram.transAxes,
        ha='right', va='top', usetex=True
    )


    clock_handle = ax_histogram.text(
        0.025, 0.975, f'{time_at_frame:>4.3f} s',
        transform=ax_histogram.transAxes,
        ha='left', va='top', fontsize=9
    )

    # ax_temp = ax_histogram.secondary_xaxis(location='top', functions=(cs_b2t, cs_t2b))
    # ax_temp.set_xlabel('Temperature (K)')
    ax_histogram.set_xlim(temp_min, temp_max)
    number_str, exponent_str = f'{max_counts:.0E}'.split('E')
    ax_histogram.set_ylim(top=10**(int(exponent_str)+1))

    frame_rate = 1. / np.mean(np.diff(time_s))
    print(f'Frame rate: {frame_rate:.2f} Hz')
    metadata = dict(title=f'{file_tag}', artist='thermography_brightness_distributions.py',
                    comment=f'frame rate: {frame_rate}')
    writer = FFMpegWriter(fps=10, metadata=metadata, codec='mpeg4')

    ani = animation.FuncAnimation(
        fig=fig_histogram, func=update_frame, frames=np.arange(0, len(time_s), 1), interval=30,
        fargs=(
            line1, clock_handle, mean_text_handle, time_s, histrogram_matrix,
            mean_brightness,
            temperature_cal,
            cs_b2t
        )
    )

    path_to_figures = Path(r'./thermography/figures') / file_tag
    path_to_figures.mkdir(parents=True, exist_ok=True)
    path_to_figure_file = path_to_figures / f'brightness_histogram_frame_{show_frame}.png'
    fig_histogram.savefig(path_to_figure_file, dpi=600)

    plt.show()

    path_to_ani = path_to_figures / f'{file_tag}_histogram_movie.mp4'
    print('saving animation to', path_to_ani)
    ani.save(filename=str(path_to_ani), dpi=600, writer=writer)

    path_to_image_figure = path_to_figures / f'brightness_frame_{show_frame}.png'
    # fig.savefig(path_to_image_figure, dpi=600)

    # Make a figure of the calibrations
    fig_cal, ax_cal = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig_cal.set_size_inches(4.5, 3.)
    ax_cal.plot(np.arange(0, 256), temperature_calibration, color='C0', alpha=0.9, lw=2, label='Calibration')
    ax_cal.plot(np.arange(0, 256), cs_b2t(np.arange(0, 256)), color='C1', alpha=0.9, lw=2, label='Modified')
    ax_cal.legend(loc='lower right', frameon=True, fontsize=9)
    # ax_cal.set_yscale('log')

    ax_cal.set_xlabel('Brightness (a.u.)')
    ax_cal.set_ylabel('Temperature (K)')

    ax_cal.set_title(f'Calibration (Integration time: {exposure_time:.0f} us)')
    path_to_cal_figure = path_to_figures / f'temperature_calibration_{exposure_time:.0}um_calibration.png'
    fig_cal.savefig(path_to_cal_figure, dpi=600)


    plt.show()











if __name__ == '__main__':
    main(
        stack_file=STACK_FILE,metadata_file=METADATA_FILE, reference_diameter=REFERENCE_ROD_DIAMETER, ellipse_radii=MEASURED_ELIPSE_RADII,
        color_map=COLOR_MAP, show_frame=SHOW_FRAME, beam_center=BEAM_CENTER, dimes_diameter=DIMES_DIAMETER
    )



