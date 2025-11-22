import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import CubicSpline, interp1d

from data_processing.misc_utils.plot_style import load_plot_style
from pathlib import Path
import tifffile
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.animation import FFMpegWriter

PATH_TO_TIFF_FILE = r'/Volumes/KINGSTON/DIII-D Omega/DiMES TV/2025 Pebble rods/203782_1x1_B-II_410nm_1ms_low.tif'
SHOT = 203785 # shot with B-II imaging and mds spectrometer data
PLOT_TIMES = [2000, 2100, 2200, 2250] # DiMES TV time steps to plot [ms]

D_SPOT_MDS = 2.3 # mds spectrometer chord L5 spot diameter [cm], aimed at DiMES
D_DIMES = 4.78 # diameter of DiMES head [cm]
R_DIMES = 1.485 # DiMES major radius [m]
PATH_TO_EMISSION_DATA = r'../mds_spectra/data/emission_flux'
RAD_BN_PX = 33.4 # BN radius [pixels]
RAD_ROD_PX = 14.2 # radius of pebble rod [pixels]
RAD_LIP_PX = 65.5 # radius of lip [pixels]
RAD_DIMES_PX = 78.8 # radius of dimes [pixels]
PIXELS_TO_CM = D_DIMES / (2 * RAD_DIMES_PX)
RAD_SPOT_PX = D_SPOT_MDS * 0.5 / PIXELS_TO_CM # radius of mds spot [pixels]
FRAME_RATE = 50

# Coordinates of the center of the rod
CENTER_X = 322 # pixels
CENTER_Y = 233 # pixels




def radial_distance_array(center_x, center_y, height=480, width=640):
    """
    Calculate radial distance from a specified center point for each pixel in an image array.

    Parameters:
    center_x: float, x-coordinate of the center point
    center_y: float, y-coordinate of the center point
    height: int, image height (default 480)
    width: int, image width (default 640)

    Returns:
    numpy array of shape (height, width) containing radial distances
    """
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Calculate radial distance from center for each pixel
    radial_distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    return radial_distances

def load_emission_data(shot, path_to_emission_data=PATH_TO_EMISSION_DATA):
    path_to_file = Path(path_to_emission_data) / f'{shot}_emission_flux_B-II.csv'
    df = pd.read_csv(str(path_to_file), comment='#').apply(pd.to_numeric, errors='coerce')
    time_ms = 1E3 * df['time (s)'].values
    line_brightness =df['Line brightness (photons/cm^2/ster/s)'].values
    flux = df['Flux BII (molecules/s)'].values
    return time_ms, line_brightness, flux

def get_circle_coordinates(center_x:int, center_y:int, radius:int):
    theta = np.linspace(0, 2 * np.pi, num=360)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return x, y

def correct_bn_reflection(img, rad_rod_px=RAD_ROD_PX, rad_bn_px=RAD_BN_PX, center_x=CENTER_X, center_y=CENTER_Y, plot=False):
    slice = img[center_y, :]
    bgnd = np.mean(np.hstack([slice[:10], slice[-10:]]), axis=0)
    slice = slice - bgnd
    idx_bn_x1 = center_x - int(rad_bn_px)
    idx_bn_x2 = center_x + int(rad_bn_px)
    idx_rod_x1 = center_x - int(rad_rod_px)
    idx_rod_x2 = center_x + int(rad_rod_px)

    # make fits to the slope before and after idx_bn
    x_fit_1 = np.arange(idx_bn_x1-50, idx_bn_x1+1)
    x_fit_2 = np.arange(idx_bn_x2, idx_bn_x2 + 51)
    idx_left, idx_right = idx_bn_x1-50, idx_bn_x1+1
    y_fit_1= slice[idx_left:idx_right]
    idx_left, idx_right = idx_bn_x2, idx_bn_x2 + 51
    y_fit_2 = slice[idx_left:idx_right]
    fit_1 = np.polyfit(x_fit_1, y_fit_1, deg=1)
    fit_2 = np.polyfit(x_fit_2, y_fit_2, deg=1)
    x1 = np.arange(idx_bn_x1 - 50, idx_rod_x1)
    x2 = np.arange(idx_rod_x2, idx_rod_x2+51)
    p1 = np.poly1d(fit_1)
    p2 = np.poly1d(fit_2)

    enhancement_factor = np.mean([slice[idx_rod_x1]/p1(idx_rod_x1), slice[idx_rod_x2]/p2(idx_rod_x2)])
    print(f'enhancement factor: {enhancement_factor}')

    if plot:
        load_plot_style()
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        fig.set_size_inches(4, 3)
        ax.plot(slice, label='Signal')

        ax.plot(x1, p1(x1), color='r', ls='--', lw=1., label='Fit')
        ax.plot(x2, p2(x2), color='r', ls='--', lw=1.)
        ax.plot([idx_rod_x1, idx_rod_x2], [slice[idx_rod_x1], slice[idx_rod_x2]], color='r', ls='none', marker='o', mfc='none')
        ax.plot([idx_rod_x1, idx_rod_x2], [p1(idx_rod_x1-1), p2(idx_rod_x2)], color='r', ls='none', marker='s', mfc='none')
        ax.axvspan(xmin=idx_bn_x1, xmax=idx_bn_x2, color='gray', alpha=0.25, ls='none', label='BN')
        ax.axvspan(xmin=idx_rod_x1, xmax=idx_rod_x2, color='blue', alpha=0.25, ls='none', label='ROD')
        ax.set_xlabel('Pixels')
        ax.set_ylabel('Intensity (a.u.)')
        ax.legend(loc='upper left', ncol=2, fontsize=9)
        plt.show(block=False)
    return enhancement_factor

def main(
    shot, plot_times=PLOT_TIMES, path_to_tiff_file=PATH_TO_TIFF_FILE, pixels_to_cm=PIXELS_TO_CM,
    rad_rod_px=RAD_ROD_PX, rad_bn_px=RAD_BN_PX, rad_lip_px=RAD_LIP_PX, rad_dimes_px=RAD_DIMES_PX,
    rad_spot_px=RAD_SPOT_PX,
    frame_rate=FRAME_RATE, center_x=CENTER_X, center_y=CENTER_Y
):
    time_flux, line_brightness, flux = load_emission_data(shot)
    cs_line_brightness = interp1d(time_flux, line_brightness, fill_value='extrapolate')

    with tifffile.TiffFile(path_to_tiff_file) as tiff:
        n_t = len(tiff.pages)
        page = tiff.pages[0].asarray()
        n_y, n_x = page.shape

    dt = 1E3 / frame_rate # in ms
    time_dimes_tv = np.arange(n_t) * dt

    idx_plot_times = np.array([np.argmin(np.abs(time_dimes_tv - time_plot)) for time_plot in plot_times])

    load_plot_style()

    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    fig.set_size_inches(6.5, 5)

    x_rod, y_rod = get_circle_coordinates(center_x=center_x, center_y=center_y, radius=RAD_ROD_PX)
    x_bn, y_bn = get_circle_coordinates(center_x=center_x, center_y=center_y, radius=RAD_BN_PX)
    x_lip, y_lip = get_circle_coordinates(center_x=center_x, center_y=center_y, radius=RAD_LIP_PX)
    x_dimes, y_dimes = get_circle_coordinates(center_x=center_x, center_y=center_y, radius=RAD_DIMES_PX)
    x_spot, y_spot = get_circle_coordinates(center_x=center_x, center_y=center_y, radius=RAD_SPOT_PX)
    structure_color = 'w'
    structure_lw = '0.5'

    distance_from_center = radial_distance_array(center_x=center_x, center_y=center_y, height=n_y, width=n_x)
    msk_rod = distance_from_center <= rad_rod_px
    msk_bn = (distance_from_center <= rad_bn_px) & ~msk_rod
    msk_spot = (distance_from_center <= rad_spot_px)


    with tifffile.TiffFile(path_to_tiff_file) as tiff:
        img_reflection = tiff.pages[idx_plot_times[1]].asarray()
        reflection_factor = correct_bn_reflection(img=img_reflection)
        # mds_spot_min, mds_spot_max = 1E50, -1E50
        # for i, page in enumerate(tiff.pages):
        #     # Remove bn reflections
        #     img = tiff.pages[i].asarray()
        #     img[msk_bn] = img[msk_bn] / reflection_factor
        #     # Get the integrated intensities around the mds_spot
        #     mds_spot_integrated = np.sum(img[msk_spot])
        #     mds_spot_min = min(mds_spot_min, mds_spot_integrated)
        #     mds_spot_max = max(mds_spot_max, mds_spot_integrated)
        min_brightness, max_brightness = 1E50, -1E50
        min_intensity, max_intensity = 1E50, -1E50
        max_brightness_factor = -1
        for i, idx in enumerate(idx_plot_times):
            img = tiff.pages[idx].asarray()
            img[msk_bn] = img[msk_bn] / reflection_factor

            # Get the expected line brightness at t
            time_i = time_dimes_tv[i]
            spot_brightness = 4 * np.pi * cs_line_brightness(time_i)
            # print(f'Spot brightness: {spot_brightness}')

            # Get the integrated intensities around the mds_spot
            mds_spot_integrated = np.sum(img[msk_spot].flatten())
            min_intensity = min(min_intensity, np.min(img[msk_spot].flatten()))
            max_intensity = max(max_intensity, np.max(img[msk_spot].flatten()))

            # print(f'MDS spot integrated: {mds_spot_integrated}')

            brightness_factor = spot_brightness / mds_spot_integrated
            max_brightness_factor = max(max_brightness_factor, brightness_factor)

            # print(f'brightness_factor: {brightness_factor}')

            img = max_brightness_factor * img
            min_brightness = min(min_intensity, np.min(img[msk_spot].flatten()))
            max_brightness = max(max_intensity, np.max(img[msk_spot].flatten()))

        norm = plt.Normalize(vmin=4E10, vmax=6E11)
        print(f'Brightness factor: {brightness_factor}')
        print(f'min_intensity: {min_intensity}, max_intensity: {max_intensity}')
        print(f'min_brightness: {min_brightness:.3E}, max_brightness: {max_brightness:.3E}')



        for i, idx in enumerate(idx_plot_times):
            img = tiff.pages[idx].asarray()
            img[msk_bn] = img[msk_bn] / reflection_factor

            # Get the expected line brightness at t
            time_i = time_dimes_tv[i]
            spot_brightness = 4 * np.pi * cs_line_brightness(time_i)
            # print(f'Spot brightness: {spot_brightness}')

            # Get the integrated intensities around the mds_spot
            mds_spot_integrated = np.sum(img[msk_spot].flatten())
            # print(f'MDS spot integrated: {mds_spot_integrated}')

            brightness_factor = spot_brightness / mds_spot_integrated
            # print(f'brightness_factor: {brightness_factor}')

            img = max_brightness_factor * img


            ax_i = i // 2
            ax_j = i % 2
            ax = axes[ax_i, ax_j]

            im = ax.imshow(img, cmap='viridis', norm=norm, interpolation='gaussian', rasterized=True)


            ax.plot(x_rod, y_rod, color=structure_color, lw=structure_lw, alpha=0.5)
            ax.plot(x_bn, y_bn, color=structure_color, lw=structure_lw, alpha=0.5)
            ax.plot(x_lip, y_lip, color=structure_color, lw=structure_lw, alpha=0.5)
            ax.plot(x_dimes, y_dimes, color=structure_color, lw=structure_lw, alpha=0.5)
            scalebar = ScaleBar(pixels_to_cm, 'cm', frameon=False, color='w', scale_loc='top', location='lower right')
            ax.add_artist(scalebar)
            ax.set_title(f'{time_dimes_tv[idx]:.0f} ms')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.plot(x_spot, y_spot, color='r', lw=1, alpha=1, ls='--')

    # cbar = fig.colorbar(im, ax=axes[:,1], shrink=0.95, extend='both')
    # cbar.set_label(r'{\sffamily Brightness (photons/cm\textsuperscript{2}/s) ', size=11, usetex=True)  # , labelpad=9)
    fig.suptitle(f'Shot #{shot} BII')
    path_to_figures = Path(r'./figures')
    path_to_figures.mkdir(parents=True, exist_ok=True)
    for extension in ['.png', '.svg', '.pdf']:
        path_to_figure = path_to_figures / f'{shot}_DiMES_TV_BII{extension}'
        fig.savefig(path_to_figure, dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main(shot=SHOT)





