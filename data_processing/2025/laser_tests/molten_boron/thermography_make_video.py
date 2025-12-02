import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tkinter.filedialog import askopenfile
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
from data_processing.misc_utils.plot_style import load_plot_style
from pathlib import Path
import tifffile
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.animation import FFMpegWriter
from scipy.ndimage import binary_dilation, median_filter, gaussian_filter
import matplotlib.patches as patches

"""
Removing spurious reflections
-----------------------------
Pick a frame in which spurious reflections are bright but does not contain pebbles ejected from the rod.
This frame will be subtracted from every other frame. However, the rod itself should not get subtracted (assume it
does not contain reflections).

Select a region around beam center (which is aimed at the rod) on all frames. Make a mask to later on avoid subtraction
of this area.

Alternatively, we could try averaging frames which contain reflections to get a larger area for the reflections without
much particles

Avoid removing the camera flare for which there are plenty of particle trajectories. 

To remove the camera flare, create a mask which would prevent subtraction of the background. The mask is defined
by looking for signal > 300 K around a cirlce of diameter 468 px, centered around FLARE_CENTER
"""

MAKE_MOVIE = True # If true, make an animation and save it as an mp4 movie

""" Uncomment for grazing incidence tests """
REFERENCE_ROD_DIAMETER = 1.27 # cm
MEASURED_ELLIPSE_RADII = [85.37, 155.85] # minor and major radius in pixels The major radius should be vertical
BEAM_CENTER = [1006, 416] # The beam center in pixels (measured in imageJ), used to draw a circular mask around it
# BEAM_CENTER = [1000, 436] # The beam center in pixels (measured in imageJ), used to draw a circular mask around it
BEAM_DIAMETER = 448 # The beam diameter in pixels.
REFLECTIONS_FRAME_REF = 59 # The number of the frame containing strong spurious reflections but no particles
FLARE_CENTER = [625, 608] # The center of the 468 px diameter circle around which we want to avoid the flare
PLOT_FRAME = 77

""" Uncomment for normal incidence """
# REFERENCE_ROD_DIAMETER = 1.0 # cm
# MEASURED_ELLIPSE_RADII = [184*0.5, 202*0.5] # minor and major radius in pixels The major radius should be vertical
# BEAM_CENTER = [446, 432] # The beam center in pixels (measured in imageJ), used to draw a circular mask around it
# BEAM_DIAMETER = 600 # The beam diameter in pixels.
# REFLECTIONS_FRAME_REF = 175 # The number of the frame containing strong spurious reflections but no particles
# FLARE_CENTER = [889, 628] # The center of the 468 px diameter circle around which we want to avoid the flare



STACK_FILE =  r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/thermal_images/LCT_POLYBORON-20_030PCT_2025-12-01_1/LCT_POLYBORON-20_030PCT_2025-12-01_1_temperature.tiff'
# STACK_FILE = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/thermal_images/LCT_R5N16-0914_060PCT_2025-09-15_1/LCT_R5N16-0914_060PCT_2025-09-15_1_temperature_stack.tif'
COLOR_MAP = 'jet'

def load_times_from_json(json_file):
    with open(json_file) as json_file:
        data = json.load(json_file)
        time_s = np.array(data['t (s)'], dtype=float)
    return time_s

def update_frame(frame, cs, clock_text, path_to_tif, time_s, img_reflections, mask_reflections, mask_beam_center):
    with tifffile.TiffFile(str(path_to_tif)) as tif:
        img = tif.pages[frame].asarray()
        # img_ref = tif.pages[REFLECTIONS_FRAME_REF].asarray()
    time_txt = f'{time_s[frame]:04.3f} s'
    # img = remove_reflections_median(img, img_reflections, mask_reflections, mask_beam_center, kernel_size=5)
    cs.set_array(img)
    clock_text.set_text(time_txt)
    return cs, clock_text

# Option 1: Replace with median filter (good for small spots)
def remove_reflections_median(img, image_reflections, mask_reflections, mask_beam_center, kernel_size=5):
    """Replace masked areas with median filtered values"""
    result = img.copy()
    # Apply median filter to entire image
    filtered = median_filter(img, size=kernel_size)
    # filtered_ref = gaussian_filter(median_filter(image_reflections, size=kernel_size), sigma=10)
    # Replace only the masked (reflection) areas
    result[mask_reflections & ~mask_beam_center] = (0.5*filtered[mask_reflections & ~mask_beam_center] + 0.5*img[mask_reflections & ~mask_beam_center]) - image_reflections[mask_reflections & ~mask_beam_center]
    msk_300 = result < 300
    result[msk_300] = 300
    return result

def movie_animation(
    time_s, path_to_stack_file, img0, temp_max, img_reflections, mask_reflections, mask_beam_center, file_tag,
    pixels_per_cm, color_map=COLOR_MAP
):
    load_plot_style()
    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=False)  # , frameon=False)
    fig.set_size_inches(4., 2.6)
    cmap = mpl.colormaps.get_cmap(color_map)
    norm = plt.Normalize(vmin=1000, vmax=np.ceil(temp_max/200)*200)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)


    cs = ax.imshow(img0, interpolation='gaussian', norm=norm, cmap=cmap, rasterized=False)  # ,
    # circle = patches.Circle((beam_center[0], beam_center[1]), beam_diameter/2, fill=False, edgecolor='red', linewidth=2)

    # Add the circle patch to the axes
    # ax.add_patch(circle)
    # ax.imshow(img0, interpolation='gaussian', cmap=cmap, rasterized=False)
    # extent=(0, frameSize[1] * px2mm, 0, frameSize[0] * px2mm))
    cbar = fig.colorbar(cs, cax=cax, extend='min')
    cbar.set_label('Temperature (K)', size=9)  # , labelpad=9)
    cbar.ax.set_ylim(1000, np.ceil(temp_max/200)*200)
    # cbar.ax.ticklabel_format(axis='x', style='sci', useMathText=True)
    cbar.ax.tick_params(labelsize=9)
    # cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    # cbar.ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    # Add a scalebar
    scalebar = ScaleBar(1/pixels_per_cm, 'cm', frameon=False, color='w', scale_loc='top', location='lower right')
    ax.add_artist(scalebar)
    # remove ticks from heat map
    ax.set_xticks([])
    ax.set_yticks([])

    clock_txt = ax.text(
        0.035, 0.95, '0.000 s',
        horizontalalignment='left',
        verticalalignment='top',
        color='w',
        transform=ax.transAxes,
        fontsize=10
    )

    fig.tight_layout()

    frame_rate = 1. / np.mean(np.diff(time_s))
    print(f'Frame rate: {frame_rate:.2f} Hz')

    metadata = dict(title=f'{file_tag}', artist='thermography_make_video.py',
                    comment=f'frame rate: {frame_rate}')
    writer = FFMpegWriter(fps=10, metadata=metadata, codec='mpeg4')

    ani = animation.FuncAnimation(
        fig=fig, func=update_frame, frames=np.arange(0, len(time_s), 1), interval=30,
        fargs=(cs, clock_txt, path_to_stack_file, time_s, img_reflections, mask_reflections, mask_beam_center)
    )
    plt.show()
    path_to_ani = path_to_stack_file.parent / f'{file_tag}_movie.mp4'
    print('saving animation to', path_to_ani)
    ani.save(filename=str(path_to_ani), dpi=600, writer=writer)

def get_distance_calibration(reference_diameter, ellipse_radii):
    radius_minor, radius_major = ellipse_radii # the ellipse radii in pixels
    pixels_per_cm = 2 * radius_major / reference_diameter
    angle = np.arccos(radius_minor / radius_major)
    return pixels_per_cm, np.degrees(angle)


def main(
    reflections_frame, beam_center, beam_diameter, stack_file=None, color_map='viridis', flare_center=FLARE_CENTER,
    make_movie=MAKE_MOVIE, reference_diameter=REFERENCE_ROD_DIAMETER, ellipse_radii=MEASURED_ELLIPSE_RADII
):
    pixels_per_cm, angle = get_distance_calibration(reference_diameter, ellipse_radii)
    print(f'Pixels per cm: {pixels_per_cm:.2f} cm, angle: {angle:.2f} degrees')
    if stack_file is None:
        file = askopenfile(title="Select laser experiment file", filetypes=[("TIF files", ".tif")])
        stack_file = file.name

    path_to_stack_file = Path(stack_file)
    path_to_json = path_to_stack_file.parent / f'{path_to_stack_file.stem}_metadata.json'
    file_tag = path_to_stack_file.stem

    temp_max = 0
    img0 = None
    n_images = 0
    with tifffile.TiffFile(str(path_to_stack_file)) as tif:
        img_for_mask = tif.pages[reflections_frame].asarray()
        n_images = len(tif.pages)
        # img_for_mask = np.zeros_like(img_for_mask)
        n = 5
        cnt = 0
        for i, page in enumerate(tif.pages):
            img = page.asarray()
            img_max = np.max(img.flatten())
            temp_max = max(temp_max, img_max)
            if i == 0:
                img0 = img
            if (i >= reflections_frame - n) and (i < reflections_frame + n):
                img_for_mask += page.asarray()
                cnt += 1
        img_for_mask /= cnt

    print(f'img_for_mask.min: {img_for_mask.flatten().min()}, img_for_mask.max: {img_for_mask.flatten().max()}')
    # Apply Gaussian filter to smooth the image before masking
    # This creates more homogeneous boundaries
    sigma = 15.0  # Adjust this value to control smoothing (higher = more smoothing)
    reflections_image_smoothened = gaussian_filter(img_for_mask, sigma=sigma)
    print(f'smoothed_image.min: {reflections_image_smoothened.flatten().min()}, smoothed_image.max: {reflections_image_smoothened.flatten().max()}')
    # bring the image back to original temperature scale
    smoothed_min, smoothed_max = np.min(reflections_image_smoothened.flatten()), np.max(reflections_image_smoothened.flatten())
    reflections_image_smoothened =gaussian_filter(median_filter(reflections_image_smoothened, size=5), sigma=10)
    reflections_image_smoothened = 300 + (temp_max - 300) / (smoothed_max - smoothed_min) * (reflections_image_smoothened - smoothed_min)
    print(f'Re-scaling smoothed_image')
    print(f'smoothed_image.min: {reflections_image_smoothened.flatten().min()}, smoothed_image.max: {reflections_image_smoothened.flatten().max()}')

    mask_reflections = reflections_image_smoothened > img_for_mask.min()*1.2

    x_pixels, y_pixels = img.shape
    distance_to_center = np.empty_like(img)
    for i in range(x_pixels):
        for j in range(y_pixels):
            distance_to_center[i, j] = np.linalg.norm([i-beam_center[1], j-beam_center[0]])
    mask_beam_center = gaussian_filter(distance_to_center, sigma=10) <= beam_diameter / 2

    # Look for pixels with temp > 300 within a circle that contain the camera flare
    distance_to_flare_center = np.empty_like(img)
    for i in range(x_pixels):
        for j in range(y_pixels):
            distance_to_flare_center[i, j] = np.linalg.norm([i-flare_center[1], j-flare_center[0]])

    mask_flare_center = gaussian_filter(distance_to_flare_center, sigma=10) <= 468 / 2
    # mask_flare_center = mask_flare_center & mask
    mask_beam_center = (mask_beam_center & mask_reflections)  | mask_flare_center
    temp_max = np.ceil(temp_max/100)*100
    print(f'Max temperature: {temp_max}')
    time_s = load_times_from_json(path_to_json)
    print(f'len(time_s): {len(time_s)}')

    print(f'len(tif.pages): {n_images}')
    if make_movie:
        movie_animation(
            time_s, path_to_stack_file, img0, temp_max, reflections_image_smoothened, mask_reflections, mask_beam_center,
            file_tag, pixels_per_cm, color_map
        )

    # output_tif = path_to_stack_file.parent / f'{file_tag}_removed_reflections.tif'


    # with tifffile.TiffFile(str(path_to_stack_file)) as tif, tifffile.TiffWriter(str(output_tif), bigtiff=True) as writer:
    #     num_pages = len(tif.pages)
    #     for i, page in enumerate(tif.pages):
    #         # Get the 2D array data for the current frame
    #         img = page.asarray()
    #         # Pass the frame to the processing function
    #         processed_frame = remove_reflections_median(img, reflections_image_smoothened, mask_reflections, mask_beam_center, kernel_size=5)
    #         # Save the processed frame immediately to the output file
    #         # The 'photometric' argument can be important for some viewers.
    #         # MINISBLACK is standard for grayscale images.
    #         # Added compress='zlib' to enable compression.
    #         writer.save(processed_frame, photometric='minisblack', compression='zlib')
    #         print(f"  - Processed and wrote frame {i + 1}/{num_pages}")





if __name__ == '__main__':
    main(
        reflections_frame=REFLECTIONS_FRAME_REF, beam_center=BEAM_CENTER, beam_diameter=BEAM_DIAMETER,
        stack_file=STACK_FILE, color_map=COLOR_MAP, flare_center=FLARE_CENTER, make_movie=MAKE_MOVIE,
        reference_diameter=REFERENCE_ROD_DIAMETER, ellipse_radii=MEASURED_ELLIPSE_RADII
    )



