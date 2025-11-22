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



""" Uncomment for grazing incidence tests """
REFERENCE_ROD_DIAMETER = 1.27 # cm
MEASURED_ELLIPSE_RADII = [85.37, 155.85] # minor and major radius in pixels The major radius should be vertical
BEAM_CENTER = [1006, 416] # The beam center in pixels (measured in imageJ), used to draw a circular mask around it
# BEAM_CENTER = [1000, 436] # The beam center in pixels (measured in imageJ), used to draw a circular mask around it
BEAM_DIAMETER = 448 # The beam diameter in pixels.
PLOT_FRAME = 84

""" Uncomment for normal incidence """
# REFERENCE_ROD_DIAMETER = 1.0 # cm
# MEASURED_ELLIPSE_RADII = [184*0.5, 202*0.5] # minor and major radius in pixels The major radius should be vertical
# BEAM_CENTER = [446, 432] # The beam center in pixels (measured in imageJ), used to draw a circular mask around it
# BEAM_DIAMETER = 600 # The beam diameter in pixels.
# REFLECTIONS_FRAME_REF = 175 # The number of the frame containing strong spurious reflections but no particles
# FLARE_CENTER = [889, 628] # The center of the 468 px diameter circle around which we want to avoid the flare



STACK_FILE =  r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/thermal_images/LCT_R5N16-0912_100PCT_2025-09-15_1/LCT_R5N16-0912_100PCT_2025-09-15_1_temperature_stack_removed_reflections.tif'
# STACK_FILE = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/thermal_images/LCT_R5N16-0914_060PCT_2025-09-15_1/LCT_R5N16-0914_060PCT_2025-09-15_1_temperature_stack.tif'
COLOR_MAP = 'jet'

def load_times_from_json(json_file):
    with open(json_file) as json_file:
        data = json.load(json_file)
        time_s = np.array(data['t (s)'], dtype=float)
    return time_s


def plot_movie_frame(
    frame, path_to_stack_file,
    pixels_per_cm, color_map=COLOR_MAP
):
    file_tag = path_to_stack_file.stem
    json_tag = file_tag.replace('_removed_reflections', '')
    path_to_json = path_to_stack_file.parent / f'{json_tag}_metadata.json'

    time_s = load_times_from_json(path_to_json)
    time_at_frame = time_s[frame]
    print(f'Stack file {path_to_stack_file.stem}, frame: {frame}, time: {time_at_frame}')

    temp_max = 0
    with tifffile.TiffFile(str(path_to_stack_file)) as tif:
        img = tif.pages[frame].asarray()
        for i, page in enumerate(tif.pages):
            imgi = page.asarray()
            imgi_max = np.max(img.flatten())
            temp_max = max(temp_max, imgi_max)


    load_plot_style()
    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=False)  # , frameon=False)
    fig.set_size_inches(4., 2.6)
    cmap = mpl.colormaps.get_cmap(color_map)
    norm = mpl.colors.Normalize(vmin=1000, vmax=np.ceil(temp_max/100)*100)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)


    cs = ax.imshow(img, interpolation='gaussian', norm=norm, cmap=cmap, rasterized=False)  # ,
    # circle = patches.Circle((beam_center[0], beam_center[1]), beam_diameter/2, fill=False, edgecolor='red', linewidth=2)

    # Add the circle patch to the axes
    # ax.add_patch(circle)
    # ax.imshow(img0, interpolation='gaussian', cmap=cmap, rasterized=False)
    # extent=(0, frameSize[1] * px2mm, 0, frameSize[0] * px2mm))
    cbar = fig.colorbar(cs, cax=cax, extend='min')
    cbar.set_label('Temperature (K)', size=9)  # , labelpad=9)
    cbar.ax.set_ylim(1000, np.ceil(temp_max/100)*100)
    # cbar.ax.ticklabel_format(axis='x', style='sci', useMathText=True)
    cbar.ax.tick_params(labelsize=9)
    # cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    # cbar.ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    # Add a scalebar
    scalebar = ScaleBar(1/pixels_per_cm, 'cm', frameon=False, color='w', scale_loc='top', location='lower left')
    ax.add_artist(scalebar)
    # remove ticks from heat map
    ax.set_xticks([])
    ax.set_yticks([])

    clock_txt = ax.text(
        0.035, 0.95, f'{time_at_frame:>4.3f} s',
        horizontalalignment='left',
        verticalalignment='top',
        color='w',
        transform=ax.transAxes,
        fontsize=10
    )

    fig.tight_layout()

    frame_rate = 1. / np.mean(np.diff(time_s))
    print(f'Frame rate: {frame_rate:.2f} Hz')

    for extension in ['svg', 'png', 'pdf']:
        path_to_figure = path_to_stack_file.parent / f'{file_tag}_frame_{frame}.{extension}'
        fig.savefig(path_to_figure, dpi=600)

    plt.show()


def get_distance_calibration(reference_diameter, ellipse_radii):
    radius_minor, radius_major = ellipse_radii # the ellipse radii in pixels
    pixels_per_cm = 2 * radius_major / reference_diameter
    angle = np.arccos(radius_minor / radius_major)
    return pixels_per_cm, np.degrees(angle)


def main(
    plot_frame, beam_center, beam_diameter, stack_file=None, color_map='viridis',
    reference_diameter=REFERENCE_ROD_DIAMETER, ellipse_radii=MEASURED_ELLIPSE_RADII
):
    pixels_per_cm, angle = get_distance_calibration(reference_diameter, ellipse_radii)
    print(f'Pixels per cm: {pixels_per_cm:.2f} cm, angle: {angle:.2f} degrees')
    if stack_file is None:
        file = askopenfile(title="Select laser experiment file", filetypes=[("TIF files", ".tif")])
        stack_file = file.name

    path_to_stack_file = Path(stack_file)

    plot_movie_frame(
        frame=plot_frame,
        path_to_stack_file=path_to_stack_file,
        pixels_per_cm=pixels_per_cm,
        color_map=color_map
    )







if __name__ == '__main__':
    main(
        plot_frame=PLOT_FRAME, beam_center=BEAM_CENTER, beam_diameter=BEAM_DIAMETER,
        stack_file=STACK_FILE, color_map=COLOR_MAP,
        reference_diameter=REFERENCE_ROD_DIAMETER, ellipse_radii=MEASURED_ELLIPSE_RADII
    )



