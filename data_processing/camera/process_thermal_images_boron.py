import sys

import numpy as np
import os
import cv2
import pandas as pd
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as manimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.util import crop
import matplotlib.pyplot as plt
from data_processing.utils import get_experiment_params
from matplotlib_scalebar.scalebar import ScaleBar
from tkinter.filedialog import askopenfile
import re
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
import platform

platform_system = platform.system()

# base_path = r'Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC'
# info_csv = r'LCT_R4N137_ROW513_100PCT_2024-01-24_1.csv'
img_extension = 'tiff'

color_map = 'viridis'
# color_map = 'coolwarm'
t_range = [1500, 3000]

"""
Remove the attenuation from carbon deposit on the windows
---------------------------------------------------------
Assumption: The material starts sublimating at t >= 0 (need to come with a better approximation)

The carbon deposit at the window has a thickness

d = deposition_rate * t (nm)

Use

I = I0 * exp(-a*d) => I0 = I * exp(a*d)
"""

# deposition_rate = 100.  # nm/s
# absorption_coefficient = 4 * np.pi * 0.3 / 656.3  # 1/nm

deposition_rate = 1.  # nm/s
absorption_coefficient = 1E-3

frame_rate = 200.0
# pixel_size = 20.4215  # pixels/mm
pixel_size = 23.6364
p = re.compile(r'.*?-(\d+)\.' + img_extension)
nmax = 200
# calibration_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\CALIBRATION_20230726\GAIN5dB\adc_calibration_curve.csv'
calibration_path = 'calibration/CALIBRATION_20231010_boron'

px2mm = 1. / pixel_size
px2cm = 0.1 * px2mm
crop_image = True
center = np.array([274, 485])
crop_r = 200  # 9x
crop_extents = {
    'left': (center[0] - crop_r),
    'top': center[1] - crop_r,
    'right': 1440 - (center[0] + crop_r),
    'bottom': 1080 - (center[1] + crop_r)
}

remove_reflection = False
reflection_extents = {
    'p1': (755, 0),
    'p2': (910, 0),
    'p3': (755, 90),
    'p4': (910, 90)
}


def get_files(base_dir: str, tag: str, ext: str = img_extension):
    files = []
    for f in os.listdir(base_dir):
        if f.startswith(tag) and f.endswith('.' + ext):
            files.append(f)
    return files


def subtract_images(img2: np.ndarray, img1: np.ndarray):
    m, n = img1.shape
    im = np.zeros_like(img1)
    for i in range(m):
        for j in range(n):
            a2 = img2[i, j]
            a1 = img1[i, j]
            d = a2 - a1 if a2 >= a1 else 0
            im[i, j] = d
    return im


def clear_pixels(img: np.ndarray, extents: dict):
    p1, p2, p3, p4 = extents['p1'], extents['p2'], extents['p3'], extents['p4']
    x1, x2 = p1[0], p2[0] + 1
    y1, y2 = p1[1], p3[1] + 1
    # print(f"x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
    # for i in range(0, 80):
    #     for j in range(763, 900):
    #         img[i, j] = 255
    img[y1:y2, x1:x2] = 0
    return img


def update_line(n, ln, file_list, t0, img0, pulse_width, cal, relative_path, file_tag, fig, image_save_path,
                emission_time):
    file = file_list[n]
    n_max = len(file_list)
    img = cv2.imread(os.path.join(relative_path, file), 0)
    # im = Image.open(os.path.join(relative_path, file))
    if crop_image:
        img = get_cropped_image(img)
        file_tag = file_tag + '_cropped'
    elif remove_reflection:
        img = clear_pixels(img=img, extents=reflection_extents)
    img = cv2.subtract(img, img0)
    m = p.match(file)
    dt = (float(m.group(1)) - t0) * 1E-9

    # if dt <= pulse_width:
    #     # tmp = img - img0
    #     # tmp[img < 0] = 0
    #     img = cv2.subtract(img, img0)
    # img = img.astype(np.uint8)
    temp_img = convert_to_temperature(img, cal, dt, emission_time)
    dt_txt = f'{dt:05.4f} s'
    ln[0].set_array(temp_img)
    ln[1].set_text(dt_txt)
    print('Updating time step {0}/{1}, T_max: {2:.0f} K, img_max: {3:d}'.format(
        n, n_max, temp_img.flatten().max(), img.flatten().max()
    ))
    fig.savefig(os.path.join(image_save_path, f'{file_tag}_{n:03d}.png'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(image_save_path, f'{file_tag}_{n:03d}.pdf'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(image_save_path, f'{file_tag}_{n:03d}.svg'), dpi=600, bbox_inches='tight')
    return ln


LN10 = np.log(10.)


def correct_for_window_deposit_intensity(img: np.ndarray, t):
    return np.exp(LN10 * absorption_coefficient * t * deposition_rate) * img


def load_calibration(calibration_csv):
    df = pd.read_csv(calibration_csv).apply(pd.to_numeric)
    return df['Temperature [K]'].values


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
            temp_img[i, j] = cali[int(s)]
    return temp_img  # .astype(np.uint8)


def get_cropped_image(img) -> np.ndarray:
    width, height = img.shape
    left = int(crop_extents['left'])  # * width)
    top = int(crop_extents['top'])  # * height)
    right = int(crop_extents['right'])  # * width)
    bottom = int(crop_extents['bottom'])  # * height)
    img2 = crop(img, ((top, bottom), (left, right)), copy=True)
    return img2

def main():
    global calibration_path, img_extension, color_map
    file = askopenfile(title="Select laser experiment file", filetypes=[("CSV files", ".csv")])
    info_csv = file.name
    base_path = os.path.dirname(info_csv)
    top_path = os.path.abspath(os.path.dirname(base_path))
    base_name = os.path.basename(info_csv)
    file_tag = os.path.splitext(base_name)[0]
    save_img_dir = os.path.abspath(os.path.join(top_path, 'thermal_images'))
    save_dir = os.path.join(save_img_dir, file_tag)
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    images_path = os.path.join(base_path, file_tag + '_images')

    image_save_path = os.path.join(save_dir, f'processed_images')
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    # print(f'Looking for .{img_extension} files in folder \'{images_path}\'')
    # print('Image folder exists:', os.path.exists(images_path))
    params = get_experiment_params(relative_path=base_path, filename=file_tag)
    pulse_length = float(params['Emission Time']['value'])
    sample_name = params['Sample Name']['value']
    exposure_time = float(params['Camera exposure time']['value'])
    calibration_csv = f'calibration_20231010_{exposure_time:.0f}_us.csv'
    # print(calibration_csv)
    temperature_calibration = load_calibration(os.path.join(calibration_path, calibration_csv))
    # print(temperature_calibration)
    # print(f'length of cal: {len(cal)}')
    # for i, c in enumerate(cal):
    #     print(f'ADC: {i}, temp: {c} K')
    image_tag = sample_name + '_IMG'
    list_of_files = get_files(base_dir=images_path, tag=image_tag)
    files_dict = {}
    p2 = re.compile(r'.*?-(\d+)-(\d+)\.' + img_extension)
    for i, f in enumerate(list_of_files):
        m2 = p2.match(f)
        fn = int(m2.group(1))
        ts = float(m2.group(2))
        files_dict[fn] = f

    frame_keys = list(files_dict.keys())
    frame_keys.sort()
    list_of_files = [files_dict[i] for i in frame_keys]
    list_of_files = list_of_files[0:nmax]
    # for f in list_of_files:
    #     print(f)
    # frameSize = (1440, 1080)
    img = cv2.imread(os.path.join(images_path, list_of_files[0]), 0)
    # img = Image.open(os.path.join(images_path, list_of_files[0]))
    print(f'Image{list_of_files[0]} has the following size:')
    frameSize = img.shape

    print(frameSize)
    if crop_image:
        img = get_cropped_image(img)
        print(f'Cropped image has the following size:')
        frameSize = img.shape
        print(frameSize)

    temp_im = convert_to_temperature(np.zeros_like(img), temperature_calibration, 0, pulse_length)

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['axes.labelpad'] = 5.

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)  # , frameon=False)
    # fig.set_size_inches(px2mm * frameSize[1] / 25.4, px2mm * frameSize[0] / 25.4)
    cmap = mpl.colormaps.get_cmap(color_map)
    scale_factor = 2.5 if crop_image else 1.
    aspect_ratio = 1.75 if crop_image else 1.5
    w, h = frameSize[1] * scale_factor * aspect_ratio * px2mm / 25.4, frameSize[0] * scale_factor * px2mm / 25.4
    fig.set_size_inches(w, h)
    norm1 = plt.Normalize(vmin=t_range[0], vmax=t_range[1])
    # ax.margins(x=0, y=0)
    # plt.autoscale(tight=True)

    # get the initial timestamp in nano seconds
    m = p.match(list_of_files[0])
    initial_timestamp = float(m.group(1))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.025)

    cs = ax.imshow(temp_im, interpolation='none', norm=norm1, cmap=cmap)  # ,
    # extent=(0, frameSize[1] * px2mm, 0, frameSize[0] * px2mm))
    cbar = fig.colorbar(cs, cax=cax, extend='min')
    cbar.set_label('Temperature (K)\n', size=9, labelpad=9)
    cbar.ax.set_ylim(t_range[0], t_range[1])
    # cbar.ax.ticklabel_format(axis='x', style='sci', useMathText=True)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    cbar.ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    # Add a scalebar
    scalebar = ScaleBar(px2cm, 'cm', frameon=False, color='w', scale_loc='top', location='lower right')
    ax.add_artist(scalebar)
    # remove ticks from heat map
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.xaxis.set_label_text("")
    # ax.yaxis.set_label_text("")
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    # ax.set_axis_off()

    # cbar.formatter.set_powerlimits((-3, 3))
    # cbar.formatter.useMathText = True
    cbar.update_ticks()
    # plt.axis('off')

    # img.close()

    time_txt = ax.text(
        0.95, 0.95, '0.0000 s',
        horizontalalignment='right',
        verticalalignment='top',
        color='w',
        transform=ax.transAxes,
        fontsize=8
    )

    # fig.tight_layout()

    line = [cs, time_txt]

    metadata = dict(title=f'{file_tag}', artist='process_thermal_images.py',
                    comment=f'frame rate: {frame_rate}')
    writer = FFMpegWriter(fps=5, metadata=metadata, codec='mpeg4')

    n_max = len(list_of_files)
    ani = manimation.FuncAnimation(
        fig, update_line, interval=100,
        repeat=False, frames=nmax,  # np.arange(0, nmax, 1),
        fargs=(
            line, list_of_files, initial_timestamp, img, pulse_length, temperature_calibration, images_path, file_tag,
            fig, image_save_path, pulse_length)
    )

    # plt.show()

    ft = file_tag + '_movie.mp4'
    if crop_image:
        ft = file_tag + '_cropped_movie.mp4'
    ani.save(os.path.join(save_dir, ft), writer=writer, dpi=300)  # dpi=pixel_size*25.4)

    # out = cv2.VideoWriter(
    #     # os.path.join(base_path, f'{file_tag}.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 5, frameSize
    #     os.path.join(save_dir, f'{ft}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), frame_rate / 10, frameSize
    # )
    #
    # for i, f in enumerate(list_of_files):
    #     m = p.match(f)
    #     dt = (float(m.group(1)) - initial_timestamp) * 1E-9
    #     txt = f'{dt:04.3f} s'
    #     # color = (0, 0, 255, 255) if dt <= 0.5 else (255, 255, 255, 255)
    #     color = (255, 255, 255, 255)
    #     img = cv2.imread(os.path.join(images_path, f))
    #     position = (10, 75)
    #     img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    #     cv2.putText(
    #         img=img,  # numpy array on which text is written
    #         text=txt,  # text
    #         org=position,  # position at which writing has to start
    #         fontFace=cv2.FONT_HERSHEY_DUPLEX ,  # font family
    #         color=color,  # font color
    #         fontScale=2,  # font stroke
    #         lineType=cv2.LINE_AA,
    #         thickness=2
    #     )
    #     out.write(img)
    #
    # out.release()


if __name__ == '__main__':
    main()
