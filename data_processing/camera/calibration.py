import logging
import json
import os
import pandas as pd
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as manimation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.io import imread, imsave
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
from scipy.interpolate import interp1d
import datetime
from data_processing.utils import get_experiment_params
import matplotlib.ticker as ticker

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\CALIBRATION_20230726'
# images_base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\LCT_GRAPHITE_100PCT_2023-02-21_1_images'
images_base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\CALIBRATION_20230726\GAIN5dB\LCT_GRAPHITE_100PCT_2023-07-28_2_images'
calibration_csv = 'temperature_data.csv'
file_tag = 'GRAPHITE_IMG'
frame_rate = 200.0
p = re.compile(r'.*?-(\d+)\.jpg')
# spot_center = np.array([36.49, 29.77])
spot_center = np.array([36.34, 29.53])
spot_center = np.array([14.18, 29.71])
pixel_size = 20.8252  # pixels/mm
px2mm = 1. / pixel_size
diameter = 2.68
radius = 0.5 * diameter
nmax = 150


def get_file_list(base_dir: str, tag: str):
    files = []
    for f in os.listdir(base_dir):
        if f.startswith(tag) and f.endswith('.jpg'):
            files.append(f)
    return files


def get_spot_indices(center):
    x = px2mm * np.arange(0, 1440)
    y = px2mm * np.arange(0, 1080)
    r2 = radius ** 2.
    idx = []
    cx, cy = center[0], px2mm * 1080 - center[1]
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            rr = (xi - cx) ** 2.0 + (yi - cy) ** 2.0
            if rr <= r2:
                idx.append({'ix': i, 'iy': j})
    return idx


def average_value(img: np.ndarray, idx_list: list) -> float:
    s = 0.
    for idx in idx_list:
        i, j = idx['ix'], idx['iy']
        s += img[j, i]
    average = int(s / len(idx_list))
    return average


def update_line(
        n, ln, file_list, t0, temperature_func: callable, brightness_func: callable, calibration: np.ndarray,
        spot_idx: np.ndarray, avg0
):
    file = file_list[n]
    img = imread(os.path.join(images_base_path, file))

    # for x in spot_idx:
    #     ix, iy = x['ix'], x['iy']
    #     img[iy, ix] = 100
    m = p.match(file)
    dt = (float(m.group(1)) - t0) * 1E-9

    avg = average_value(img, spot_idx)
    avg = max(0, avg)
    # if dt < 0.5:
    #     avg -= avg0

    temperature_c = np.round(temperature_func(dt), 0)
    b = brightness_func(dt)
    calibration[n] = (dt, avg, b, temperature_c)
    dt_txt = f'{dt:05.4f} s'
    ln[0].set_array(img)
    ln[1].set_text(dt_txt)
    print(f'{dt:>5.4f} s, average val: {avg:>3d}, {temperature_c:>4.0f} °C, step {n:>d}/{nmax}')
    return ln


def main():
    calibration_df = pd.read_csv(os.path.join(base_dir, calibration_csv)).apply(pd.to_numeric)
    time_s = calibration_df['Time (s)'].values
    temperature_c = calibration_df['Temperature (°C)'].values
    brightness = calibration_df['Brightness at 900 nm (W/ster/cm^2)'].values

    func = interp1d(time_s, temperature_c)
    func_brightness = interp1d(time_s, brightness)
    # center[0] = 1080 * mm2px - center[0]
    center = spot_center[::]
    # center[1] = 1080 * px2mm - center[1]
    spot_indices = get_spot_indices(center)

    list_of_files = get_file_list(base_dir=images_base_path, tag=file_tag)
    files_dict = {}
    p2 = re.compile(r'.*?-(\d+)-(\d+)\.jpg')
    for i, f in enumerate(list_of_files):
        m2 = p2.match(f)
        fn = int(m2.group(1))
        ts = float(m2.group(2))
        files_dict[fn] = f

    frame_keys = list(files_dict.keys())
    frame_keys.sort()
    list_of_files = [files_dict[i] for i in frame_keys]
    # list_of_files = list_of_files[30:35]
    list_of_files = list_of_files[0:nmax]

    calibration_adc = np.empty(nmax, dtype=np.dtype([
        ('Time (s)', 'd'), ('Average ADC', 'i'), ('Brightness at 900 mm (W/ster/cm^2)', 'd'), ('Temperature (°C)', 'd')
    ]))

    img = imread(os.path.join(images_base_path, list_of_files[0]))
    avg0 = average_value(img, spot_indices)
    frameSize = img.shape
    m = p.match(list_of_files[0])
    initial_timestamp = float(m.group(1))
    print(f'Frame size: {frameSize}')
    print(f'Initial timestamp: {initial_timestamp:.0f}')

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1)
    fig.set_size_inches(px2mm * frameSize[1] / 25.4, px2mm * frameSize[0] / 25.4)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    plt.axis('off')
    norm1 = plt.Normalize(vmin=0, vmax=255)

    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="7%", pad="2%")

    cs = ax.imshow(img, interpolation='none',
                   norm=norm1, extent=(0, frameSize[1] * px2mm, 0, frameSize[0] * px2mm))
    # cbar = fig.colorbar(cs, ax=cax)
    # cbar.ax.set_ylabel('ADC value')
    # cbar.ax.set_ylim(0, 255)
    # cbar.formatter.set_powerlimits((-3, 3))
    # cbar.formatter.useMathText = True
    # cbar.update_ticks()
    plt.axis('off')

    circle = plt.Circle(center, 0.5 * diameter, ec='r', fill=False, clip_on=False, ls=(0, (1, 1)), lw=0.75)
    ax.add_patch(circle)

    time_txt = ax.text(
        0.95, 0.95, '0.0000 s',
        horizontalalignment='right',
        verticalalignment='top',
        color='w',
        transform=ax.transAxes
    )

    line = [cs, time_txt, circle]

    metadata = dict(title=f'{file_tag}', artist='Erick',
                    comment=f'frame rate: {frame_rate}')
    writer = FFMpegWriter(fps=5, metadata=metadata)

    ani = manimation.FuncAnimation(
        fig, update_line, interval=100,
        repeat=False, frames=np.arange(0, nmax, 1),
        fargs=(line, list_of_files, initial_timestamp, func, func_brightness, calibration_adc, spot_indices, avg0)
    )

    # plt.show()

    # ln, file_list, t0, temperature_func: callable, calibration: np.ndarray
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d")
    ft = file_tag + f'_temperature_spot_movie_{now_str}_5dB.mp4'
    # save_dir = os.path.dirname(base_dir)
    ani.save(os.path.join(base_dir, ft), writer=writer, dpi=pixel_size * 25.4)

    adc_df = pd.DataFrame(data=calibration_adc)
    adc_df.to_csv(os.path.join(base_dir, f'adc_calibration_{now_str}_5dB.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
