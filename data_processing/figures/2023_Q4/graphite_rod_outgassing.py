import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import platform
from skimage.util import crop
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats.distributions import t
from data_processing.utils import get_experiment_params, latex_float
import json
import cv2

platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'


image_file = 'gt001688_1.9cm_ROW403_IMG-144-4301570437080.jpg'
experiment_info = 'LCT_gt001688_1.9cm_ROW403_100PCT_2023-10-05_1'
rod_diameter = 1.9 #cm

calibration_path = r'Documents\ucsd\Postdoc\research\thermal camera\calibration\CALIBRATION_20231010'


"""
Parameters to estimate the graphite sublimation rates
E_a: 8.2321 eV
r_0: 2.7424E+18 C/s/nm^2
"""
e_a = 8.231
r_0_c = 2.7424E+18
torrL2atoms = 3.219E19
r_0 = 2.7424 / 32.19 * 1E18  # (Torr-L/s/m^2)

crop_image = True
center = np.array([280, 484])
crop_r = 200  # 9x
crop_extents = {
    'left': (center[0] - crop_r),
    'top': center[1] - crop_r,
    'right': 1440 - (center[0] + crop_r),
    'bottom': 1080 - (center[1] + crop_r)
}

pixel_size = 17.25  # pixels/mm
px2mm = 1. / pixel_size
px2cm = 0.1 * px2mm


def load_calibration(calibration_csv):
    df = pd.read_csv(calibration_csv).apply(pd.to_numeric)
    return df['Temperature [K]'].values

def convert_to_temperature(signal, cali: np.ndarray) -> np.ndarray:
    n, m = signal.shape
    temp_img = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            s = signal[i, j]
            temp_img[i, j] = cali[int(s)]
    return temp_img  # .astype(np.uint8)


def get_sublimation_rate(temperature_k, r0=r_0, ea=e_a):
    return r0 * np.exp(-ea / (8.617333262e-05 * temperature_k))


def sublimation_rate_thermal_img(t_img, sample_area, threshold=0):
    n, m = t_img.shape
    g_rate = 0.
    cnt = 0
    hot_area = 0
    for i in range(n):
        for j in range(m):
            temp_i = t_img[i, j]
            if temp_i > threshold:
                cnt += 1
                sr = get_sublimation_rate(t_img[i, j])
                hot_area += px2cm ** 2.
                g_rate += sr
    return g_rate / cnt,  hot_area


def get_cropped_image(img) -> np.ndarray:
    width, height = img.shape
    left = int(crop_extents['left'])  # * width)
    top = int(crop_extents['top'])  # * height)
    right = int(crop_extents['right'])  # * width)
    bottom = int(crop_extents['bottom'])  # * height)
    img2 = crop(img, ((top, bottom), (left, right)), copy=True)
    return img2


def get_img_msk(signal, threshold=0) -> np.ndarray:
    # n, m = signal.shape
    # temp_img = np.zeros((n, m), dtype=float)
    # for i in range(n):
    #     for j in range(m):
    #         temp_img[i, j] = cali[int(s)]
    return signal > threshold  # .astype(np.uint8)

def threshold_image(signal, threshold=0):
    msk = get_img_msk(signal, threshold)
    n, m = signal.shape
    temp_img = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            if msk[i, j]:
                temp_img[i, j] = signal[i, j]
    return temp_img

def normalize_path(the_path):
    global platform_system, drive_path
    the_path = os.path.join(drive_path, the_path)
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    return the_path

def main():
    global image_file, experiment_info, calibration_path
    calibration_path = normalize_path(calibration_path)

    params = get_experiment_params('./', experiment_info)
    exposure_time = float(params['Camera exposure time']['value'])
    pulse_length = float(params['Emission Time']['value'])
    calibration_csv = f'calibration_20231010_{exposure_time:.0f}_us.csv'
    temperature_calibration = load_calibration(os.path.join(calibration_path, calibration_csv))
    img = cv2.imread(image_file, 0)
    if crop_image:
        img = get_cropped_image(img)
        # print(f'Cropped image has the following size:')
        # frameSize = img.shape
        # print(frameSize)
    temp_im = convert_to_temperature(img, temperature_calibration)
    frameSize = img.shape

    p = re.compile(r'IMG\-(\d+)\-\d+.jpg')
    m = p.findall(image_file)
    fid = m[0]
    t_frame = float(fid) * 5E-3

    lp = 35.
    info_txt = f'{lp:.0f} ' + r'MW/m$^{\mathregular{2}}$' + '\n'
    info_txt += f'{t_frame:.3f} s'
    output_file_tag = f'{experiment_info}_thermal'

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    norm1 = plt.Normalize(vmin=1500, vmax=3200)

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)  # , frameon=False)
    # ax.margins(x=0, y=0)
    # plt.autoscale(tight=True)
    scale_factor = 3.5 if crop_image else 1.5
    aspect_ratio = 1.75 if crop_image else 1.5
    w, h = frameSize[1] * scale_factor * aspect_ratio * px2mm / 25.4, frameSize[
        0] * scale_factor * px2mm / 25.4
    fig.set_size_inches(w, h)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.025)

    cs = ax.imshow(temp_im, interpolation='none', norm=norm1)  # ,
    # extent=(0, frameSize[1] * px2mm, 0, frameSize[0] * px2mm))
    cbar = fig.colorbar(cs, cax=cax, extend='both')
    cbar.set_label('Temperature (K)\n', size=9, labelpad=9)
    cbar.ax.set_ylim(1500, 3200)
    cbar.ax.ticklabel_format(axis='x', style='sci', useMathText=True)
    cbar.ax.tick_params(labelsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar.update_ticks()

    fig2, ax2 = plt.subplots(ncols=1, nrows=1, constrained_layout=True)  # , frameon=False)
    w, h = frameSize[1] * scale_factor * aspect_ratio * px2mm / 25.4, frameSize[
        0] * scale_factor * px2mm / 25.4
    fig2.set_size_inches(w, h)
    cs2 = ax2.imshow(threshold_image(temp_im, threshold=temp_im.max()*0.95), interpolation='none', norm=norm1)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="7%", pad=0.025)

    # extent=(0, frameSize[1] * px2mm, 0, frameSize[0] * px2mm))
    cbar2 = fig2.colorbar(cs2, cax=cax2, extend='both')
    cbar2.set_label('Temperature (K)\n', size=9, labelpad=9)
    cbar2.ax.set_ylim(1500, 3200)
    cbar2.ax.ticklabel_format(axis='x', style='sci', useMathText=True)
    cbar2.ax.tick_params(labelsize=9)
    ax2.set_xticks([])
    ax2.set_yticks([])
    cbar2.update_ticks()


    rod_area = 0.25 * np.pi * (rod_diameter ** 2.)
    # print(temp_im, get_img_msk(img), rod_area)

    gs_rate, area_hot = sublimation_rate_thermal_img(t_img=temp_im, threshold=temp_im.max()*0.95, sample_area=rod_area)
    print(f'Sublimation rate: {gs_rate:.3E} Torr-L/s/m2')

    info_img_txt = info_txt + f'\nT$_{{\\mathregular{{max}}}}$ = {temp_im.max():.0f} K'
    info_img_txt += f'\n A: {rod_area:.2f} cm$^{{\mathregular{{2}}}}$'
    time_txt = ax.text(
        0.95, 0.95, info_img_txt,
        horizontalalignment='right',
        verticalalignment='top',
        color='w',
        transform=ax.transAxes,
        fontsize=11
    )

    g_s_txt = f'\nGraphite sublimation:\n{gs_rate:.1f} Torr-L/s/m$^{{\mathregular{{2}}}}$'
    g_s_txt += f'\nArea hot: {area_hot:.2f} cm$^{{\mathregular{{2}}}}$'
    time_txt = ax.text(
        0.05, 0.05, g_s_txt,
        horizontalalignment='left',
        verticalalignment='bottom',
        color='w',
        transform=ax.transAxes,
        fontsize=11
    )

    hot_spot_txt = ax2.text(
        0.05, 0.05, 'Hot spots',
        horizontalalignment='left',
        verticalalignment='bottom',
        color='w',
        transform=ax2.transAxes,
        fontsize=11
    )
    # transmission_df.loc[i, 'Graphite sublimation (Torr-L/s/m^2)'] = gs_rate
    fig.savefig(output_file_tag + '.png', dpi=600)
    fig2.savefig(output_file_tag + 'hot_spots_.png', dpi=600)
    # fig.savefig(os.path.join(base_path, os.path.splitext(image_file)[0] + '_temp.svg'), dpi=600)
    # fig.savefig(os.path.join(base_path, os.path.splitext(image_file)[0] + '_temp.pdf'), dpi=600)
    # fig.savefig(os.path.join(base_path, os.path.splitext(image_file)[0] + '_temp.eps'), dpi=600)
    # plt.close(fig)
    plt.show()


if __name__ == '__main__':
    main()