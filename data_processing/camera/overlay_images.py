import numpy as np
import os
import cv2
import pandas as pd
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as manimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from data_processing.utils import get_experiment_params
from matplotlib_scalebar.scalebar import ScaleBar
import re
import matplotlib as mpl
import json

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC'
images_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC\LCT_R4N55_100PCT_2023-03-16_1_processed_images\threshold'
info_csv = r'LCT_R4N55_100PCT_2023-03-16_1.csv'
frame_rate = 200.0
pixel_size = 20.4215  # pixels/mm
p = re.compile(r'.*?-(\d+)\.jpg')
nmax = 120
calibration_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\adc_calibration_curve.csv'
px2mm = 1. / pixel_size
px2cm = 0.1 * px2mm
center_mm = np.array([18.41, 26.77])


trajectories_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC\LCT_R4N55_100PCT_2023-03-16_1_trackpoints_revised.csv'
fitted_trajectories_csv = trajectories_csv

exclude_trajectories = [10,11,12]

crop_image = False
center = np.array([318, 496])
crop_r = 100 #9x
crop_extents = {
    'left': (center[0] - crop_r),
    'top': center[1] - crop_r,
    'right': 1440 - (center[0] + crop_r),
    'bottom': 1080 - (center[1] + crop_r)
}

def get_files(base_dir: str, tag: str):
    files = []
    for f in os.listdir(base_dir):
        if f.startswith(tag) and f.endswith('.jpg'):
            files.append(f)
    return files

def get_cropped_image(img) -> np.ndarray:
    width, height = img.shape
    left = int(crop_extents['left'])  # * width)
    top = int(crop_extents['top'])  # * height)
    right = int(crop_extents['right'])  # * width)
    bottom = int(crop_extents['bottom'])  # * height)
    w, h = right - left, top - bottom
    # img2 = crop(img, ((top, bottom), (left, right)), copy=True)
    return img[120:width, 0:height-120]

def load_calibration():
    df = pd.read_csv(calibration_csv).apply(pd.to_numeric)
    return df['Temperature (°C)'].values


def convert_to_temperature(img: np.ndarray, cali: np.ndarray) -> np.ndarray:
    n, m = img.shape
    temp_img = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            temp_img[i, j] = cali[int(img[i, j])]
    return temp_img


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


def main():
    params = get_experiment_params(relative_path=base_path, filename=os.path.splitext(info_csv)[0])
    pulse_length = float(params['Emission Time']['value'])
    file_tag = os.path.splitext(info_csv)[0]
    # images_path = os.path.join(base_path, file_tag + '_images')
    sample_name = params['Sample Name']['value']
    image_tag = sample_name + '_IMG'
    cal = load_calibration()
    # print(f'length of cal: {len(cal)}')
    # for i, c in enumerate(cal):
    #     print(f'ADC: {i}, temp: {c} °C')
    list_of_files = get_files(base_dir=images_path, tag=image_tag)
    files_dict = {}
    p2 = re.compile(r'.*?_IMG-(\d+)\.jpg')
    for i, f in enumerate(list_of_files):
        m2 = p2.match(f)
        fn = int(m2.group(1))
        files_dict[fn] = f

    frame_keys = list(files_dict.keys())
    frame_keys.sort()
    list_of_files = [files_dict[i] for i in frame_keys]
    list_of_files = list_of_files[0:nmax]

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)
    # mpl.rcParams['axes.labelpad'] = -0.5
    # mpl.rcParams['axes.titlepad'] = 0.

    # img0 = cv2.imread(os.path.join(images_path, list_of_files[19]), 0)
    img0 = cv2.imread(os.path.join(images_path, list_of_files[0]), 0)
    if crop_image:
        img0 = get_cropped_image(img0)

    frameSize = img0.shape

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    # fig.set_size_inches(px2mm * frameSize[1] / 25.4, px2mm * frameSize[0] / 25.4)
    d = 2.0
    w, h = d * 1.95, d
    fig.set_size_inches(4.0, 3.)
    im_ratio = w / h  # px2mm * frameSize[1] / 25.4 / px2mm * frameSize[0] / 25.4
    norm1 = mpl.colors.LogNorm(vmin=0, vmax=255)


    # temp0 = convert_to_temperature(img0, cal)
    # temp_img = convert_to_temperature(np.zeros_like(img0), cal)
    # temp_min = temp0.flatten().min()

    img_sum = np.zeros_like(img0, dtype=np.uint8)

    selected_frames = list_of_files[0:120]
    n_selected = len(selected_frames)

    for i, f in enumerate(selected_frames):
        m = p.match(f)
        img = cv2.imread(os.path.join(images_path, f), 0)
        if crop_image:
            img = get_cropped_image(img)
        w1 = 1.0 - 0.75*np.exp(-0.25*(i + 1))
        img = cv2.subtract(img, img0)
        img_sum = cv2.addWeighted(img_sum, 1., img, 0.75, 0.)
        # if dt <= pulse_length:
        #     img = subtract_images(img, img0)
        # img = cv2.subtract(img, img0)
        # temp_img += subtract_images(convert_to_temperature(img, cal), temp0)
        # temp_img -= temp_min
        print(f'Updating time step {i + 1:>3d}/{n_selected}, file: {f:>33s}')

    # temp_img /= n_selected
    img_sum = 255*np.array(img_sum / n_selected, dtype=np.uint8)
    # frameSize = (1440, 1080)

    cs = ax.imshow(img_sum, interpolation='lanczos', cmap=plt.cm.cividis)#, norm=norm1)
    # cs = ax.imshow(temp_img, interpolation='none', norm=norm1,
    #                extent=(0, frameSize[1] * px2cm, 0, frameSize[0] * px2cm))
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.025)
    # cbar = fig.colorbar(cs, cax=cax)
    # cbar.set_label('\nTemperature (°C)', size=9)
    # cbar.ax.set_ylim(1400, 2500)
    # cbar.ax.tick_params(labelsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    # cbar.formatter.set_powerlimits((-3, 3))
    # cbar.formatter.useMathText = True
    # cbar.update_ticks()
    # ax.set_aspect('equal', adjustable='box', anchor='C')
    # plt.axis('off')
    scalebar = ScaleBar(px2cm, 'cm', frameon=False, color='w', label_loc='top', location='lower right')
    ax.add_artist(scalebar)
    ax.set_title('Pebble trajectories')

    fitted_trajectories_df = pd.read_csv(os.path.join(base_path, fitted_trajectories_csv)).apply(pd.to_numeric)
    fitted_trajectories_df['x (px)'] = fitted_trajectories_df['x (cm)'] * 10. * pixel_size + center_mm[0] * pixel_size
    fitted_trajectories_df['y (px)'] = 1080 - (10. * pixel_size * fitted_trajectories_df['y (cm)'])

    trajectories_df = pd.read_csv(os.path.join(base_path, trajectories_csv)).apply(pd.to_numeric)
    trajectory_ids = trajectories_df['TID'].unique()
    for tid in trajectory_ids:
        if tid not in exclude_trajectories:
            points_df = trajectories_df[trajectories_df['TID'] == tid]
            # points_df = points_df[1::]  # first frame is usually just expansion (no kinematics here)
            x_vector = points_df['x [pixel]'].values
            y_vector = points_df['y [pixel]'].values
            ax.plot(x_vector, y_vector, ls='-', lw=1.0, color='w', alpha=0.5)


    # trajectories_df = pd.read_csv(trajectories_csv).apply(pd.to_numeric)
    fig.savefig(os.path.join(base_path, file_tag + '_overlaid_trajectories.svg'), dpi=600)
    fig.savefig(os.path.join(base_path, file_tag + '_overlaid_trajectories.pdf'), dpi=600)
    fig.savefig(os.path.join(base_path, file_tag + '_overlaid_trajectories.png'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
