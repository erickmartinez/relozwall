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
import re
import matplotlib as mpl
import json

data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\LCT_GRAPHITE_100PCT_2023-03-10_3_images'
info_csv = r'LCT_GRAPHITE_100PCT_2023-03-10_3.csv'
file_tag = 'GRAPHITE_IMG'
frame_rate = 200.0
pixel_size = 20.8252  # pixels/mm
p = re.compile(r'.*?-(\d+)\.jpg')
nmax = 200
calibration_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\adc_calibration_curve.csv'
px2mm = 1. / pixel_size

def get_files(base_dir: str, tag: str):
    files = []
    for f in os.listdir(base_dir):
        if f.startswith(tag) and f.endswith('.jpg'):
            files.append(f)
    return files

def subtract_images(img2: np.ndarray, img1:np.ndarray):
    m, n = img1.shape
    im = np.zeros_like(img1)
    for i in range(m):
        for j in range(n):
            a2 = img2[i, j]
            a1 = img1[i, j]
            d = a2 - a1 if a2 >= a1 else 0
            im[i, j] = d
    return im



def update_line(n, ln, file_list, t0, img0, pulse_width, cal):
    file = file_list[n]
    n_max = len(file_list)
    img = imread(os.path.join(data_path, file))
    m = p.match(file)
    dt = (float(m.group(1)) - t0) * 1E-9
    if dt <= pulse_width:
        # tmp = img - img0
        # tmp[img < 0] = 0
        img = subtract_images(img, img0)
    # img = img.astype(np.uint8)
    temp_img = convert_to_temperature(img, cal)
    dt_txt = f'{dt:05.4f} s'
    ln[0].set_array(temp_img)
    ln[1].set_text(dt_txt)
    print('Updating time step {0}/{1}, T_max: {2:.0f} °C, img_max: {3:d}'.format(
        n, n_max, temp_img.flatten().max(), img.flatten().max()
    ))
    return ln


def load_calibration():
    df = pd.read_csv(calibration_csv).apply(pd.to_numeric)
    return df['Temperature (°C)'].values

def convert_to_temperature(img: np.ndarray, cali: np.ndarray) -> np.ndarray:
    n, m = img.shape
    temp_img = np.zeros((n,m), dtype=float)
    for i in range(n):
        for j in range(m):
            temp_img[i, j] = cali[int(img[i,j])]
    return temp_img #.astype(np.uint8)


def main():
    params = get_experiment_params(relative_path=os.path.dirname(data_path), filename=os.path.splitext(info_csv)[0])
    pulse_length = float(params['Emission Time']['value'])
    cal = load_calibration()
    # print(f'length of cal: {len(cal)}')
    # for i, c in enumerate(cal):
    #     print(f'ADC: {i}, temp: {c} °C')
    list_of_files = get_files(base_dir=data_path, tag=file_tag)
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
    list_of_files = list_of_files[0:nmax]
    for f in list_of_files:
        print(f)
    # frameSize = (1440, 1080)
    img = imread(os.path.join(data_path, list_of_files[0]))
    img_shape = img.shape
    temp_im = convert_to_temperature(np.zeros_like(img), cal)


    frameSize = img.shape

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['axes.labelpad'] = -0.5
    mpl.rcParams['axes.titlepad'] = 0.

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    # fig.set_size_inches(px2mm * frameSize[1] / 25.4, px2mm * frameSize[0] / 25.4)
    d = 2.0
    w, h = d*1.95, d
    fig.set_size_inches(w, h)
    im_ratio = w / h #px2mm * frameSize[1] / 25.4 / px2mm * frameSize[0] / 25.4
    norm1 = plt.Normalize(vmin=1300, vmax=2500)


    # get the initial timestamp in nano seconds
    m = p.match(list_of_files[0])
    initial_timestamp = float(m.group(1))

    # img = cv2.imread(os.path.join(base_path, list_of_files[0]))
    # height, width, layers = img.shape
    # frameSize = (width, height)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.025)

    cs = ax.imshow(temp_im, interpolation='none', norm=norm1,
                   extent=(0, frameSize[1] * px2mm, 0, frameSize[0] * px2mm))
    # cbar = fig.colorbar(cs, fraction=0.046*im_ratio, pad=-0.06)
    cbar = fig.colorbar(cs, cax=cax)
    cbar.set_label('\nTemperature (°C)', size=9)
    cbar.ax.set_ylim(1400, 2500)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    # cbar.formatter.set_powerlimits((-3, 3))
    # cbar.formatter.useMathText = True
    cbar.update_ticks()
    # plt.axis('off')


    time_txt = ax.text(
        0.95, 0.95, '0.0000 s',
        horizontalalignment='right',
        verticalalignment='top',
        color='w',
        transform=ax.transAxes,
        fontsize=8
    )

    # fig.tight_layout()
    plt.show()

    line = [cs, time_txt]

    metadata = dict(title=f'{file_tag}', artist='Erick',
                    comment=f'frame rate: {frame_rate}')
    writer = FFMpegWriter(fps=5, metadata=metadata)

    n_max = len(list_of_files)
    ani = manimation.FuncAnimation(
        fig, update_line, interval=100,
        repeat=False, frames=np.arange(0, nmax, 1),
        fargs=(line, list_of_files, initial_timestamp, img, pulse_length, cal)
    )

    ft = file_tag + '_movie.mp4'
    save_dir = os.path.dirname(data_path)
    ani.save(os.path.join(save_dir, ft), writer=writer, dpi=200) # dpi=pixel_size*25.4)

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
    #     img = cv2.imread(os.path.join(base_path, f))
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
