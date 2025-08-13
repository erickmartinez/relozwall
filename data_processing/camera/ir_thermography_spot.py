import logging
import json
import os
import sys

sys.path.append('../')
import data_processing.confidence as cf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage.filters as filters
from skimage.io import imread, imsave
import itertools
from scipy.optimize import least_squares

from skimage.util import img_as_ubyte
import matplotlib.ticker as ticker

data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration'
image_file = 'ir_thermography_spot_size_20.8252px_per_mm.png'
# base_path = r'G:\Shared drives\ARPA-E Project\Lab\Data\Laser Tests\CAMERA\BEAM_PROFILING_20221212'
center = np.array([13.97, 12.03])
pixel_size = 20.8252  # pixels/mm
diameter = 2.68


if __name__ == '__main__':
    img = imread(os.path.join(data_path, image_file))
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    img_shape = img.shape

    norm1 = plt.Normalize(vmin=0, vmax=255)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4.0, 3.0), constrained_layout=True)
    ax.imshow(img, interpolation='none', norm=norm1, extent=(0, img_shape[1]/pixel_size, 0, img_shape[0]/pixel_size))
    circle = plt.Circle(center, 0.5*diameter, ec='r', fill=False, clip_on=False, ls=(0, (1, 1)), lw=1.0)

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    ax.set_title('IR thermography spot size', fontweight='regular')
    wz_text = f'Ø: {diameter:.2f} mm'

    # ax.text(
    #     0.95, 0.05, wz_text, color='w',
    #     transform=ax.transAxes, va='bottom', ha='right',
    #     fontsize=11
    # )

    q = 45.0
    x1 = center[0] + 0.5 * diameter * np.cos(q)
    y1 = center[1] - 0.5 * diameter * np.sin(q)

    x2 = center[0] + 2.0 * diameter * np.cos(q) + 1.0
    y2 = center[1] - 2.0 * diameter * np.sin(q)

    connectionstyle = "angle,angleA=0,angleB=-90,rad=0"

    ax.annotate(
        wz_text,
        xy=(x1, y1), xycoords='data',
        xytext=(x2, y2), textcoords='data',
        color='w', ha='left', va='center',
        arrowprops=dict(
            arrowstyle="->", color="w",
            shrinkA=-30, shrinkB=2,
            patchA=None, patchB=None,
            connectionstyle=connectionstyle,
        )
    )

    ax.add_patch(circle)

    ax.set_xlim(0, img_shape[1]/pixel_size)
    ax.set_ylim(top=0, bottom=img_shape[0]/pixel_size)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(100))

    fig.savefig(os.path.join(data_path, 'result.png'), dpi=600)
    plt.show()