# import os
# import random
# from itertools import count
# from re import T
# from tkinter import Scale
import math
import matplotlib as mpl
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from matplotlib.animation import FuncAnimation
# import matplotlib.image as mpimg
# from matplotlib.ticker import (
#     MultipleLocator, FormatStrFormatter, AutoMinorLocator)
# from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox, AnchoredOffsetbox
# from pandas.io.parquet import FastParquetImpl
data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\manuscripts\paper1\inl\data_and_script_for_figures\data_and_script_for_figures\laser_heating'


def dropZero(df):
    for i in range(len(df.index)):
        if (df.iloc[i, 1] < 1e-6):
            df.drop(index=df.index[i], axis=0, inplace=True)
            break

def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


if __name__ == '__main__':
    load_plt_style()
    out = pd.read_csv(os.path.join(data_path, 'laserHeating_out_oct6.csv'))
    # columns: [time, abs_rate, avg_front_temp, avg_min_y, current_rate, flux_bottom, flux_mvg_bnd]
    cur_rate = out.loc[:, ['time', 'current_rate']]
    cur_rate.dropna(inplace=True)
    dropZero(cur_rate)

    abs_rate = out.loc[:, ['time', 'abs_rate']]
    abs_rate.dropna(inplace=True)
    dropZero(abs_rate)

    avg_temp = out.loc[:, ['time', 'avg_front_temp']]
    avg_temp.dropna(inplace=True)

    fig, ax = plt.subplots(ncols=1, nrows=2, constrained_layout=True)
    fig.set_size_inches(4.0, 5.5)

    ax[0].plot(out['time'], out['avg_front_temp'], 'r-', lw=1.75)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('K')
    ax[0].set_title('Average front surface temperature')
    ax[0].set_ylim(500.0, 5000.0)
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(1000.0))
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(500.0))

    ax[1].plot(out['time'], out['abs_rate']*100.0, 'C0-', lw=1.75,
             label='Average')
    ax[1].plot(out['time'], out['current_rate']*100.0, 'r-', lw=1.75,
             label='Current')
    ax[1].set_ylim(0.0, 1.5)
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('cm/s')
    ax[1].set_title('Disintegration rate')
    ax[1].legend(loc='upper right', prop={'size': 9})

    for i, axi in enumerate(ax.flatten()):
        axi.set_xlim(0, 3.0)
        axi.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        axi.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        panel_label = chr(ord('`') + i + 1)
        # axi.text(
        #     -0.15, 1.15, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
        #     va='top', ha='right'
        # )


    # # ax.set_yticks([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    # # ax.set_yscale('log')
    # ax.set_xlim(0, 1)
    # # ax.set_ylim(1, 5)
    # ax.ticklabel_format(style='sci')
    fig.savefig(os.path.join(data_path, 'avg_front_temp.svg'), dpi=600, format='svg')
    fig.savefig(os.path.join(data_path, 'avg_front_temp.eps'), dpi=600, format='eps')
    fig.savefig(os.path.join(data_path, 'avg_front_temp.png'), dpi=600)
    plt.show()