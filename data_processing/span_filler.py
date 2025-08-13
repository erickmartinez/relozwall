from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from matplotlib.ticker import ScalarFormatter
import re

data_path = r"C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\beam_expander"
filename = 'BINDER_SCAN_PLOT_EROSION'
csv_file = 'span_filler.csv'

if __name__ == "__main__":
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3.0)

    df = pd.read_csv(os.path.join(data_path, csv_file))
    binder_content = np.array(df['Binder %'], dtype=float)
    erosion_rate = np.array(df['Erosion Rate (cm/s)'], dtype=float)
    erosion_rate_err = erosion_rate * 50
    debris_velocity = np.array(df['Mean Velocity (cm/s)'], dtype=float)
    debris_velocity_std = np.array(df['Velocity std (cm/s)'], dtype=float)
    label = df['label']
    x_pos = np.arange(0, len(label))

    leg = ax.legend(frameon=True, loc='best', fontsize=8)
    ax.bar(binder_content, erosion_rate, align='center', alpha=0.5, ecolor='black', capsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(data_path, f'{filename}_erosion.png'), dpi=600)

    plt.show()
