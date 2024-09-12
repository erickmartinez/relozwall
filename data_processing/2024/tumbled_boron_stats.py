import pandas as pd
from scipy.stats.distributions import t, lognorm
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json

data_csv = './data/tumbled_boron_diameters/B7-20240807.csv'
num_bins = 20


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    rcParams.update(plot_style)


def main():
    global data_csv, num_bins
    file_tag = os.path.splitext(os.path.basename(data_csv))[0]
    df = pd.read_csv(data_csv)
    area_mm = df['Area (mm^2)'].values
    # load_plot_style()
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 6.0)
    d_est = np.round(2.*np.sqrt(area_mm/np.pi), decimals=2)
    d_est_log = np.log(d_est)
    n, bins, patches = axes[0].hist(d_est, num_bins, density=True)
    n_log, bins_log, patches_log = axes[1].hist(d_est_log
                                                , num_bins, density=True)
    axes[0].set_xlabel('Estimated d (mm)')
    axes[0].set_ylabel('Counts')
    plt.show()


if __name__ == '__main__':
    main()
