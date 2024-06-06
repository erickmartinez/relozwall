import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import json

color_map = 'viridis'
t_range = [1500, 4000]


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

def main():
    global color_map, t_range
    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 1.)
    cmap = plt.colormaps.get_cmap(color_map)
    norm1 = plt.Normalize(vmin=t_range[0], vmax=t_range[1])
    cb = mpl.colorbar.ColorbarBase(
        ax, orientation='horizontal',
        cmap=color_map,
        norm=norm1,
        extend='both',
    )
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Temperature (K)', fontsize=16)
    fig.savefig(f'{color_map}_{t_range[0]:.0f}-{t_range[1]}.svg', dpi=600)

    plt.show()

if __name__ == '__main__':
    main()
