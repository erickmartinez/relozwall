import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import platform
import os
import json
from scipy.interpolate import interp1d

files = [
    {'csv': 'INL_ft60.csv', 'ft': 60}, {'csv': 'INL_ft70.csv', 'ft': 70},
    {'csv': 'INL_ft80.csv', 'ft': 80},
]


platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'

data_path = 'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/fig_bending_tests'


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

def normalize_path(the_path):
    global platform_system, drive_path
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    else:
        the_path = the_path.replace('/', '\\')
    return os.path.join(drive_path, the_path)

def main():
    global data_path, files
    base_path = normalize_path(base_path)
    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3.)

    ax.set_xlabel('Deformation (mm)')
    ax.set_ylabel('Load (N)')

    ax.set_xlim(-0.1, 0.4)
    ax.set_ylim(0., 3.)

    dx = 0.0025
    n_points = 0.25 / dx + 1
    new_x = np.arange(0, n_points) * dx

    output_df = pd.DataFrame(data={'Deformation (mm)': new_x})

    colors = {60:'tab:red', 70:'tab:green', 80:'k'}

    for file in files:
        ft = int(file['ft'])
        csv = file['csv']
        df: pd.DataFrame = pd.read_csv(os.path.join(base_path, csv), comment='#').apply(pd.to_numeric)
        df = df.sort_values(by=['Deformation (mm)'])
        df['Deformation (mm)'] -= df['Deformation (mm)'].min()
        f = interp1d(x=df['Deformation (mm)'].values, y=df['Load (N)'].values)

        new_y = f(new_x)
        colname = f'Load at ft={ft:d} (N)'
        output_df[colname] = new_y

        ax.plot(new_x, new_y, c=colors[ft], label=rf'Simulation $f_{{\mathregular{{t}}}}$ = {ft}')

    print(output_df)
    output_df.to_csv(os.path.join(base_path, 'interpolated_load.csv'), index=False)
    plt.show()


if __name__ == '__main__':
    main()

