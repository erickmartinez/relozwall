import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from matplotlib.ticker import ScalarFormatter

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests'
csv_file = 'R3N3_R3N4_BAKING_TEMP.csv'

if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer=os.path.join(base_path, csv_file), sep=",")
    # df = df.drop(columns=df.columns[0])
    df = df.iloc[:,2:].apply(pd.to_numeric)
    df = df.groupby('Baking Temperature (C)').agg(
        {'Erosion Rate (cm/s)': ['mean', 'std'], 'Mean particle velocity (cm/s)': ['mean'],
         'Particle velocity std (cm/s)': 'mean',
         'Particle velocity mode': 'mean',
         'Baking Temperature (C)': 'mean'})
    # print(df.columns)
    # df
    print(df.info())
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    baking_temp = df[('Baking Temperature (C)', 'mean')].values
    erosion_rate = df[('Erosion Rate (cm/s)', 'mean')].values
    particle_velocity_mean = df[('Mean particle velocity (cm/s)', 'mean')].values

    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-2, 3))

    fig, ax = plt.subplots()
    fig.set_size_inches(5., 3.5)

    ax.plot(baking_temp, erosion_rate, ls='-', marker='o', label='Erosion Rate', color='C0')
    ax2 = ax.twinx()

    ax2.plot(baking_temp, particle_velocity_mean, ls='--', marker='s', label='Particle velocity', color='C1')

    ax.set_xlabel('Baking Temperature (°C)')
    ax.set_ylabel('Erosion Rate (cm/s)', color='C0')

    ax.tick_params(axis='y', labelcolor='C0')
    ax2.tick_params(axis='y', labelcolor='C1')

    ax2.set_ylabel('Debris velocity (cm/s)', color='C1')

    fig.tight_layout()
    plt.show()

