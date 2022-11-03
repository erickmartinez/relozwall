import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
from matplotlib import ticker

csv_file = '../data/friction_measurements.csv'

if __name__ == '__main__':
    df = pd.read_csv(csv_file)
    columns = df.columns
    df[columns[2:-1]]=df[columns[2:-1]].apply(pd.to_numeric)
    df['Date'] = df['Date'].apply(pd.to_datetime)
    df.sort_values(by=['Target Speed (cm/s)', 'Baking Temperature (C)'], inplace=True)
    temperatures = df['Baking Temperature (C)'].unique()
    sample_length = df['Length Mean (cm)'].mean()
    mean_friction = []

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1, nrows=1)  # , constrained_layout=True)
    fig.set_size_inches(4.5, 3.5)

    norm = mpl.colors.Normalize(vmin=25.0, vmax=1000)
    cmap = plt.cm.jet

    markers = ['o', 's', '^']

    for i, temperature in enumerate(temperatures):
        print(f'Plotting line for temperature: {temperature}')
        temp_df = df[df['Baking Temperature (C)'] == temperature]

        force = temp_df['Average Friction Force (N)'].values
        force_err = force*0.09#temp_df['Friction Force Std (N)'].values
        area = temp_df['Contact Area (cm2)'].values
        area_err = temp_df['Contact Area Error (cm2)'].values
        force_n = force / area
        force_n_err = np.linalg.norm([area_err / area, force_err / force])
        mean_friction.append(force_n)
        ax.errorbar(
            temp_df['Target Speed (cm/s)'], force_n, force_n_err, color=cmap(norm(temperature)),
            marker=markers[i], ms=9, mew=1.25, mfc='none', label=f'{temperature:.0f} °C',
            capsize=2.75, elinewidth=1.25, lw=1.5
        )

    mean_friction = np.array(mean_friction)
    print(f'Mean friction: {mean_friction.mean()} N/cm^2')
    ax.set_xlabel('Speed (cm/s)')
    ax.set_ylabel('Friction (N/cm$^{\mathregular{2}}$)')
    ax.set_title(f'Sample length: {sample_length:.1f} cm')
    ax.legend(loc='lower right', frameon=True)

    ax.set_xlim(0.0, 1.5)

    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(csv_file), 'friction_tests.svg'), dpi=600)
    fig.savefig(os.path.join(os.path.dirname(csv_file), 'friction_tests.png'), dpi=600)
    plt.show()