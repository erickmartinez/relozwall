import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import json

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\manuscripts\paper1\inl'
temperature_csv = 'average_surface_temperature.csv'
current_disintegration_csv = 'breakup_plot_no_legend.csv'


def prepare_dataset():
    df = pd.read_csv(os.path.join(base_path, temperature_csv), comment='#').apply(pd.to_numeric)
    time_s = df['Time (s)'].values
    temperature_k = df['Temperature (K)'].values
    f = interp1d(time_s, temperature_k)
    dt = 0.05
    t_min, t_max = 0.1, 1.0
    time_steps = int((t_max - t_min) / dt + 1)
    time_interp = dt * np.arange(0, time_steps) + t_min
    temperature_interp_k = f(time_interp)
    interp_df = pd.DataFrame(data={
        'Time (s)': time_interp, 'Temperature (K)': temperature_interp_k
    })
    base_name = os.path.splitext(temperature_csv)[0]
    interp_df.to_csv(os.path.join(base_path, base_name + '_interp.csv'), index=False)
    return interp_df


def prepare_disintegration_dataset():
    df = pd.read_csv(os.path.join(base_path, current_disintegration_csv), comment='#').apply(pd.to_numeric)
    # df = df.loc[df['Time (s)'].round(2).drop_duplicates().index]
    print(df.describe())
    time_s = df['Time (s)'].values
    disintegration_rate = df['Current disintegration rate (mm/s)'].values
    f = interp1d(time_s, disintegration_rate)
    dt = 0.025
    t_min, t_max = 0.05, 1.0
    time_steps = int((t_max - t_min) / dt + 1)
    time_interp = dt * np.arange(0, time_steps) + t_min
    disintegration_interp = f(time_interp)
    average_disintegration = np.zeros_like(disintegration_interp)

    for i, di in enumerate(disintegration_interp):
        average_disintegration[i] = disintegration_interp[0:i + 1].mean()

    interp_df = pd.DataFrame(data={
        'Time (s)': time_interp, 'Current disintegration rate (mm/s)': disintegration_interp,
        'Average disintegration rate (mm/s)': average_disintegration
    })
    base_name = os.path.splitext(current_disintegration_csv)[0]
    interp_df.to_csv(os.path.join(base_path, base_name + '_interp.csv'), index=False)
    return interp_df


def load_plt_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


if __name__ == '__main__':
    average_temp_df = prepare_dataset()
    disintegration_df = prepare_disintegration_dataset()
    load_plt_style()
    fig, ax = plt.subplots(ncols=1, nrows=2, constrained_layout=True)
    fig.set_size_inches(4.0, 5.5)
    ax[0].plot(
        average_temp_df['Time (s)'].values, average_temp_df['Temperature (K)'].values,
        color='tab:red', lw=1.5
    )

    ax[0].set_title('Average temperature')
    ax[0].set_ylabel('K')
    ax[0].set_ylim(bottom=2900, top=3900)
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax[0].set_xlabel('Time (s)')


    ax[1].plot(
        disintegration_df['Time (s)'].values, disintegration_df['Current disintegration rate (mm/s)'].values,
        color='tab:red', ls='-', lw=1.5, label='Current'
    )
    ax[1].plot(
        disintegration_df['Time (s)'].values, disintegration_df['Average disintegration rate (mm/s)'].values,
        color='k', ls='-', lw=1.5, label='Average'
    )

    ax[1].set_title('Disintegration rate')
    ax[1].set_ylabel('mm/s')
    ax[1].set_ylim(bottom=0.0, top=20.0)
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    ax[1].set_xlabel('Time (s)')
    ax[1].legend(
        loc='upper right', frameon=True, prop={'size': 9}
    )

    for i, axi in enumerate(ax.flatten()):
        axi.set_xlim(0, 1)
        axi.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        axi.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        panel_label = chr(ord('`') + i + 1)
        axi.text(
            -0.15, 1.15, f'({panel_label})', transform=axi.transAxes, fontsize=13, fontweight='bold',
            va='top', ha='right'
        )

    plt.show()
