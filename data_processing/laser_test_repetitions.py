import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os

from matplotlib import ticker

from utils import get_experiment_params

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\REPETITIONS'
files = [
    'GT001688_050PCT_30X_5.0s_2022-07-15_1',
    'GT001688_050PCT_5X_5.0s_2022-07-16_1'
]


if __name__ == '__main__':

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1, nrows=2)  # , constrained_layout=True)
    fig.set_size_inches(4.5, 5.2)

    colors = ['C0', 'C1']

    for i, fn in enumerate(files):
        ffn = os.path.join(base_path, fn + '.csv')
        params = get_experiment_params(relative_path=base_path, filename=fn)
        pulse_length = float(params['Pulse length']['value'])
        pulse_delay = float(params['Pulse delay']['value'])
        lbl = f'Pulse: {pulse_length:.1f} s, Period: {pulse_delay:.1f} s'
        df = pd.read_csv(ffn, comment='#').apply(pd.to_numeric)
        time_s = df['Measurement Time (s)'].values
        pressure = df['Pressure (Torr)'].values * 1000.0
        trigger = df['Trigger Pulse (V)'].values
        trigger /= trigger.max()

        ax[i].set_title(lbl)
        ax[i].set_ylabel('Pressure (mTorr)', color=colors[i])
        ax[i].tick_params(axis='y', labelcolor=colors[i])
        ax_t = ax[i].twinx()
        ax_t.fill_between(time_s, y1=trigger, y2=0.0, color='tab:red', alpha=0.25)
        ax_t.set_ylabel('Laser Pulse', color='tab:red')
        ax_t.tick_params(axis='y', labelcolor='tab:red', right=False, labelsize=0, length = 0)

        ax[i].plot(
            time_s, pressure, c=colors[i], label=lbl, lw=1.2
        )



    ax[1].set_xlabel('Time (s)')

    # ax[0].legend(
    #     loc='best', frameon=True, prop={'size': 10}
    # )
    ax[0].set_xlim(left=0.0, right=300.0)
    ax[1].set_xlim(left=0.0, right=300.0)
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(50))

    fig.tight_layout()
    fig.savefig(os.path.join(base_path, 'repeated_firing_test.png'), dpi=600)
    plt.show()

