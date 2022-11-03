import matplotlib as mpl
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import utils



base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\SPHERE_DIAMETER'
base_filename = 'LT_R3N05_005PCT_2022-10-11'

n = 9


def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


if __name__ == '__main__':
    cmap = plt.cm.jet
    colors = plt.cm.jet(np.linspace(0, 1, n))
    load_plt_style()
    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)
    ax_t = ax.twinx()


    for i in range(n):
        csv = os.path.join(base_path, f'{base_filename}_{i+1}.csv')
        print(f'Opening {csv}')
        df = pd.read_csv(csv, comment='#').apply(pd.to_numeric)
        ax.plot(
            df['Measurement Time (s)']-0.5,
            df['Pressure (Torr)']*1000.0,
            lw=1.25,
            color=colors[i],
            zorder=10
        )

    trigger = df['Trigger (V)'].values
    pulse = np.zeros_like(trigger)
    pulse[trigger>1.0] = 1.0

    ax_t.plot(
        df['Measurement Time (s)'],
        pulse,
        color='tab:grey', lw=1.0, zorder=1, ls='--'
    )

    ax_t.tick_params(
        axis='y', which='both', labelright=False, right=False
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure (mTorr)')
    ax.set_title('Chamber pressure')

    ax.set_xlim(0.0, 30.0)
    ax.set_ylim(0.0, 20.0)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))

    fig.savefig(
        os.path.join(base_path, 'laser_heating.png'),
        dpi=600
    )
    plt.show()