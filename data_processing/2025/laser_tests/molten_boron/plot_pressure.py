import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from data_processing.misc_utils.plot_style import load_plot_style
from data_processing.utils import get_experiment_params
from pathlib import Path

EXPERIMENT_CSV = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/MOLTEN_BORON/LCT_POLYBORON-20_030PCT_2025-12-01_1.csv'

def main(experiment_csv = EXPERIMENT_CSV):
    path_to_csv = Path(experiment_csv)

    experiment_params = get_experiment_params(relative_path=str(path_to_csv.parent), filename=path_to_csv.stem)
    sample_name = experiment_params['Sample Name']['value']
    data_df = pd.read_csv(experiment_csv, comment='#').apply(pd.to_numeric, errors='coerce')
    time_s = data_df['Measurement Time (s)'].values
    pressure_s = data_df['Pressure (Torr)'].values
    laser_power = data_df['Laser output peak power (W)'].values

    msk_power_on = laser_power > 0
    t_on = time_s[msk_power_on].min()
    idx_on = np.argmin(np.abs(time_s - t_on)) - 1

    time_s = time_s[idx_on:]
    pressure_s = pressure_s[idx_on:]
    laser_power = laser_power[idx_on:]

    pmax = np.max(pressure_s)
    idx_pmax = np.argmin(np.abs(pressure_s - pmax))
    t_pmax = time_s[idx_pmax]

    load_plot_style()

    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True, sharex=True)
    fig.set_size_inches(4., 4.94)

    axes[0].plot(time_s, pressure_s*1000, color='C0')
    axes[1].plot(time_s, laser_power, color='C1')

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_ylabel('Pressure (mTorr)')
    axes[1].set_ylabel('Laser power (W)')

    axes[-1].set_xlim(0, 30)
    axes[0].set_ylim(bottom=0, top=175)
    axes[1].set_ylim(bottom=0, top=1000)


    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(50))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(10))

    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(200))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(100))

    for ax in axes:
        ax.axvline(x=t_pmax, color='k', linestyle='--')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    axes[0].text(
        t_pmax, pressure_s[idx_pmax]*1000,
        f'$t={t_pmax:.2f}\;\mathrm{{s}}$\t',
        ha='right', va='center',
    )

    path_to_plots = Path('./pressure_plots')
    for extension in ['png', 'svg']:
        fig.savefig(path_to_plots / f'pressure_{sample_name}.{extension}', dpi=300)


    plt.show()

if __name__ == '__main__':
    main(experiment_csv=EXPERIMENT_CSV)
