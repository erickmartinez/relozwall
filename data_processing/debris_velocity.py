import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from matplotlib.ticker import ScalarFormatter
from scipy import stats


# base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\Sample_50'
# csv_file = 'Sample50_debris_distribution_3kW_1s.csv'
base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\R3N4'
csv_file = 'R3N4_particle_distribution_2.csv'

label = "R3N4 3 kW, 0.5 s"

height = 5. * 2.54  # cm
x_center = 19.0 / 2.0  # Assume that the sample holder is at 1/3 of the x position wrt the tray
g = 9.8E2  # cm/s^2

if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer=os.path.join(base_path, csv_file), sep=",").reset_index(drop=True)
    df = df.drop(columns=df.columns[0])
    df = df.apply(pd.to_numeric)
    xm = df['XM'].values - x_center
    ym = df['YM'].values
    ym = ym - ym.min()
    rm = np.sqrt(xm**2.0 + ym**2.0)
    vx = rm * np.sqrt(0.5 * g / height)
    vx_mean = vx.mean()
    vx_std = vx.std()


    # Load plotting style
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)



    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-2, 3))

    fig, ax = plt.subplots()
    fig.set_size_inches(5., 4.)
    number_of_bins = 10
    n, bins, patches = ax.hist(vx, number_of_bins, density=True, facecolor='C0', alpha=0.95)
    ax.set_xlabel(f"Particle Velocity (cm/s)")
    ax.set_ylabel(f"Probability")
    ax.set_title(f"{label}")
    ax.yaxis.set_major_formatter(xfmt)

    mode_index = n.argmax()
    vx_mode = 0.5 * (bins[mode_index]+bins[mode_index+1])

    print(vx_mode)
    results = f'$v_{{\\mathrm{{mean}}}} = {vx_mean:.0f}$ cm/s\n' \
              f'$v_{{\\mathrm{{std}}}}  \\quad= {vx_std:.0f}$ cm/s\n' \
              f'$v_{{\\mathrm{{mode}}}} = {vx_mode:.0f}$ cm/s'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(
        0.05,
        0.95,
        results,
        fontsize=10,
        transform=ax.transAxes,
        va='top', ha='left',
        bbox=props
    )

    fig.tight_layout()
    fig.savefig(os.path.join(base_path, 'histograms.png'), dpi=600)
    plt.show()



