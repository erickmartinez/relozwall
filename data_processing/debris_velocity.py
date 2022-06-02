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
# base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\GC_GRAPHITE'
base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\beam_expander\MULTIPLE_FIRINGS'
# csv_file = r'R3N40_-2.5h_DEGASSING_20220517_230330504_results.csv'
csv_file = 'R3N21_100PCT_POWER_20220415_160521140_RESULTS.csv'
tray_width_mm = 21.0 * 0.5 * 25.4
units = 'mm'

label = "100% Laser Power"

height = 5. * 2.54  # cm
# x_center = 0.5 * 8 * 2.54  #
x_center = 0.5 * 9.0 * 2.54  #
y_center = 0.5 * 1.0 * 2.54
g = 9.8E2  # cm/s^2

if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer=os.path.join(base_path, csv_file), sep=",").reset_index(drop=True)
    df = df.drop(columns=df.columns[0])
    df = df.apply(pd.to_numeric)
    xm = df['XM'].values - x_center
    ym = df['YM'].values
    ym = ym.max() - ym - y_center
    rm = np.sqrt(xm**2.0 + ym**2.0)
    if units == 'mm':
        rm *= 0.1

    vx = rm * np.sqrt(0.5 * g / height)
    vx_mean = vx.mean()
    vx_std = vx.std()

    print(f"Velocity Mean: {vx_mean:5.1f} cm/s")



    # Load plotting style
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)



    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-2, 3))

    fig, ax = plt.subplots()
    fig.set_size_inches(4., 3.)
    number_of_bins = 10
    n, bins, patches = ax.hist(vx, number_of_bins, density=False, facecolor='C0', alpha=0.95)
    ax.set_xlabel(f"Particle Velocity (cm/s)")
    ax.set_ylabel(f"Counts")
    ax.set_title(f"{label}")
    ax.yaxis.set_major_formatter(xfmt)

    mode_index = n.argmax()
    vx_mode = 0.5 * (bins[mode_index]+bins[mode_index+1])

    print(f'Velocity Mode: {vx_mode:.1f} cm/s')
    results = f'$v_{{\\mathrm{{mean}}}} = {vx_mean:.0f}$ cm/s\n' \
              f'$v_{{\\mathrm{{std}}}}  \\quad= {vx_std:.0f}$ cm/s\n' \
              f'$v_{{\\mathrm{{mode}}}} = {vx_mode:.0f}$ cm/s\n' \
              f'$N    = {vx.size}$'

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
    output_filename = os.path.splitext(csv_file)[0]
    fig.savefig(os.path.join(base_path, f'{output_filename}_histograms.png'), dpi=600)
    plt.show()



