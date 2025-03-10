import numpy as np
from utils import latex_float, get_experiment_params
import matplotlib.pylab as plt
import pandas as pd
import os
from matplotlib import ticker, gridspec
import matplotlib as mpl
from scipy.signal import savgol_filter
import json
from scipy.interpolate import interp1d

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\extrusion setup\outgassing'
output_filename = 'extrusion_front_surface_outgassing_summary_202301'


extrusion_temperature = 800

files = [
    'EXTRUSION_R4N35_BECKOPOX_0800C2023-01-24_1_outgassing',
    'EXTRUSION_R4N34_CYREZ_0800C2023-01-24_1_outgassing',
    'EXTRUSION_R4N36_200-400UM_CYREZ_0800C2023-01-25_1_outgassing',
    'EXTRUSION_R4N38_CYREZ_0800C2023-01-27_1_outgassing'
]

labels = [
    r'$d_{p}$ = 850 $\mathregular{\mu}$m, Epoxy, $A_{v}$ = 2.9 mm$\mathregular{^2}$',
    r'$d_{p}$ = 850 $\mathregular{\mu}$m, Amine, $A_{v}$ = 2.9 mm$\mathregular{^2}$',
    r'$d_{p}$ = 220 $\mathregular{\mu}$m, Amine, $A_{v}$ = 2.9 mm$\mathregular{^2}$',
    r'$d_{p}$ = 850 $\mathregular{\mu}$m, Amine, $A_{v}$ = 5.9 mm$\mathregular{^2}$',
]

if __name__ == '__main__':
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    xfmt1 = ticker.ScalarFormatter(useMathText=True)
    xfmt1.set_powerlimits((-2, 2))

    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(4.5, 5.5)

    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)  # , height_ratios=[1.618, 1.618, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    colors = ['C0', 'C2', 'C1', 'C3', 'C4']
    norm = mpl.colors.Normalize(vmin=25.0, vmax=1000)
    cmap = plt.cm.jet

    for i, fn, lbl in zip(range(len(files)), files, labels):
        df = pd.read_csv(os.path.join(base_dir, fn + '.csv')).apply(pd.to_numeric)
        if fn == 'EXTRUSION_SACRIFICIAL_20220705_R3N51_FONT_0800C2022-07-07_1_CORRECTED_ISBAKING_outgassing':
            df = df[df['Time (s)'] > 890]
        time_s = df['Time (s)'].values
        time_s -= time_s.min()
        temperature_c = df['Baking Temperature (C)'].values
        outgassing_rate = df['Outgassing (Torr*L/m^2 s)'].values
        try:
            outgassing_rate = df['S*p/A (Torr*L/m^2 s)'].values
        except KeyError as e:
            print(fn)
            raise(e)

        if i == 2:
            outgassing_rate *= 0.02

        ax1.plot(
            time_s / 60.0, temperature_c, label=lbl, color=colors[i],
        )

        ax2.plot(
            time_s / 60.0, outgassing_rate, label=lbl, color=colors[i],
        )


    # ax1.set_xlabel('Time (min)')
    ax2.set_xlabel('Time (min)')
    ax1.set_ylabel('°C')
    ax2.set_ylabel('Torr $\\cdot$ L / (m$^{\mathregular{2}}$ s)')
    ax1.set_title('Baking temperature')
    ax2.set_title('Front surface outgassing')

    ax1.set_xlim(left=0, right=15)
    ax2.set_xlim(left=0, right=15)

    ax1.set_ylim(bottom=20, top=1000)
    ax2.set_ylim(bottom=0, top=50)

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2.5))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2.5))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(5.0))

    ax2.legend(
        # bbox_to_anchor=(0., 1.02, 1., .102),
        loc='upper right', ncol=1, #mode="expand", borderaxespad=0.,
        prop={'size': 8}
    )

    # ax2.set_yscale('log')
    # ax2.set_ylim(bottom=1E-3, top=1E1)

    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, output_filename + '.png'), dpi=600)
    fig.savefig(os.path.join(base_dir, output_filename + '.svg'), dpi=600)
    fig.savefig(os.path.join(base_dir, output_filename + '.pdf'), dpi=600)
    fig.savefig(os.path.join(base_dir, output_filename + '.eps'), dpi=600)
    plt.show()