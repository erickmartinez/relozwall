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
output_filename = 'extrusion_fron_surface_outgassing_summary'


files = [
    'EXTRUSION_R3N49_1_40V_350C_350C2022-06-02_1_outgassing',
    'EXTRUSION_R3N45_1_100V_800C2022-05-31_3_outgassing',
    'EXTRUSION_R3N48_1_150V-1000c_1000C2022-06-01_2_outgassing',
    'EXTRUSION_SACRIFICIAL_20220705_R3N51_FONT_0800C2022-07-07_1_CORRECTED_ISBAKING_outgassing',
    'EXTRUSION_R3N57_0715C2022-07-07_1_outgassing'

]

labels = [
    '300 °C Ramp', '715 °C Ramp', '800 °C Ramp', '800 °C Preheat', '715 °C Preheat - Outgassed GC'
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

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

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

        ax1.plot(
            time_s / 60.0, temperature_c, label=lbl, color=colors[i],
        )

        ax2.plot(
            time_s / 60.0, outgassing_rate, label=lbl, color=colors[i],
        )


    # ax1.set_xlabel('Time (min)')
    ax2.set_xlabel('Time (min)')
    ax1.set_ylabel('$T$ (°C)')
    ax2.set_ylabel('$Q_{\mathrm{tot}}$ (Torr $\\cdot$ L / m${^2}$ s)')

    ax1.set_xlim(left=0, right=15)
    ax2.set_xlim(left=0, right=15)

    ax1.set_ylim(bottom=20, top=1000)
    ax2.set_ylim(bottom=0, top=20)

    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(5.0))

    ax1.legend(
        bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.,
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