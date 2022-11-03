import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
from matplotlib import ticker

csv_file = '../data/friction_measurements.csv'

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\extrusion setup\outgassing'
output_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\manuscripts\paper1\figures'
output_filename = 'figure_3'


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

    fig, ax = plt.subplots(ncols=1, nrows=3, constrained_layout=True)
    fig.set_size_inches(3.5, 6.0)

    norm = mpl.colors.Normalize(vmin=150.0, vmax=950.0)
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
        ax[0].errorbar(
            temp_df['Target Speed (cm/s)'], force_n, force_n_err, color=cmap(norm(temperature)),
            marker=markers[i], ms=9, mew=1.25, mfc='none', label=f'{temperature:.0f} °C',
            capsize=2.75, elinewidth=1.25, lw=1.5
        )

    mean_friction = np.array(mean_friction)
    print(f'Mean friction: {mean_friction.mean()} N/cm^2')
    ax[0].set_xlabel('Speed (cm/s)')
    ax[0].set_ylabel('N/cm$^{\mathregular{2}}$')
    # ax[0].set_title(f'Sample length: {sample_length:.0f} cm')
    ax[0].set_title(f'Friction ({sample_length:.0f} cm-sample)')
    ax[0].legend(loc='lower right', frameon=True)

    ax[0].set_xlim(0.0, 1.5)
    ax[0].set_ylim(bottom=-0.2, top=0.4)

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
            raise (e)

        c = cmap(norm(temperature_c.max()))
        ax[1].plot(
            time_s / 60.0, temperature_c, label=lbl, color=c,
        )

        ax[2].plot(
            time_s / 60.0, outgassing_rate, label=lbl, color=c,
        )

    ax[1].set_xlabel('Time (min)')
    ax[2].set_xlabel('Time (min)')
    ax[1].set_ylabel('°C')
    ax[2].set_ylabel('Torr-L / (m$^{\mathregular{2}}$ s)')
    ax[1].set_title('Baking temperature')
    ax[2].set_title('Front surface outgassing')

    ax[1].set_xlim(left=0, right=15)
    ax[2].set_xlim(left=0, right=15)

    ax[1].set_ylim(bottom=20, top=1000)
    ax[2].set_ylim(bottom=0, top=20)

    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(2.5))
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(250))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(125))

    ax[2].xaxis.set_major_locator(ticker.MultipleLocator(2.5))
    ax[2].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax[2].yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax[2].yaxis.set_minor_locator(ticker.MultipleLocator(5.0))

    ax[1].legend(
        # bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.,
        loc='lower right', ncol=2,
        prop={'size': 8}
    )

    # Add panel labels out of the box
    panel_labels = ['a', 'b', 'c']
    for axi, al in zip(ax, panel_labels):
        axi.text(
            -0.15, 1.15, f'({al})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )

    # fig.tight_layout()
    fig.savefig(os.path.join(output_dir, output_filename + '.svg'), dpi=600)
    # fig.savefig(os.path.join(output_dir, output_filename + '.eps'), dpi=600)
    plt.show()