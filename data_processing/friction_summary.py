import numpy as np
import datetime
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import json
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

base_path = r"G:\Shared drives\ARPA-E Project\Lab\Data\Extruder\Friction"
database_csv = r"friction_force_database.csv"

if __name__ == "__main__":
    database_df: pd.DataFrame = pd.read_csv(os.path.join(base_path, database_csv))
    column_names = database_df.columns
    database_df[column_names[2:]] = database_df[column_names[2:]].apply(pd.to_numeric)

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(4.5, 5.5)

    gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)#, height_ratios=[1.618, 1.618, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    samples_df = database_df[database_df['Sample'] != 'BACKGROUND'].sort_values(
        by='Target Speed (cm/s)', ascending=True,
    )
    samples_df.reset_index(inplace=True)

    background_df = database_df[database_df['Sample'] == 'BACKGROUND'].sort_values(
        by='Target Speed (cm/s)', ascending=True,
    )
    average_speeds = background_df['Average Speed (cm/s)'].values
    background_df.reset_index(inplace=True)
    sample_colors = plt.cm.Set1(np.linspace(0, 1, len(samples_df)))
    bgd_colors = plt.cm.Blues(np.linspace(0.5, 1, len(background_df)))
    # sample_colors = plt.cm.brg(mpl.colors.Normalize(vmin=average_speeds.min(), vmax=average_speeds.max()))
    markers = ['o', 's', '^']
    speed_markers = {r['Speed Setpoint']: markers[i] for i, r in background_df.iterrows()}

    for i, row in samples_df.iterrows():
        data_df = pd.read_csv(os.path.join(base_path, row['csv'])).apply(pd.to_numeric)
        speed_setting = row['Speed Setpoint']
        bkg_csv_df = background_df[background_df['Speed Setpoint'] == speed_setting]
        bkg_csv = bkg_csv_df['csv'].iloc[0]
        bkg_data_df = pd.read_csv(os.path.join(base_path, bkg_csv)).apply(pd.to_numeric)
        position = data_df['Position (cm)'].values
        force = data_df['Force (N)'].values - bkg_data_df['Force (N)'].values

        position = position[:-1]
        force = force[:-1]

        speed = row['Average Speed (cm/s)']
        temperature = row['Baking Temperature (C)']
        lbl = rf'{speed:3.2f} cm/s'
        if int(temperature) == 250:
            axi = ax1
        else:
            axi = ax2

        axi.plot(
            position, force,
            color=sample_colors[i], fillstyle='none', marker=speed_markers[speed_setting],
            ls='-', lw=1.75,
            label=lbl,
            zorder=i
        )

    for i, row in background_df.iterrows():
        data_df = pd.read_csv(os.path.join(base_path, row['csv'])).apply(pd.to_numeric)
        position = data_df['Position (cm)'].values
        force = data_df['Force (N)'].values
        position = position[:-1]
        force = force[:-1]
        speed = row['Target Speed (cm/s)']
        lbl = f'{speed} cm/s'
        ax3.plot(
            position, force,
            color=bgd_colors[i], fillstyle='none', marker=markers[i],
            ls='-', lw=1.75,
            label=lbl,
            zorder=i
        )

    ax1.set_title('250 °C')
    ax2.set_title('1000 °C')
    ax3.set_title('Background')
    ax1.set_ylabel('Force (N)')
    ax2.set_ylabel('Force (N)')
    ax3.set_ylabel('Force (N)')
    ax3.set_xlabel('Position (cm)')

    ax3.set_ylim(-2.0, 1.0)

    ax1.legend(loc="upper left", prop={'size': 9}, frameon=False, ncol=3)
    ax2.legend(loc="upper right", prop={'size': 9}, frameon=False, ncol=3)
    ax3.legend(loc="upper left", prop={'size': 9}, frameon=False, ncol=3)
    # ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10})
    # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10})
    today = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filetag = f"friction_measurements_{today}"
    fig.savefig(os.path.join(base_path, filetag + '.png'), dpi=600)
    plt.show()

