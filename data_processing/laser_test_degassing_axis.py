import json
import os
from utils import lighten_color
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import ticker, patches

input_file = '../data/degassing_database_gc_graphite.csv'
k_B = 1.380649E-23  # J / K
temperature = 20.0  # K
chamber_volume = 31571.33  # cm^3
mean_dt = 1.4 # s


def number_density(chamber_pressure, chamber_temperature):
    """
    kb = 1.380649e-23  # J/k = N * m / K
    pressure = 133.322 * pressure   # N / m^2
    n = pressure / (kb * (temperature + 273.15))  # (N / m^2) / (N * m ) = (#/m^3)
    n = n * 1E-6
    """
    p_k = 133.322 * chamber_pressure / 1.380649e-17
    return p_k / (chamber_temperature + 273.15)


if __name__ == '__main__':
    outgassing_df: pd.DataFrame = pd.read_csv(input_file)
    # outgassing_df = outgassing_df[(outgassing_df['Laser Power Setpoint (%)'] == 100) & (outgassing_df['Sample ID'] != 'R3N40')]
    column_names = outgassing_df.columns
    outgassing_df[column_names[2::]] = outgassing_df[column_names[2::]].apply(pd.to_numeric)
    outgassing_highlighted_df = outgassing_df[outgassing_df['Sample ID'] == 'GT001688']
    outgassing_pebble_df = outgassing_df[outgassing_df['Sample ID'] != 'GT001688']
    # outgassing_df['dP (mTorr)'] = outgassing_df['Peak Pressure (mTorr)'] - outgassing_df['Base Pressure (mTorr)'].min()
    n = len(outgassing_df)
    outgassed_particles = np.empty(n, dtype=np.float64)
    # outgassing_df['Gas Concentration x10^15 (1/cm^3)'] = 1E-15*number_density(outgassing_df['dP (mTorr)']*1E-3, temperature)
    # print(number_density(outgassing_df['dP (mTorr)']*1E-3, temperature))
    # print(outgassing_df[['Sample', 'dP (mTorr)', 'Erosion Rate (cm/s)', 'Outgassing Rate (Torr L / s m2)']])
    degassing_time = outgassing_df['Degassing Time (h)'].values
    base_pressure = outgassing_df['Base Pressure (mTorr)'].values
    mean_erosion_rate = outgassing_df.loc[outgassing_df['Sample ID'] != 'GT0016888', 'Erosion Rate (cm/s)'].mean()
    mean_pebble_velocity = outgassing_pebble_df['Particle velocity mode (cm/s)'].mean()
    print(f'Errosion Rate Mean for Outgassing < 1E4 Torr L: {mean_erosion_rate:.1f}')


    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    # fig, (ax, cbar_ax) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [5, 1]})  # , constrained_layout=True)

    fig, ax = plt.subplots(ncols=1, nrows=2)  # , constrained_layout=True)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.set_size_inches(3.5, 4.5)

    colors = ['C0', 'C1']

    ax[0].errorbar(
        outgassing_df['Outgassing Rate (Torr L / s m2)'],
        outgassing_df['Erosion Rate (cm/s)'],
        yerr=outgassing_df['Erosion Rate Error (cm/s)'],
        capsize=2.5, mew=1.25, marker='o', ms=8, elinewidth=1.25,
        ls='none', c=colors[0], fillstyle='none'
    )

    # Highlighted points
    ax[0].errorbar(
        outgassing_highlighted_df['Outgassing Rate (Torr L / s m2)'],
        outgassing_highlighted_df['Erosion Rate (cm/s)'],
        yerr=outgassing_highlighted_df['Erosion Rate Error (cm/s)'],
        capsize=2.5, mew=1.25, marker='o', ms=8, elinewidth=1.25,
        ls='none', c='tab:orange', fillstyle='none'
    )


    ax[1].set_xlabel(r'Outgassing (Torr $\cdot$ L / s m$^{\mathregular{2}}$)')
    ax[0].set_ylabel('cm/s', color='k')
    ax[0].tick_params(axis='y', labelcolor='k')
    ax[0].set_xscale('log')
    ax[0].set_title('Erosion rate', fontsize=12)

    ax[0].set_xlim(left=200, right=20000)
    ax[0].set_ylim(bottom=0.0, top=0.5)
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))


    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax[0].text(
        0.05, 0.95,
        f'Mean Erosion Rate: {mean_erosion_rate:.1f} cm/s',
        fontsize=10,
        transform=ax[0].transAxes,
        va='top', ha='left',
        color='tab:red'
    )

    # ax2 = ax.twinx()

    # ax.set_zorder(1)
    # ax2.set_zorder(0)
    # ax.patch.set_visible(False)

    ax[1].errorbar(
        outgassing_pebble_df['Outgassing Rate (Torr L / s m2)'],
        outgassing_pebble_df['Particle velocity mode (cm/s)'],
        yerr=outgassing_pebble_df['Particle velocity std (cm/s)'],
        capsize=2.5, mew=1.25, marker='s', ms=8, elinewidth=1.25,
        ls='none', c=colors[0], fillstyle='none', zorder=5
    )

    ax[1].errorbar(
        outgassing_pebble_df['Outgassing Rate (Torr L / s m2)'],
        outgassing_pebble_df['Particle velocity mode (cm/s)'],
        yerr=outgassing_pebble_df['Particle velocity std (cm/s)'],
        capsize=2.5, mew=1.25, marker='s', ms=8, elinewidth=1.25,
        ls='none', c=colors[0], fillstyle='none', zorder=5
    )

    ax[1].set_title('Pebble velocity', fontsize=12)
    ax[1].set_ylabel('cm/s', color='k')
    ax[1].tick_params(axis='y', labelcolor='k')
    ax[1].set_xscale('log')
    ax[1].set_xlim(left=100, right=20000)
    ax[1].set_ylim(bottom=0, top=80)
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax[1].text(
        0.05, 0.95,
        f'Mean velocity: {mean_pebble_velocity:.0f} cm/s',
        fontsize=10,
        transform=ax[1].transAxes,
        va='top', ha='left',
        color='tab:red', zorder=6
    )

    outgassing_graphite = outgassing_highlighted_df['Outgassing Rate (Torr L / s m2)'].values
    velocity_graphite = outgassing_highlighted_df['Erosion Rate (cm/s)'].values
    outgassing_graphite = outgassing_graphite[0]
    velocity_graphite = velocity_graphite[0]
    print(outgassing_graphite, velocity_graphite)

    ax1_xlim = ax[0].get_xlim()
    ax1_ylim = ax[0].get_ylim()
    xy = (ax1_xlim[0], ax1_ylim[0])
    p_width = 2.0*outgassing_graphite
    p_height = ax1_ylim[1]
    rect = patches.Rectangle(xy, p_width, p_height, linewidth=1, edgecolor=lighten_color('r', 0.35), facecolor=lighten_color('orange', 0.35), zorder=1)
    ax[0].add_patch(rect)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[0].annotate(
        "Graphite",
        xy=(outgassing_graphite, velocity_graphite), xycoords="data",
        xytext=(0.01, 25.0), textcoords='offset points',
        color='k', fontsize=10,
        arrowprops=dict(
            arrowstyle="->", color="k",
            shrinkA=5, shrinkB=5,
            # patchA=None, patchB=None,
            connectionstyle='angle,angleA=-90,angleB=180,rad=5',

        ),
        rotation=90, ha="center",
        zorder=7
    )

    # Add panel labels out of the box
    ax[0].text(
        -0.1, 1.15, '(a)', transform=ax[0].transAxes, fontsize=14, fontweight='bold',
        va='top', ha='right'
    )
    ax[1].text(
        -0.1, 1.15, '(b)', transform=ax[1].transAxes, fontsize=14, fontweight='bold',
        va='top', ha='right'
    )

    fig.tight_layout()

    basename = os.path.splitext(os.path.basename(input_file))[0]
    path = os.path.dirname(input_file)
    fig.savefig(os.path.join(path, basename + '_erosion_rates.png'), dpi=600)
    fig.savefig(os.path.join(path, basename + '_erosion_rates.svg'), dpi=600)
    fig.savefig(os.path.join(path, basename + '_erosion_rates.pdf'), dpi=600)
    fig.savefig(os.path.join(path, basename + '_erosion_rates.eps'), dpi=600)
    plt.show()
