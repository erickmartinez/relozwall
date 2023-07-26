import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import numpy as np
from matplotlib import ticker
from data_processing.utils import lighten_color

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\surface_temperature\equilibrium_redone\pebble_sample'
csv_outgassing = r'GC_GRAPHITE_POWER_SCAN_OUTGASSING.csv'
csv_disintegration = r'velocity_database.csv'
csv_soot_deposition = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests' \
                      r'\surface_temperature\equilibrium_redone\slide_transmission_smausz.csv'

csv_degassing = '../../data/degassing_database_gc_graphite.csv'
output_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\manuscripts\paper1\figures'

nir_hl = np.array([45.])
nir_v0 = np.array([118.])
nir_v0_lb = nir_v0 - np.array([19.])
nir_v0_ub = np.array([724.]) - nir_v0

p0 = 4.7E3
p0_err_pct = 12.5
time_constant = 1.68
beam_diameter = 0.8164 * 1.5  # cm
sample_diameter = 0.92

"""
Correction factor from second peak for revision number 1 JAP - Estimation lead to pumping speed of ~20 L/s
Peaks estimated to be 6000 vs 1500 Torr-L/m^2/s for first peak compared to second peak
"""
outgassing_factor_20230331 = 0.103
outgassing_error_factor = 0.525


def csv_match_size(df1, df2):
    if len(df1) == len(df2):
        return True
    return False


def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


if __name__ == '__main__':
    outgassing_df = pd.read_csv(os.path.join(base_dir, csv_outgassing))
    outgassing_columns = outgassing_df.columns
    outgassing_df[outgassing_columns[1:]] = outgassing_df[outgassing_columns[1:]].apply(pd.to_numeric)
    outgassing_df.sort_values(by=['Laser power setpoint (%)'], ascending=True, inplace=True)
    disintegration_df = pd.read_csv(os.path.join(base_dir, csv_disintegration))
    disinteration_columns = disintegration_df.columns
    disintegration_df[disinteration_columns[1:]] = disintegration_df[disinteration_columns[1:]].apply(pd.to_numeric)
    disintegration_df.sort_values(by=['Laser power setpoint (%)'], ascending=True, inplace=True)
    disintegration_agg_df = disintegration_df.groupby('Laser power setpoint (%)').agg({
        'Particle velocity mode (cm/s)': ['mean', 'min'],
        'Particle velocity std (cm/s)': ['mean'],
        'Erosion rate (cm/s)': ['mean', 'min'],
        'Erosion rate error (cm/s)': ['mean', 'min'],
    })
    soot_deposition_df = pd.read_csv(csv_soot_deposition)
    soot_deposition_columns = soot_deposition_df.columns
    soot_deposition_df[soot_deposition_columns[1:]] = soot_deposition_df[soot_deposition_columns[1:]].apply(
        pd.to_numeric)
    soot_deposition_pebble_df = soot_deposition_df[soot_deposition_df['Sample'] != 'GT001688']
    soot_deposition_graphite_df = soot_deposition_df[soot_deposition_df['Sample'] == 'GT001688']
    soot_deposition_pebble_df.sort_values(by=['Laser Power (%)'], ascending=True)
    soot_deposition_graphite_df.sort_values(by=['Laser Power (%)'], ascending=True)

    degassing_df: pd.DataFrame = pd.read_csv(csv_degassing)
    degassing_column_names = degassing_df.columns
    degassing_df[degassing_column_names[2::]] = degassing_df[degassing_column_names[2::]].apply(pd.to_numeric)
    degassing_graphite_df = degassing_df[degassing_df['Sample ID'] == 'GT001688']
    degassing_prebaked_df = degassing_df[degassing_df['Sample ID'] == 'R3N58']


    if not csv_match_size(outgassing_df, disintegration_df):
        raise IndexError(f'Number of rows in both datasets must mathc.\n'
                         f'{csv_outgassing} has {len(outgassing_df)} rows.\n'
                         f'{csv_disintegration} has {len(disintegration_df)} rows.')

    merged_df = pd.merge(disintegration_df, outgassing_df, how='inner', on=['Laser power setpoint (%)'])
    print(merged_df)
    laser_power_setting = merged_df['Laser power setpoint (%)'].values
    sample_area = 0.25 * np.pi * sample_diameter ** 2.0
    heat_flux_factor = 0.01 * (1.0 - np.exp(-2.0 * (sample_diameter / beam_diameter) ** 2.0)) / sample_area
    heat_load = p0 * laser_power_setting * heat_flux_factor * 0.01
    heat_load_err = heat_load * (0.01 * p0_err_pct)
    heat_load_agg = p0 * disintegration_agg_df.index * heat_flux_factor * 0.01

    laser_power_setting_sublimation_graphite = soot_deposition_graphite_df['Laser Power (%)'].values
    laser_power_setting_sublimation_pebble = soot_deposition_pebble_df['Laser Power (%)'].values
    heat_load_sublimation_pebble = p0 * laser_power_setting_sublimation_pebble * heat_flux_factor * 0.01
    heat_load_sublimation_pebble_err = heat_load_sublimation_pebble * (0.01 * p0_err_pct)

    heat_load_sublimation_graphite = p0 * laser_power_setting_sublimation_graphite * heat_flux_factor * 0.01
    heat_load_sublimation_graphite_err = heat_load_sublimation_graphite * (0.01 * p0_err_pct)

    erosion_rate = disintegration_agg_df['Erosion rate (cm/s)']['mean'].values
    erosion_rate_err = disintegration_agg_df['Erosion rate error (cm/s)']['mean'].values
    outgassing_rate = merged_df['Outgassing rate (Torr L / s / m^2)'].values
    pebble_velocity = disintegration_agg_df['Particle velocity mode (cm/s)']['mean'].values
    pebble_velocity_err = disintegration_agg_df['Particle velocity std (cm/s)']['mean'].values

    # correct for outgassing rate with re-estimation in 2023/03/30
    outgassing_rate *= outgassing_factor_20230331


    film_thickness_pebble = soot_deposition_pebble_df['Thickness (nm)'].values
    film_thickness_pebble_err = film_thickness_pebble * soot_deposition_pebble_df['Error %'].values
    flattop_time_pebble = soot_deposition_pebble_df['Flat top time (s)'].values
    sublimation_rate_pebble = film_thickness_pebble / flattop_time_pebble
    de_by_de_pebble = np.zeros_like(sublimation_rate_pebble, dtype=np.float64)
    msk_pebble = film_thickness_pebble > 0.0
    de_by_de_pebble[msk_pebble] = film_thickness_pebble_err[msk_pebble] / film_thickness_pebble[msk_pebble]
    sublimation_rate_pebble_err = sublimation_rate_pebble * np.sqrt(
        de_by_de_pebble ** 2.0 + 0.1 ** 2.0
    )

    film_thickness_graphite = soot_deposition_graphite_df['Thickness (nm)'].values
    film_thickness_graphite_err = film_thickness_graphite * soot_deposition_graphite_df['Error %'].values
    flattop_time_graphite = soot_deposition_graphite_df['Flat top time (s)'].values
    msk_graphite = (film_thickness_graphite > 0.0) & (flattop_time_graphite > 0.0)
    sublimation_rate_graphite = np.zeros_like(flattop_time_graphite)
    sublimation_rate_graphite[msk_graphite] = film_thickness_graphite[msk_graphite] / flattop_time_graphite[
        msk_graphite]
    de_by_de_graphite = np.zeros_like(sublimation_rate_graphite, dtype=np.float64)
    de_by_de_graphite[msk_graphite] = film_thickness_graphite_err[msk_graphite] / film_thickness_graphite[msk_graphite]
    sublimation_rate_graphite_err = sublimation_rate_graphite * np.sqrt(
        de_by_de_graphite ** 2.0 + 0.1 ** 2.0
    )

    outgassing_rate_graphite = degassing_graphite_df['Outgassing Rate (Torr L / s m2)'].values
    outgassing_rate_prebaked = degassing_prebaked_df['Outgassing Rate (Torr L / s m2)'].values
    erosion_rate_graphite = degassing_graphite_df['Erosion Rate (cm/s)'].values
    erosion_rate_err_graphite = degassing_graphite_df['Erosion Rate Error (cm/s)'].values
    erosion_rate_prebaked = degassing_prebaked_df['Erosion Rate (cm/s)'].values
    erosion_rate_err_prebaked = degassing_prebaked_df['Erosion Rate Error (cm/s)'].values
    heat_load_graphite = heat_load.max() * np.ones_like(outgassing_rate_graphite)
    heat_load_prebaked = heat_load.max() * np.ones_like(outgassing_rate_prebaked)

    outgassing_rate_graphite *= outgassing_factor_20230331
    outgassing_rate_prebaked *= outgassing_factor_20230331

    load_plt_style()

    fig, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    fig.set_size_inches(6.5, 4.0)

    ax[0, 0].errorbar(
        heat_load_agg, erosion_rate, yerr=erosion_rate_err, ls='none',
        color='C0', marker='o', ms=8, fillstyle='none',  # label='Pebble sample',
        capsize=2.5, mew=1.25, elinewidth=1.25,
    )

    ax[0, 0].errorbar(
        heat_load_graphite, erosion_rate_graphite, yerr=erosion_rate_err_graphite, ls='none',
        color='goldenrod', marker='o', ms=8, fillstyle='full', label='Graphite',
        capsize=2.5, mew=1.25, elinewidth=1.25,
    )

    ax[0, 0].errorbar(
        heat_load_prebaked, erosion_rate_prebaked, yerr=erosion_rate_err_prebaked, ls='none',
        color='magenta', marker='h', ms=8, fillstyle='none', label='Outgassed pebbles',
        capsize=2.5, mew=1.25, elinewidth=1.25,
    )

    ax[0, 1].errorbar(
        heat_load_agg, pebble_velocity, yerr=pebble_velocity_err, ls='none',
        color='C1', marker='s', ms=8, fillstyle='none', ecolor=lighten_color('C1', 0.25),
        capsize=2.5, mew=1.25, elinewidth=1.25, label='Final positions'
    )

    # ax_nir = ax[0, 1].twinx()
    ax[0, 1].errorbar(
        nir_hl, nir_v0, yerr=(nir_v0_lb, nir_v0_ub), ls='none',
        color='saddlebrown', marker='>', ms=8, fillstyle='full', ecolor=lighten_color('saddlebrown', 0.25),
        capsize=2.5, mew=1.25, elinewidth=1.25, label='NIR imaging'
    )

    ax[1, 0].errorbar(
        heat_load, outgassing_rate, yerr=(outgassing_rate * outgassing_error_factor, outgassing_rate * outgassing_error_factor), ls='none',
        color='C2', marker='^', ms=8, fillstyle='none',
        capsize=2.5, mew=1.25, elinewidth=1.25,
    )

    ax[1, 0].errorbar(
        heat_load_graphite, outgassing_rate_graphite, yerr=(outgassing_rate_graphite * outgassing_error_factor, outgassing_rate_graphite * outgassing_error_factor),
        ls='none',
        color='goldenrod', marker='o', ms=8, fillstyle='full', label='Graphite',
        capsize=2.5, mew=1.25, elinewidth=1.25,
    )

    ax[1, 0].errorbar(
        heat_load_prebaked, outgassing_rate_prebaked, yerr=(outgassing_rate_prebaked * outgassing_error_factor, outgassing_rate_prebaked * outgassing_error_factor),
        ls='none',
        color='magenta', marker='h', ms=8, fillstyle='none', label='Outgassed pebbles',
        capsize=2.5, mew=1.25, elinewidth=1.25,
    )

    ax[1, 1].errorbar(
        heat_load_sublimation_pebble, sublimation_rate_pebble, yerr=sublimation_rate_pebble_err,
        ls='-', lw=1.5,
        color='tab:red', marker='D', ms=8, fillstyle='none', label='Pebble sample',
        capsize=2.5, mew=1.25, elinewidth=1.25,
    )

    ax[1, 1].errorbar(
        heat_load_sublimation_graphite, sublimation_rate_graphite, yerr=sublimation_rate_graphite_err,
        ls='-', lw=1.5,
        color='navy', marker='^', ms=8, fillstyle='none', label='Graphite',
        capsize=2.5, mew=1.25, elinewidth=1.25,
    )

    ax[1, 0].set_yscale('log')

    for axi in ax[1, :]:
        axi.set_xlabel(r'Heat load (MW/m$^{\mathregular{2}}$)')

    ax[0, 0].set_ylabel('cm/s')
    ax[0, 0].set_ylim(bottom=1E-4, top=1.0)
    ax[0, 0].set_title('Erosion rate')
    ax[0, 0].set_yscale('log')
    ax[0, 0].yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax[0, 0].yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * .1, numticks=20))

    ax[0, 1].set_ylabel('v$_{\mathregular{0}}$ (cm/s)', color='k')
    ax[0, 1].set_ylim(bottom=-10.0, top=150)
    ax[0, 1].set_title('Pebble velocity')
    ax[0, 1].yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax[0, 1].yaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax[0, 1].tick_params(axis='y', labelcolor='k')
    # ax_nir.set_ylabel('v$_{\mathregular{0, method~2}}$ (cm/s)', color='saddlebrown')
    # ax_nir.tick_params(axis='y', labelcolor='saddlebrown')
    # ax_nir.set_ylim(-10., 1000)
    # ax_nir.yaxis.set_major_locator(ticker.MultipleLocator(250))
    # ax_nir.yaxis.set_minor_locator(ticker.MultipleLocator(50))

    ax[1, 0].set_ylabel('Torr-L/s m$^{\mathregular{2}}$')
    ax[1, 0].set_ylim(bottom=1.0, top=1E4)
    ax[1, 0].set_title('Outgassing rate')
    ax[1, 0].yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax[1, 0].yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * .1, numticks=20))

    ax[1, 1].set_ylabel('nm/s')
    ax[1, 1].set_ylim(bottom=-25.0, top=250)
    ax[1, 1].set_title('Sublimation rate')
    ax[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax[1, 1].yaxis.set_minor_locator(ticker.MultipleLocator(25))

    ax[0, 0].legend(
        loc='upper left', frameon=True,
        prop={'size': 8}
    )

    ax[0, 1].legend(
        loc='upper left', frameon=True,
        prop={'size': 8}
    )

    ax[1, 0].legend(
        loc='upper left', frameon=True,
        prop={'size': 8}
    )

    ax[1, 1].legend(
        loc='upper left', frameon=True,
        prop={'size': 8}
    )

    for i, axi in enumerate(ax.flatten()):
        axi.set_xlim(left=0.0, right=50.0)
        axi.xaxis.set_major_locator(ticker.MultipleLocator(10.0))
        axi.xaxis.set_minor_locator(ticker.MultipleLocator(5.0))
        panel_label = chr(ord('`') + i + 1)
        axi.text(
            -0.15, 1.15, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )

    fig.savefig(os.path.join(output_dir, 'figure_6_revised.png'), dpi=600)
    fig.savefig(os.path.join(output_dir, 'figure_6_revised.eps'), dpi=600)
    fig.savefig(os.path.join(output_dir, 'figure_6_revised.svg'), dpi=600)

    plt.show()
