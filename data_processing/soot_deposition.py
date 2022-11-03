import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
import pandas as pd

from matplotlib import ticker

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\surface_temperature\equilibrium_redone'
transmission_csv = 'slide_transmission_smausz.csv'

distance_sample_substrate_in = 9.0
sublimation_angle = 35.0

"""
Density of graphite rod used
https://www.graphitestore.com/core/media/media.nl?id=6310&c=4343521&h=Tz5uoWvr-nhJ13GL1b1lG8HrmYUqV1M_1bOTFQ2MMuiQapxt
"""
density_graphite = 1.81  # g/cm^3
molar_mass_graphite = 12.011  # g/mol
"""
A. L. Marshall and Francis J. Norton
VAPOR PRESSURE AND HEAT OF VAPORIZATION OF GRAPHITE
https://pubs.acs.org/doi/pdf/10.1021/ja01328a513
"""
heat_of_sublimation = 177.0  # kcal/mol
sample_diameter = 2.54 * 3.0 / 8.0
laser_power_100pct = 4.7E3  # kW
beam_diameter = 1.5 * 0.8165

colors = {'g': 'navy', 'p': 'tab:red'}

if __name__ == '__main__':
    thickness_df = pd.read_csv(os.path.join(base_dir, transmission_csv))
    column_names = thickness_df.columns
    thickness_df[column_names[1:]] = thickness_df[column_names[1:]].apply(pd.to_numeric)

    d_ss = distance_sample_substrate_in * 2.54
    radius = d_ss * np.tan(sublimation_angle * np.pi / 180.0)
    area = np.pi * radius * radius
    sample_area = 0.25 * np.pi * (sample_diameter ** 2.0)
    print(f'Deposition area: {area:.2f} cm2')

    graphite_df = thickness_df[thickness_df['Sample'] == 'GT001688']
    pebble_df = thickness_df[thickness_df['Sample'] != 'GT001688']

    laser_power_g = graphite_df['Laser Power (%)'].values
    laser_power_p = pebble_df['Laser Power (%)'].values
    thickness_g = graphite_df['Thickness (nm)'].values
    thickness_p = pebble_df['Thickness (nm)'].values
    flat_top_time_g = graphite_df['Flat top time (s)'].values
    flat_top_time_p = pebble_df['Flat top time (s)'].values

    heat_flux_p = laser_power_p * (0.01 * laser_power_100pct)
    heat_flux_g = laser_power_g * (0.01 * laser_power_100pct)

    heat_flux_factor = 0.01 * (1.0 - np.exp(-2.0 * (sample_diameter / beam_diameter) ** 2.0)) / sample_area
    heat_flux_p *= heat_flux_factor
    heat_flux_g *= heat_flux_factor

    moles_g = (thickness_g * 1E-7 * area) * density_graphite / molar_mass_graphite
    moles_p = (thickness_p * 1E-7 * area) * density_graphite / molar_mass_graphite
    heat_sublimation_film_g = 4184 * heat_of_sublimation * moles_g
    heat_sublimation_film_p = 4184 * heat_of_sublimation * moles_p

    cooling_power_sublimation_g = np.zeros_like(heat_sublimation_film_g)
    cooling_power_sublimation_p = np.zeros_like(heat_sublimation_film_p)

    msk_non_zero_g = flat_top_time_g > 0
    msk_non_zero_p = flat_top_time_p > 0

    cooling_power_sublimation_g[msk_non_zero_g] = heat_sublimation_film_g[msk_non_zero_g] / flat_top_time_g[msk_non_zero_g] / sample_area * 0.01
    cooling_power_sublimation_p[msk_non_zero_p] = heat_sublimation_film_p[msk_non_zero_p] / flat_top_time_p[msk_non_zero_p] / sample_area * 0.01


    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1, nrows=2 , constrained_layout=True)
    fig.set_size_inches(5.0, 5.0)

    mew = 1.5
    ls = '-'
    ax[0].plot(
        heat_flux_g, thickness_g, c=colors['g'],
        marker='o', fillstyle='none', ls=ls, mew=mew,
        label='Graphite'
    )

    ax[0].plot(
        heat_flux_p, thickness_p, c=colors['p'],
        marker='s', fillstyle='none', ls=ls, mew=mew,
        label='Sample'
    )

    ax[1].plot(
        heat_flux_g, cooling_power_sublimation_g, c=colors['g'],
        marker='o', fillstyle='none', ls=ls, mew=mew
    )

    ax[1].plot(
        heat_flux_p, cooling_power_sublimation_p, c=colors['p'],
        marker='s', fillstyle='none', ls=ls, mew=mew
    )

    ax[1].set_xlabel('Heat Flux (MW/m$^{\mathregular{2}}$)')
    ax[0].set_ylabel('nm')
    ax[1].set_ylabel('MW/m$^{\mathregular{2}}$')
    ax[0].set_title('Deposition thickness')
    ax[1].set_title('Sublimation cooling')

    ax[0].legend(
        loc='upper left', frameon=True
    )

    ax[0].set_xlim(left=0.0, right=50.0)
    ax[1].set_xlim(left=0.0, right=50.0)
    # ax[0].set_ylim(top=150.0)

    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(5.0))
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(5.0))

    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(50.0))
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(25.0))
    # ax[1].yaxis.set_major_locator(ticker.MultipleLocator(2.0))
    # ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(1.0))

    fig.savefig(os.path.join(base_dir, 'sublimation_heat_plot.png'), dpi=600)


    plt.show()
