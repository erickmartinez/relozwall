import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
import pandas as pd
from utils import lighten_color
from matplotlib import ticker

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\surface_temperature\equilibrium_redone'
transmission_csv = 'slide_transmission_smausz.csv'

distance_sample_substrate_in = 9.0
sublimation_angle = 35.0

"""
Density of graphite rod used
https://www.graphitestore.com/core/media/media.nl?id=6310&c=4343521&h=Tz5uoWvr-nhJ13GL1b1lG8HrmYUqV1M_1bOTFQ2MMuiQapxt
"""
density_graphite = 1.723  # g/cm^3
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

"""
K. Shinzato and T. Baba
A LASER FLASH APPARATUS FOR THERMAL
DIFFUSIVITY AND SPECIFIC HEAT CAPACITY
MEASUREMENTS
https://link.springer.com/content/pdf/10.1023/A:1011594609521.pdf
"""
specific_heat_p = 0.714
specific_heat_g = 0.6752  # Markelov, Volga, et al., 1973
rho_g = 1.81 # https://www.graphitestore.com/core/media/media.nl?id=6310&c=4343521&h=Tz5uoWvr-nhJ13GL1b1lG8HrmYUqV1M_1bOTFQ2MMuiQapxt
rho_p = 0.918
sample_length_g = 3.5
sample_length_p = 3.15

k_g = 85E-2  # W / (cm K)
a_g = k_g / rho_g /  specific_heat_g
a_p = 1.29E-2  # cm^2/s

cte_gc = 1E-6 * np.array([4.4, 3.6, 5.4]).mean()  # /°C
cte_g = 4.6E-6  # /°C

sb = 5.670374419E-8 # W/m2 K4
reflectance = 40.4
emissivity = 1.0 - (reflectance / 100.0)
ambient_temperature = 20.0

colors = {'g': 'navy', 'p': 'tab:red'}

cmap = plt.cm.tab20b_r


if __name__ == '__main__':
    colors = cmap(np.linspace(0, 1, 4))

    thickness_df = pd.read_csv(os.path.join(base_dir, transmission_csv))
    column_names = thickness_df.columns
    thickness_df[column_names[1:]] = thickness_df[column_names[1:]].apply(pd.to_numeric)
    ambient_temperature_k = ambient_temperature + 273.15

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
    erosion_rate_g = graphite_df['Erosion rate (cm/s)'].values
    erosion_rate_p = pebble_df['Erosion rate (cm/s)'].values
    max_temperautre_g = graphite_df['Max surface temperature (C)'].values
    max_temperautre_p = pebble_df['Max surface temperature (C)'].values
    base_pressure_g = graphite_df['Base Pressure (mTorr)'].values
    base_pressure_p = pebble_df['Base Pressure (mTorr)'].values

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

    cooling_power_erosion_g = 1E-2 * erosion_rate_g * rho_g * (max_temperautre_g - ambient_temperature) * specific_heat_g
    cooling_power_erosion_p = 1E-2 * erosion_rate_p * rho_p * (max_temperautre_p - ambient_temperature) * specific_heat_p

    cooling_radiation_g = sb * 1.0 * ((max_temperautre_g + 273.15) ** 4.0 - ambient_temperature_k ** 4.0) * 1E-6
    cooling_radiation_p = sb * 1.0 * ((max_temperautre_p + 273.15) ** 4.0 - ambient_temperature_k ** 4.0) * 1E-6

    cooling_conduction_g = np.zeros_like(heat_sublimation_film_g)
    cooling_conduction_p = np.zeros_like(heat_sublimation_film_p)
    cooling_conduction_g[msk_non_zero_g] = 2E-2 * specific_heat_g * rho_g * np.sqrt(a_g/flat_top_time_g[msk_non_zero_g]) * (max_temperautre_g[msk_non_zero_g] - ambient_temperature)
    cooling_conduction_p[msk_non_zero_p] = 2E-2 * specific_heat_p * rho_p * np.sqrt(a_p/ flat_top_time_p[msk_non_zero_p]) * (max_temperautre_p[msk_non_zero_p] - ambient_temperature)

    sample_volume_g = np.zeros_like(heat_sublimation_film_g)
    sample_volume_p = np.zeros_like(heat_sublimation_film_p)
    sample_volume_g = sample_area * 2.0 * np.sqrt(a_g*flat_top_time_g)
    sample_volume_p = sample_area * 2.0 * np.sqrt(a_p*flat_top_time_p)

    expansion_factor_g = 1.0 + cte_g * (max_temperautre_g - ambient_temperature)
    expansion_factor_p = 1.0 + cte_gc * (max_temperautre_p - ambient_temperature)
    dv_g = sample_volume_g * (expansion_factor_g ** 3.0 - 1.0) * 1E-6
    dv_p = sample_volume_p * (expansion_factor_p ** 3.0 - 1.0) * 1E-6

    output_g_df = pd.DataFrame(data={
        'Heat flux (MW/m^2)': heat_flux_g,
        'Conduction cooling (MW/m^2)' : cooling_conduction_g,
        'Sublimation cooling (MW/m^2)': cooling_power_sublimation_g,
        'Erosion cooling (MW/m^2)': cooling_power_erosion_g,
        'Radiation cooling (MW/m^2)': cooling_radiation_g
    })

    output_p_df = pd.DataFrame(data={
        'Heat flux (MW/m^2)': heat_flux_p,
        'Conduction cooling (MW/m^2)': cooling_conduction_p,
        'Sublimation cooling (MW/m^2)': cooling_power_sublimation_p,
        'Erosion cooling (MW/m^2)': cooling_power_erosion_p,
        'Radiation cooling (MW/m^2)': cooling_radiation_p
    })

    output_g_df.to_csv(
        os.path.join(base_dir, 'graphite_heat_analysis.csv'),
        index=False
    )

    output_p_df.to_csv(
        os.path.join(base_dir, 'pebble_sample_heat_analysis.csv'),
        index=False
    )

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1, nrows=2, constrained_layout=True)
    fig.set_size_inches(4.0, 4.5)

    mew = 1.5
    ls = '-'
    width = 3.0
    error_kw = dict(lw=1.0, capsize=3.5, capthick=1.0, ecolor='tab:grey')

    ax[0].bar(
        heat_flux_g, cooling_conduction_g, width,
        color=colors[0],
        hatch='////', lw=0.5, ec=lighten_color(colors[0], 1.5),
        ls=ls, label='Conduction'
    )

    ax[0].bar(
        heat_flux_g, cooling_power_sublimation_g, width,
        bottom=cooling_conduction_g,
        color=colors[1], lw=0.5, ec=lighten_color(colors[1], 1.5),
        ls=ls, label='Sublimation'
    )

    ax[0].bar(
        heat_flux_g, cooling_power_erosion_g, width,
        bottom=cooling_conduction_g+cooling_power_sublimation_g,
        color=colors[2],
        hatch=r'\\\\', lw=0.5, ec=lighten_color(colors[2], 1.5),
        ls=ls, label='Erosion'
    )

    ax[0].bar(
        heat_flux_g, cooling_radiation_g, width,
        # yerr=(cooling_radiation_g+cooling_conduction_g+cooling_power_sublimation_g+cooling_power_erosion_g)*0.5,
        error_kw=error_kw,
        bottom=cooling_conduction_g+cooling_power_sublimation_g+cooling_power_erosion_g,
        color=colors[3],
        lw=0.5, ec=lighten_color(colors[3], 1.5),
        ls=ls, label='Radiation'
    )


    xlims = ax[0].get_xlim()
    input_power = np.linspace(xlims[0], xlims[1], 100)

    ax[0].plot(
        input_power, input_power, lw=1.0, ls='--', color=(0.8, 0.8, 0.9),
        label='Input', fillstyle='none', mew=1.0, ms=6,
    )

    # ax[0].bar(
    #     heat_flux_g, cooling_expansion_g, width,
    #     bottom=cooling_conduction_g + cooling_power_sublimation_g + cooling_power_erosion_g + cooling_radiation_g,
    #     color='C4',
    #     ls=ls, label='Expansion'
    # )

    ax[1].bar(
        heat_flux_p, cooling_conduction_p, width,
        color=colors[0],
        hatch='////', lw=0.5, ec=lighten_color(colors[0], 1.5),
        ls=ls, label='Conduction'
    )

    ax[1].bar(
        heat_flux_p, cooling_power_sublimation_p, width,
        color=colors[1],
        lw=0.5, ec=lighten_color(colors[1], 1.5),
        bottom=cooling_conduction_p,
        ls=ls, label='Sublimation'
    )

    ax[1].bar(
        heat_flux_p, cooling_power_erosion_p, width,
        bottom=cooling_conduction_p+cooling_power_sublimation_p,
        color=colors[2],
        hatch=r'\\\\', lw=0.5, ec=lighten_color(colors[2], 1.5),
        ls=ls, label='Erosion'
    )

    ax[1].bar(
        heat_flux_p, cooling_radiation_p, width,
        # yerr=(cooling_radiation_p + cooling_conduction_p + cooling_power_sublimation_p + cooling_power_erosion_p) * 0.5,
        error_kw=error_kw,
        bottom=cooling_conduction_p+cooling_power_sublimation_p + cooling_power_erosion_p,
        color=colors[3],
        lw=0.5, ec=lighten_color(colors[3], 1.5),
        ls=ls, label='Radiation'
    )

    ax[1].plot(
        input_power, input_power, lw=1.0, ls='--', color=(0.8, 0.8, 0.9),
        label='Input', fillstyle='none', mew=1.0, ms=6,
    )

    # ax[1].bar(
    #     heat_flux_p, cooling_expansion_p, width,
    #     bottom=cooling_conduction_p + cooling_power_sublimation_p + cooling_power_erosion_p + cooling_radiation_p,
    #     color='C4',
    #     ls=ls, label='Expansion'
    # )

    # ax[0].set_xlabel('Heat Flux (MW/m$^{\mathregular{2}}$)')
    ax[1].set_xlabel('Heat load (MW/m$^{\mathregular{2}}$)')
    ax[0].set_ylabel('MW/m$^{\mathregular{2}}$')
    ax[1].set_ylabel('MW/m$^{\mathregular{2}}$')
    ax[0].set_title('Graphite')
    ax[1].set_title('Pebble sample')

    ax[0].legend(
        loc='upper left', frameon=True, prop={'size': 9}, ncol=2
    )
    ax[1].legend(
        loc='upper left', frameon=True, prop={'size': 9}, ncol=2
    )

    ax[0].set_xlim(left=0.0, right=50.0)
    ax[1].set_xlim(left=0.0, right=50.0)
    ax[0].set_ylim(top=50.0)
    ax[1].set_ylim(top=50.0)
    ax[0].tick_params(axis='x', which='both', direction='out')
    ax[1].tick_params(axis='x', which='both', direction='out')

    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(5.0))
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(5.0))

    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(5.0))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(5.0))

    # Add panel labels out of the box
    ax[0].text(
        -0.1, 1.15, '(a)', transform=ax[0].transAxes, fontsize=14, fontweight='bold',
        va='top', ha='right'
    )
    ax[1].text(
        -0.1, 1.15, '(b)', transform=ax[1].transAxes, fontsize=14, fontweight='bold',
        va='top', ha='right'
    )

    fig.savefig(os.path.join(base_dir, 'heat_balance_plot.png'), dpi=600)
    fig.savefig(os.path.join(base_dir, 'heat_balance_plot.eps'), dpi=600)
    fig.savefig(os.path.join(base_dir, 'heat_balance_plot.svg'), dpi=600)


    plt.show()
