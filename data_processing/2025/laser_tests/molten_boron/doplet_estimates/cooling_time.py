import numpy as np
from scipy import constants
from astropy import units as u

PRESSURE = 200 # mTorr
KINETIC_DIAMETER_AR = 0.34 # nm
TEMPERATURE_GAS = 300

BORON_HEAT_OF_FUSION = 50.2 # kJ/mol
BORON_MOLAR_MASS = 10.811 # g/mol
BORON_HEAT_OF_FUSION = BORON_HEAT_OF_FUSION / BORON_MOLAR_MASS * 1E6 # J/kg
BORON_SPECIFIC_HEAT = 1.15 # J/g/K
BORON_DENSITY = 2.34 # g/cm^3
BORON_TEMPERATURE_MELTING = 2349. # K (~ 2076 + 273.15)
BORON_EMISSIVITY = 0.4

"""
CRUCIBLE DATA  
"""
W_SPECIFIC_HEAT = 0.13 # J/g/K
W_EMISSIVITY = 0.35



def gas_mean_path(pressure, temperature, atom_diameter) -> float:
    """
    Estimates the mean paht of the gas

    Parameters
    ----------
    pressure: float
        Pressure of the gass in Torr
    temperature: float
        Temperature of the gas in Kelvin
    atom_diameter: float
        Diameter of the gas in nm

    Returns
    -------
    float:
        Mean path of the gas in mm
    """
    kB = constants.value('Boltzmann constant') # J/K
    pressure = pressure * 1E-3 * 133.32 # Pa
    d = 1E-9 * atom_diameter # m
    return kB * temperature / ((2**0.5) * np.pi * (d ** 2) * pressure) * 1E3

def estimate_droplet_volume(diameter_mm) -> float:
    """
    Estimates the volume of the droplet in m^3

    Parameters
    ----------
    diameter_mm: float
        Diameter of the droplet in mm

    Returns
    -------
    float:
        Volume of the droplet in m^3
    """
    return 1E-9 * np.pi * diameter_mm ** 3 / 6

def estimate_droplet_area(diameter_mm) -> float:
    """
    Estimates the area of the droplet in m^2

    Parameters
    ----------
    diameter_mm: float
        Diameter of the droplet in mm

    Returns
    -------
    float:
        Area of the droplet in m^2
    """
    return 1E-6 * np.pi * diameter_mm ** 2

def estimate_radiative_heat_transfer(temperature, area, emissivity, temperature_ambient=300) -> float:
    """
    Estimates the radiative heat transfer between temperature and emissivity

    Parameters
    ----------
    temperature: float
        Temperature of the droplet in Kelvin
    area: float
        Area of the droplet in m^2
    emissivity: float
        Emissivity of the boron
    temperature_ambient: float
        Temperature of the ambient

    Returns
    -------
    float:
        Radiative heat transfer from the droplet at the target temperature in W
    """
    sigma = constants.value('Stefan-Boltzmann constant') # W/m^2K^4
    return emissivity * sigma * area * (temperature ** 4 - temperature_ambient ** 4)


def estimate_heat_to_reach_temperature(
    mass_g:float, temperature_initial:float, temperature_final:float=TEMPERATURE_GAS,
    specific_heat:float=BORON_SPECIFIC_HEAT
):
    """
    Estimates the heat required to reach the target temperature

    Parameters
    ----------
    mass_g: float
        Mass of the material in g
    temperature_initial: float
        Initial temperature of the material in K
    temperature_final: float
        Final temperature of the material in K
    specific_heat: float
        Specific heat of the material in J/g/K

    Returns
    -------
    float:
        Heat required to reach the target temperature in J
    """
    return specific_heat * mass_g * (temperature_final - temperature_initial)


if __name__ == "__main__":
    boron_source_volume = 1. # cm^3
    droplet_diameter_mm = 1.
    exposed_source_surface_area_cm = 1 # cm2

    # crucible_mass_g = 40. # g
    crucible_mass_g = 0. # <- Consider boron in BN crucible laser sourced aimed at boron
    # crucible_area_cm = 13. # cm2 # < tungsten crucible
    crucible_area_cm = 0. # <- Consider boron in BN crucible laser sourced aimed at boron
    heat_source_power = 1000. # W
    heat_source_efficiency = 0.9
    # heat_conduction_losses_supports_and_base = 300. # W
    heat_conduction_losses_supports_and_base = 100.  # W < BN crucible, laser heating


    boron_source_mass = boron_source_volume * BORON_DENSITY
    heat_to_reach_melting_point = estimate_heat_to_reach_temperature(
        mass_g=boron_source_mass, temperature_initial=TEMPERATURE_GAS, temperature_final=BORON_TEMPERATURE_MELTING,
    )
    heat_to_reach_melting_temperature_crucible = estimate_heat_to_reach_temperature(
        mass_g=crucible_mass_g, temperature_initial=TEMPERATURE_GAS, temperature_final=BORON_TEMPERATURE_MELTING,
    )
    heat_of_fusion_source = BORON_HEAT_OF_FUSION * boron_source_mass * 1E-3 # convert source mass from g to kg
    total_heat_to_melt_source = heat_to_reach_melting_point + heat_of_fusion_source + heat_to_reach_melting_temperature_crucible

    source_radiative_loss = estimate_radiative_heat_transfer(
        temperature=BORON_TEMPERATURE_MELTING, area=exposed_source_surface_area_cm*1E-4, emissivity=BORON_EMISSIVITY,
    )
    crucible_radiative_loss = estimate_radiative_heat_transfer(
        temperature=BORON_TEMPERATURE_MELTING, area=crucible_area_cm*1E-4, emissivity=W_EMISSIVITY
    )

    heating_radiative_loss = source_radiative_loss + crucible_radiative_loss
    total_heat_losses = heating_radiative_loss + heat_conduction_losses_supports_and_base
    effective_source_power = heat_source_power*heat_source_efficiency - total_heat_losses
    time_to_reach_melting_point = total_heat_to_melt_source / effective_source_power


    mean_free_path = gas_mean_path(PRESSURE, TEMPERATURE_GAS, KINETIC_DIAMETER_AR)
    knudsen_number = mean_free_path / droplet_diameter_mm
    droplet_volume = estimate_droplet_volume(droplet_diameter_mm)
    droplet_area = estimate_droplet_area(droplet_diameter_mm)
    droplet_mass = droplet_volume * (BORON_DENSITY * 1000) # kg
    radiative_heat_transfer = estimate_radiative_heat_transfer(
        temperature=BORON_TEMPERATURE_MELTING, area=droplet_area, emissivity=BORON_EMISSIVITY
    )
    # How much heat needed to solidify droplet
    heat_solid = BORON_HEAT_OF_FUSION * droplet_mass

    # At estimated radiative heat transfer, how long does it take for heat of solidification to be removed:
    time_solidification_radiative = heat_solid / radiative_heat_transfer

    print("-" * 20)
    print(f'Boron source volume: {boron_source_mass} cm^3')
    print(f'Boron source mass: {boron_source_mass} g')
    print(f'Heat to reach melting point: {heat_to_reach_melting_point*1E-3:.2f} kJ')
    print(f'Heat required for crucible to reach {BORON_TEMPERATURE_MELTING-273:.0f}: '
          f'{heat_to_reach_melting_temperature_crucible*1E-3:.2f} J')
    print(f'Heat to melt source: {heat_of_fusion_source * 1E-3:.2f} J')
    print(f'Total heat to melt {boron_source_mass:.2f} g of boron: {total_heat_to_melt_source*1E-3:.2f} kJ')
    print(
        f'Radiative loss from boron exposed area ({exposed_source_surface_area_cm:.2f} cm^2: '
        f'{source_radiative_loss * 1E-3:.2f} kW)'
    )
    print(f'Radiative loss from crucible area ({crucible_area_cm:.2f} cm^2: {crucible_radiative_loss * 1E-3:.2f} kW)')
    print(f'Heat conduction losses through support/base: {heat_conduction_losses_supports_and_base*1E-3:.2f} kW')
    print(f'Heat source power: {heat_source_power:.2f} W')
    print(f'Total heat losses: {total_heat_losses:.2f} W')
    print(f'Effective source power: {effective_source_power*1E-3:.2f} kW')
    print(f'Time to reach melting point: {time_to_reach_melting_point:.2f} s')
    print("-"*20)




    print(f'Droplet volume: {droplet_volume:.2E} m^3')
    print(f'Droplet area: {droplet_area:.2E} m^2')
    print(f'Droplet mass: {droplet_mass:.2E} kg')
    print(f'Ar mean free path: {mean_free_path:.3f} mm')
    print(f'Heat of fusion: {BORON_HEAT_OF_FUSION:.2E} J/kg')
    print(f'Radiative heat transfer: {radiative_heat_transfer:.2f} W')
    print(f'Heat removed to become solid: {heat_solid:.2f} J')
    print(f'Time to solidification (radiative only): {time_solidification_radiative:.2f} s')
    print(f'Knudsen number: {knudsen_number:.2f}')



