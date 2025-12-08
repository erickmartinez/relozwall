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

