import pandas as pd
from data_processing.utils import get_laser_power_mapping
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import json

SAMPLE_DIAMETER = 1.016 # cm
BEAM_DIAMETER = 0.8164  # cm

def load_plot_style():
    with open(r'plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    rcParams.update(plot_style)
    rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}')


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)

def estimate_incident_heat_load(beam_diameter: float, sample_diameter: float, laser_power_setting: float):
    """

    Parameters
    ----------
    beam_diameter: float
        The diameter of the laser beam in cm
    sample_diameter: float
        The diameter of the sample in cm
    laser_power_setting: float
        The perentage of power on the laser diode

    Returns
    -------
    float:
        The incident heat load in MW/m²

    """
    d_s = sample_diameter * 0.5
    d_b = beam_diameter * 0.5

    power_mapping = get_laser_power_mapping(csv=r'')


