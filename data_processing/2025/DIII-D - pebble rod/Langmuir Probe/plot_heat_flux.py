import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate as integrate
from scipy.integrate import simpson, trapezoid
from scipy.interpolate import make_smoothing_spline, CubicSpline
from scipy import ndimage
from pathlib import Path
from pybaselines import Baseline
from data_processing.misc_utils.plot_style import load_plot_style
from data_processing.utils import latex_float
from typing import Dict
import h5py

SHOT = 203782

def load_lp_data(shot, path_to_folder):
    path_to_folder = Path(path_to_folder)
    with h5py.File( path_to_folder / f'{shot}_LP.h5', 'r') as h5:
        dimes_gp = h5['/LANGMUIR_DIMES']
        t_ms = np.array(dimes_gp.get('time')) * 1E-3
        T_eV = np.array(dimes_gp.get('TeV'))
        qpara = np.array(dimes_gp.get('qpara'))
    return t_ms, T_eV, qpara