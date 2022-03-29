import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from matplotlib.ticker import ScalarFormatter

import numpy as np
from scipy import constants


def spectral_radiance(wavelength_nm, temperature: float):
    """
    Estimates the blackbody radiation as a function of wavelength and temperature
    .. math::
        B_{\nu}(T) = \frac{2 h c^2 / \lambda^5}{e^{h c/\lambda kT} - 1}
    where :math:`\nu = c/\lambda`

    Parameters
    ----------
    wavelength_nm: float
        The wavelength in nanometers
    temperature: float
        The temperature in Kelvin

    Returns
    -------
    float:
        The spectral radiance in W / (cm^2 nm sr)
    """

    hc = 6.62607015 * 2.99792458  # x 1E -34 (J s) x 1E8 (m/s) = 1E-26 (J m)
    hc2 = hc * 2.99792458  # x 1E -34 (J s) x 1E16 (m/s)^2 = 1E-18 (J m^2 s^{-1})
    hc_by_kt_lambda = 1E6 * hc / 1.380649 / temperature / wavelength_nm
    radiance = 2.0 * hc2 * np.power(wavelength_nm, -5.0)  # 1E-18 (J m^2 s^{-1}) / 1E-45 (m^5) = 1E27 J / (m^3 s)
    radiance *= 1E14 / (np.exp(hc_by_kt_lambda) - 1.0)  # 1E27 J/m^3 s = 1E18 J / (m^2 nm s) = 1E14 J / cm^2 nm
    return radiance


def temperature_at_radiance(radiance: float, wavelength_nm: float):
    """
    Estimates the blackbody temperature in Kelvin for a source at a given wavelength in nm and a known spectral radiance
    ..math::
        T(B, \lambda) = \frac{h c}{\lambda k} \left\{ \ln\left[\frac{hc^2}{\lambda^5 B} + 1\right]\right\}^{-1}

    Parameters
    ----------
    radiance: float
        The spectral radiance in W / cm^2 / nm / s / sr
    wavelength_nm: float
        The wavelength in nm


    Returns
    -------
    float:
        The estimated temperature in K
    """
    hc = 6.62607015 * 2.99792458  # x 1E -34 (J s) x 1E8 (m/s) = 1E-26 (J m)
    hc2 = hc * 2.99792458  # x 1E -34 (J s) x 1E16 (m/s)^2 = 1E-18 (J m^2 s^{-1})
    arg = 2.0 * hc2 * np.power(wavelength_nm, -5.0)  # 1E18 J / (m^2 nm s)
    arg = 1.0E14 * arg /radiance  #  1E14 [J / (cm^2 nm s)] / [J / cm^2 nm s] = 1
    arg += 1.0
    temperature = 1E6 * hc / wavelength_nm / 1.380649  # 1E-26 (J m) / 1E-9 m / 1E-23 J/K
    temperature = temperature / np.log(arg)
    return temperature


if __name__ == "__main__":
    calibration_df = pd.read_csv('https://raw.githubusercontent.com/erickmartinez/relozwall/main/ir_thermography/photodiode_brightness_calibration_202202.csv')
    calibration_df[:] = calibration_df[:].apply(pd.to_numeric)
    print(calibration_df.columns)
    brightness = calibration_df['Labsphere Brightness at 900 nm [W/ster/cm^2/nm]'].values
    wavelength_nm = 900.0 * np.ones_like(brightness)
    calibration_df['Temperature at radiance (K)'] = temperature_at_radiance(
        brightness,
        wavelength_nm
    )
    print(calibration_df)