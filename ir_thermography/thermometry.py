import numpy as np
import pandas as pd
import sys
sys.path.append('../')

DEFAULT_CALIBRATION = '../ir_thermography/pd_brightness_processed.csv'
TRANSMISSION_WINDOW = 0.912
TRANSMISSION_SLIDE = 0.934


def temperature_at_radiance(radiance: float, wavelength_nm: float):
    """
    Estimates the blackbody temperature in Kelvin for a source at a given wavelength in nm and a known spectral radiance
    ..math::
        T(B, \lambda) = \frac{h c}{\lambda k} \left\{ \ln\left[\frac{hc^2}{\lambda^5 B} + 1\right]\right\}^{-1}

    Parameters
    ----------
    radiance: float
        The spectral radiance in W / cm^2 / nm / s / ster
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
    arg = 1.0E14 * arg / radiance  # 1E14 [J / (cm^2 nm s)] / [J / cm^2 nm s] = 1
    arg += 1.0
    temperature = 1E6 * hc / wavelength_nm / 1.380649  # 1E-26 (J m) / 1E-9 m / 1E-23 J/K
    temperature = temperature / np.log(arg)
    return temperature


class PDThermometer:
    __calibration_df: pd.DataFrame
    __gain: int = 0
    __valid_gains = None
    __wavelength: float = 900.0
    __transmission_factor: float = TRANSMISSION_WINDOW

    def __init__(self, calibration_url: str = DEFAULT_CALIBRATION):
        self.__calibration_df = pd.read_csv(calibration_url).apply(pd.to_numeric)
        self.__valid_gains = self.__calibration_df['Photodiode Gain (dB)'].unique()
        self.__transmission_factor = TRANSMISSION_WINDOW * TRANSMISSION_SLIDE
        print(self.__calibration_df)

    @property
    def gain(self) -> float:
        return self.__gain

    @gain.setter
    def gain(self, value):
        if value in self.__valid_gains:
            self.__gain == value

    @property
    def transmission_factor(self) -> float:
        return self.__transmission_factor

    @transmission_factor.setter
    def transmission_factor(self, value: float):
        value = abs(value)
        if (value != 0) and (value is not np.nan) and (value is not np.inf):
            self.__transmission_factor = value

    @property
    def calibration_factor(self) -> float:
        df = self.__calibration_df[self.__calibration_df['Photodiode Gain (dB)'] == self.gain]
        # brightness = df['Labsphere Brightness at 900 nm (W/ster/cm^2/nm)'].mean()
        # return brightness / df['Signal out (V)'].mean()
        print(df['Calibration Factor (W/ster/cm^2/nm/V) at 900 nm and 2900 K'].values)
        return df['Calibration Factor (W/ster/cm^2/nm/V) at 900 nm and 2900 K'].mean()

    def get_temperature(self, voltage: np.ndarray) -> np.ndarray:
        brightness = voltage * self.calibration_factor / self.transmission_factor
        return temperature_at_radiance(radiance=brightness, wavelength_nm=self.__wavelength)
