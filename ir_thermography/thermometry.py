import numpy as np
import pandas as pd
import sys

sys.path.append('../')

DEFAULT_CALIBRATION = 'https://raw.githubusercontent.com/erickmartinez/relozwall/main/ir_thermography/pd_brightness_processed.csv'
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


def radiance_at_temperature(temperature: float, wavelength_nm: float) -> float:
    hc = 6.62607015 * 2.99792458  # x 1E -34 (J s) x 1E8 (m/s) = 1E-26 (J m)
    hc2 = hc * 2.99792458  # x 1E -34 (J s) x 1E16 (m/s)^2 = 1E-18 (J m^2 s^{-1})
    factor = 2. * 1E14 * hc2 * np.power(wavelength_nm, -5.0) # W / cm^2 / nm
    arg = 1E6 * hc / wavelength_nm / 1.380649 / temperature  # 26 (J m) / 1E-9 m / 1E-23 J/K
    return factor / (np.exp(arg) - 1.)


class PDThermometer:
    __calibration_df: pd.DataFrame
    __gain: int = 0.0
    __valid_gains = None
    __emissivity = 0.8
    __wavelength: float = 900.0
    __transmission_factor: float = TRANSMISSION_WINDOW
    __noise_level: float = 0.08

    def __init__(self, calibration_url: str = DEFAULT_CALIBRATION):
        self.__calibration_df = pd.read_csv(calibration_url).apply(pd.to_numeric)
        self.__valid_gains = self.__calibration_df['Photodiode Gain (dB)'].unique()
        self.__transmission_factor = TRANSMISSION_WINDOW * TRANSMISSION_SLIDE
        self.__gain = 0

    @property
    def noise_level(self) -> float:
        return self.__noise_level

    @noise_level.setter
    def noise_level(self, val):
        self.__noise_level = abs(float(val))

    @property
    def emissivity(self) -> float:
        return self.__emissivity

    @emissivity.setter
    def emissivity(self, value):
        value = float(value)
        if 0.0 < value < 1.0:
            self.__emissivity = value

    @property
    def valid_gains(self):
        gains = self.__calibration_df['Photodiode Gain (dB)'].unique()
        return [int(g) for g in gains]

    @property
    def gain(self) -> float:
        return self.__gain

    @gain.setter
    def gain(self, value: int):
        value = int(value)
        if value in self.valid_gains:
            self.__gain = value
        else:
            print(f"{value} is an invalid gain.")

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
        return df['Calibration Factor (W/ster/cm^2/nm/V) at 900 nm and 2900 K'].mean()

    def get_temperature_at_brightness(self, brightness: np.ndarray) -> np.ndarray:
        return temperature_at_radiance(radiance=brightness, wavelength_nm=self.__wavelength)

    def get_temperature(self, voltage: np.ndarray) -> np.ndarray:
        brightness = voltage * self.calibration_factor / self.transmission_factor / self.emissivity
        return temperature_at_radiance(radiance=brightness, wavelength_nm=self.__wavelength)

    def get_voltage_at_temp(self, temperature:float) -> np.ndarray:
        brightness = radiance_at_temperature(temperature=temperature, wavelength_nm=self.__wavelength)
        voltage = brightness / self.calibration_factor * self.transmission_factor * self.emissivity
        return voltage
