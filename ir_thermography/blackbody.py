import numpy as np
from scipy import constants


def spectral_radiance(wavelength_nm, temperature: float):
    """
    Estimates the blackbody radiation as a function of wavelength and temperature
    .. math::
        B_{\nu}(T) = \frac{8 \pi h c}{\lambda^5} \frac{1}{e^{h c/\lambda kT} - 1}
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
        The spectral radiance in J * s / (cm^2 nm sr)
    """

    hc = 6.62607015 * 2.99792458  # x 1E -34 (J s) x 1E8 (m/s) = 1E-26 (J m)
    hc2 = hc * 6.62607015 # x 1E -34 (J s) x 1E16 (m/s)^2 = 1E-18 (J m^2 s^{-1})
    hc_by_kt = hc / 1.3806491E-3 / temperature
    radiance = 2 * hc2 * np.power(wavelength_nm, -5.0)  # 1E-18 (J m^2 s^{-1}) / 1E-45 (m^5) = 27 J/m^3 s
    radiance *= np.power(np.exp(hc_by_kt) - 1.0, -1.0)
    radiance *= 2 * 2.99792458 / 1E18
    return radiance


def temperature_at_radiacnce(radiance: float, wavelength_nm: float):
    v = 2.99792458 / wavelength_nm  # (1E8 m/s) x (1E-9 m) = 1E17 (1/s)
    hv = 6.62607015 * v  # x 1E -34 (J s) x 1E17 (1/s) = 1E-17 (J)
