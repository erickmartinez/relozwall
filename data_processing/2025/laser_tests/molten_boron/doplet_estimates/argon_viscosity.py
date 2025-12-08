import numpy as np
from scipy import constants

class ArgonViscosity:
    def __init__(self):
        """
        Class constructor, takes temperature, pressure

        """

        # Define the coefficients for the contribution to the viscosity in the dilute-gas limit, where only two-body
        # molecular interactions
        self._ai = np.array([
            8.395115E1, -1.062564E-1, 1.065796E-2, 1.879809E-2, -8.881774E-3, -9.613779E-5, 1.404406E-3,
            -4.321739E-4, -2.544782E-5, 4.398471E-5, -9.997908E-6, 7.753453E-7
        ])

        self._eps_by_kb: float = 143.235 # scaled temperature that will depend on the intermolecular potential
        self._sigma_param: float = 0.33501 # intermolecular potential distance parameter (nm)

        # Define the coefficients for the reduced second viscosity virial coefficient
        self._ci: np.ndarray = np.array([
            -0.2571, 3.033, 1.144, -5.586, 3.089, -0.8824, -0.03856
        ])

    @property
    def pressure_pascal(self):
        return self.pressure * 133.322

    @pressure_pascal.setter
    def pressure_pascal(self, value):
        self.pressure = value / 133.322

    def estimate_eta0(self, temperature) -> float:
        """
        Estimates the contribution to the viscosity in the dilute-gas limit, where only two-body molecular interactions
        occur.

        Parameters
        ----------
        temperature: float
            Temperature in Kelvin (K)

        Returns
        -------
        float:
            eta_0 in Pa * s
        """
        argument = 0
        # eta_0(298.15) = 22.5666 uPa * s
        n0 = 22.5666E-6 # Pa * s
        for i, ai in enumerate(self._ai):
            argument += ai * (np.log(temperature/298.15)*(i+1))

        return n0*np.exp(argument)


    def estimate_b_eta(self, temperature: float) -> float:
        """
        The second viscosity virial coefficient

        Parameters
        ----------
        temperature: float
            Temperature in Kelvin (K)

        Returns
        -------
        float:
            B_eta
        """
        # First estimate the reduced second viscosity virial coefficient
        T_red = temperature / self._eps_by_kb
        B_eta_red = 0.
        for i, ci in enumerate(self._ci):
            B_eta_red += ci / (T_red ** i)

        return B_eta_red * constants.NA * self._sigma_param * 1E-27

    def estimate_eta_1(self, temperature) -> float:
        """
        Estimates the linear-in-density term

        Parameters
        ----------
        temperature: float
            Temperature in Kelvin (K)

        Returns
        -------
        float:
            eta_1
        """
        return self.estimate_eta0(temperature) * self.estimate_b_eta(temperature)




