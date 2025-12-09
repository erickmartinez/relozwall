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
            8.395115E-1, -1.062564E-1, 1.065796E-2, 1.879809E-2, -8.881774E-3, -9.613779E-5, 1.404406E-3,
            -4.321739E-4, -2.544782E-5, 4.398471E-5, -9.997908E-6, 7.753453E-7
        ])

        self._eps_by_kb: float = 143.235 # scaled temperature that will depend on the intermolecular potential
        self._sigma_param: float = 0.33501 # intermolecular potential distance parameter (nm)

        # Define the coefficients for the reduced second viscosity virial coefficient
        self._ci: np.ndarray = np.array([
            -0.2571, 3.033, 1.144, -5.586, 3.089, -0.8824, -0.03856
        ])

        # Define the coefficients f_i for the residual viscosity term
        self._fi = np.array([
            3.62648753859904, 6.655428299399591, 0.397511608257391, 2.6697983930209, 0.0472018570860789
        ])

        self._Tc = 150.687 # Critical temperature [K]
        self._Pc = 4.863 # Critical pressure [Pa]
        self._rho_c = 535.6 # Critical density [kg / m^3]
        self._T_tp = 83.8058 # Triple point temperature [K]
        self._M = 39.948 # Molar mass [g / mol]
        self._R = 8.31451 # Molar mass constant [J / mol / K]


    @property
    def pressure_pascal(self):
        return self.pressure * 133.322

    @pressure_pascal.setter
    def pressure_pascal(self, value):
        self.pressure = value / 133.322

    def estimate_viscosity(self, rho: float, temperature: float) -> float:
        """
        Estimate viscosity using Sotiriadou et al. 2025

        Parameters
        ----------
        rho: float
            The density in kg/m3
        temperature: float
            The temperature in K

        Returns
        -------
        float:
            The viscosity estimate in uPa * s
        """
        # The contribution to the viscosity in the dilute-gas limit, where only two-body molecular interactions occur
        eta_0 = self.estimate_eta0(temperature)
        # The initial density dependence term
        eta_1 = self.estimate_eta_1(temperature)
        # The residual viscosity term
        delta_eta = self.estimate_delta_eta(rho=rho, temperature=temperature)
        # Since data close to the critical point are unavailable,
        # Δη_c(ρ,Τ) will be set to zero
        return eta_0 + eta_1 * rho + delta_eta


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
        n0 = 22.5666 # Pa * s
        for i, ai in enumerate(self._ai):
            argument += ai * (np.log(temperature/298.15))**(i+1)

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

        return B_eta_red * constants.N_A * (self._sigma_param **3) * 1E-27 * 1E-3 / self._M

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

    def estimate_delta_eta(self, rho: float, temperature: float) -> float:
        """
        Estimate the residual viscosity term

        Parameters
        ----------
        rho: float
            The density in kg / m^3
        temperature: float
            Temperature in Kelvin (K)

        Returns
        -------
        float:
            delta_eta
        """

        T_r = temperature / self._Tc  # Reduced temperature
        rho_r = rho / self._rho_c # Reduced pressure
        f = self._fi

        delta_eta = (rho_r ** (2/3)) * ( T_r ** 0.5 ) * (f[0] * rho_r
                                                         + f[1] * (rho_r ** 2) / T_r
                                                         + (f[0] * rho_r - (rho_r ** 2)) / (T_r ** 5)
                                                         + (rho_r - f[2] * (rho_r ** 5)) / (rho_r - f[3] - T_r)
                                                         - f[4])
        return delta_eta




if __name__ == "__main__":
    check_values_data = np.array([
        [300, 0, 22.6840],
        [300, 4, 22.7334],
        [300, 700, 49.3360],
        [150, 3.2255, 12.095]])
    check_values = np.empty(len(check_values_data), dtype=np.dtype([
        ('T [K]', 'd'), ('rho (kg/m3)', 'd'), ('eta (uPa-s)', 'd')
    ]))
    for i, row in enumerate(check_values_data):
        check_values['T [K]'][i] = row[0]
        check_values['rho (kg/m3)'][i] = row[1]
        check_values['eta (uPa-s)'][i] = row[2]

    viscosity = ArgonViscosity()

    for i, row in enumerate(check_values):
        eta_estimate = viscosity.estimate_viscosity(temperature=row['T [K]'], rho=row['rho (kg/m3)'])
        print(f'Estimated eta: {eta_estimate:>6.4f}, expected: {row["eta (uPa-s)"]:>6.4f}')