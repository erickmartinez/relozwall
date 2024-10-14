import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from scipy import interpolate
from scipy.interpolate import interp1d

plasma_params = pd.DataFrame(data={
    'Sample': ['Boron rod', 'Boron pebble rod'],
    'Date': ['2024/08/15', '2024/08/27'],
    'T_e (eV)': [5.85, 5.3],
    'n_e 10^18 (/m^3)': [0.212, 0.2475],
    'phi_i 10^22 (/m^3/s)': [0.2125, 0.2375],
    'E_i (eV)': [62.6, 61.25],
    'Temp (K)': [458, 472]
})

def x_rate(T_e: np.ndarray) -> np.ndarray:
    """
    Estimates the excitation rate coefficient from the
    ground state of B-H for the transition:

    .. math::\Chi^1 \Sigma^+ \to \mathrm{A}^1\Pi

    as a function of the electron temperature.

    This relationship corresponds to the modified Arrhenius function
    .. math:: k = A T_e^n\exp\left(-\frac{E_{\mathrm{act}}{T_e}\right)

    described in Kawate et al. Plasma Sources Sci. Technol. 32, 085006 (2023)
    doi: 10.1088/1361-6595/acec0c


    Parameters
    ----------
    T_e: np.ndarray
        The electron temperature in eV

    Returns
    -------
    np.ndarray:
        The excitation rate coefficient in cm^3/s

    """
    return 5.62E-8 * np.power(T_e, 0.021) * np.exp(-3.06 / T_e)

def s_rate(T_e: np.ndarray) -> np.ndarray:
    """
        Estimates the ionization rate coefficient from the
        ground state of B-H for the transition:

        .. math::\Chi^1 \Sigma^+ \to \mathrm{A}^1\Pi

        as a function of the electron temperature.

        This relationship corresponds to the modified Arrhenius function
        .. math:: k = A T_e^n\exp\left(-\frac{E_{\mathrm{act}}{T_e}\right)

        described in Kawate et al. Plasma Sources Sci. Technol. 32, 085006 (2023)
        doi: 10.1088/1361-6595/acec0c


        Parameters
        ----------
        T_e: np.ndarray
            The electron temperature in eV

        Returns
        -------
        np.ndarray:
            The ionization rate coefficient in cm^3/s

        """
    return 1.46E-8 * np.power(T_e, 0.690) * np.exp(-9.38 / T_e)

adas_te = np.array([
    1.000E+00, 2.000E+00, 3.000E+00, 4.000E+00, 5.000E+00, 7.000E+00,  1.000E+01, 1.500E+01, 2.000E+01,
    3.000E+01, 4.000E+01, 5.000E+01,  7.000E+01, 1.000E+02, 1.500E+02, 2.000E+02, 3.000E+02, 4.000E+02,
    5.000E+02, 7.000E+02, 1.000E+03, 2.000E+03, 5.000E+03, 1.000E+04
])

adas_k_ion = np.array([
    2.629E-12, 3.189E-10, 1.803E-09, 4.520E-09, 8.059E-09, 1.618E-08, 2.832E-08, 4.526E-08, 5.807E-08, 7.539E-08,
    8.620E-08, 9.339E-08, 1.019E-07, 1.079E-07, 1.110E-07, 1.111E-07, 1.088E-07, 1.058E-07, 1.028E-07, 9.758E-08,
    9.128E-08, 7.803E-08, 6.079E-08, 4.915E-08
])


# Constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
m_B = 10.811 * 1.66053906660e-27  # Mass of boron atom (kg)
m_e = 9.10938356e-31  # Mass of electron (kg)
m_D = 2.014 * 1.66053906660e-27  # Mass of deuterium (kg)

def thermal_velocity(T, m):
    """
    Calculate thermal velocity
    T: temperature (K)
    m: mass of particle (kg)
    """
    global k_B
    # return np.sqrt(8 * k_B * T / (np.pi * m))
    return np.sqrt(2. * k_B * T / m)

def thermal_velocity_ion_energy(ei_ev):
    kT = (2./3.) * ei_ev * 1.60217663 # x1E-19
    arg = 2. * kT / 10.811 / 1.6605390671 # x (1E-19) x (1E27) = 1E8
    result = np.sqrt(arg) * 1E4
    return result



def main():
    global adas_te, adas_k_ion, plasma_params, m_B
    f_interp = interpolate.interp1d(x=adas_te, y=adas_k_ion)
    v_th = thermal_velocity(T=500+273.15, m=m_B)
    v_th_ei = thermal_velocity_ion_energy(ei_ev=6.71E-5)
    v_th_bh = thermal_velocity(T=500+273.15, m=(m_B + m_D))
    print(f'Boron v_th (based on 500 K temperature): {v_th*100.:.3e} cm/s')
    # print(f'Boron v_th (based on ei=6.71E-5 eV): {v_th_ei:.3e} m/s')
    print(f'Boron v_th (based on ei=6.71E-5 eV): {v_th_ei*100:.3e} cm/s')
    print(f'B-D v_th (based on 500 K temperature): {v_th_bh*100:.3e} m/s')
    print(f'Ionization rate of boron at {0.2E11:.3E} /cm^3: {f_interp(5):.2E} 1/s')
    print(f'Ionization rate of  coefficient of B-H (Kawate) at 5 eV: {s_rate(T_e=5.):.2E}')
    plasma_params['S_B_ion (cm^3/s)'] = f_interp(plasma_params['T_e (eV)'])
    plasma_params['B ionisation rate (1/s)'] = f_interp(plasma_params['T_e (eV)']) * plasma_params['n_e 10^18 (/m^3)'] * 1E12
    plasma_params['vth_B (cm/s)'] = thermal_velocity(500., m_B) * 1E2
    plasma_params['B mean free path (cm)'] = plasma_params['vth_B (cm/s)'] / plasma_params['B ionisation rate (1/s)']
    plasma_params['S_k BH (cm^3/s)'] = s_rate(plasma_params['T_e (eV)'])
    plasma_params['B-H ionisation rate (1/s)'] = plasma_params['S_k BH (cm^3/s)'] * plasma_params[
        'n_e 10^18 (/m^3)'] * 1E12
    plasma_params['BH mean free path (cm)'] = 100. * v_th_bh / plasma_params['B ionisation rate (1/s)']
    print(plasma_params)
    plasma_params.to_csv(r'../PISCES-A/mean_free_path_estimate.csv', index=False)

if __name__ == '__main__':
    main()



