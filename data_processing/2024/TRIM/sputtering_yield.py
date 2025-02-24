"""
This code estimates the sputtering yield as a function of the ion energy using the
Eckstein-Preuss model

Eckstein W, Preuss R. J Nucl Mater 2003;320:209.
Eckstein W, Vacuum 2008;82;930

Erick R Martinez Loran
erickrmartinez@gmail.com
"""
import numpy as np


def estimate_gamma(m1, m2):
    """
    Estimates the coefficient determining the maximum energy transfer between the ion and the target atom

    Parameters
    ----------
    m1: float
        The mass of the ion (amu)
    m2: float
        The mass of the target atom (amu)

    Returns
    -------
    float:
        The gamma parameter
    """
    return 4. * m1 * m2 / (m1 + m2) ** 2.

def estimate_eth(m1, m2, sbe):
    """
    Estimates the threshold energy before which, no sputtering is expected.

    Parameters
    ----------
    m1: float
        The mass of the ion (amu).
    m2: float
        The mass of the target atom  (amu).
    sbe: float
        The surface binding energy of the target (eV)

    Returns
    -------
    float
    """
    g = estimate_gamma(m1, m2)
    if m1 >= m2:
        return 6.7 * sbe / g
    return (1 + 5.7 * (m1 / m2)) * sbe / g

def estimate_lindhard_screeining_length(z1, z2):
    """
    Estimates the Lindhard screening length
    Parameters
    ----------
    z1: float
        The atomic number of the ion
    z2: float
        The atomic number of the target atom

    Returns
    -------
    float

    """
    a_B = 0.529177210544 # Bohr's radius in Å
    return ((9 * np.pi ** 2 / 128) ** (1/3)) * a_B * (z1 ** (2/3) + z2 ** (2/3)) ** (-1/2)

def estimate_lindhard_reduced_energy(E0, m1, m2, z1, z2):
    a_L = estimate_lindhard_screeining_length(z1, z2)
    e_L = E0 * (m2 / (m1 + m2)) * (a_L / (z1*z2))
    return e_L


def estimate_w(e_l):
    """
    Estimates the w in Eckstein-Preuss model (see Eckstein W, Vacuum 2008;82;930 equation 2)
    Parameters
    ----------
    e_l: float
        The Lindhard reduced energy

    Returns
    -------
    float
    """
    return e_l + 0.1728 * np.sqrt(e_l) + 0.008*(e_l**0.1505)

def estimate_KrC_stopping_power(e_l, w=None):
    """
    Estimates the Krypton-Carbon stopping power
    Parameters
    ----------
    e_l: float
        The Lindhard reduced energy
    w: float
        The w parameter from Eckstein-Preuss model. If not provided it will estimate it.

    Returns
    -------
    float
    """
    if w is None:
        w = estimate_w(e_l)
    return 0.5 * np.log(1 + 1.2288 * e_l) / w


def estimate_yield_eckstein_preuss(E0, q, lam, E_th, mu, m1, m2, z1, z2, sbe):
    """
    Estimates the sputtering yield according to the Eckstein-Preuss model

    Parameters
    ----------
    E0: float, np.ndarray
        The energy of the ions
    q: float
        Fitting parameter
    lam: float
        Fitting parameters
    E_th: float
        The threshold energy for sputtering (usually a fitting parameter)
    mu: float
        A fitting parameter
    m1: float
        The mass of the ion (amu)
    m2: float
        The mass of the target atom (amu)
    z1: float
        The atomic number of the ion
    z2: float
        The atomic number of the target atom
    sbe: float
        The surface binding energy of the target atom (eV)

    Returns
    -------
    float:
        The sputtering yield
    """
    # E_th_val = estimate_eth(m1, m2, sbe)
    e_L = estimate_lindhard_reduced_energy(E0, m1, m2, z1, z2)
    w = estimate_w(e_L)
    snkr = estimate_KrC_stopping_power(e_L, w)
    msk_above_eth = E0 > E_th
    E0Eth = np.zeros_like(E0)*1E-8
    E0Eth[msk_above_eth] = (E0[msk_above_eth]/E_th - 1.0) ** mu
    y = q * snkr * E0Eth / (lam / w + E0Eth)
    return y
