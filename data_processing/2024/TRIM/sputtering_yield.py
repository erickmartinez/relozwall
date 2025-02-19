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
    g = estimate_gamma(m1, m2)
    if m1 >= m2:
        return 6.7 * sbe / g
    return (1 + 5.7 * (m1 / m2)) * sbe / g

def estimate_lindhard_screeining_length(z1, z2):
    a_B = 0.529177210544 # Bohr's radius in Å
    return ((9 * np.pi ** 2 / 128) ** (1/3)) * a_B * (z1 ** (2/3) + z2 ** (2/3)) ** (-1/2)

def estimate_lindhard_reduced_energy(E0, m1, m2, z1, z2):
    e = 1.602176634e-19  # Elementary charge in Coulombs
    a_L = estimate_lindhard_screeining_length(z1, z2)
    e_L = E0 * (m2 / (m1 + m2)) * (a_L / (z1*z2))
    return e_L


def estimate_w(e_l):
    """

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
    if w is None:
        w = estimate_w(e_l)
    return 0.5 * np.log(1 + 1.2288 * e_l) / w


def estimate_yield_eckstein_preuss(E0, q, lam, E_th, mu, m1, m2, z1, z2, sbe):
    # E_th_val = estimate_eth(m1, m2, sbe)
    e_L = estimate_lindhard_reduced_energy(E0, m1, m2, z1, z2)
    w = estimate_w(e_L)
    snkr = estimate_KrC_stopping_power(e_L, w)
    msk_above_eth = E0 > E_th
    E0Eth = np.zeros_like(E0)*1E-8
    E0Eth[msk_above_eth] = (E0[msk_above_eth]/E_th - 1.0) ** mu
    y = q * snkr * E0Eth / (lam / w + E0Eth)
    return y
