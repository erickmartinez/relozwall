import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.ticker as ticker
import json

# Atomic numbers
Z1 = 1  # Deuterium
# Z2 = 26  # Iron
Z2 = 5 # Boron

# Atomic masses (amu)
M1 = 2.014  # Deuterium
# M2 = 55.845  # Iron
M2 = 10.811

# Set other parameters
U = 5.73  # surface binding energy for Fe (eV)
N = 50 # normalization factor to be adjusted to match experimental data
Einc = 50  # incident energy in eV

# Get the prediction at the ion energies of interest for the beam composition
ION_ENERGIES = np.array([11.67, 14., 16.67, 17.5, 21., 25.0, 35, 42., 50])

def calculate_physical_parameters(z1, z2, m1, m2):
    """
    Calculate physical parameters for ion -> target sputtering

    Parameters
    ----------
    z1: float
        The atomic number of the incident ions
    z2: float
        The atomic number of the target atoms
    m1: float
        The mass of the ions
    m2: float
        The mass of the target atoms

    Returns
    -------

    """


    # Calculate gamma (energy transfer factor)
    gamma = 4 * m1 * m2 / (m1 + m2) ** 2

    # Calculate K_L (equation 9)
    # K_L = 1.216e-2 * Z1**(7/6) * Z2 / M1**(1/2) * (Z1**(2/3) + Z2**(2/3))**(3/2)
    K_L = 1.216e-2 * (z1 ** (7 / 6)) * z2 * (m1 ** (-1 / 2)) * ((z1 ** (2 / 3) + z2 ** (2 / 3)) ** (3 / 2))

    # Calculate C1/2 (after equation 11-II)
    lambda_m = 0.276  # for region II
    a = 0.04685 * (z1 ** (2 / 3) + z2 ** (2 / 3)) ** (1 / 2)  # nm
    C_half = np.pi * lambda_m * a ** 2 * (m1 / m2) ** (1 / 2) * (2 * z1 * z2 * e ** 2 / a) ** 2 / 2

    # Calculate B (after equation 14-II)
    B = 2 * C_half * (gamma * (1 - gamma)) ** (1 / 2) / K_L

    return gamma, K_L, B


def normalized_yield_region_I(E1, Einc, gamma, U, N):
    """
    Calculate normalized yield for region I using equation 15-I
    E1: sputtered atom energy (eV)
    Einc: incident ion energy (eV)
    gamma: energy transfer factor
    U: surface binding energy (eV)
    N: normalization factor
    """
    term1 = N * E1 * (E1 + U) ** (-9 / 4)
    term2 = (gamma * (1 - gamma) * Einc) ** (1 / 4) - (E1 + U) ** (1 / 4)
    return term1 * term2


def normalized_yield_region_II(E1, Einc, gamma, U, N, B):
    """
    Calculate normalized yield for region II using equation 15-II
    """
    term1 = N * E1 * (E1 + U) ** (-5 / 2)
    term2 = np.log((B + Einc ** (1 / 2)) / (B + (E1 + U) ** (1 / 2) / (gamma * (1 - gamma)) ** (1 / 2)))
    return term1 * term2


def thompson_distribution(E, U):
    """
    Calculate the Thompson energy distribution for sputtered atoms.

    Parameters:
    -----------
    E : float or numpy.ndarray
        Kinetic energy of sputtered atoms (eV)
    U : float
        Surface binding energy (eV)

    Returns:
    --------
    float or numpy.ndarray
        Probability density of atoms with energy E
    """
    return (E / (E + U) ** 3) #* (1 - np.sqrt(U / E))

def ono_distribution_stats(z1, z2, m1, m2, Einc, Eb, N=1, E_range=None):
    """
    Calculate statistical moments of the sputtering distribution

    Parameters:
    -----------
    z1: float
        The atomic number of the incident ions
    z2: float
        The atomic number of the target atoms
    m1: float
        The mass of the ions
    m2: float
        The mass of the target atoms
    Einc : float
        Incident ion energy in eV
    Eb: float
        The surface binding energy in eV
    N: float
        Normalization factor
    E_range : tuple, optional
        (min_energy, max_energy) range for integration in eV

    Returns:
    --------
    dict containing mean, mode, variance, std, skewness, and kurtosis
    """
    # Calculate physical parameters
    gamma, K_L, B = calculate_physical_parameters(z1, z2, m1, m2)
    U = Eb  # surface binding energy (eV)

    if E_range is None:
        E_range = (0.1, min(10 * U, 2. * Einc))  # reasonable energy range

    # Create energy array for calculations
    E1 = np.linspace(E_range[0], E_range[1], 1000)
    dE = E1[1] - E1[0]

    # Calculate distribution
    y = np.array([normalized_yield_region_II(e, Einc, gamma, U, N, B) for e in E1])
    msk_positive = y >= 0
    y = y[msk_positive]
    E1 = E1[msk_positive]
    if len(y) == 0:
        return {
            'mean': 0,
            'mode': 0,
            'variance': 0,
            'std': 0,
            'skewness': 0,
            'kurtosis': 0
        }

    # Normalize to get proper probability distribution
    norm = integrate.simpson(y, x=E1)
    if norm > 0:
        y = y / norm
    else:
        raise ValueError("Distribution normalization failed")

    # Calculate mode (energy at maximum yield)
    mode_idx = np.argmax(y)
    mode = E1[mode_idx]

    # Calculate mean (first moment)
    mean = integrate.simpson(E1 * y, x=E1)

    # Calculate central moments
    var = integrate.simpson((E1 - mean) ** 2 * y, x=E1)  # second central moment
    std = np.sqrt(var)

    # Calculate skewness (third standardized moment)
    skew = integrate.simpson((E1 - mean) ** 3 * y, x=E1) / std ** 3

    # Calculate kurtosis (fourth standardized moment)
    kurt = integrate.simpson((E1 - mean) ** 4 * y, x=E1) / var ** 2

    return {
        'mean': mean,
        'mode': mode,
        'variance': var,
        'std': std,
        'skewness': skew,
        'kurtosis': kurt
    }


def thompson_distribution_stats(U, E_range=(0, 100), num_points=1000):
    """
    Calculate statistical properties of the Thompson distribution.

    Parameters:
    -----------
    U : float
        Surface binding energy (eV)
    E_range : tuple
        Range of energies to plot (eV)
    num_points : int
        Number of points to calculate

    Returns:
    --------
    dict
        Dictionary containing mean, mode, standard deviation, skewness, and kurtosis
    """
    # Generate energy values
    E1 = np.linspace(E_range[0], E_range[1], num_points)
    y = thompson_distribution(E1, U)

    # Normalize to get proper probability distribution
    norm = integrate.simpson(y, x=E1)
    if norm > 0:
        y = y / norm
    else:
        raise ValueError("Distribution normalization failed")

    # Calculate mode (energy at maximum yield)
    mode_idx = np.argmax(y)
    mode = E1[mode_idx]

    # Calculate mean (first moment)
    mean = integrate.simpson(E1 * y, x=E1)

    # Calculate central moments
    var = integrate.simpson((E1 - mean) ** 2 * y, x=E1)  # second central moment
    std = np.sqrt(var)

    # Calculate skewness (third standardized moment)
    skew = integrate.simpson((E1 - mean) ** 3 * y, x=E1) / std ** 3

    # Calculate kurtosis (fourth standardized moment)
    kurt = integrate.simpson((E1 - mean) ** 4 * y, x=E1) / var ** 2

    return {
        'mean': mean,
        'mode': mode,
        'std': std,
        'skewness': skew,
        'kurtosis': kurt
    }

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')



if __name__ == '__main__':
    # Physical constants
    e = 1.60218e-19  # elementary charge in Coulombs

    # Calculate parameters
    gamma, K_L, B = calculate_physical_parameters(z1=Z1, z2=Z2, m1=M1, m2=M2)

    # Create energy range for sputtered atoms
    E1_range = np.linspace(0.01, 20, 1000)  # eV

    # Calculate yields
    yields_II = np.array([normalized_yield_region_II(E, Einc, gamma, U, N, B) for E in E1_range])
    yields_thompson = thompson_distribution(U=U, E=E1_range)

    mean_sputtered_energy = np.zeros(len(ION_ENERGIES))
    for i, ion_energy in enumerate(ION_ENERGIES):
        print(ion_energy)
        ion_energy_stats = ono_distribution_stats(z1=Z1, z2=Z2, m1=M1, m2=M2, Einc=ion_energy, Eb=U, N=N)
        mean_sputtered_energy[i] = ion_energy_stats['mean']

    mean_ion_energies_df = pd.DataFrame(data={
        'Ion energy (eV)': ION_ENERGIES,
        'Mean sputtered energy (eV)': mean_sputtered_energy
    })

    mean_ion_energies_df.to_csv(r'./data/pisces_mean_sputtered_energy.csv', index=False)

    stats_ono = ono_distribution_stats(z1=Z1, z2=Z2, m1=M1, m2=M2, Einc=Einc, Eb=U, N=N)
    stats_thompson = thompson_distribution_stats(U=U, E_range=(0.01, 50), num_points=10000)
    print(f"\nStatistics for {Einc} eV incident energy:")
    print("+------------------+-------+------------+")
    print("| Stat             |  Ono  |  Thompson  |")
    print("+---------"
          "---------+-------+------------+")
    print(f"| Mean energy (eV) | {stats_ono['mean']:>5.2f} | {stats_thompson['mean']:>10.2f} |")
    print(f"| Mode energy (eV) | {stats_ono['mode']:>5.2f} | {stats_thompson['mode']:>10.2f} |")
    print(f"| Stdev (eV):      | {stats_ono['std']:>5.2f} | {stats_thompson['std']:>10.2f} |")
    print(f"| Skewness:        | {stats_ono['skewness']:>5.2f} | {stats_thompson['skewness']:>10.2f} |")
    print(f"| Kurtosis:        | {stats_ono['kurtosis']:>5.2f} | {stats_thompson['kurtosis']:>10.2f} |")
    print("+------------------+-------+------------+")

    e_mean_ono = stats_ono['mean']
    e_mean_thompson = stats_thompson['mean']
    e_mean_trim = 3.573E+00

    yields_II *= e_mean_trim / e_mean_ono

    y_mean_ono = normalized_yield_region_II(e_mean_ono, Einc, gamma, U, N, B) * e_mean_trim / e_mean_ono
    y_mode_ono = normalized_yield_region_II(stats_ono['mode'], Einc, gamma, U, N, B) * e_mean_trim / e_mean_ono

    y_mean_thompson = thompson_distribution(U=U, E=e_mean_thompson) * np.max(yields_II) / np.max(yields_thompson)
    y_mode_thompson = thompson_distribution(U=U, E=stats_thompson['mode']) * np.max(yields_II) / np.max(yields_thompson)


    yields_thompson *= np.max(yields_II) / np.max(yields_thompson)

    # Create plot
    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.5)
    ax.plot(E1_range, yields_II, 'b-', label='Ono 2004')
    ax.plot(E1_range, yields_thompson, 'r--', label='Thompson')



    ax.plot([e_mean_ono], [y_mean_ono], marker='o', mfc='k', mec='b')
    ax.plot([stats_ono['mode']], [y_mode_ono], marker='s', mfc='k', mec='b')

    ax.plot([e_mean_thompson], [y_mean_thompson], marker='o', mfc='k', mec='r')
    ax.plot([stats_thompson['mode']], [y_mode_thompson], marker='s', mfc='k', mec='r')

    ax.set_xlabel('Sputtered atom energy (eV)')
    ax.set_ylabel('Normalized Yield')
    ax.set_title(rf'{{\sffamily D\textsuperscript{{+}} → B Sputtering (E\textsubscript{{inc}} = {Einc} eV)}}', usetex=True)
    ax.legend()
    # ax.grid(True)
    ax.set_ylim(bottom=0.)  # Adjust based on Fig. 2 for 100 eV
    ax.set_xlim(0, 15)

    # Print physical parameters for verification
    print(f"Calculated parameters:")
    print(f"gamma = {gamma:.3f}")
    print(f"K_L = {K_L:.3e}")
    print(f"B = {B:.3e}")

    fig.savefig(r"./figures/sputtered_energy_distribution.svg", dpi=600)

    # Show plot
    # plt.tight_layout()
    plt.show()

