import numpy as np
import matplotlib.pyplot as plt
import re
from Ono_sputtering_yield_model import load_plot_style
from scipy import stats

PATH_TO_SPUTTER_FILE = r'./data/SRIM/SPUTTER.txt'
SBE = 5.73 # The surface biding energy in eV

def load_energies_from_file(path_to_sputter_file):
    pattern = re.compile(r"^\w+\s*\d+\s*\d+\s*(\d*\.\d*E?\+?\-?\d+)")
    energies = []
    with open(path_to_sputter_file, 'r') as f:
        for line in f.readlines():
            match = pattern.match(line)
            if match:
                energies.append(float(match.group(1)))

    return np.array(energies)

def main(path_to_sputter_file, sbe):
    energies = load_energies_from_file(path_to_sputter_file)
    energies -= sbe

    grid = np.linspace(min(energies), max(energies), 1000)

    # Calculate basic statistics
    energy_stats = {
        'mean': np.mean(energies),
        'mode': stats.mode(energies, keepdims=False)[0],
        'std_dev': np.std(energies),
        'kurtosis': stats.kurtosis(energies),
        'skewness': stats.skew(energies),
        'moment1': stats.moment(energies, moment=1),
        'moment2': stats.moment(energies, moment=2),
        'moment3': stats.moment(energies, moment=3)
    }

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3.)

    n_bins = 30
    hist, bin_edges = np.histogram(energies, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create KDE object
    bandwidth = 'scott'
    kernel = 'gaussian'
    # Get bandwidth using specified method
    if isinstance(bandwidth, str):
        if bandwidth == 'scott':
            bw = len(energies) ** (-1 / 5) * np.std(energies)
        elif bandwidth == 'silverman':
            bw = (len(energies) * 3 / 4) ** (-1 / 5) * np.std(energies)
        else:
            raise ValueError("bandwidth method must be 'scott' or 'silverman'")
    else:
        bw = float(bandwidth)

    # Define kernel functions
    kernels = {
        'gaussian': lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2),
        'tophat': lambda x: np.where(np.abs(x) <= 1, 0.5, 0),
        'epanechnikov': lambda x: np.where(np.abs(x) <= 1, 3 / 4 * (1 - x ** 2), 0),
        'exponential': lambda x: np.where(np.abs(x) <= 1, 0.5 * np.exp(-np.abs(x)), 0),
        'linear': lambda x: np.where(np.abs(x) <= 1, 1 - np.abs(x), 0),
        'cosine': lambda x: np.where(np.abs(x) <= 1, np.pi / 4 * np.cos(np.pi / 2 * x), 0)
    }

    if kernel not in kernels:
        raise ValueError(f"kernel must be one of {list(kernels.keys())}")

    # Compute KDE manually
    pdf = np.zeros_like(grid)
    kernel_func = kernels[kernel]

    for x in energies:
        pdf += kernel_func((grid - x) / bw)
    pdf = pdf / (len(energies) * bw)  # Normalize

    # Find the most probable energy
    most_probable_energy = grid[np.argmax(pdf)]



    ax.hist(energies, bins=n_bins, density=True, label='SRIM')
    ax.plot(grid, pdf, 'r-', label='KDE')

    ax.set_xlabel('Sputtered atom energy (eV)')
    ax.set_ylabel('Probability density')

    ax.axvline(x=energy_stats['mode'], color='r', ls='--')
    ax.axvline(x=energy_stats['mean'], color='k', ls='--')

    ax.legend(loc='upper right')
    # for e in energies:
    #     print(e)
    print(energy_stats)
    plt.show()


if __name__ == '__main__':
    main(path_to_sputter_file=PATH_TO_SPUTTER_FILE, sbe=SBE)

