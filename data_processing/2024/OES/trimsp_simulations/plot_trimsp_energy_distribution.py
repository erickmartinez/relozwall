import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfile


def read_energy_distribution(filename):
    with open(filename, 'r') as f:
        content = f.readlines()

    # Find the energy distribution section
    start_idx = None
    end_idx = None

    for i, line in enumerate(content):
        if "LOG ENERGY - COS OF EMISSION ANGLE" in line:
            # Skip the header line and search for first data line
            for j in range(i + 1, len(content)):
                if any(c.isdigit() for c in content[j]) and 'E+' in content[j]:
                    start_idx = j
                    break
        elif "ENERGY(E/E0 IN %)" in line:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        raise ValueError("Could not find energy distribution section")

    # Process the data
    energies = []
    counts = []
    yields = []

    for line in content[start_idx:end_idx]:
        parts = line.split()
        if len(parts) >= 22:  # Each line should have energy, angle bins, and total
            try:
                energy = float(parts[0])
                count = int(parts[-2])  # Total count is second to last column
                dyield = float(parts[-1])  # Differential yield is last column
                energies.append(energy)
                counts.append(count)
                yields.append(dyield)
            except (ValueError, IndexError) as e:
                continue

    return np.array(energies), np.array(counts), np.array(yields)


def thompson_distribution(E, U):
    """Calculate Thompson energy distribution
    E: energy points
    U: surface binding energy in eV
    """
    return E / (E + U) ** 3


def plot_energy_distribution(filename):
    # Read and process data
    energies, counts, yields = read_energy_distribution(filename)

    if len(counts) == 0:
        raise ValueError("No data points found")

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Particle counts
    counts_normalized = counts / np.max(counts)
    U = 5.73  # Surface binding energy for Boron in eV
    thompson = thompson_distribution(energies, U)
    scale_factor = 0.8 * np.max(counts_normalized)
    thompson_normalized = thompson * (scale_factor / np.max(thompson))

    ax1.semilogx(energies, counts_normalized, 'bo-', label='TRIM.SP Data', markersize=4)
    ax1.semilogx(energies, thompson_normalized, 'r--', label='Thompson Distribution')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Normalized Count')
    ax1.set_title('Particle Count Distribution')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()
    ax1.set_xlim(0.1, 40)
    ax1.set_ylim(0, 1.1)

    # Plot 2: Differential yield
    yields_normalized = yields / np.max(yields)
    thompson = thompson_distribution(energies, U)
    scale_factor = np.max(yields_normalized)
    thompson_normalized = thompson * (scale_factor / np.max(thompson))

    ax2.semilogx(energies, yields_normalized, 'bo-', label='TRIM.SP dY/dE', markersize=4)
    ax2.semilogx(energies, thompson_normalized, 'r--', label='Thompson Distribution')
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Normalized dY/dE')
    ax2.set_title('Differential Yield Distribution')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    ax2.set_xlim(0.1, 40)
    ax2.set_ylim(0, 1.1)

    plt.suptitle('Energy Distribution of Sputtered B Atoms\nD+ on B, E = 40 eV')
    plt.tight_layout()

    # Print statistics
    mean_energy = np.average(energies, weights=counts)
    peak_energy = energies[np.argmax(counts)]
    print(f"\nStatistics:")
    print(f"Mean Energy: {mean_energy:.2f} eV")
    print(f"Peak Energy: {peak_energy:.2f} eV")
    print(f"Total Sputtered Particles: {np.sum(counts)}")
    print(f"Energy range: {min(energies):.2f} - {max(energies):.2f} eV")

    plt.show()


# Usage
# plot_energy_distribution('D_on_B_E500_IAEA.out')
if __name__ == '__main__':
    file = askopenfile(title="Select TRIM .out file", filetypes=[("Text files", ".out")])
    print(file.name)
    plot_energy_distribution(file.name)