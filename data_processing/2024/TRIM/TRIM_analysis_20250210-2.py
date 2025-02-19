import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re

PATH_TO_TRIM_OUTPUT = r'./data/simulations/D_ON_B/20250210/D_on_B_E50_20250210.out'


def read_trim_stats(filename):
    """Read sputtering statistics from TRIM.SP output file."""
    # Initialize default values
    stats = {
        'yield': None,
        'mean': None,
        'variance': None,
        'skewness': None,
        'kurtosis': None,
        'moments': None,
        'Eb': None
    }

    with open(filename, 'r') as f:
        content = f.read()

    # Find the sputtering statistics section
    pattern = r"SPUTTERING YIELD\(1\) =\s*([0-9.E+-]+)\s*SPUTTERED ENERGY\(1\) =\s*([0-9.E+-]+)\s*REL\.MEAN ENERGY\(1\) =\s*([0-9.E+-]+)\s*MEAN ENERGY\(1\) =\s*([0-9.E+-]+)"
    matches = re.search(pattern, content)

    if matches:
        stats['yield'] = float(matches.group(1))
        stats['mean'] = float(matches.group(4))
    else:
        print("Warning: Could not find sputtering yield and mean energy")

    # Find variance, skewness, kurtosis
    pattern = r"ENERGY\(1\)\s*([0-9.E+-]+)\s*([0-9.E+-]+)\s*([0-9.E+-]+)\s*([0-9.E+-]+)"
    matches = re.search(pattern, content)

    if matches:
        stats['mean'] = float(matches.group(1))
        stats['variance'] = float(matches.group(2))
        stats['skewness'] = float(matches.group(3))
        stats['kurtosis'] = float(matches.group(4))
    else:
        print("Warning: Could not find variance, skewness, and kurtosis")

    # Find moments
    pattern = r"ENERGY\(1\)\s*([0-9.E+-]+)\s*([0-9.E+-]+)\s*([0-9.E+-]+)\s*([0-9.E+-]+)\s*([0-9.E+-]+)\s*([0-9.E+-]+)"
    matches = re.search(pattern, content)

    if matches:
        stats['moments'] = [float(matches.group(i)) for i in range(1, 7)]
    else:
        print("Warning: Could not find moments")

    # Find surface binding energy
    # First try to find it in the standard location
    pattern = r"SBE\(LAYER,ELEMENT\)\s*\*\*\*\s*([0-9.]+)"
    matches = re.search(pattern, content)

    if matches:
        stats['Eb'] = float(matches.group(1))
    else:
        # Try alternative location in input data section
        pattern = r"ED\(LAYER,ELEMENT\)\s*\*\*\*\s*([0-9.]+)"
        matches = re.search(pattern, content)
        if matches:
            stats['Eb'] = float(matches.group(1))
        else:
            print("Warning: Could not find surface binding energy, using default value of 5.73 eV")
            stats['Eb'] = 5.73  # Default value for many materials

    # Verify essential values are present
    if any(v is None for v in [stats['mean'], stats['variance'], stats['Eb']]):
        raise ValueError("Could not find all required values in the TRIM.SP output file")

    return stats


def thompson_distribution(E, Em, Eb):
    """Calculate Thompson energy distribution."""
    return E / np.power(E + Eb, 3) * np.exp(-E / (2 * Em))


def thompson_distribution(E, Em, Eb):
    """Calculate Thompson energy distribution.

    Args:
        E: Energy values (array)
        Em: Mean energy
        Eb: Surface binding energy

    Returns:
        Normalized probability distribution
    """
    # Thompson's formula: N(E) = E/(E + Eb)^3
    N = E / np.power(E + Eb, 3)

    # Add cutoff at higher energies based on mean energy
    cutoff = np.exp(-E / (2 * Em))
    dist = N * cutoff

    # Proper normalization
    norm = np.trapz(dist, E)
    return dist / norm


def analyze_thompson_distribution(E, prob):
    """Analyze the statistical properties of the Thompson distribution."""
    # prob should already be normalized from thompson_distribution()

    # Calculate statistical moments
    # Mean (1st moment)
    mean = np.trapz(E * prob, E)

    # Variance (2nd central moment)
    variance = np.trapz((E - mean) ** 2 * prob, E)

    # Skewness (3rd standardized moment)
    skewness = np.trapz(((E - mean) / np.sqrt(variance)) ** 3 * prob, E)

    # Kurtosis (4th standardized moment)
    kurtosis = np.trapz(((E - mean) / np.sqrt(variance)) ** 4 * prob, E)

    # Calculate raw moments up to 6th order
    moments = []
    for k in range(1, 7):
        moment = np.trapz(E ** k * prob, E)
        moments.append(moment)

    return {
        'mean': mean,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'moments': moments
    }


def plot_sputtered_distribution(stats):
    """Plot the sputtered atoms energy distribution and analyze it."""
    # Use finer grid and wider range for better numerical accuracy
    E = np.linspace(0, 30, 2000)  # Increased range and resolution
    prob = thompson_distribution(E, stats['mean'], stats['Eb'])

    # Analyze the simulated distribution
    sim_stats = analyze_thompson_distribution(E, prob)

    # Calculate plotting range based on the statistics
    plot_max = max(5 * stats['mean'], 15)  # Show at least up to 5 times mean energy
    plot_mask = E <= plot_max

    plt.figure(figsize=(10, 6))
    plt.plot(E[plot_mask], prob[plot_mask], 'b-', label='Thompson Distribution')

    # Plot statistical markers
    plt.axvline(stats['mean'], color='r', linestyle='--', label=f'TRIM Mean = {stats["mean"]:.2f} eV')
    plt.axvline(sim_stats['mean'], color='g', linestyle='--', label=f'Sim Mean = {sim_stats["mean"]:.2f} eV')

    plt.xlabel('Energy (eV)')
    plt.ylabel('Normalized Probability')
    plt.title('Sputtered Atoms Energy Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add comparison textbox
    comparison_text = (
        'TRIM vs Simulation:\n'
        f'Mean: {stats["mean"]:.2f} vs {sim_stats["mean"]:.2f} eV\n'
        f'Std Dev: {np.sqrt(stats["variance"]):.2f} vs {np.sqrt(sim_stats["variance"]):.2f} eV\n'
        f'Skewness: {stats["skewness"]:.3f} vs {sim_stats["skewness"]:.3f}\n'
        f'Kurtosis: {stats["kurtosis"]:.3f} vs {sim_stats["kurtosis"]:.3f}'
    )

    plt.text(0.98, 0.98, comparison_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Print detailed moment comparison
    print("\nDetailed Statistical Comparison:")
    print("Parameter      TRIM          Simulation    Rel. Diff (%)")
    print("-" * 55)
    print(
        f"Mean          {stats['mean']:10.3e}  {sim_stats['mean']:10.3e}  {100 * (sim_stats['mean'] - stats['mean']) / stats['mean']:10.2f}")
    print(
        f"Variance      {stats['variance']:10.3e}  {sim_stats['variance']:10.3e}  {100 * (sim_stats['variance'] - stats['variance']) / stats['variance']:10.2f}")
    print(
        f"Skewness      {stats['skewness']:10.3e}  {sim_stats['skewness']:10.3e}  {100 * (sim_stats['skewness'] - stats['skewness']) / stats['skewness']:10.2f}")
    print(
        f"Kurtosis      {stats['kurtosis']:10.3e}  {sim_stats['kurtosis']:10.3e}  {100 * (sim_stats['kurtosis'] - stats['kurtosis']) / stats['kurtosis']:10.2f}")

    print("\nMoment Comparison:")
    print("Order         TRIM          Simulation    Rel. Diff (%)")
    print("-" * 55)
    for i, (trim_moment, sim_moment) in enumerate(zip(stats['moments'], sim_stats['moments']), 1):
        rel_diff = 100 * (sim_moment - trim_moment) / trim_moment
        print(f"{i:5d}         {trim_moment:10.3e}  {sim_moment:10.3e}  {rel_diff:10.2f}")

# Example usage
filename = PATH_TO_TRIM_OUTPUT
stats = read_trim_stats(filename)
plot_sputtered_distribution(stats)

print("\nDetailed Statistics:")
print(f"Mean Energy: {stats['mean']:.2f} eV")
print(f"Standard Deviation: {np.sqrt(stats['variance']):.2f} eV")
print(f"Skewness: {stats['skewness']:.3f}")
print(f"Surface Binding Energy: {stats['Eb']:.2f} eV")
print(f"Sputtering Yield: {stats['yield']:.3e} atoms/ion")