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


def plot_sputtered_distribution(stats):
    """Plot the sputtered atoms energy distribution."""
    E = np.linspace(0, 15, 1000)
    prob = thompson_distribution(E, stats['mean'], stats['Eb'])

    # Normalize
    prob = prob / np.max(prob)

    # Calculate statistics
    sigma = np.sqrt(stats['variance'])
    median = stats['mean'] - (0.4 * sigma * stats['skewness'])
    mode = E[np.argmax(prob)]

    # Calculate moments of the generated distribution for comparison
    generated_moments = []
    for k in range(1, 7):
        moment = np.trapz(E ** k * prob, E) / np.trapz(prob, E)
        generated_moments.append(moment)

    plt.figure(figsize=(10, 6))
    plt.plot(E, prob, 'b-', label='Thompson Distribution')

    # Plot statistical markers
    plt.axvline(stats['mean'], color='r', linestyle='--', label=f'Mean = {stats["mean"]:.2f} eV')
    plt.axvline(median, color='g', linestyle='--', label=f'Median ≈ {median:.2f} eV')
    plt.axvline(mode, color='m', linestyle='--', label=f'Mode ≈ {mode:.2f} eV')

    plt.xlabel('Energy (eV)')
    plt.ylabel('Normalized Probability')
    plt.title('Sputtered Atoms Energy Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add statistics textbox
    stats_text = f'Statistics:\n' \
                 f'Mean = {stats["mean"]:.2f} eV\n' \
                 f'Std Dev = {np.sqrt(stats["variance"]):.2f} eV\n' \
                 f'Skewness = {stats["skewness"]:.3f}\n' \
                 f'Kurtosis = {stats["kurtosis"]:.3f}\n' \
                 f'Yield = {stats["yield"]:.3e} atoms/ion'

    plt.text(0.98, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Print moment comparison
    print("\nMoment Comparison (TRIM vs Generated):")
    print("Moment  TRIM        Generated")
    print("------  ----------  ----------")
    for i, (trim_moment, gen_moment) in enumerate(zip(stats['moments'], generated_moments), 1):
        print(f"{i:6d}  {trim_moment:10.3e}  {gen_moment:10.3e}")


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