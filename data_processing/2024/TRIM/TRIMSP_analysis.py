import re

PATH_TO_TRIM_FILE = r'./data/simulations/D_ON_B/20250211/D_on_B_E50_20250211.out'

def parse_trim_sputter_output(path_to_file):
    # Dictionary to store the extracted values
    results = {}

    file_content = ""
    with open(path_to_file, 'r') as f:
        file_content = f.read()

    # Extract sputtering yield and energies
    sputter_pattern = r'SPUTTERING YIELD\(1\) =\s+(\d+\.\d+E[-+]?\d+)\s+SPUTTERED ENERGY\(1\) =\s+(\d+\.\d+E[-+]?\d+)\s+REL\.MEAN ENERGY\(1\) =\s+(\d+\.\d+E[-+]?\d+)\s+MEAN ENERGY\(1\) =\s+(\d+\.\d+E[+-]?\d+)'
    sputter_match = re.search(sputter_pattern, file_content)

    if sputter_match:
        results['sputtering_yield'] = float(sputter_match.group(1))
        results['sputtered_energy'] = float(sputter_match.group(2))
        results['rel_mean_energy'] = float(sputter_match.group(3))
        results['mean_energy'] = float(sputter_match.group(4))

    # Extract ENERGY(1) statistics
    energy_stats_pattern = r'ENERGY\(1\)\s+(\d+\.\d+E[-+]?\d+)\s+(\d+\.\d+E[-+]?\d+)\s+(\d+\.\d+E[-+]?\d+)\s+(\d+\.\d+E[-+]?\d+)\s+(\d+\.\d+E[-+]?\d+)\s+(\d+\.\d+E[-+]?\d+)\s+(\d+\.\d+E[-+]?\d+)\s+(\d+\.\d+E[-+]?\d+)'
    energy_stats_match = re.search(energy_stats_pattern, file_content)

    if energy_stats_match:
        results['energy_stats'] = {
            'mean': float(energy_stats_match.group(1)),
            'variance': float(energy_stats_match.group(2)),
            'skewness': float(energy_stats_match.group(3)),
            'kurtosis': float(energy_stats_match.group(4)),
            'sigma': float(energy_stats_match.group(5)),
            'error_1M': float(energy_stats_match.group(6)),
            'error_2M': float(energy_stats_match.group(7)),
            'error_3M': float(energy_stats_match.group(8))
        }

    # Extract ENERGY(1) moments
    moments_pattern = r'ENERGY\(1\)\s+(\d+\.\d+E[-+]?\d+)\s+(\d+\.\d+E[-+]?\d+)\s+(\d+\.\d+E[-+]?\d+)\s+(\d+\.\d+E[-+]?\d+)\s+(\d+\.\d+E[-+]?\d+)\s+(\d+\.\d+E[-+]?\d+)'
    moments_match = re.search(moments_pattern, file_content, re.MULTILINE)

    if moments_match:
        results['energy_moments'] = {
            'moment_1': float(moments_match.group(1)),
            'moment_2': float(moments_match.group(2)),
            'moment_3': float(moments_match.group(3)),
            'moment_4': float(moments_match.group(4)),
            'moment_5': float(moments_match.group(5)),
            'moment_6': float(moments_match.group(6))
        }

    return results


# Example usage:
def main(path_to_file):
    # Read the file content

    # Parse the content
    results = parse_trim_sputter_output(path_to_file)

    # Print the results in a formatted way
    print("Sputtering Analysis Results:")
    print("-" * 30)
    print(f"Sputtering Yield: {results['sputtering_yield']:.6e}")
    print(f"Sputtered Energy: {results['sputtered_energy']:.6e}")
    print(f"Relative Mean Energy: {results['rel_mean_energy']:.6e}")
    print(f"Mean Energy: {results['mean_energy']:.6e}")

    print("\nEnergy Statistics:")
    print("-" * 30)
    for key, value in results['energy_stats'].items():
        print(f"{key.replace('_', ' ').title()}: {value:.6e}")

    print("\nEnergy Moments:")
    print("-" * 30)
    for key, value in results['energy_moments'].items():
        print(f"{key.replace('_', ' ').title()}: {value:.6e}")


if __name__ == "__main__":
    main(PATH_TO_TRIM_FILE)