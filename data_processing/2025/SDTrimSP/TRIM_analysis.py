import re
import sys
from typing import Dict, Optional, Tuple


def parse_sdtrimsp_output(filename: str) -> Dict[str, float]:
    """
    Parse SDTrimSP output file to extract sputtering yield data for boron (particle 2).

    Args:
        filename (str): Path to the SDTrimSP output file

    Returns:
        Dict containing sputtering yield data and energies
    """

    results = {
        'total_projectiles': 0,
        'sputtered_by_particle_1': 0,
        'sputtered_by_particle_2': 0,
        'total_sputtered': 0,
        'sputtering_yield_by_1': 0.0,
        'sputtering_yield_by_2': 0.0,
        'total_sputtering_yield': 0.0,
        'energy_per_sputtered_by_1': 0.0,
        'energy_per_sputtered_by_2': 0.0,
        'total_energy_per_sputtered': 0.0,
        'total_sputtered_energy': 0.0
    }

    try:
        with open(filename, 'r') as file:
            content = file.read()

        # Extract total number of projectiles (particle 1)
        projectiles_match = re.search(r'PROJECTILES\(\s*1\)\s*=\s*(\d+)', content)
        if projectiles_match:
            results['total_projectiles'] = int(projectiles_match.group(1))

        # Extract sputtered particles by particle 1 (D -> B)
        sput_by_1_match = re.search(r'BACK\.SPUTTERED PARTIC\.\(\s*2 BY\s*1\)\s*=\s*(\d+)\s+ENERGY=\s*([\d.E+-]+)\s*EV',
                                    content)
        if sput_by_1_match:
            results['sputtered_by_particle_1'] = int(sput_by_1_match.group(1))
            energy_1 = float(sput_by_1_match.group(2))
            if results['sputtered_by_particle_1'] > 0:
                results['energy_per_sputtered_by_1'] = energy_1 / results['sputtered_by_particle_1']

        # Extract sputtered particles by particle 2 (B -> B, self-sputtering)
        sput_by_2_match = re.search(r'BACK\.SPUTTERED PARTIC\.\(\s*2 BY\s*2\)\s*=\s*(\d+)\s+ENERGY=\s*([\d.E+-]+)\s*EV',
                                    content)
        if sput_by_2_match:
            results['sputtered_by_particle_2'] = int(sput_by_2_match.group(1))
            energy_2 = float(sput_by_2_match.group(2))
            if results['sputtered_by_particle_2'] > 0:
                results['energy_per_sputtered_by_2'] = energy_2 / results['sputtered_by_particle_2']

        # Extract total backward sputtered recoils (verification)
        total_sput_match = re.search(r'BACKW\. SPUTTER\.\s*RECOILS\(\s*2\s*\)\s*=\s*(\d+)\s+ENERGY=\s*([\d.E+-]+)\s*EV',
                                     content)
        if total_sput_match:
            results['total_sputtered'] = int(total_sput_match.group(1))
            results['total_sputtered_energy'] = float(total_sput_match.group(2))

        # Calculate sputtering yields
        if results['total_projectiles'] > 0:
            results['sputtering_yield_by_1'] = results['sputtered_by_particle_1'] / results['total_projectiles']
            results['sputtering_yield_by_2'] = results['sputtered_by_particle_2'] / results['total_projectiles']
            results['total_sputtering_yield'] = (results['sputtered_by_particle_1'] + results[
                'sputtered_by_particle_2']) / results['total_projectiles']

        # Calculate total energy per sputtered particle
        total_sputtered_particles = results['sputtered_by_particle_1'] + results['sputtered_by_particle_2']
        if total_sputtered_particles > 0:
            results['total_energy_per_sputtered'] = results['total_sputtered_energy'] / total_sputtered_particles

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return results
    except Exception as e:
        print(f"Error reading file: {e}")
        return results

    return results


def print_results(results: Dict[str, float]) -> None:
    """Print the sputtering analysis results in a formatted way."""

    print("=" * 60)
    print("SDTrimSP Sputtering Yield Analysis")
    print("=" * 60)

    print(f"\nInput Parameters:")
    print(f"  Total projectiles (D): {results['total_projectiles']:,}")

    print(f"\nSputtering Results (Boron particles):")
    print(f"  Sputtered by D (particle 1): {results['sputtered_by_particle_1']:,}")
    print(f"  Sputtered by B (particle 2): {results['sputtered_by_particle_2']:,}")
    print(f"  Total sputtered: {results['total_sputtered']:,}")

    print(f"\nSputtering Yields:")
    print(f"  Yield by D: {results['sputtering_yield_by_1']:.4f}")
    print(f"  Yield by B: {results['sputtering_yield_by_2']:.4f}")
    print(f"  Total yield: {results['total_sputtering_yield']:.4f}")

    print(f"\nEnergy Analysis:")
    print(f"  Energy per sputtered (by D): {results['energy_per_sputtered_by_1']:.2f} eV")
    print(f"  Energy per sputtered (by B): {results['energy_per_sputtered_by_2']:.2f} eV")
    print(f"  Average energy per sputtered: {results['total_energy_per_sputtered']:.2f} eV")
    print(f"  Total sputtered energy: {results['total_sputtered_energy']:.2e} eV")

    # Verification with detailed analysis
    calculated_total = results['sputtered_by_particle_1'] + results['sputtered_by_particle_2']
    if calculated_total == results['total_sputtered']:
        print(f"\n✓ Verification: Sum matches total sputtered particles")
    else:
        difference = results['total_sputtered'] - calculated_total
        print(f"\n⚠ Warning: Sum ({calculated_total}) ≠ total ({results['total_sputtered']})")
        print(f"   Difference: {difference} particles ({difference / results['total_sputtered'] * 100:.2f}% of total)")
        print(f"   Possible reasons:")
        print(f"   • Rounding differences in output")
        print(f"   • Additional sputtering mechanisms not captured")
        print(f"   • Different energy threshold criteria")
        print(f"   • Statistical fluctuations in Monte Carlo simulation")


def main():
    """Main function to run the analysis."""

    # Default filename - can be changed or passed as argument
    filename = "./grazing angle/erosion_simulation_results/1000/output.dat"

    # Allow filename as command line argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    print(f"Analyzing SDTrimSP output file: {filename}")

    # Parse the file
    results = parse_sdtrimsp_output(filename)

    # Print results
    print_results(results)

    # Return key values for potential further use
    return results


if __name__ == "__main__":
    results = main()