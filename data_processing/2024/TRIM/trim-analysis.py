import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
import re

# Example usage
PATH_TO_TRIM_OUTPUT = r'./data/simulations/D_ON_B/20250207/D_on_B_E42_SRIM.out' # Replace with your TRIM.SP output file

def read_trim_output(filename):
    """Read relevant sputtering data from TRIM.SP output file."""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract sputtering yield
    yield_match = re.search(r'SPUTTERING YIELD\(1\)\s*=\s*(\d+\.\d+E[+-]\d+)', content)
    sputter_yield = float(yield_match.group(1)) if yield_match else None
    
    # Extract energy statistics
    energy_stats = {}
    stats_pattern = r'ENERGY\(1\)\s+(\d+\.\d+E[+-]\d+)\s+(\d+\.\d+E[+-]\d+)\s+([+-]?\d+\.\d+E[+-]\d+)\s+(\d+\.\d+E[+-]\d+)'
    stats_match = re.search(stats_pattern, content)
    
    if stats_match:
        energy_stats['mean'] = float(stats_match.group(1))
        energy_stats['variance'] = float(stats_match.group(2))
        energy_stats['skewness'] = float(stats_match.group(3))
        energy_stats['kurtosis'] = float(stats_match.group(4))
    
    return sputter_yield, energy_stats

def plot_energy_distribution(stats, save_path=None):
    """
    Plot the energy distribution using the statistical parameters.
    Uses skewnorm from scipy to approximate the distribution.
    """
    mean = stats['mean']
    std = np.sqrt(stats['variance'])
    skewness = stats['skewness']
    
    # Create energy range for plotting
    e_range = np.linspace(0, mean + 4*std, 200)
    
    # Calculate shape parameter for skewnorm
    # This is an approximation based on the skewness
    a = 4 * np.sign(skewness) * np.power(np.abs(skewness/2), 1/3)
    
    # Create the distribution
    dist = skewnorm(a, loc=mean, scale=std)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(e_range, dist.pdf(e_range), 'b-', lw=2, label='Energy Distribution')
    
    # Add vertical line for mean energy
    plt.axvline(x=mean, color='r', linestyle='--', label=f'Mean Energy: {mean:.2f} eV')
    
    # Customize plot
    plt.xlabel('Energy (eV)')
    plt.ylabel('Probability Density')
    plt.title('Energy Distribution of Sputtered Atoms')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistical information as text
    stats_text = f'Mean: {mean:.2f} eV\n'
    stats_text += f'Std Dev: {std:.2f} eV\n'
    stats_text += f'Skewness: {skewness:.2f}'
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main(path_to_file):

    try:
        sputter_yield, energy_stats = read_trim_output(path_to_file)
        
        print(f"Sputtering Yield: {sputter_yield}")
        print("\nEnergy Statistics:")
        for key, value in energy_stats.items():
            print(f"{key}: {value}")
        
        plot_energy_distribution(energy_stats, save_path='energy_distribution.png')
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main(PATH_TO_TRIM_OUTPUT)
