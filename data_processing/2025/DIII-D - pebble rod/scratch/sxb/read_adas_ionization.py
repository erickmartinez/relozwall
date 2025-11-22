import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from matplotlib.colors import LogNorm  # Import LogNorm for the color scale


def parse_adas_scd(filepath):
    """
    Parses an ADAS 'scd' file (ADF11 format) for ionization rate coefficients.
    This version is robust against text footers and intermediate separators.

    Args:
        filepath (str): The path to the ADAS scd file.

    Returns:
        tuple: A tuple containing:
            - log_te (np.ndarray): 1D array of log10(Te) in eV.
            - log_ne (np.ndarray): 1D array of log10(ne) in cm⁻³.
            - log_rates (np.ndarray): 2D array of log10(rate) in cm³/s.
                                      Shape is (len(log_ne), len(log_te)).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_parts = lines[0].split()
    n_ne, n_te = int(header_parts[1]), int(header_parts[2])

    all_numbers_as_strings = []
    for line in lines[1:]:
        line_clean = line.strip()
        if not line_clean.startswith('C') and '---' not in line_clean:
            all_numbers_as_strings.extend(line_clean.split())

    total_numbers_expected = n_ne + n_te + (n_ne * n_te)
    numerical_words = all_numbers_as_strings[:total_numbers_expected]
    data_values = np.array(numerical_words, dtype=float)

    if len(data_values) != total_numbers_expected:
        raise ValueError(
            f"Data parsing error: Expected {total_numbers_expected} numerical "
            f"values, but found {len(data_values)}."
        )

    log_ne = data_values[0:n_ne]
    log_te = data_values[n_ne: n_ne + n_te]
    log_rates_flat = data_values[n_ne + n_te:]
    log_rates = log_rates_flat.reshape(n_te, n_ne).T

    print("✅ Successfully parsed ADAS data.")
    return log_te, log_ne, log_rates


def create_ionization_interpolator(filepath):
    """
    Creates a 2D interpolator for ionization rates from an ADAS file.
    """
    log_te, log_ne, log_rates = parse_adas_scd(filepath)
    interpolator = RectBivariateSpline(log_ne, log_te, log_rates, kx=1, ky=1)
    return interpolator


def plot_adas_rates_log_scale(log_te, log_ne, log_rates):
    """
    Plots the ionization rate coefficients with a logarithmic color normalization.
    """
    Te_grid = 10 ** log_te
    Ne_grid = 10 ** log_ne
    Rates_2d = 10 ** log_rates

    min_r, max_r = np.floor(np.log10(Rates_2d.flatten().min())), np.ceil(np.log10(Rates_2d.flatten().max()))
    print(f'min_r={min_r}, max_r={max_r}')

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- FIX IS HERE ---
    # Extend the color bar to much lower values to see the full range.
    # From -16 down to -30, and added more levels for a smoother gradient.
    contour = ax.contourf(Te_grid, Ne_grid, Rates_2d,
                          norm=LogNorm(),
                          levels=np.logspace(min_r, max_r, 24))  # Changed from (-16, -7, 10)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('$T_e$ (eV)', fontsize=14)
    ax.set_ylabel('$n_e$ (cm$^{-3}$)', fontsize=14)
    ax.set_title('B I Ionization Rate Coefficient (Full Logarithmic Scale)', fontsize=16)

    cbar = fig.colorbar(contour)
    cbar.set_label('Rate Coefficient (cm³/s)', fontsize=14)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("boron_ionization_rates_log_color_FIXED.png", dpi=300)
    plt.show()


def plot_ne_slice(log_te, log_ne, log_rates):
    """
    Extracts and plots a slice of the ionization rate at ne = 1e13 cm⁻³.
    """
    target_log_ne = 13.0

    # Find the index of the density grid closest to our target
    ne_index = np.argmin(np.abs(log_ne - target_log_ne))

    # Get the actual density value at that index to display in the title
    slice_ne_val = 10 ** log_ne[ne_index]

    # Extract the 1D slice of rates at that density index
    rate_slice = 10 ** log_rates[ne_index, :]

    # Get the corresponding temperature grid
    Te_grid = 10 ** log_te

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(Te_grid, rate_slice, marker='o', linestyle='-', color='b')

    ax.set_xlabel('$T_e$ (eV)', fontsize=14)
    ax.set_ylabel('Ionization Rate Coefficient (cm³/s)', fontsize=14)
    ax.set_title(f'B I Ionization Rate at $n_e$ = {slice_ne_val:.1e} cm$^{{-3}}$', fontsize=16)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("boron_ionization_rate_slice.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    adas_file = '../../mds_spectra/sxb/scd89_b.dat'

    try:
        log_te_grid, log_ne_grid, log_rates_grid = parse_adas_scd(adas_file)

        # Plot 1: 2D contour plot with a logarithmic colorbar
        print("\nGenerating 2D contour plot with logarithmic color scale...")
        plot_adas_rates_log_scale(log_te_grid, log_ne_grid, log_rates_grid)

        # Plot 2: 1D slice at a specific electron density
        print("Generating 1D slice plot at ne = 1e13 cm⁻³...")
        plot_ne_slice(log_te_grid, log_ne_grid, log_rates_grid)

    except FileNotFoundError:
        print(f"❌ ERROR: The file '{adas_file}' was not found.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")