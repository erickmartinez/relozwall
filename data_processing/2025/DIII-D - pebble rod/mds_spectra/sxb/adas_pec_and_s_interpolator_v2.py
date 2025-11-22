import re
import numpy as np
import h5py
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt


def list_wavelengths(filepath):
    """
    Scans an ADAS PEC file and returns a list of available wavelengths.

    Args:
        filepath (str): The path to the ADAS PEC file.

    Returns:
        list: A list of wavelengths (floats) found in the file.
    """
    # Regex to find lines starting with a floating point number followed by ' A'
    wavelength_regex = re.compile(r"^\s*(\d+\.\d*)\s+A")
    wavelengths = []
    with open(filepath, 'r') as f:
        for line in f:
            match = wavelength_regex.match(line)
            if match:
                wavelengths.append(float(match.group(1)))
    return wavelengths


def parse_adas_pec(filepath, wavelength):
    """
    Parses an ADAS PEC file for a specific wavelength to extract electron
    temperature, density, and the PEC data matrix.

    Args:
        filepath (str): The path to the ADAS PEC file.
        wavelength (float): The specific wavelength to extract data for.

    Returns:
        tuple: A tuple containing (log_te, log_ne, log_pec_rates).
               Returns (None, None, None) if the wavelength is not found.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the starting line for the requested wavelength
    start_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{wavelength}"):
            start_index = i
            break

    if start_index == -1:
        print(f"Error: Wavelength {wavelength} not found in {filepath}")
        return None, None, None

    # --- Data Extraction ---
    # Get n_ne and n_te from the wavelength line
    header_parts = lines[start_index].split()
    n_ne, n_te = int(header_parts[2]), int(header_parts[3])

    # Read all data values into a single list, starting from the line after the header
    data_lines = lines[start_index + 1:]
    all_values = []
    for line in data_lines:
        # Stop if we hit the start of the next data block or a comment
        if 'C---' in line or re.match(r"^\s*\d+\.\d*\s+A", line):
            break
        all_values.extend(map(float, line.split()))

    # Extract ne, te, and pec rates from the flat list of values
    ne_values = np.array(all_values[0:n_ne])
    te_values = np.array(all_values[n_ne: n_ne + n_te])
    pec_rates = np.array(all_values[n_ne + n_te: n_ne + n_te + (n_ne * n_te)])

    # The ADAS file lists the data with temperature as the fast index for each
    # density. So, we first reshape to (number_of_densities, number_of_temperatures).
    pec_matrix = pec_rates.reshape(n_ne, n_te)

    # Take the log of the data for interpolation
    # Add a small floor to avoid log(0) errors if data contains zeros
    log_ne = np.log10(ne_values)
    log_te = np.log10(te_values)

    # The RectBivariateSpline expects the data grid Z to have a shape of
    # (len(x), len(y)), which corresponds to (len(te), len(ne)).
    # Therefore, we need to transpose the PEC matrix before taking the log.
    log_pec_rates = np.log10(np.maximum(pec_matrix.T, 1e-50))

    return log_te, log_ne, log_pec_rates


def parse_adas_scd(filepath, iz_stage=1):
    """
    Parses an ADAS 'scd' file (ADF11 format) for ionization rate coefficients.
    This version can handle files with multiple ionization stages.

    Args:
        filepath (str): Path to the ADAS scd file.
        iz_stage (int): The initial charge state to extract data for (e.g., 1 for B0 -> B1+).
                        ADAS files are 1-indexed for charge states.

    Returns:
        tuple: A tuple containing (log_te, log_ne, log_ion_rates).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_parts = lines[0].split()
    izmax = int(header_parts[0])  # Maximum charge state in file + 1
    n_ne = int(header_parts[1])
    n_te = int(header_parts[2])

    if not (1 <= iz_stage <= izmax):
        print(f"Error: Requested charge state {iz_stage} is out of range. File contains stages 1 to {izmax}.")
        return None, None, None

    all_numbers_as_strings = []
    # Data starts after the '---' line
    for line in lines[2:]:
        line_clean = line.strip()
        if not line_clean.startswith('C') and '---' not in line_clean:
            all_numbers_as_strings.extend(line_clean.split())

    data_values = np.array(all_numbers_as_strings, dtype=float)

    # The data is ordered as log_ne, then log_te, then all log_rates blocks
    log_ne = data_values[0:n_ne]
    log_te = data_values[n_ne: n_ne + n_te]

    # All rate coefficients for all ionization stages are flattened
    all_log_rates_flat = data_values[n_ne + n_te:]

    # Calculate the slice for the requested ionization stage.
    block_size = n_ne * n_te
    start_index = (iz_stage - 1) * block_size
    end_index = start_index + block_size

    if end_index > len(all_log_rates_flat):
        print(f"Error: Data for charge state {iz_stage} not found or incomplete.")
        return None, None, None

    log_rates_flat_stage = all_log_rates_flat[start_index:end_index]

    # In this format, Te is the fast-varying index. We reshape to (n_ne, n_te)
    # where each row corresponds to a single density.
    log_rates_matrix = log_rates_flat_stage.reshape(n_te, n_ne)

    # We then transpose the matrix to get a shape of (n_te, n_ne) as required
    # by RectBivariateSpline where x=Te and y=Ne.
    log_ion_rates = log_rates_matrix

    print(f"✅ Successfully parsed stage {iz_stage} from ADAS ionization file '{filepath}'")
    return log_te, log_ne, log_ion_rates


def create_pec_interpolator(filepath, wavelength, k_te=1, k_ne=1):
    """
    Creates a 2D interpolator from an ADAS PEC file for a given wavelength.
    We map x -> Te and y -> ne.

    Args:
        filepath (str): Path to the ADAS PEC file.
        wavelength (float): The wavelength for which to create the interpolator.
        k_te (int): The spline degree for the Te dimension.
        k_ne (int): The spline degree for the ne dimension.

    Returns:
        RectBivariateSpline: The interpolator object.
    """
    log_te, log_ne, log_rates = parse_adas_pec(filepath, wavelength)
    if log_te is None:
        return None
    interpolator = RectBivariateSpline(log_te, log_ne, log_rates, kx=k_te, ky=k_ne)
    return interpolator


def create_ionization_interpolator(filepath, iz_stage=1, k_te=1, k_ne=1):
    """
    Creates a 2D interpolator from an ADAS ionization file for a specific charge state.
    We map x -> Te and y -> ne.

    Args:
        filepath (str): Path to the ADAS scd file.
        iz_stage (int): The initial charge state to create the interpolator for.
        k_te (int): The spline degree for the Te dimension.
        k_ne (int): The spline degree for the ne dimension.

    Returns:
        RectBivariateSpline: The interpolator object.
    """
    log_te, log_ne, log_rates = parse_adas_scd(filepath, iz_stage=iz_stage)
    if log_te is None:
        return None
    interpolator = RectBivariateSpline(log_te, log_ne, log_rates, kx=k_te, ky=k_ne)
    return interpolator


def save_interpolator_to_h5(interpolator, k_te, k_ne, filepath="adas_interpolator.h5"):
    """
    Saves the core components of a RectBivariateSpline interpolator to an HDF5 file.

    Args:
        interpolator (RectBivariateSpline): The object to save.
        k_te (int): The spline degree in the Te-dimension.
        k_ne (int): The spline degree in the ne-dimension.
        filepath (str): The path for the output HDF5 file.
    """
    log_te_knots, log_ne_knots = interpolator.get_knots()
    coeffs = interpolator.get_coeffs()

    with h5py.File(filepath, 'w') as hf:
        hf.create_dataset("log_te_knots", data=log_te_knots)
        hf.create_dataset("log_ne_knots", data=log_ne_knots)
        hf.create_dataset("coeffs", data=coeffs)
        hf.attrs['k_te'] = k_te
        hf.attrs['k_ne'] = k_ne

    print(f"✅ Interpolator saved successfully to '{filepath}'")


def load_interpolator_from_h5(filepath="adas_interpolator.h5"):
    """
    Loads interpolator components from an HDF5 file and reconstructs the object.

    Args:
        filepath (str): The path to the HDF5 file.

    Returns:
        RectBivariateSpline: The reconstructed interpolator object.
    """
    with h5py.File(filepath, 'r') as hf:
        log_te_knots = hf['log_te_knots'][:]
        log_ne_knots = hf['log_ne_knots'][:]
        coeffs = hf['coeffs'][:]
        k_te = hf.attrs['k_te']
        k_ne = hf.attrs['k_ne']

    tck = (log_te_knots, log_ne_knots, coeffs, k_te, k_ne)
    loaded_interpolator = RectBivariateSpline._from_tck(tck)

    print(f"✅ Interpolator loaded successfully from '{filepath}'")
    return loaded_interpolator


def plot_pec_vs_te(interpolator, n_e_val, wavelength):
    """
    Plots the Photon Emissivity Coefficient (PEC) as a function of electron
    temperature (T_e) for a fixed electron density (n_e).

    Args:
        interpolator (RectBivariateSpline): The loaded interpolator object.
        n_e_val (float): The fixed electron density (in cm^-3) for the plot.
        wavelength (float): The wavelength of the transition for the title.
    """
    te_range = np.logspace(np.log10(0.1), np.log10(200), 100)
    log_te_range = np.log10(te_range)
    log_ne_val = np.log10(n_e_val)
    log_pec_values = interpolator(log_te_range, log_ne_val, grid=False)
    pec_values = 10 ** log_pec_values

    plt.figure(figsize=(10, 6))
    plt.plot(te_range, pec_values, lw=2, color='royalblue')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Electron Temperature, $T_e$ (eV)', fontsize=12)
    plt.ylabel('PEC (photons cm$^{3}$ s$^{-1}$)', fontsize=12)
    title = f'Photon Emissivity Coefficient vs. $T_e$\n'
    title += f'Wavelength = {wavelength} Å, $n_e$ = {n_e_val:.1e} cm$^{{-3}}$'
    plt.title(title, fontsize=14)
    plt.ylim(bottom=1e-25)
    plt.show()


def plot_ionization_vs_te(interpolator, n_e_values, species="Boron", charge_state=0):
    """
    Plots the Ionization Rate Coefficient (S) vs. electron temperature (T_e)
    for a list of fixed electron densities (n_e).

    Args:
        interpolator (RectBivariateSpline): The loaded interpolator object.
        n_e_values (list of float): List of electron densities (in cm^-3) for the plot.
        species (str): The atomic species for the plot title.
        charge_state (int): The initial charge state (e.g., 0 for neutral).
    """
    te_range = np.logspace(np.log10(0.1), np.log10(200), 100)
    log_te_range = np.log10(te_range)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(n_e_values)))

    for i, n_e_val in enumerate(n_e_values):
        log_ne_val = np.log10(n_e_val)
        log_s_values = interpolator(log_te_range, log_ne_val, grid=False)
        s_values = 10 ** log_s_values
        plt.plot(te_range, s_values, lw=2, color=colors[i], label=f'$n_e$ = {n_e_val:.1e} cm$^{{-3}}$')

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Electron Temperature, $T_e$ (eV)', fontsize=12)
    plt.ylabel('Ionization Rate Coefficient, S (cm$^{3}$ s$^{-1}$)', fontsize=12)
    title = f'Ionization Rate: {species}$^{{+{charge_state}}}$ → {species}$^{{+{charge_state + 1}}}$'
    plt.title(title, fontsize=14)
    plt.legend()
    plt.show()


def main():
    """
    Main function to demonstrate creating, saving, loading, and plotting interpolators.
    """
    # --- Part 1: Photon Emissivity Coefficient (PEC) ---
    print("--- Processing PEC Data ---")
    pec_adas_file = 'pec93#b_llu#b0.dat'
    pec_output_h5 = 'boron_pec_interpolator.h5'

    try:
        available_wavelengths = list_wavelengths(pec_adas_file)
        if not available_wavelengths:
            print(f"No wavelengths found in {pec_adas_file}.")
        else:
            target_wavelength = available_wavelengths[0]
            target_wavelength = 8148.2
            k_te, k_ne = 3, 3
            print(f"Creating PEC interpolator for wavelength: {target_wavelength} Å")
            pec_interpolator = create_pec_interpolator(pec_adas_file, target_wavelength, k_te=k_te, k_ne=k_ne)

            if pec_interpolator:
                save_interpolator_to_h5(pec_interpolator, k_te, k_ne, filepath=pec_output_h5)
                loaded_pec_interpolator = load_interpolator_from_h5(filepath=pec_output_h5)
                plot_ne = 1e11
                print(f"Generating PEC plot for n_e = {plot_ne:.1e} cm^-3...")
                plot_pec_vs_te(loaded_pec_interpolator, n_e_val=plot_ne, wavelength=target_wavelength)
    except FileNotFoundError:
        print(f"Error: PEC data file '{pec_adas_file}' not found. Skipping PEC part.")
    except Exception as e:
        print(f"An error occurred during PEC data processing: {e}")

    # --- Part 2: Ionization Rate Coefficient (S) ---
    print("\n--- Processing Ionization Data ---")
    try:
        ion_adas_file = 'scd89_b.dat'
        ion_output_h5 = 'boron_ionization_interpolator.h5'
        k_te, k_ne = 3, 3

        target_iz_stage = 1  # For B0 -> B1+

        print(f"Creating ionization interpolator from '{ion_adas_file}' for stage {target_iz_stage}")
        ion_interpolator = create_ionization_interpolator(ion_adas_file, iz_stage=target_iz_stage, k_te=k_te, k_ne=k_ne)

        if ion_interpolator:
            save_interpolator_to_h5(ion_interpolator, k_te, k_ne, filepath=ion_output_h5)
            loaded_ion_interpolator = load_interpolator_from_h5(filepath=ion_output_h5)

            plot_ne_values = [1e11, 1e13, 1e15]
            print(f"Generating Ionization plot for densities: {plot_ne_values} cm^-3...")
            plot_ionization_vs_te(loaded_ion_interpolator, n_e_values=plot_ne_values, species="B",
                                  charge_state=target_iz_stage - 1)

    except FileNotFoundError:
        print(f"Error: Ionization data file '{ion_adas_file}' not found. Skipping ionization part.")
    except Exception as e:
        print(f"An error occurred during ionization data processing: {e}")


if __name__ == "__main__":
    main()

