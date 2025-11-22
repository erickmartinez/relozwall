import re
import numpy as np
import h5py
from scipy.interpolate import RectBivariateSpline


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

    # The PEC data is in (n_te, n_ne) order, so reshape accordingly
    pec_matrix = pec_rates.reshape(n_te, n_ne)

    # Take the log of the data for interpolation
    # Add a small floor to avoid log(0) errors if data contains zeros
    log_ne = np.log10(ne_values)
    log_te = np.log10(te_values)
    log_pec_rates = np.log10(np.maximum(pec_matrix, 1e-50))

    return log_te, log_ne, log_pec_rates


def create_pec_interpolator(filepath, wavelength, kx=1, ky=1):
    """
    Creates a 2D interpolator from an ADAS PEC file for a given wavelength.

    Args:
        filepath (str): Path to the ADAS PEC file.
        wavelength (float): The wavelength for which to create the interpolator.
        kx (int): The spline degree in the x-dimension (log_ne).
        ky (int): The spline degree in the y-dimension (log_te).

    Returns:
        RectBivariateSpline: The interpolator object.
    """
    log_te, log_ne, log_rates = parse_adas_pec(filepath, wavelength)
    if log_te is None:
        return None
    # Note: Scipy's RectBivariateSpline expects (y, x) ordering for knots,
    # so we pass (log_te, log_ne) and the correctly shaped log_rates matrix.
    interpolator = RectBivariateSpline(log_te, log_ne, log_rates, kx=kx, ky=ky)
    return interpolator


def save_interpolator_to_h5(interpolator, kx, ky, filepath="adas_pec_interpolator.h5"):
    """
    Saves the core components of a RectBivariateSpline interpolator to an HDF5 file.

    Args:
        interpolator (RectBivariateSpline): The object to save.
        kx (int): The spline degree in the x-dimension.
        ky (int): The spline degree in the y-dimension.
        filepath (str): The path for the output HDF5 file.
    """
    # The knots are returned in (y, x) order, which corresponds to (Te, ne)
    log_te_knots, log_ne_knots = interpolator.get_knots()
    coeffs = interpolator.get_coeffs()

    with h5py.File(filepath, 'w') as hf:
        hf.create_dataset("log_ne_knots", data=log_ne_knots)
        hf.create_dataset("log_te_knots", data=log_te_knots)
        hf.create_dataset("coeffs", data=coeffs)
        hf.attrs['kx'] = kx
        hf.attrs['ky'] = ky

    print(f"✅ Interpolator saved successfully to '{filepath}'")


def load_interpolator_from_h5(filepath="adas_pec_interpolator.h5"):
    """
    Loads interpolator components from an HDF5 file and reconstructs the object.

    Args:
        filepath (str): The path to the HDF5 file.

    Returns:
        RectBivariateSpline: The reconstructed interpolator object.
    """
    with h5py.File(filepath, 'r') as hf:
        log_ne_knots = hf['log_ne_knots'][:]
        log_te_knots = hf['log_te_knots'][:]
        coeffs = hf['coeffs'][:]
        kx = hf.attrs['kx']
        ky = hf.attrs['ky']

    # The TCK tuple must be in (y_knots, x_knots, coeffs, ky, kx) order for reconstruction.
    # Note the reversal of kx and ky here is intentional for the internal tuple format.
    tck = (log_te_knots, log_ne_knots, coeffs, ky, kx)
    loaded_interpolator = RectBivariateSpline._from_tck(tck)

    print(f"✅ Interpolator loaded successfully from '{filepath}'")
    return loaded_interpolator


def main():
    """
    Main function to demonstrate creating, saving, and loading the PEC interpolator.
    """
    adas_file = '../../mds_spectra/sxb/pec93#b_llu#b0.dat'
    output_h5_file = '../../mds_spectra/sxb/boron_pec_interpolator.h5'

    # 1. Find available wavelengths in the file
    available_wavelengths = list_wavelengths(adas_file)
    print(f"Available wavelengths in '{adas_file}':")
    print(available_wavelengths)

    if not available_wavelengths:
        print("No wavelengths found. Exiting.")
        return

    # --- Parser Verification Step ---
    print("\n--- Parser Verification ---")
    target_wavelength = available_wavelengths[0]
    log_te_grid, log_ne_grid, log_pec_rates = parse_adas_pec(adas_file, target_wavelength)
    if log_te_grid is None:
        return
    print(f"Parsing for wavelength {target_wavelength} Å:")
    print(f"Electron temperature grid points (log10 eV): {len(log_te_grid)}")
    print(f"Electron density grid points (log10 cm^-3): {len(log_ne_grid)}")
    print(f"Shape of the PEC data matrix (Te, ne): {log_pec_rates.shape}")

    # Check for n_e dependency in the raw data at a moderate temperature
    mid_temp_index = len(log_te_grid) // 2
    pec_row = 10 ** log_pec_rates[mid_temp_index, :]
    print(f"\nChecking raw PEC data at T_e = {10 ** log_te_grid[mid_temp_index]:.2f} eV:")
    print(f"PEC values across all densities: {pec_row}")
    if np.allclose(pec_row, pec_row[0]):
        print("NOTE: For this specific Te, the raw data shows weak dependency on n_e.")
    else:
        print("Conclusion: The raw data clearly shows that PEC varies with electron density.")

    # 2. Select a wavelength and create the interpolator
    print(f"\nCreating interpolator for wavelength: {target_wavelength} Å")
    # Using cubic splines for smoother interpolation
    kx, ky = 3, 3
    pec_interpolator = create_pec_interpolator(adas_file, target_wavelength, kx=kx, ky=ky)

    if pec_interpolator is None:
        return

    # 3. Save the interpolator to an HDF5 file
    save_interpolator_to_h5(pec_interpolator, kx, ky, filepath=output_h5_file)

    # 4. Load the interpolator from the file
    loaded_pec_interpolator = load_interpolator_from_h5(filepath=output_h5_file)

    # 5. Test the loaded interpolator at two different densities
    print(f"\n--- Interpolator Test ---")
    test_te = 10  # eV
    test_ne_1 = 1e12  # cm^-3
    test_ne_2 = 1e15  # cm^-3

    log_te_val = np.log10(test_te)

    # Test case 1
    log_ne_val_1 = np.log10(test_ne_1)
    log_pec_val_1 = loaded_pec_interpolator(log_te_val, log_ne_val_1, grid=False)
    pec_val_1 = 10 ** log_pec_val_1
    print(f"Interpolated PEC at T_e = {test_te} eV and n_e = {test_ne_1:.1e} cm^-3:")
    print(f"-> PEC = {pec_val_1:.4e} photons/cm^3/s")

    # Test case 2
    log_ne_val_2 = np.log10(test_ne_2)
    log_pec_val_2 = loaded_pec_interpolator(log_te_val, log_ne_val_2, grid=False)
    pec_val_2 = 10 ** log_pec_val_2
    print(f"\nInterpolated PEC at T_e = {test_te} eV and n_e = {test_ne_2:.1e} cm^-3:")
    print(f"-> PEC = {pec_val_2:.4e} photons/cm^3/s")

    if not np.isclose(pec_val_1, pec_val_2):
        print("\nConclusion: The interpolated PEC value correctly changes with electron density.")
    else:
        print("\nConclusion: At this specific T_e, the interpolated PEC shows negligible dependency on n_e.")


if __name__ == "__main__":
    main()

