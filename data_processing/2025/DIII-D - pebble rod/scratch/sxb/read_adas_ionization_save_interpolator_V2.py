import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from matplotlib.colors import LogNorm
import h5py


def parse_adas_scd(filepath):
    """
    Parses an ADAS 'scd' file (ADF11 format) for ionization rate coefficients.
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
        raise ValueError("Data parsing error")
    log_ne = data_values[0:n_ne]
    log_te = data_values[n_ne: n_ne + n_te]
    log_rates_flat = data_values[n_ne + n_te:]
    log_rates = log_rates_flat.reshape(n_te, n_ne).T
    print("✅ Successfully parsed ADAS data.")
    return log_te, log_ne, log_rates


def create_ionization_interpolator(filepath, kx=1, ky=1):
    """
    Creates a 2D interpolator from an ADAS file with specified spline degrees.
    """
    log_te, log_ne, log_rates = parse_adas_scd(filepath)
    interpolator = RectBivariateSpline(log_ne, log_te, log_rates, kx=kx, ky=ky)
    return interpolator


def save_interpolator_to_h5(interpolator, k_te, k_ne, filepath="adas_boron_ionization_interpolator.h5"):
    """
    Saves the core components of a RectBivariateSpline interpolator to an HDF5 file.

    Args:
        interpolator (RectBivariateSpline): The object to save.
        kx (int): The spline degree in the x-dimension.
        ky (int): The spline degree in the y-dimension.
        filepath (str): The path for the output HDF5 file.
    """
    log_ne_knots, log_te_knots = interpolator.get_knots()
    coeffs = interpolator.get_coeffs()

    with h5py.File(filepath, 'w') as hf:
        hf.create_dataset("log_ne_knots", data=log_ne_knots)
        hf.create_dataset("log_te_knots", data=log_te_knots)
        hf.create_dataset("coeffs", data=coeffs)
        # Save kx and ky, which are passed explicitly
        hf.attrs['k_te'] = k_te
        hf.attrs['k_ne'] = k_ne

    print(f"✅ Interpolator saved successfully to '{filepath}'")


def load_interpolator_from_h5(filepath="boron_ion_interpolator.h5"):
    """
    Loads interpolator components from an HDF5 file and reconstructs the object.
    """
    with h5py.File(filepath, 'r') as hf:
        log_te_knots = hf['log_te_knots'][:]
        log_ne_knots = hf['log_ne_knots'][:]
        coeffs = hf['coeffs'][:]
        k_te = hf.attrs['k_te']
        k_ne = hf.attrs['k_ne']

    # The TCK tuple must be in (tx, ty, c, kx, ky) order for reconstruction.
    # tx -> te_knots, ty -> ne_knots, kx -> k_te, ky -> k_ne
    tck = (log_te_knots, log_ne_knots, coeffs, k_te, k_ne)
    loaded_interpolator = RectBivariateSpline._from_tck(tck)

    print(f"✅ Interpolator loaded successfully from '{filepath}'")
    return loaded_interpolator


if __name__ == '__main__':
    adas_file = '../../mds_spectra/sxb/scd89_b.dat'
    h5_file = 'adas_boron_ionization_interpolator.h5'

    try:
        # Define spline degrees here
        spline_kx, spline_ky = 3, 3

        # Step 1: Create and Save
        print("--- Creating interpolator from source ADAS file ---")
        original_interpolator = create_ionization_interpolator(adas_file, kx=spline_kx, ky=spline_ky)
        # --- FIX IS HERE: Pass the kx, ky values to the save function ---
        save_interpolator_to_h5(original_interpolator, k_te=spline_kx, k_ne=spline_ky, filepath=h5_file)

        # Step 2: Load and Use
        print("\n--- Loading interpolator from HDF5 file for use ---")
        loaded_interpolator = load_interpolator_from_h5(h5_file)

        # Step 3: Test
        te_eV = 25.0
        ne_cm3 = 1e13
        log_te_val = np.log10(te_eV)
        log_ne_val = np.log10(ne_cm3)
        log_rate = loaded_interpolator(log_ne_val, log_te_val, grid=False)
        rate = 10 ** log_rate

        print(f"\n---> Test Calculation with LOADED Interpolator <---")
        print(f"For Te = {te_eV:.1f} eV and ne = {ne_cm3:.1e} cm⁻³:")
        print(f"  - The ionization rate is: {rate:.3e} cm³/s")

    except FileNotFoundError:
        print(f"❌ ERROR: The file '{adas_file}' was not found.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")