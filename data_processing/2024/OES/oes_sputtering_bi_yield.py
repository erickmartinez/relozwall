"""
Estimate the sputtering rate of boron neutral from the BI line at ~820 nm
We need to estimate the emissivity

ε(x,y,z) = n_B(x,y,z) n_e(x,y,z) S_photo(λ)

We don't know n_B, but we assume that it is distribution is matches the shape 𝜌_B in
PARTICLE_DENS_FILE, and it is proportional to a constant:

n_B = A 𝜌_B(x,y,z)

To get the intensity of this line we integrate ε over the volume of the cylinder
defined by the spectrometer spot line of view (get excitations/s) and divide over
the area of the spot (give excitations/cm^2/s).

I_sim = 4π ∫ ε(x,y,z) dx dy dz / (π r_s^2)

where π r_s^2 is the area of the spot

Then

I_sim = 4A S_photo(λ) ∫ 𝜌_B(x,y,z) n_e(x,y,z) dx dy dz / r^2

The simulated intensity I_sim, must be equal to the measured brightness I_measured.
Then

A = I_measured r^2 / (S_photo(λ) ∫ 𝜌_B(x,y,z) n_e(x,y,z) dx dy dz)

Once A is determined, we have

n_B(x,y,z) = A 𝜌_B(x,y,z)

We want to know the sputtering rate (flux of B leaving the surface at z=0).

𝛤_B = n_B(x,y,z) v_thermal = A 𝜌_B(x,y,z) v_thermal

where v_thermal is the thermal velocity of B in the plasma:

v_thermal = √(3kT/m), where T is the temperature and m is the mass of the particle

"""
import os.path
import re
import pandas as pd
import numpy as np
from emissivity_simulator import GridEmissivityCalculator
from emissivity_simulator import load_plot_style
from pathlib import Path
from scipy.stats.distributions import t


PARTICLE_DENS_FILE = r"./data/emissivity_simulations/20241224_particle_density.hd5"
PA_PROBE_MEANS_FILE = r"./data/PA_probe_surface_mean.xlsx"
INTENSITY_FILE = r"./data/bi_lorentzian.xlsx"
S_PEC = 5.1E-11 # cm^3/s
MASS_TARGET_PARTICLE = 10.811

# Probe cylindrical volume parameters
CYLINDER_DIAMETER = 2.0  # cm <-- The probe diameter
CYLINDER_AXIS = [(0, 1.0, 0.0), (0, -1.0, 0.0)]  # Points defining cylinder axis

TRIMSP_DATA_DF = pd.DataFrame(data={
    'ion': ['D+', 'D2+', 'D3+'],
    'ion_composition': [0.41, 0.22, 0.37],
    'sputtering_yield': [0.016705392, 0.005855387, 0.0],
    'sputtered_energy_eV': [1.952335105, 0.0, 0.0]
})

def estimate_vth_trim(trimsp_df: pd.DataFrame, mass_amu, surface_temperature_k):
    trimsp_df['velocity (cm/s)'] = estimate_velocity_energy(
        trimsp_df['sputtered_energy_eV'], mass_amu, surface_temperature_k
    )
    trimsp_df.loc[trimsp_df['sputtered_energy_eV'] == 0 , 'velocity (cm/s)'] = 0.
    v_sputtered = trimsp_df['velocity (cm/s)'].values
    ion_composition = trimsp_df['ion_composition'].values
    sputtreing_yield = trimsp_df['sputtering_yield'].values
    v_sputtered_mean = np.dot(v_sputtered, ion_composition * sputtreing_yield) / np.dot(sputtreing_yield, ion_composition)
    v2_sputtered_mean = np.dot(v_sputtered*v_sputtered, ion_composition * sputtreing_yield) / np.dot(sputtreing_yield, ion_composition)
    # Estimate the t-val for a confidence level 0f 95%
    alpha = 1 - 0.95
    n = len(trimsp_df)
    tval = t.ppf(1. - alpha/2, n-1)
    v_sputtered_std = np.sqrt(np.abs(v2_sputtered_mean - v_sputtered_mean ** 2.)) * np.sqrt(n / (n-1))
    v_sputtered_se = v_sputtered_std * tval / np.sqrt(n)
    # print(trimsp_df)
    return v_sputtered_mean, v_sputtered_se

def estimate_velocity_energy(energy_ev, mass_au, surface_temperature_k):
    v_trimsp = 13891.388 * np.sqrt(energy_ev / mass_au ) * 100.
    vth_surface_temperature = estimate_thermal_velocity(temperature_k=surface_temperature_k, mass_au=mass_au)
    return np.array([max(v, vth_surface_temperature) for v in v_trimsp])

def estimate_thermal_velocity(temperature_k, mass_au):
    dalton = 1.660539068E-27 # kg
    # The most probable speed
    return np.sqrt(1.380649e-23 * temperature_k / (mass_au * dalton) ) * 100.

def get_gamma_d_from_file(path_to_file):
    with open(path_to_file, 'r') as f:
        for line in f:
            if 'Gamma_D_mean' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    value = float(parts[1].split('-/+')[0].strip())
                    uncertainty = float(parts[1].split('-/+')[1].replace('1/cm^2/s','').strip())
                    break
    return value, uncertainty

def load_probe_data(path_to_file) -> pd.DataFrame:
    df = pd.read_excel(path_to_file, sheet_name=0)
    return df

def load_intensity_data(path_to_file: str, sheet:str=0) -> pd.DataFrame:
    df = pd.read_excel(
        path_to_file,
        sheet_name=sheet,
    )
    usecols = [
        'Folder', 'File', 'BI H (photons/cm^2/s)', 'BI H lb (photons/cm^2/s)', 'BI H ub (photons/cm^2/s)',
        'Elapsed time (s)', 'Timestamp', 'Temperature (K)'

    ]
    return df[usecols].sort_values(by=['Folder', 'File']).reset_index(drop=True)


def find_closest_timestamp(df, target_timestamp, timestamp_column='Timestamp'):
    """
    Find the row in a DataFrame with the timestamp closest to the target timestamp.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the timestamp column
    target_timestamp (str or pandas.Timestamp): The timestamp to compare against
    timestamp_column (str): Name of the timestamp column in the DataFrame

    Returns:
    pandas.Series: The row with the closest timestamp
    """
    # Convert target_timestamp to pandas Timestamp if it's not already
    if not isinstance(target_timestamp, pd.Timestamp):
        target_timestamp = pd.to_datetime(target_timestamp)

    # Convert the timestamp column to pandas datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Calculate the absolute difference between each timestamp and the target
    df['time_diff'] = abs(df[timestamp_column] - target_timestamp)

    # Get the index of the minimum difference
    closest_idx = df['time_diff'].idxmin()

    # Drop the temporary time_diff column
    df.drop('time_diff', axis=1, inplace=True)

    # Return the row with the closest timestamp
    return df.loc[closest_idx]


def main(
    intensity_file, pa_probe_db_file, particle_density_hd5, cylinder_diameter, cylinder_axis, target_particle_mass,
    photoemission_rate_coeff, trimsp_df
):
    probe_df = load_probe_data(pa_probe_db_file)
    intensity_df = load_intensity_data(intensity_file, 'Lorentzian peaks')

    # Initialize calculator
    calculator = GridEmissivityCalculator(photoemission_coeff=photoemission_rate_coeff)


    sputtering_yield_df = intensity_df[['Folder', 'File', 'Elapsed time (s)', 'Timestamp', 'Temperature (K)']].copy()
    n_points = len(sputtering_yield_df)
    sputtering_yield_df['Gamma_B (1/cm^2/s)'] = np.zeros(n_points)
    sputtering_yield_df['Gamma_B error (1/cm^2/s)'] = np.zeros(n_points)
    sputtering_yield_df['Sputtering yield'] = np.zeros(n_points)
    sputtering_yield_df['Sputtering yield error'] = np.zeros(n_points)

    for i, row in intensity_df.iterrows():
        timestamp_intensity = row['Timestamp']
        intensity_oes = row['BI H (photons/cm^2/s)']
        intensity_oes_ub = row['BI H ub (photons/cm^2/s)']
        intensity_oes_delta = intensity_oes_ub - intensity_oes
        intensity_error_pct = intensity_oes_delta / intensity_oes
        # Get the row from the probe database with the timestamp closest to `timestamp_intensity`
        probe_at_ts_df = find_closest_timestamp(df=probe_df, target_timestamp=timestamp_intensity,
                                                timestamp_column='Datetime')
        # Get the corresponding file from the probe analysis containing the coefficients to the
        # polynomial fit to the electron density of the plasma n_e(x,y)
        probe_folder = probe_at_ts_df['Folder']
        probe_file = probe_at_ts_df['File'] + '_fit.csv'
        path_to_pa_file = Path('./data') / Path(probe_folder) / 'langprobe_results' / 'symmetrized' / probe_file

        # Load electron density
        r, ne = calculator.load_electron_density(path_to_pa_file)

        # Calculate emissivity on grid
        X, Y, Z, emissivity = calculator.calculate_grid_emissivity(
            density_file=particle_density_hd5
        )

        x, y, z = X[:, 0, 0], Y[0, :, 0], Z[0, 0, :]

        DZ = z[1] - z[0]
        # New z grid for extension
        z_ext = np.arange(-1.0, z.min() - DZ, DZ)
        # Calculate number of points in extension
        nz_ext = len(z_ext)
        # Create zero-valued extension
        e_extension = np.zeros((len(x), len(y), nz_ext))
        # Concatenate original and extension along z-axis
        emissivity_full = np.concatenate((e_extension, emissivity), axis=2)

        # Create full z grid
        z_full = np.concatenate((z_ext, z))
        # Create meshgrid for visualization
        X_full, Y_full, Z_full = np.meshgrid(x, y, z_full, indexing='ij')

        # Calculate integrated intensity
        simulated_intensity = calculator.integrate_cylinder(
            X_full, Y_full, Z_full, emissivity_full,
            cylinder_diameter,
            cylinder_axis[0], cylinder_axis[1]
        ) * 4. * np.pi

        intensity_factor = intensity_oes / simulated_intensity

        n_sim, (X, Y, Z) = calculator.load_density_grid(particle_density_hd5)
        n_b = n_sim * intensity_factor

        # Assume z=zmin at idx 1
        # DZ = z[1] - z[0]
        # idx_z0 = int(0.252 // DZ)
        nb_mean = np.mean(n_b)
        nb_mean_uncertainty = nb_mean * intensity_error_pct
        # print(f'<n_b>: {nb_mean:.3E} -/+ {nb_mean_uncertainty:.3E}')

        # v_thermal = estimate_thermal_velocity(temperature_k=row['Temperature (K)'], mass_au=target_particle_mass)
        v_thermal, v_thermal_error = estimate_vth_trim(
            trimsp_df=trimsp_df, mass_amu=target_particle_mass, surface_temperature_k=row['Temperature (K)']
        )

        # print(f"v_th: {v_thermal:.3E} -/+ {v_thermal_error:.4E} cm/s")
        # Consider a 5% error in the temperture
        # v_thermal_error = 0.5 * v_thermal * 0.05 <- Using value from std of the estimated ion compoistion instead
        gamma_b = v_thermal * nb_mean
        gamma_b_error = gamma_b * np.linalg.norm([intensity_error_pct, 0.5*0.05])

        sputtering_yield_df.loc[i, 'Gamma_B (1/cm^2/s)'] = gamma_b
        sputtering_yield_df.loc[i, 'Gamma_B error (1/cm^2/s)'] = gamma_b_error


        # Get the mean D flux on the surface
        gamma_d, gamma_d_error = get_gamma_d_from_file(path_to_pa_file)

        sputtering_yield_df.loc[i, 'Sputtering yield'] = gamma_b / gamma_d
        sputtering_yield_df.loc[i, 'Sputtering yield error'] = sputtering_yield_df.loc[i, 'Sputtering yield'] * np.linalg.norm([gamma_d_error/gamma_d, gamma_b_error/gamma_b])

        print(f'Gamma_B: {gamma_b:.3E} -/+ {gamma_b_error:.3E}, '
              f'Gamma_D: {gamma_d:.3E} -/+ {gamma_d_error:.3E} 1/cm^2/s, '
              f"Y_B/D: {sputtering_yield_df.loc[i, 'Sputtering yield']:.4f} "
              f"-/+ {sputtering_yield_df.loc[i, 'Sputtering yield error']:.4f}, "
              f"v_th: {v_thermal:.4E}")

        sputtering_yield_df.to_csv('./data/boron_physical_sputtering_yields.old.csv', index=False, lineterminator='\n')





if __name__ == '__main__':
    main(
        intensity_file=INTENSITY_FILE, pa_probe_db_file=PA_PROBE_MEANS_FILE, particle_density_hd5=PARTICLE_DENS_FILE,
        cylinder_diameter=CYLINDER_DIAMETER, cylinder_axis=CYLINDER_AXIS, target_particle_mass=MASS_TARGET_PARTICLE,
        photoemission_rate_coeff=S_PEC, trimsp_df=TRIMSP_DATA_DF
    )







