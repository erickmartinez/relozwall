import numpy as np
import h5py
from pathlib import Path
import pandas as pd
from typing import Dict
from scipy.interpolate import interp1d, CubicSpline, make_smoothing_spline
import matplotlib.pyplot as plt
from data_processing.misc_utils.plot_style import load_plot_style
import scipy.ndimage as ndimage
from pybaselines import Baseline

PEBBLE_ROD_DETAILS_XLS = r'../recession rate model/pebble_rod_exposure.xlsx'
PATH_TO_CURRENT_DATA = r'../recession rate model/data'
PATH_TO_LP_DATA = r'../Langmuir Probe/data/dimes_lp'
SHOTS = np.arange(203780, 203786).astype(int)
SPUTTERING_YIELD = 0.09325 # From SDTrimSP at 100 eV, 45 degree incidence angle


def deuterium_flux_from_jsat(I_sat, area):
    """
    Calculate deuterium flux from saturation current (first order approximation).

    To first order: Γ_D ≈ j_sat / e

    Parameters:
    -----------
    I_sat : float
        Saturation current in Amperes
    area : float
        Target area in cm^2

    Returns:
    --------
    flux : float
        Deuterium flux in particles/(cm^2*s)
    j_sat : float
        Saturation current density in A/cm^2
    """

    # Constants
    e = 1.602e-19  # elementary charge (C)

    # Calculate saturation current density
    j_sat = I_sat / area  # A/cm^2

    # First order approximation: Γ = j_sat / e
    # j_sat is in A/cm^2 = C/(s*cm^2)
    # Dividing by e gives particles/(cm^2*s)
    flux = j_sat / e

    return flux, j_sat


def calculate_divertor_deuterium_flux(n_e, T_e, T_i=None, gamma_i=1.0, sheath_coefficient=0.5):
    """
    Calculate deuterium flux to divertor target in a tokamak.

    Parameters:
    -----------
    n_e : float
        Electron density in cm^-3 (from Langmuir probe at divertor)
    T_e : float
        Electron temperature in eV (from Langmuir probe at divertor)
    T_i : float, optional
        Ion temperature in eV. If None, assumes T_i ≈ T_e (common in divertor)
    gamma_i : float, optional
        Adiabatic index for ions (default=1.0 for isothermal flow along field lines)
        For divertor plasmas: typically 1.0 (isothermal) due to strong parallel losses
    sheath_coefficient : float, optional
        Sheath transmission coefficient (default=0.5)
        For divertor analysis: often 0.5-0.61, sometimes up to 1.0 depending on model

    Returns:
    --------
    flux : float
        Deuterium flux in particles/(cm^2*s)
    c_s : float
        Sound speed in m/s
    heat_flux_e : float
        Electron heat flux in MW/m^2 (using sheath heat transmission factor)
    heat_flux_i : float
        Ion heat flux in MW/m^2
    """

    # Constants
    e = 1.602e-19  # elementary charge (C)
    m_D = 3.344e-27  # deuterium mass (kg)

    # In divertor, if T_i not measured, often assume T_i ≈ T_e
    if T_i is None:
        T_i = T_e
        print(f"Note: T_i not provided, assuming T_i ≈ T_e = {T_e} eV (common in divertor)")

    # Convert temperatures from eV to Joules
    T_e_J = T_e * e
    T_i_J = T_i * e

    # Calculate sound speed with ion temperature
    # c_s = sqrt((T_e + gamma_i * T_i) / m_D)
    c_s = np.sqrt((T_e_J + gamma_i * T_i_J) / m_D)

    # Calculate particle flux: Γ = sheath_coefficient * n_e * c_s
    # c_s is in m/s, n_e is in cm^-3, multiply by 100 to get flux in particles/(cm^2*s)
    flux = sheath_coefficient * n_e * c_s * 100

    # Calculate heat fluxes (important for divertor analysis)
    # Electron heat flux: q_e = γ_e * n_e * T_e * c_s
    # where γ_e ≈ 4-5 for electrons (sheath heat transmission factor)
    gamma_e = 4.5  # Typical value for electrons
    n_e_m3 = n_e * 1e6  # Convert to m^-3
    q_e = gamma_e * n_e_m3 * T_e * e * c_s  # W/m^2
    heat_flux_e = q_e / 1e6  # MW/m^2

    # Ion heat flux: q_i = γ_i * n_i * T_i * c_s + 0.5 * n_i * m_i * c_s^3
    # For simplicity with γ_i, kinetic energy term
    gamma_i_heat = 2.5  # Typical value for ions
    q_i = gamma_i_heat * n_e_m3 * T_i * e * c_s + 0.5 * n_e_m3 * m_D * c_s ** 3
    heat_flux_i = q_i / 1e6  # MW/m^2

    return flux, c_s, heat_flux_e, heat_flux_i

def load_lp_data(shot, path_to_folder):
    path_to_folder = Path(path_to_folder)
    with h5py.File( path_to_folder / f'{shot}_LP.h5', 'r') as h5:
        dimes_gp = h5['/LANGMUIR_DIMES']
        t_ms = np.array(dimes_gp.get('time'))
        T_eV = np.array(dimes_gp.get('TeV'))
        n_e = np.array(dimes_gp.get('ne')) * 1E13
        qpara = np.array(dimes_gp.get('qpara'))
    data = {
        'time_ms': t_ms,
        'n_e': n_e,
        'T_e': T_eV,
        'qpara': qpara,
    }
    return data

def load_current_data(shot, path_to_current_data):
    path_to_current_data = Path(path_to_current_data)
    path_to_current_csv = path_to_current_data / f'{shot}_voltage_and_rvsout.csv'
    return pd.read_csv(path_to_current_csv).apply(pd.to_numeric, errors='coerce')

def load_pebble_rod_details(shot, xlsx=PEBBLE_ROD_DETAILS_XLS) -> Dict[str, float]:
    """
    Return a pandas dataframe with the details of the pebble rod in the given shot.

    Parameters
    ----------
    shot: int, float
        The shot number.
    xlsx: str, Path
        Path to the excel file containing the pebble rod details.

    Returns
    -------
    Dict[str, float]:
        Dictionary with the details of the pebble rod in the given shot.
    """
    details_df = pd.read_excel(xlsx, sheet_name='pebble rod details')
    shots_df = pd.read_excel(xlsx, sheet_name='shots')

    df = pd.merge(details_df, shots_df, on=['sample id'], how='left')
    df = df[df['shot'] == shot].reset_index(drop=True)
    columns = df.columns.tolist()
    details = {}
    for column in columns:
        details[column] = df.loc[0, column]

    return details

def remove_spikes_zscore(spectrum, threshold=3, window_size=5):
    """
    Remove spikes using Z-score method with local statistics.

    Parameters:
    -----------
    spectrum : array-like
        Input spectrum/signal
    threshold : float, default=3
        Z-score threshold above which points are considered spikes
    window_size : int, default=5
        Size of the local window for calculating statistics

    Returns:
    --------
    cleaned_spectrum : ndarray
        Spectrum with spikes removed
    spike_mask : ndarray
        Boolean array indicating spike locations
    """
    spectrum = np.array(spectrum)
    cleaned_spectrum = spectrum.copy()

    # Calculate local median and MAD (Median Absolute Deviation)
    local_median = ndimage.median_filter(spectrum, size=window_size)
    mad = ndimage.median_filter(np.abs(spectrum - local_median), size=window_size)

    # Calculate modified Z-score using MAD
    with np.errstate(divide='ignore', invalid='ignore'):
        modified_z_score = 0.6745 * (spectrum - local_median) / mad

    # Identify spikes
    spike_mask = np.abs(modified_z_score) > threshold

    # Replace spikes with local median
    cleaned_spectrum[spike_mask] = local_median[spike_mask]

    return cleaned_spectrum, spike_mask

def main(shots, path_to_current_data, path_to_lp_data, sputtering_yield, pebble_rod_details_xls):
    load_plot_style()
    fig, axes = plt.subplots(nrows=len(shots), ncols=1, constrained_layout=True, sharex=True)
    fig.set_size_inches(4., 7)

    fig_h, axes_h  = plt.subplots(nrows=len(shots), ncols=1, constrained_layout=True, sharex=True)
    fig_h.set_size_inches(4., 7)

    axes[-1].set_xlabel('Time (s)')
    fig.supylabel('$\Gamma_{\mathregular{D}}$ (D/cm$^{\mathregular{2}}$/s)')

    axes_h[-1].set_xlabel('Time (s)')
    fig_h.set_size_inches(4., 6)

    fig_sb, axes_sb = plt.subplots(nrows=len(shots), ncols=1, sharex=True, constrained_layout=True)
    fig_sb.set_size_inches(4., 7)
    fig_sb.suptitle('Sputtered boron (atoms/s)')



    for i, shot in enumerate(shots):
        dro_df = load_current_data(shot, path_to_current_data)
        lp_data = load_lp_data(shot, path_to_lp_data)

        time_dro1 = dro_df['t_ms'].values
        current_dro1 = dro_df['current_rcsmooth'].values

        # Assume a flat baseline corresponding to the data from 0 to 600 ms

        msk_baseline = time_dro1 <= 600
        # print(f'Current: {current[msk_baseline]}')
        baseline_fitter = Baseline(x_data=time_dro1[msk_baseline])
        bkgd_1, params_1 = baseline_fitter.modpoly(current_dro1[msk_baseline], poly_order=1)

        current_baselined = current_dro1 - bkgd_1.mean()

        time_lp = lp_data['time_ms']
        n_e = lp_data['n_e']
        T_e = lp_data['T_e']
        qpara = lp_data['qpara']

        msk = (n_e > 0) & (T_e > 0)
        time_lp = time_lp[msk]
        n_e = n_e[msk]
        T_e = T_e[msk]
        qpara = qpara[msk]

        T_e_despiked, _ = remove_spikes_zscore(spectrum=T_e, threshold=1, window_size=50)
        spl_T_e = make_smoothing_spline(x=time_lp, y=T_e_despiked, lam=None)

        n_e_despiked, _ = remove_spikes_zscore(spectrum=n_e, threshold=1, window_size=50)
        spl_n_e = make_smoothing_spline(x=time_lp, y=n_e_despiked, lam=None)

        pebble_rod_details = load_pebble_rod_details(shot)
        diameter, diameter_delta = pebble_rod_details['diameter (cm)'], pebble_rod_details['diameter error (cm)']
        area = 0.25 * np.pi * diameter ** 2

        flux_jsat, j_sat = deuterium_flux_from_jsat(current_baselined, area)
        flux_lp, c_s, heat_flux_e, heat_flux_i = calculate_divertor_deuterium_flux(n_e=spl_n_e(time_lp), T_e=spl_T_e(time_lp), sheath_coefficient=0.61)

        sputtering_B_jsat = flux_jsat * sputtering_yield
        sputtering_B_lp = flux_lp * sputtering_yield

        cs_flux = CubicSpline(time_lp, flux_lp)
        cs_heat_flux = CubicSpline(time_lp, qpara)
        cs_heat_flux_2 = CubicSpline(time_lp, heat_flux_e + heat_flux_i)
        axes[i].plot(time_dro1, cs_flux(time_dro1), label='From LP', color='C0')
        axes[i].plot(time_dro1, flux_jsat, label='From Jsat', color='C1')

        cs_sputtering_B_lp = CubicSpline(time_lp, sputtering_B_lp)

        axes[i].set_title(f'Shot #{shot}')
        axes_h[i].set_title(f'Shot #{shot}')
        axes_sb[i].set_title(f'Shot #{shot}')


        axes_h[i].plot(time_dro1, cs_heat_flux(time_dro1), color='C0', label='Qpara')
        axes_h[i].plot(time_dro1, cs_heat_flux_2(time_dro1), color='C1', label='From LP')
        axes[i].legend(loc='upper left')
        axes_h[i].legend(loc='upper left')

        axes_sb[i].plot(time_dro1, cs_flux(time_dro1)*sputtering_yield, color='C0', label='From LP')
        axes_sb[i].plot(time_dro1, flux_jsat * sputtering_yield, color='C2', label='From Jsat')

        output_dir = Path(r'./data')
        output_dir.mkdir(parents=True, exist_ok=True)

        path_to_d_flux_csv = output_dir / f'{shot}_d_flux.csv'
        outout_df = pd.DataFrame(data={
            'time_ms': time_dro1,
            'D_flux (D/cm^2/s)': cs_flux(time_dro1),
            'B_sputtered_jsat (B/cm^2/s)': sputtering_B_jsat,
            'B_sputtering_lp (B/cm^2/s)': cs_sputtering_B_lp(time_dro1)
        })

        outout_df.to_csv(path_to_d_flux_csv, index=False)

    path_to_figures = Path(r'./figures')
    path_to_figures.mkdir(parents=True, exist_ok=True)
    path_to_d_flux_figure = path_to_figures / f'{shot}_d_flux_figure.png'
    path_to_heatflux_figure = path_to_figures / f'{shot}_heatflux_figure.png'
    fig.savefig(path_to_d_flux_figure, dpi=600)
    fig_h.savefig(path_to_heatflux_figure, dpi=600)

    plt.show()





# Example usage
if __name__ == "__main__":
    main(
        shots=SHOTS, path_to_lp_data=PATH_TO_LP_DATA, path_to_current_data=PATH_TO_CURRENT_DATA,
        pebble_rod_details_xls=PEBBLE_ROD_DETAILS_XLS, sputtering_yield=SPUTTERING_YIELD
    )
    # # Example parameters
    # I_sat = 0.1  # Saturation current in A
    # area = 10.0  # Target area in cm^2
    # T_e = 20.0  # Electron temperature in eV
    #
    # # Compare different coefficients
    # for coeff in [0.5, 0.61]:
    #     flux, j_sat, c_s, n_e = calculate_deuterium_flux(I_sat, area, T_e, coeff)
    #
    #     print(f"\n{'=' * 50}")
    #     print(f"Sheath coefficient: {coeff}")
    #     print(f"{'=' * 50}")
    #     print(f"  Current density: {j_sat:.4f} A/cm^2")
    #     print(f"  Sound speed: {c_s:.2e} m/s")
    #     print(f"  Electron density: {n_e:.2e} m^-3")
    #     print(f"  Deuterium flux: {flux:.2e} particles/(cm^2*s)")