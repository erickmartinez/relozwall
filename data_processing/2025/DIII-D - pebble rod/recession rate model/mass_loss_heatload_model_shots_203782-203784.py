import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate as integrate
from scipy.integrate import simpson, trapezoid
from scipy.interpolate import make_smoothing_spline, CubicSpline
from scipy import ndimage
from pathlib import Path
from pybaselines import Baseline
from data_processing.misc_utils.plot_style import load_plot_style
from data_processing.utils import latex_float
from typing import Dict
import h5py

BORON_MOLAR_MASS = 10.811 # g / mol
PEBBLE_ROD_DETAILS_XLS = r'./pebble_rod_exposure.xlsx'

LP_FOLDER = r'../Langmuir Probe/data/dimes_lp'
CURRENT_FOLDER = r'./data'
FIGURES_FOLDER = r'./figures'

SHOTS = [203782, 203783, 203784]
T_RANGE = [1500, 3500]
TAU = 10 # Time constant for the RC filter (ms)
PLASMA_ANGLE = 1.5 # deg
BORON_MOLAR_MASS = 10.811 # g / mol


def load_lp_data(shot, path_to_folder):
    path_to_folder = Path(path_to_folder)
    with h5py.File( path_to_folder / f'{shot}_LP.h5', 'r') as h5:
        dimes_gp = h5['/LANGMUIR_DIMES']
        t_ms = np.array(dimes_gp.get('time')) * 1E-3
        T_eV = np.array(dimes_gp.get('TeV'))
    return t_ms, T_eV

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

def rc_smooth(x: np.ndarray, y:np.ndarray, tau: float = 0.1) -> np.ndarray:
    """
    RC filter smoothing (simulates hardware RC low-pass filter)

    Parameters
    ----------
    x: np.ndarray
        Input signal x
    y: np.ndarray
        Input signal y
    tau: float
        Time constant

    Returns
    -------
    np.ndarray
        Filtered signal
    """

    # Average time step
    dt = np.mean(np.diff(x))

    alpha = dt / (tau + dt) # Smoothing factor

    y_smooth = np.zeros_like(y)
    y_smooth[0] = y[0]

    for i in range(1, len(x)):
        y_smooth[i] = alpha * y[i] + (1 - alpha) * y_smooth[i - 1]

    return y_smooth


def load_current_data(shot, data_dir=CURRENT_FOLDER):
    path_to_data = Path(data_dir) / f'{shot}_voltage_and_rvsout.csv'
    df = pd.read_csv(path_to_data).apply(pd.to_numeric, errors='coerce')
    return df

def solve_for_h(t_s, current, T_e, theta, r, mu, mu_delta, h0, rho):
    # Group constants
    a = 0.5 * np.pi * r * np.sin(theta)
    b = np.pi * (r ** 2) * np.cos(theta)
    h1 = np.zeros_like(t_s)
    h2 = np.zeros_like(t_s)
    h_delta = np.zeros_like(t_s)
    for i in range(0, len(t_s)):
        t_int = t_s[0:i]
        c_int = current[0:i]
        Te_int = T_e[0:i]
        # int_i = integral_at_t(t_int, c_int*Te_int)
        if i < 5:
            int_i = trapezoid(y=c_int*Te_int, x=t_int)
        else:
            int_i = integrate.simpson(y=c_int*Te_int, x=t_int)


        d = int_i * mu / (rho * np.pi * r ** 2)
        c = -(np.pi * (r ** 2) * np.cos(theta) * h0 + 0.5 * np.pi * r * np.sin(theta) * h0 ** 2 + d)
        h1[i] = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
        h2[i] = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        h_delta[i] = 2 * mu_delta / np.sqrt(b ** 2 - 4 * a * c)
    return h1, h2, h_delta


def main(
    shots, lp_folder, t_range, tau=10, plasma_angle=PLASMA_ANGLE, current_folder=CURRENT_FOLDER,
    path_to_figures=FIGURES_FOLDER, boron_molar_mass=BORON_MOLAR_MASS
):
    load_plot_style()
    fig, axes = plt.subplots(nrows=3, ncols=3, constrained_layout=True, sharex=False)
    fig.set_size_inches(8, 6)
    t0 = 0
    total_int = 0
    t_shots = []
    T_eV_shots = []
    current_shots = []
    t0_shots = []
    for i, shot in enumerate(shots):
        print(f'Loading shot {shot}...')
        t0_shots.append(t0)
        t_left, t_right = 1E-3*t_range[0], 1E-3*t_range[1]
        t_lp, T_eV = load_lp_data(shot, lp_folder)
        msk_time = (t_left <= t_lp) & (t_lp <= t_right)
        t_lp = t_lp[msk_time] #+ t0
        T_eV = T_eV[msk_time]
        T_eV_despiked, _ = remove_spikes_zscore(spectrum=T_eV, threshold=5, window_size=50)
        spl_TeV = make_smoothing_spline(x=t_lp, y=T_eV_despiked, lam=None)
        T_eV_smooth = spl_TeV(t_lp)

        T_eV_shots.append(T_eV_smooth)

        current_df = load_current_data(shot)
        t_current = (current_df['t_ms'].values) * 1E-3  # + t0
        current = current_df['current'].values

        # Assume a flat baseline corresponding to the data from 0 to 600 ms

        msk_baseline = t_current <= 0.6
        # print(f'Current: {current[msk_baseline]}')
        baseline_fitter = Baseline(x_data=t_current[msk_baseline])
        bkgd_1, params_1 = baseline_fitter.modpoly(current[msk_baseline], poly_order=1)

        current_df = current_df[current_df['t_ms'].between(t_range[0], t_range[1])]
        t_current = 1E-3 * current_df['t_ms'].values
        current = current_df['current'].values

        current_rcsmooth = rc_smooth(t_current, current, tau*1E-3)
        current_baselined = current_rcsmooth - bkgd_1.mean()

        cs_current = CubicSpline(t_current, current_baselined)
        current_interp = cs_current(t_lp)

        current_shots.append(current_interp)
        t_shots.append(t_lp)


        I_x_TeV = current_interp * T_eV_despiked
        int_i_TeV = simpson(x=t_lp, y=I_x_TeV)
        total_int += int_i_TeV

        axes[0,i].plot(t_lp+t0, current_interp, c='C0', label=f'dro1')
        axes[1,i].plot(t_lp+t0, T_eV, c='C1', label=f'LP')
        axes[1, i].plot(t_lp+t0, T_eV_smooth, c='tab:red', label=f'Smoothened')
        axes[2,i].plot(t_lp+t0, I_x_TeV, c='C2', label=f'Product')

        text_int = r'\begin{equation*}'
        text_int += f'\int_{{t={t_lp[0]+t0:.1f}}}^{{{t_lp[-1]+t0:.1f}}} I(t)T_e(t) dt = {latex_float(int_i_TeV)} '
        text_int += r'\end{equation*}'
        if i == 0:
            axes[0,i].set_title(f'Shot #{shot}')
        axes[2, i].text(
            0.025, 0.975, text_int,
            ha='left',
            va='top',
            transform=axes[2, i].transAxes,
            fontsize=10,
            usetex=True
        )
        t0 = t0 + t_lp.max() - t_lp.min()

    for ax in axes.flatten():
        # ax.set_xlim(0, t_range[1]*1E-3)
        ax.legend(loc='upper right', fontsize=9)
    fig.supxlabel('Time (s)')

    for ax in axes[0,:]:
        ax.set_ylim(0, 6)
    for ax in axes[1,:]:
        ax.set_ylim(0, 80)
    for ax in axes[2,:]:
        ax.set_ylim(0, 300)

    for ax in axes[:, 0]:
        ax.set_xlim(1.5, 3.5)
    for ax in axes[:, 1]:
        ax.set_xlim(3.5, 5.5)

    axes[0, 0].set_ylabel('I (A)')
    axes[1, 0].set_ylabel(r'{\sffamily T\textsubscript{e} (eV)}', usetex=True)
    axes[2, 0].set_ylabel(r'{\sffamily I x T\textsubscript{e} (eV$\cdot$C/s)}', usetex=True)

    fig_h, axes_h = plt.subplots(nrows=3, ncols=3, constrained_layout=True, sharex=True)
    fig_h.set_size_inches(8, 6.5)

    N_A = 6.02214076e+23
    N_A = 6.02214076E1

    def grams_per_second_to_atoms_per_second(mass_g):
        return mass_g / boron_molar_mass * N_A

    def atoms_per_second_to_grams_per_second(atoms):
        return atoms / N_A * boron_molar_mass


    # Estimate the constant for the model
    pebble_rod_details = load_pebble_rod_details(shots[0])
    rho, rho_delta = pebble_rod_details['density (g/cm3)'], pebble_rod_details['density error (g/cm3)']
    h0 = 0.1 * pebble_rod_details['protrusion (mm)']
    h0_error = 0.01 # 0.1 * pebble_rod_details['protrusion error (mm)']
    theta = np.pi / 2 - np.radians(plasma_angle)
    diameter, diameter_delta = pebble_rod_details['diameter (cm)'], pebble_rod_details['diameter error (cm)']
    mass_loss, mass_loss_error = pebble_rod_details['mass loss (g)'], pebble_rod_details['mass loss error (g)']
    kappa = mass_loss / total_int
    kappa_delta = mass_loss_error / total_int


    h0_i = h0
    h0_i_error = h0_error
    m_loss_i = 0
    total_mass_loss = 0
    total_height_loss = 0
    total_height_loss_error = 0

    for i, shot in enumerate(shots):
        current_shot = current_shots[i]
        t_shot = t_shots[i]
        T_ev_shot = T_eV_shots[i]
        t0 = t0_shots[i]

        dm_dt = kappa * current_shot * T_ev_shot
        dm_dt_error = kappa_delta * current_shot * T_ev_shot
        mass_loss_shot = simpson(y=dm_dt, x=t_shot)
        total_mass_loss += mass_loss_shot
        mass_loss_txt = f'Mass loss: {mass_loss_shot:.3f} ± {np.abs(kappa_delta/kappa)*mass_loss_shot:.3f} g'
        axes_h[0,i].text(
            0.975, 0.975, mass_loss_txt,
            ha='right', va='top', transform=axes_h[0,i].transAxes,
            fontsize=10,
        )

        h = np.full_like(dm_dt, fill_value=h0)
        mass_loss_at_time = np.zeros_like(dm_dt)
        for j in range(len(dm_dt)):
            if j < 5:
                mass_loss_at_time[j] = trapezoid(y=dm_dt[0:j], x=t_shot[0:j])
                h[j] = h0_i - 4 * mass_loss_at_time[j]/(rho * diameter ** 2)
            else:
                mass_loss_at_time[j] = simpson(y=dm_dt[0:j], x=t_shot[0:j])
            h[j]= h0_i - 4 * mass_loss_at_time[j]/(rho * diameter ** 2)

        mass_loss_at_time += m_loss_i
        m_loss_i = mass_loss_at_time[-1]
        ones_h = np.ones_like(h)
        h_delta = h * np.linalg.norm(
            [kappa_delta / kappa * ones_h, diameter_delta / diameter * ones_h, rho_delta / rho * ones_h,
             h0_i_error/h], axis=0
        )
        m_delta = mass_loss_at_time * np.abs(kappa_delta / kappa)
        h0_i = h[-1]
        h0_i_error = h_delta[-1]
        height_loss = h[0] - h[-1]
        height_loss_error = np.linalg.norm(h_delta[0] + h_delta[-1])
        total_height_loss += height_loss
        total_height_loss_error += height_loss_error**2
        height_loss_txt = f'Height loss: {height_loss*10:.1} ± {height_loss_error*10:.2f} mm'

        axes_h[2, i].text(
            0.025, 0.025, height_loss_txt,
            ha='left', va='bottom', transform=axes_h[2, i].transAxes,
            fontsize=10,
        )


        axes_h[0, i].plot(t_shot, dm_dt, c='C0', label=f'Shot #{shot}')
        axes_h[0, i].fill_between(t_shot+t0, (dm_dt - dm_dt_error), (dm_dt + dm_dt_error), color='C0', alpha=0.2)

        axes_h[0, i].set_xlabel('Time (s)')
        axes_h[0, i].set_ylabel('dm/dt (g/s)')
        axes_h[0, i].set_xlim(t_shot.min(), t_shot.max())

        # axes_h[0, i].legend(loc='upper right')
        axes_h[0, i].set_title(f'Shot #{shot}')

        axes_h[1, i].plot(t_shot, mass_loss_at_time, c='C3', label=f'Shot #{shot}')
        axes_h[1, i].fill_between(t_shot, (mass_loss_at_time - m_delta), (mass_loss_at_time + m_delta), color='C3', alpha=0.2)
        axes_h[1, i].set_xlabel('Time (s)')
        axes_h[1, i].set_ylabel('{\sffamily $\Delta$m(t) (g)}', usetex=True)
        axes_h[1, i].set_ylim(bottom=0, top=0.5)

        axes_h[2, i].plot(t_shot, h*10, c='C4', label=f'Shot #{shot}')
        axes_h[2, i].fill_between(t_shot, (h-h_delta)*10, (h+h_delta)*10, color='C4', alpha=0.2)
        axes_h[2, i].set_xlabel('Time (s)')
        axes_h[2, i].set_ylabel('h(t) (mm)')
        axes_h[2, i].set_ylim(0, 6)
        # axes_h[1, i].set_title(f'Shot #{shot}')

    ax_g1 = axes_h[0, 0].secondary_yaxis('right', functions=(
        grams_per_second_to_atoms_per_second, atoms_per_second_to_grams_per_second
    ))

    ax_g2 = axes_h[0, 1].secondary_yaxis('right', functions=(
        grams_per_second_to_atoms_per_second, atoms_per_second_to_grams_per_second
    ))

    for ax in [ax_g1, ax_g2]:
        ax.set_ylabel(r'{\sffamily x10\textsuperscript{22} B atoms/s}', usetex=True)

    for i in range(2):
        axes_h[0, i].set_ylim(0, 1.2)

    path_to_figures = Path(path_to_figures) / 'heatload_model_v2'
    path_to_figures.mkdir(parents=True, exist_ok=True)
    path_to_fig1 = path_to_figures / 'current_and_Te.png'
    path_to_fig2 = path_to_figures / 'height_model.png'

    fig.savefig(path_to_fig1, dpi=600)
    fig_h.savefig(path_to_fig2, dpi=600)

    # Save the model data
    path_to_model_results_folder = Path(current_folder) / 'model_results'
    path_to_model_results_folder.mkdir(parents=True, exist_ok=True)
    shots_txt = f'{shots[0]}-{shots[-1]}'
    path_to_model_folder = path_to_model_results_folder / f'{shots_txt}_mass_loss_model.h5'
    with h5py.File(str(path_to_model_folder), 'w') as f:
        for i, shot in enumerate(shots):
            current_shot = current_shots[i]
            t_shot = t_shots[i]
            T_ev_shot = T_eV_shots[i]
            dm_dt = grams_per_second_to_atoms_per_second(kappa * current_shot * T_ev_shot)*1E22
            dm_dt_error = grams_per_second_to_atoms_per_second(kappa_delta * current_shot * T_ev_shot)*1E22
            shot_group = f.create_group(str(shot))
            time_ds = shot_group.create_dataset('time', data=t_shots[i])
            time_ds.attrs['units'] = 's'
            mass_loss_rate_ds = shot_group.create_dataset('mass_loss_rate', data=dm_dt, compression='gzip')
            mass_loss_rate_error_ds = shot_group.create_dataset('mass_loss_error', data=dm_dt_error, compression='gzip')
            mass_loss_rate_ds.attrs['units'] = 'atoms/s'


    plt.show()







if __name__ == '__main__':
    main(
        shots=SHOTS, lp_folder=LP_FOLDER, t_range=T_RANGE, tau=TAU, plasma_angle=PLASMA_ANGLE,
        current_folder=CURRENT_FOLDER, path_to_figures=FIGURES_FOLDER, boron_molar_mass=BORON_MOLAR_MASS,
    )