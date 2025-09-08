import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_processing.misc_utils.plot_style import load_plot_style
from scipy import integrate
from scipy.optimize import least_squares, OptimizeResult
from pathlib import Path
import data_processing.confidence as cf
from data_processing.utils import latex_float_with_error
from data_processing.misc_utils.latex_value import format_value_uncertainty_latex

DATA_DIR = r'./data/integrated_sweep'
FIGURES_DIR = r'./figures/integrated_current_full_sweep'
SHOTS = [203780, 203781]
THETA_BEAM = 1.5 # DEG

PEBBLE_ROD_DETAILS_XLS = r'./pebble_rod_exposure.xlsx'


def get_pebble_rod_details(shot, xls=PEBBLE_ROD_DETAILS_XLS):
    details_df = pd.read_excel(xls, sheet_name='pebble rod details')
    shots_df = pd.read_excel(xls, sheet_name='shots')

    df = pd.merge(details_df, shots_df, on=['sample id'], how='left')
    df = df[df['shot'] == shot].reset_index(drop=True)
    return df

def calculate_ion_beam_interaction_area(diameter, height, theta_degrees):
    """
    Calculate the effective area of ion beam interaction with a vertically aligned cylinder.
    This represents the physical cross-sectional area that intercepts the ion beam.

    Parameters:
    diameter (float): Diameter of the cylinder
    height (float): Height of the cylinder
    theta_degrees (float): Angle in degrees between cylinder axis and ion beam direction

    Returns:
    tuple: (total_interaction_area, top_face_area, curved_surface_area)
    """
    # Convert angle to radians
    theta = np.radians(theta_degrees)

    # Radius of cylinder
    radius = diameter / 2

    # Calculate ion beam interaction areas

    # 1. Top circular face - flux-weighted effective area
    # When beam hits at angle theta, effective area = original_area * cos(theta)
    # This is also equal to the geometric projection of the circle
    top_face_area = np.pi * radius ** 2 * abs(np.cos(theta))

    # 2. Curved cylindrical surface - only the half facing the beam
    # The curved surface has a "frontal cross-section" visible to the beam
    # This is NOT the full circumference, but only the portion facing the beam
    # Effective area = (π * radius) * height * sin(theta)
    # where π * radius is the "frontal width" (half circumference)
    curved_surface_area = np.pi * radius * height * abs(np.sin(theta))

    # 3. Bottom face is completely shadowed - no contribution
    bottom_face_area = 0

    # Total ion beam interaction area
    total_interaction_area = top_face_area + curved_surface_area

    return total_interaction_area, top_face_area, curved_surface_area

def compute_charge(time, current, min_points_simpson=5):
    """
    Compute charge q = ∫I(t)dt from 0 to t using adaptive integration.

    Parameters:
    -----------
    time : array_like
        Time points (must be sorted)
    current : array_like
        Current values I(t) at each time point
    min_points_simpson : int, optional
        Minimum number of points required for Simpson's rule (default=5)

    Returns:
    --------
    charge : ndarray
        Cumulative charge at each time point
    """
    time = np.asarray(time)
    current = np.asarray(current)

    if len(time) != len(current):
        raise ValueError("Time and current arrays must have the same length")

    if len(time) < 2:
        raise ValueError("Need at least 2 data points for integration")

    # Initialize charge array
    charge = np.zeros_like(time)

    # For the first few points, use trapezoidal rule
    for i in range(1, min(min_points_simpson, len(time))):
        # Cumulative trapezoidal integration from start to current point
        charge[i] = integrate.trapezoid(current[:i + 1], time[:i + 1])

    # For remaining points with sufficient data, use Simpson's rule
    for i in range(min_points_simpson, len(time)):
        # Use Simpson's rule on the entire interval from start to current point
        # Simpson requires odd number of intervals (even number of points)
        n_points = i + 1

        if n_points % 2 == 1:  # Odd number of points (even intervals) - perfect for Simpson
            charge[i] = integrate.simpson(current[:n_points], time[:n_points])
        else:  # Even number of points - use Simpson on n-1 points + trapezoid for last interval
            charge[i] = (integrate.simpson(current[:n_points - 1], time[:n_points - 1]) +
                         integrate.trapezoid(current[n_points - 2:n_points], time[n_points - 2:n_points]))

    return charge

def model(t ,b, diameter, theta_degrees, h0, nu):
    theta = np.radians(theta_degrees)
    b0, b1 = b
    r = diameter / 2
    qi = (b0 * np.pi * (r ** 2) * abs(np.cos(theta)) * t
          + b1 *  np.pi * r * abs(np.sin(theta)) * h0  * t
          - b1 *  np.pi * r * abs(np.sin(theta)) * (nu / 2) * t ** 2)
    return qi

def residual(b, t, q, diameter, theta_degrees, h0, nu):
    q_model = model(t, b, diameter, theta_degrees, h0, nu)
    return q_model - q

def jacobian(b, t, q, diameter, theta_degrees, h0, nu):
    theta = np.radians(theta_degrees)
    m, n = len(t), len(b)
    jac = np.zeros((m, n))
    r = diameter / 2

    jac[:, 0] =  np.pi * (r ** 2) * abs(np.cos(theta)) * t
    jac[:, 1] =  np.pi * r * abs(np.sin(theta)) * (h0 * t - nu * 0.5 *  t ** 2 )

    return jac

def fit_charge(t_s, current, diameter, theta_degrees, h0, nu, x0=(1E-3, 1E-3), loss='soft_l1', f_scale=0.1):
    eps = np.finfo(np.float64).eps
    tol = eps
    # tol = 1E-3
    fit_result = least_squares(
        residual, x0=x0, args=(t_s, current, diameter, theta_degrees, h0, nu), loss=loss, f_scale=f_scale, #jac='3-point',
        max_nfev=100000,  xtol=tol, gtol=tol, ftol=tol, verbose=2, jac=jacobian,
    )

    return fit_result

def round_for_lim(value, factor):
    if value < 0:
        return np.floor(value * factor) / factor
    return np.ceil(value * factor) / factor


def main(shots, data_dir, fig_dir):
    t_s = np.array([])
    qi = np.array([])
    path_to_data = Path(data_dir)
    t0, q0 = 0, 0
    pebble_rod_details = get_pebble_rod_details(shots[0])
    diameter = pebble_rod_details.loc[0, 'diameter (cm)']
    diameter_error = pebble_rod_details.loc[0, 'diameter error (cm)']
    protrusion = pebble_rod_details.loc[0, 'protrusion (mm)']
    experiment_id = pebble_rod_details.loc[0, 'experiment id']
    density = pebble_rod_details.loc[0, 'density (g/cm3)']
    density_error = pebble_rod_details.loc[0, 'density error (g/cm3)']
    mass_loss = pebble_rod_details.loc[0, 'mass loss (g)']
    mass_loss_error = pebble_rod_details.loc[0, 'mass loss error (g)']

    shot_end_time = []


    for shot in shots:
        q_df = pd.read_csv(path_to_data / f'{shot}_integrated_current.csv', comment='#').apply(pd.to_numeric)
        t_shot =  q_df['t (s)'].values + t0
        q_shot = q_df['q (C)'].values + q0
        t_s = np.hstack((t_s,t_shot))
        qi = np.hstack([qi, q_shot])
        t0, q0 = t_shot.max(), q_shot.max()
        shot_end_time.append(t0)

    t_fit = t_s[::1]
    qi_fit = qi[::1]
    shot_end_time = np.array(shot_end_time)[:-1]

    nu = protrusion / (t_fit.max() - t_fit.min()) * 0.1
    nu_error = np.abs(nu) * np.abs(pebble_rod_details.loc[0, 'protrusion error (mm)'] / protrusion)

    # nu = 4*mass_loss / density / np.pi / diameter ** 2 /  (t_fit.max() - t_fit.min())
    # nu_error = nu * np.linalg.norm([mass_loss_error/mass_loss, 2*diameter_error/diameter, density_error/density])


    fit_result : OptimizeResult = fit_charge(
        t_s=t_fit, current=qi_fit, diameter=diameter, theta_degrees=(90 - THETA_BEAM), h0=protrusion*0.1, nu=nu,
        loss='soft_l1', f_scale=0.1
    )
    popt = fit_result.x
    ci = cf.confidence_interval(res=fit_result)
    popt_delta = np.abs(popt - ci[:, 0])

    for i, popt_i in enumerate(popt):
        print(f'popt[{i}]: {popt_i:.4E} -/+ {popt_delta[i]:.5E}')

    def model_constrained(t_model, b):
        m = model(t_model, b, diameter, 90 - THETA_BEAM, protrusion*0.1, nu)
        return m

    def jac_constrained(b, t, q):
        return jacobian(b, t, q, diameter, 90 - THETA_BEAM, protrusion*0.1, nu)

    q_pred, delta = cf.prediction_intervals(
        x_pred=t_fit, model=model_constrained, ls_res=fit_result, new_observation=True,
        jac=jac_constrained
    )



    print(f'Delta h = {protrusion:.2f} mm')
    print(f't_max = {t_s.max():.4E} s')

    load_plot_style()
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.0)

    ax.plot(t_fit, qi_fit, label='Data')
    ax.plot(t_fit, q_pred, color='r', label='Model')
    ax.fill_between(t_fit, (q_pred-delta), (q_pred+delta), color='r', alpha=0.1)

    # for t_end_shot in shot_end_time:
    #     print(f't_end = {t_end_shot:.4E} s')
    #     ax.axvline(x=t_end_shot, color='k', linestyle='--', alpha=0.5)

    ax.text(
        0.05, 0.975, r'\begin{equation*}q(t) =\int_0^t  \left[ b_0 \pi r^2 \| \cos \theta \|  +  b_1 \pi r \| \sin\theta \|  (h_0 -\nu t) \right]dt\end{equation*}',
        ha='left', va='top', transform=ax.transAxes,
        fontsize=11, usetex=True
    )

    rate_txt = (f'$\Delta h = \\nu t, \\nu = {format_value_uncertainty_latex(nu, nu_error)}$ cm/s\n'
                f'$b_0 = {format_value_uncertainty_latex(popt[0], popt_delta[0])}$\n'
                f'$b_1 = {format_value_uncertainty_latex(popt[1], popt_delta[1])}$\n'
                f'$\\theta = {90-THETA_BEAM:.1f}^{{\circ}}$')
    ax.text(
        0.5, 0.2, rate_txt,
        ha='left', va='bottom', transform=ax.transAxes,
        fontsize=10, usetex=True
    )

    ax.set_xlabel('Time (s)', usetex=False)
    ax.set_ylabel('$q(t) = \int_0^t I dt$ (C)', usetex=True)
    ax.set_xlim(round_for_lim(t_s.min(), factor=5), round_for_lim(t_s.max(), factor=5))
    ax.set_ylim(round_for_lim(qi.min(), factor=5), round_for_lim(qi.max()*1.2, factor=5))
    ax.set_title(rf'L-mode charge (sample {experiment_id})')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)

    ax.legend(loc='lower right', frameon=True)
    path_to_fig = Path(fig_dir)
    path_to_fig.mkdir(parents=True, exist_ok=True)

    fig.savefig(path_to_fig / f'L-mode_current_#{experiment_id}.png', dpi=600)

    # Save fit results
    path_to_fit_results_folder = path_to_data / 'fit_results'
    path_to_fit_results_folder.mkdir(parents=True, exist_ok=True)
    path_tof_fit_results_file = path_to_fit_results_folder / f'L-mode_current_#{experiment_id}.txt'

    with open(path_tof_fit_results_file, 'w') as f:
        f.write(r'#'*40)
        f.write("\n")
        f.write(f"# L-mode charge\n")
        f.write(f"# Experiment id: {experiment_id}\n")
        f.write(f'# theta: {90 - THETA_BEAM:.1f} deg\n')
        f.write(f'# Shots in experiment:\n')
        for shot in shots:
            f.write(f'# \tShot #: {shot}\n')
        f.write(r'#' * 40)
        f.write("\n")
        for i in range(len(popt)):
            f.write(f'b_{i} = {popt[i]:.4E} -/+ {popt_delta[i]:.4E} C/cm^2/s\n')
        f.write(r'#' * 40)
        f.write("\n")

    plt.show()

if __name__ == '__main__':
    main(shots=SHOTS, data_dir=DATA_DIR, fig_dir=FIGURES_DIR)


