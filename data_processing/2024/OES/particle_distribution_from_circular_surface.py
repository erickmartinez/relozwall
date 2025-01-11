import numpy as np
import pandas as pd
from scipy import interpolate
from n_e_profile_fit import load_plot_style
from scipy.optimize import least_squares, differential_evolution, OptimizeResult


eps = float(np.finfo(np.float64).eps)
def cosn_model(x, params):
    global eps
    y0, n = params
    return y0 * (np.cos(x) + eps) ** n

def residuals_cosn(params, x, y):
    return cosn_model(x, params) - y

def jac_cosn(params, x, y):
    y0, n = params
    xx = np.cos(x) + eps
    dy0 = xx ** n
    dn = y0 * np.log(xx) * xx ** n
    return np.vstack([dy0, dn]).T

def res_cosn_de(params, x, y):
    return 0.5 * np.linalg.norm(residuals_cosn(params, x, y))

def fit_cosn(x, y, loss='linear', f_scale=1.0, tol=eps) -> OptimizeResult:
    p0 = np.array([np.max(y), 1.])
    bounds = ([-20, -20], [20, 20])

    res_de: OptimizeResult = differential_evolution(
        func=res_cosn_de,
        args=(x, y),
        x0=p0,
        bounds=[(-20, 20), (-20, 20)],
        maxiter=10000 * len(p0),
        tol=tol,
        atol=tol,
        workers=-1,
        updating='deferred',
        recombination=0.5,
        strategy='best1bin',
        mutation=(0.5, 1.5),
        init='sobol',
        polish=False,
        disp=True
    )

    result = least_squares(
        residuals_cosn,
        x0=res_de.x,
        jac=jac_cosn,
        bounds=bounds,
        args=(x, y),
        method='trf',
        loss=loss,
        f_scale=f_scale,
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=2,
        x_scale='jac',
        max_nfev=10000 * len(p0)
    )

    return result

def load_angle_distribution(filename):
    """
    Load and process the angle distribution from CSV file
    """
    df = pd.read_csv(filename)
    angles = df['angle (rad)'].values
    counts = df['n particles'].values


    # Create finer grid
    fine_angles = np.linspace(angles.min(), angles.max(), 1000)

    # Fit a cosn law for the angles
    fit_result_cosn: OptimizeResult = fit_cosn(x=angles, y=counts / np.sum(counts))

    # Linear interpolation of PDF
    # pdf_interp = interpolate.interp1d(
    #     angles, counts, kind='linear', bounds_error=False, fill_value=(counts[0], counts[-1]),
    # )
    pdf_interp = lambda x: cosn_model(x, fit_result_cosn.x)
    fine_particles = pdf_interp(fine_angles)
    fine_particles = np.maximum(fine_particles, 0)

    # Direct normalization
    norm_fine_particles = fine_particles / (np.sum(fine_particles) * np.abs(np.diff(fine_angles)[0]))
    # norm_fine_particles = fine_particles / np.trapz(fine_particles, fine_angles)

    # Create CDF with exact bounds
    # cdf = np.concatenate(([0], np.cumsum(norm_fine_particles) * np.diff(fine_angles)[0]))
    cdf = np.concatenate(([0], np.cumsum(norm_fine_particles) * np.abs(np.diff(fine_angles)[0])))
    cdf = cdf / cdf[-1]
    fine_angles = np.concatenate(([0.], fine_angles))

    # Generate samples
    n_samples = 1000000
    random_uniform = np.random.uniform(0, 1, n_samples)
    inv_cdf = interpolate.interp1d(cdf, fine_angles)

    return inv_cdf

def generate_particle_distribution(num_particles, radius, max_distance, angle_distribution_file):
    """
    Generate particles emanating from a circular surface following:
    - Cosine law distribution for polar angle
    - 1/r^2 decay from the surface

    Parameters:
    -----------
    num_particles : int
        Number of particles to generate
    radius : float
        Radius of the circular surface
    max_distance : float
        Maximum distance from surface for particle placement
    angle_distribution_file : str
        Path to CSV file containing angle distribution

    Returns:
    --------
    positions : ndarray
        Array of shape (num_particles, 3) containing particle positions
    directions : ndarray
        Array of shape (num_particles, 3) containing particle direction vectors
    """

    # Generate random positions on the circular surface
    # Using rejection sampling for uniform distribution on circle
    surface_positions = np.zeros((num_particles, 2))
    accepted = 0
    while accepted < num_particles:
        # x = np.random.uniform(-radius, radius, 2 * (num_particles - accepted))
        # y = np.random.uniform(-radius, radius, 2 * (num_particles - accepted))
        x = np.random.normal(loc=0, scale=4, size=2 * (num_particles - accepted))
        y = np.random.normal(loc=0, scale=4, size=2 * (num_particles - accepted))
        mask = x * x + y * y <= radius * radius
        remaining = num_particles - accepted
        surface_positions[accepted:accepted + min(remaining, np.sum(mask))] = \
            np.column_stack([x[mask], y[mask]])[:remaining]
        accepted += min(remaining, np.sum(mask))

    # Add z=0 coordinate for the surface
    surface_positions = np.column_stack([surface_positions, np.zeros(num_particles)])

    # Load and sample from the custom angle distribution
    inv_cdf = load_angle_distribution(angle_distribution_file)
    xi = np.random.uniform(0, 1, num_particles)
    theta = inv_cdf(xi)

    # Generate uniform azimuthal angles
    phi = np.random.uniform(0, 2 * np.pi, num_particles)

    # Calculate direction vectors
    directions = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    # Generate distances following 1/r^2 decay
    # Using inverse transform sampling for PDF ∝ 1/r^2
    # CDF: F(r) = 1 - 1/r, r ∈ [1, max_distance]
    # Inverse: r = 1/(1-ξ)
    min_distance = 1.  # Added minimum distance
    xi = np.random.uniform(0, 1, num_particles)
    # distances = 1 / (1 - xi * (1 - 1 / max_distance))
    # distances = 1 / (1 / min_distance - xi * (1 / min_distance - 1 / max_distance))
    distances = -min_distance + 1 / (1 / min_distance - xi * (1 / min_distance - 1 / (max_distance + min_distance)))

    # Calculate final positions
    positions = surface_positions + directions * distances[:, np.newaxis]

    return positions, directions


# Example usage
if __name__ == "__main__":
    # Parameters
    NUM_PARTICLES = 10000
    SURFACE_RADIUS = 0.5
    MAX_DISTANCE = 20.0
    ANGLE_DIST_FILE = r'trimsp_simulations/d_on_b_40keV_polar_angle_dist.csv'

    # n_particles = np.array([0, 5, 16, 22, 29, 25, 41, 49, 54, 65, 63, 72, 63, 70, 82, 87, 97, 73, 92, 90])
    # cos_beta = np.arange(0, 20) * 0.05 + 0.05
    # beta = np.arccos(cos_beta)
    # df = pd.DataFrame(data={'angle (rad)': np.arccos(cos_beta), 'n particles': n_particles})


    # Generate particles
    positions, directions = generate_particle_distribution(
        NUM_PARTICLES, SURFACE_RADIUS, MAX_DISTANCE, ANGLE_DIST_FILE
    )

    # Optional: Visualize using matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    load_plot_style()

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    # Plot particles
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=np.linalg.norm(positions, axis=1),
        cmap='viridis',
        alpha=0.6
    )

    # Plot surface circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x = SURFACE_RADIUS * np.cos(theta)
    y = SURFACE_RADIUS * np.sin(theta)
    ax.plot(x, y, np.zeros_like(x), 'r-', lw=2)

    # Add colorbar
    plt.colorbar(scatter, label='Distance from origin')

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Particle Distribution from Circular Surface')
    ax.set_aspect('equal')

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.show()