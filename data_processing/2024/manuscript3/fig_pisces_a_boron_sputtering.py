import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import least_squares, differential_evolution, OptimizeResult
from scipy import interpolate
from matplotlib.patches import Circle
import matplotlib.image as mpimg
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker


ANGLE_DIST_FILE = r'./data/d_on_b_40keV_polar_angle_dist.csv'
ELECTRON_DENSITY_FIT_FILE = r"./data/probe_data/20240815/lang_results_gamma_ivdata0004_symmetrized_fit.csv"
ELECTRON_DENSITY_FILE = r"./data/probe_data/20240815/lang_results_gamma_ivdata0004_symmetrized.csv"
PARTICLE_DENS_FILE = r"./data/20241224_particle_density.hd5"
PISCES_A_PLASMA_PNG = r"./figures/pisces-a_plasma.png"
ECHELLE_DATA = r"./data/echelle_20240815/MechelleSpect_007.csv"

def load_plot_style():
    with open('./plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')

def load_density_grid(filename):
    """
    Load particle density and grid points from HDF5 file

    Returns
    -------
    tuple
        (density, (grid_x, grid_y, grid_z))
    """
    import h5py

    with h5py.File(filename, 'r') as f:
        density = f['density'][:]
        grid_x = f['grid_x'][:]
        grid_y = f['grid_y'][:]
        grid_z = f['grid_z'][:]

    return density, (grid_x, grid_y, grid_z)


def deg2rad(x):
    return x * np.pi / 180

def rad2deg(x):
    return x * 180 / np.pi

def over_r2(x, params):
    a, b, c = params
    return a / (x + b) ** 2. + c

def residual_or2(params, x, y):
    return over_r2(x, params) - y

def jacobian_or2(params, x, y):
    a, b, c = params
    da = 1 / ( x + b) ** 2
    db = -2 * a / (x + b) ** 3
    dc = np.ones_like(x)
    return np.vstack([da, db, dc]).T

def fit_r2(x, y, loss='linear', f_scale=1.0, tol=None) -> OptimizeResult:
    a0 = np.max(y)
    b0 = 1E-15
    c0 = np.min(y)
    p0 = np.array([a0, b0, c0])
    eps = float(np.finfo(np.float64).eps)
    if tol is None:
        tol = eps
    bounds = ([eps, eps, eps], [np.inf, np.inf, np.inf])
    result = least_squares(
        residual_or2,
        x0=p0,
        jac=jacobian_or2,
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
    p0 = np.array([np.max(y), 1.0])
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

def gaussian(x, params):
    """
    Gaussian function with constant baseline: A * exp(-(x - mu)^2 / (2 * sigma^2)) + baseline

    Parameters:
    x: array-like, independent variable
    params: array-like (A, mu, sigma, baseline)
        A: amplitude
        mu: mean
        sigma: standard deviation
        baseline: constant offset
    """
    A, mu, sigma, baseline = params
    # A, sigma, baseline = params
    # mu = 0.
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + baseline



def residuals_gaussian(params, x, y, w=1):
    """Calculate residuals between observed data and the Gaussian model"""
    return (gaussian(x, params) - y) * w


def jacobian_gaussian(params, x, y, w=1.):
    """
    Analytical Jacobian matrix for the Gaussian function with baseline
    Returns partial derivatives with respect to (A, mu, sigma, baseline)
    """
    A, mu, sigma, baseline = params
    # A, sigma, baseline = params
    # mu=0.
    exp_term = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    # Partial derivatives
    d_A = w * exp_term
    d_mu = A * exp_term * (x - mu) / sigma ** 2
    d_sigma = w * A * exp_term * (x - mu) ** 2 / sigma ** 3
    d_baseline = w * np.ones_like(x)  # Derivative with respect to baseline

    return np.vstack([d_A, d_mu, d_sigma, d_baseline]).T
    # return np.vstack([d_A, d_sigma, d_baseline]).T

def gaussian_symmetric(x, params):
    """
    Gaussian function with constant baseline: A * exp(-(x - mu)^2 / (2 * sigma^2)) + baseline

    Parameters:
    x: array-like, independent variable
    params: array-like (A, mu, sigma, baseline)
        A: amplitude
        mu: mean
        sigma: standard deviation
        baseline: constant offset
    """
    # A, mu, sigma, baseline = params
    A, sigma, baseline = params
    mu = 0.
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + baseline


def residuals_gaussian_symmetric(params, x, y, w=1):
    """Calculate residuals between observed data and the Gaussian model"""
    return (gaussian_symmetric(x, params) - y) * w


def jacobian_gaussian_symmetric(params, x, y, w=1.):
    """
    Analytical Jacobian matrix for the Gaussian function with baseline
    Returns partial derivatives with respect to (A, mu, sigma, baseline)
    """
    # A, mu, sigma, baseline = params
    A, sigma, baseline = params
    mu=0.
    exp_term = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    # Partial derivatives
    d_A = w * exp_term
    d_mu = A * exp_term * (x - mu) / sigma ** 2
    d_sigma = w * A * exp_term * (x - mu) ** 2 / sigma ** 3
    d_baseline = w * np.ones_like(x)  # Derivative with respect to baseline

    # return np.vstack([d_A, d_mu, d_sigma, d_baseline]).T
    return np.vstack([d_A, d_sigma, d_baseline]).T


EPS = float(np.finfo(np.float64).eps)
def fit_gaussian(x, y, dy=None, p0=None, symmetric=False, loss='linear', f_scale=1.0, tol=EPS) -> OptimizeResult:
    """
    Fit Gaussian profile to data using least_squares with analytical Jacobian

    Parameters:
    x: array-like, independent variable
    y: array-like, dependent variable
    p0: initial guess for parameters (A, mu, sigma)
    symmetric: bool, if symmetric, make mu=0


    Returns:
    OptimizeResult object containing the fitted parameters
    """
    global residuals_gaussian, residuals_gaussian_symmetric, jacobian_gaussian, jacobian_gaussian_symmetric
    residuals:callable = residuals_gaussian
    jac:callable = jacobian_gaussian
    if p0 is None:
        # Make educated guesses for initial parameters
        baseline = np.min(y)  # Estimate baseline as minimum y value
        A = np.max(y) - baseline  # Estimate amplitude above baseline
        mu = x[np.argmax(y)]
        sigma = np.std(x) / 2
        p0 = np.array([A, mu, sigma, baseline])
        bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
        if symmetric:
            p0 = np.array([A, sigma, baseline])
            residuals:callable = residuals_gaussian_symmetric
            jac:callable = jacobian_gaussian_symmetric
            bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

    if dy is None:
        weights = 1
    else:
        weights = np.abs(1. / (dy + np.median(dy)/10))

    result = least_squares(
        residuals,
        x0=p0,
        jac=jac,
        bounds=bounds,
        args=(x, y, weights),
        method='trf',
        loss = loss,
        f_scale = f_scale,
        xtol = tol,
        ftol = tol,
        gtol = tol,
        verbose = 2,
        x_scale='jac',
        max_nfev = 10000 * len(p0)
    )

    return result

def main(angle_dist_file, particle_dens_file):
    # Load pre-calculated density and grid
    n_particles, (X, Y, Z) = load_density_grid(particle_dens_file)
    # normalize density to 1
    density_norm = n_particles / np.sum(n_particles)
    mid_y = density_norm.shape[1] // 2
    density_proj = np.sum(density_norm, axis=1)
    # Read the image
    # img = mpimg.imread(pisces_a_plasma_img)
    angle_dist_df: pd.DataFrame = pd.read_csv(angle_dist_file).apply(pd.to_numeric)
    polar_angle = angle_dist_df['angle (rad)'].values
    n_p_theta = angle_dist_df['n particles'].values / angle_dist_df['n particles'].sum()

    # Fit a cosn law for the angles
    fit_result_cosn: OptimizeResult = fit_cosn(x=polar_angle, y=n_p_theta)
    theta_pred = np.linspace(polar_angle.min(), polar_angle.max(), 1000)
    prob_pred = cosn_model(theta_pred, fit_result_cosn.x)

    # Create finer grid
    fine_angles = np.linspace(polar_angle.min(), polar_angle.max(), 1000)

    # Linear interpolation of PDF
    pdf_interp = interpolate.interp1d(
        polar_angle, n_p_theta, kind='linear', bounds_error=False, fill_value=(n_p_theta[0], n_p_theta[-1]),
    )
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
    interp_inverse_cdf = interpolate.interp1d(cdf, fine_angles)
    sampled_angles = interp_inverse_cdf(random_uniform)

    # Create histogram. The number of bins controls the height of the histogram
    hist, bins = np.histogram(sampled_angles, bins=17, density=False)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Generate distances following 1/r^2 decay
    # Using inverse transform sampling for PDF ∝ 1/r^2
    # CDF: F(r) = 1 - 1/r, r ∈ [1, max_distance]
    # Inverse: r = 1/(1-ξ)
    max_distance = 1.
    min_distance = 1.  # Added minimum distance
    xi = np.random.uniform(0, 1, n_samples)
    # distances = 1 / (1 - xi * (1 - 1 / max_distance))
    distances = -min_distance + 1 / (1 / min_distance - xi * (1 / min_distance - 1 / (max_distance + min_distance)))

    # Create histogram with more bins for better resolution
    counts_r, bins_r = np.histogram(distances, bins=20, density=False)
    bin_centers_r = (bins_r[:-1] + bins_r[1:]) / 2

    fit_result_r2: OptimizeResult = fit_r2(x=bin_centers_r, y=counts_r / n_samples)
    popt_r2 = fit_result_r2.x
    x_pred = np.linspace(bin_centers_r.min(), bin_centers_r.max(), 1000)
    y_pred = over_r2(x_pred, popt_r2)

    n_rho = 1000000
    rho = np.zeros(n_rho)
    accepted = 0
    while accepted < n_samples:
        x = np.abs(np.random.normal(loc=0, scale=1.5, size=40*(n_rho - accepted)))
        mask = x <= 0.5
        remaining = n_rho - accepted
        rho[accepted:accepted+min(remaining, np.sum(mask))] = x[mask][:remaining]
        accepted += min(remaining, np.sum(mask))

    # Create histogram for sqrt(x^2+y^2)
    counts_rho, bins_rho = np.histogram(rho, bins=30, density=False)
    bin_centers_rho = (bins_rho[:-1] + bins_rho[1:]) / 2

    fit_result_gauss:OptimizeResult = fit_gaussian(x=bin_centers_rho, y=counts_rho/n_rho, symmetric=True)
    x_pred_g = np.linspace(bin_centers_rho.min(), bin_centers_rho.max(), 1000)
    y_pred_g = gaussian_symmetric(x_pred_g, fit_result_gauss.x)


    fig, axes = plt.subplots(nrows=2, ncols=2)#, layout='constrained')
    fig.subplots_adjust(left=0.12, top=0.9, right=0.95, bottom=0.1, hspace=0.4, wspace=0.4)
    fig.set_size_inches(6.75, 6.)
    axes[0,0].plot(polar_angle, n_p_theta, marker='o', ls='none', color='C0', mfc='none', mew=1.25, label='TRIM.SP')
    axes[0,0].plot(theta_pred, prob_pred,  ls='--', color='C0', label=r'$\cos^n(\theta)$', lw=1.25)
    axes[0,0].set_xlabel('Polar angle (deg)')
    axes[0,0].set_ylabel('Probability density')
    secax = axes[0,0].secondary_xaxis(location='top', functions=(rad2deg, deg2rad))
    secax.set_xlabel('$\\theta$ (deg)', usetex=True)
    axes[0,0].legend(
        loc='lower left', frameon=True
    )

    axes[0,1].plot(bin_centers_r, counts_r/n_samples, marker='^', mfc='none', color='C2', ls='none', label='Sampled r')
    axes[0,1].plot(x_pred, y_pred, marker='none', color='C2', ls='--', label='$1/(r+1)^2$', lw=1.25)
    axes[0,1].set_xlabel(r'r (cm)', usetex=False)
    axes[0,1].set_ylabel('Probability density')
    axes[0,1].legend(
        loc='upper right', frameon=True
    )

    # Use ScalarFormatter to format the y-axis in scientific notation
    formatter = ticker.ScalarFormatter(useMathText=True, useOffset=3.2E-2)
    formatter.set_powerlimits((-2, 2))  # Use scientific notation for numbers outside 10^-2 to 10^2
    axes[1,0].plot(bin_centers_rho, counts_rho/n_rho, marker='D', ls='none', color='C3', mfc='none', label='Sampled data')
    axes[1,0].plot(x_pred_g, y_pred_g, ls='--', lw=1.25, color='C3', label='Normal distribution')
    axes[1,0].set_xlabel(r"$\sqrt{x^2 + y^2}$ {\sffamily (cm)}", usetex=True)
    axes[1,0].set_ylabel('Probability density')
    axes[1,0].yaxis.set_major_formatter(formatter)
    axes[1,0].legend(
        loc='lower left', frameon=True
    )

    # axes[0].tick_params(
    #     axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    # )
    cs = axes[1,1].pcolormesh( Z[:, mid_y, :], X[:, mid_y, :],
                         density_proj, shading='auto')
    axes[1,1].set_xlabel(r'z (cm)')
    axes[1,1].set_ylabel(r'y (cm)')
    axes[1,1].set_aspect('equal')

    divider = make_axes_locatable(axes[1,1])
    cax = divider.append_axes("right", size="8%", pad=0.025)

    cbar = fig.colorbar(cs, cax=cax, extend='neither')
    cbar.set_label(r'Particle density (cm$^{\mathregular{-3}}$)', size=12, labelpad=9)
    # cbar.ax.set_ylim(t_range[0], t_range[1])
    cbar.ax.ticklabel_format(axis='y', style='sci', useMathText=True, scilimits=(-2,2))
    cbar.ax.tick_params(labelsize=10)
    # cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    # cbar.ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))

    for i, axi in enumerate(axes.flatten()):
        axi.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        axi.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        panel_label = chr(ord('`') + i + 1) # starts from a
        # panel_label = chr(ord('`') + i + 3)
        axi.text(
            -0.2, 1.1, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )
    fig.savefig(r'./figures/fig_boron_sputtered_density.png', dpi=600)
    fig.savefig(r'./figures/fig_boron_sputtered_density.pdf', dpi=600)
    plt.show()








if __name__ == "__main__":
    load_plot_style()
    main(angle_dist_file=ANGLE_DIST_FILE, particle_dens_file=PARTICLE_DENS_FILE)
