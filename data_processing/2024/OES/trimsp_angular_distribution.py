import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import json
import pandas as pd
from scipy import interpolate
from scipy.optimize import least_squares, OptimizeResult, differential_evolution


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}')

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


def main():
    # Results from trimsp simulation
    cos_beta = np.arange(0,20) * 0.05 + 0.05
    angles = np.arccos(cos_beta)
    # n_particles = np.array([15, 29, 53, 108, 187, 258, 397, 550, 743, 897, 1180, 1367, 1686, 1932, 2170, 2436, 2682, 2985, 3289, 3751])
    particles = np.array( [157, 502, 812, 1158, 1396, 1685, 2090, 2381, 2639, 2917, 3228, 3460, 3768, 4043, 4158, 4391, 4587, 4654, 4923, 5185])

    particle_density = particles / np.sum(particles)
    # Fit a cosn law for the angles
    fit_result_cosn: OptimizeResult = fit_cosn(x=angles, y=particle_density)
    theta_pred = np.linspace(angles.min(), angles.max(), 1000)
    prob_pred = cosn_model(theta_pred, fit_result_cosn.x)
    print(fit_result_cosn.x)


    # Create finer grid
    fine_angles = np.linspace(angles.min(), angles.max(), 1000)

    # Linear interpolation of PDF
    pdf_interp = interpolate.interp1d(
        angles, particle_density, kind='linear', bounds_error=False, fill_value=(particle_density[0], particle_density[-1]),
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

    # Create histogram for distances
    counts_r, bins_r = np.histogram(distances, bins=20, density=False)
    bin_centers_r = (bins_r[:-1] + bins_r[1:]) / 2

    fit_result_r2: OptimizeResult = fit_r2(x=bin_centers_r, y=counts_r/n_samples)
    popt_r2 = fit_result_r2.x
    x_pred = np.linspace(bin_centers_r.min(), bin_centers_r.max(), 1000)
    y_pred = over_r2(x_pred, popt_r2)

    fit_a, fit_b, fit_c = popt_r2

    load_plot_style()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 6.5)
    ax1.plot(angles, particle_density, marker='o', mfc='none', color='C0', label='TRIM.SP', ls='none')
    ax1.plot(theta_pred, prob_pred, ls='--', color='C0', label=r'$cos^n(\theta)$', lw=1.)
    ax1.plot(bin_centers, hist/n_samples, label='Sampled distribution', marker='s', mfc='none', color='C1', ls='none')
    ax1.set_xlabel('$\\theta$ (rad)', usetex=True)
    fig.supylabel('Particle density')
    ax1.legend(
        loc='lower left', frameon=True
    )
    # ax1.set_xlim(0, np.pi*0.5)

    ax2.plot(bin_centers_r, counts_r/n_samples, marker='^', mfc='none', color='C2', ls='none', label='Sampled r')
    ax2.plot(x_pred, y_pred, marker='none', color='C2', ls='--', label='$1/(r+1)^2$')
    ax2.set_xlabel(r'$r$ {\sffamily (cm)}', usetex=True)

    ax2.legend(
        loc='upper right', frameon=True
    )

    secax = ax1.secondary_xaxis(location='top', functions=(rad2deg, deg2rad))
    secax.set_xlabel('$\\theta$ (deg)', usetex=True)
    fig.savefig('./figures/trimsp_angle_distribution.png', dpi=600)

    # Save distribution to a csv_file
    angle_distribution_df: pd.DataFrame = pd.DataFrame(data={
        'angle (rad)': angles,
        'n particles': particles
    })

    ax1.set_title('Polar angle')
    ax2.set_title('Radial distribution')
    # angle_distribution_df.to_csv(r'trimsp_simulations/d_on_b_40keV_polar_angle_dist.csv', index=False)
    plt.show()


if __name__ == '__main__':
    main()