import numpy as np
import pandas as pd
from scipy.optimize import least_squares, OptimizeResult, differential_evolution
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import pickle
import os
import data_processing.confidence as cf
from data_processing.utils import latex_float, lighten_color
from data_processing.misc_utils.plot_style import load_plot_style


TRANSMISSION_XLS = r'./data/2025-S0803.xlsx'
SUBSTRATE_SOURCE_DISTANCE_CM = 3.8 # cm
EXPOSURE_TIME = 1.0 # In seconds


def modified_knudsen(r_, h0_, n_):
    cos_q = h0_ / np.sqrt(r_ ** 2. + h0_ ** 2.)
    return np.power(cos_q, n_ + 2.)


def fobj(b, r_, h0_, d_, w_=1.):
    return w_ * (modified_knudsen(r_, h0_, b[0]) - d_)


def fobj_de(b, r_, h0_, d_, w_=1.):
    return 0.5 * np.linalg.norm(fobj(b, r_, h0_, d_, w_))


def jac(b, r_, h0_, d_, w_=1.0):
    cos_q = h0_ / np.sqrt(r_ ** 2. + h0_ ** 2.)
    jj = np.empty((len(r_), 1))
    jj[:, 0] = np.log(cos_q) * modified_knudsen(r_, h0_, b[0]) * w_
    return jj

def pseudo_voigt(x, fwhm, mixing):
    """Calculate pseudo-Voigt profile"""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    center, amplitude = 0., 1.
    # Gaussian component
    gaussian = amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
    # Lorentzian component
    lorentzian = amplitude * (fwhm ** 2 / 4) / ((x - center) ** 2 + (fwhm ** 2 / 4))
    # Pseudo-Voigt is a weighted sum
    return mixing * lorentzian + (1 - mixing) * gaussian

def pseudo_voigt_derivatives(x, fwhm, mixing):
    """Calculate partial derivatives of pseudo-Voigt function with respect to parameters"""
    center, amplitude = 0., 1.
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm / 2

    # Precomputed terms for Gaussian
    diff_squared = (x - center) ** 2
    exp_term = np.exp(-diff_squared / (2 * sigma ** 2))
    gaussian = amplitude * exp_term

    # Precomputed terms for Lorentzian
    denom = diff_squared + gamma ** 2
    lorentzian = amplitude * gamma ** 2 / denom

    # Derivative with respect to fwhm
    # For Gaussian part
    d_sigma_d_fwhm = 1 / (2 * np.sqrt(2 * np.log(2)))
    d_gaussian_d_sigma = gaussian * diff_squared / (sigma ** 3)
    d_gaussian_d_fwhm = d_gaussian_d_sigma * d_sigma_d_fwhm

    # For Lorentzian part
    d_gamma_d_fwhm = 0.5
    d_lorentzian_d_gamma = lorentzian * (2 * gamma / denom - 2 * gamma ** 3 / denom ** 2)
    d_lorentzian_d_fwhm = d_lorentzian_d_gamma * d_gamma_d_fwhm

    d_fwhm = mixing * d_lorentzian_d_fwhm + (1 - mixing) * d_gaussian_d_fwhm

    # Derivative with respect to mixing
    d_mixing = lorentzian - gaussian

    return d_fwhm, d_mixing

def model_function(x, params):
    """Calculate the sum of multiple pseudo-Voigt profiles plus baseline"""
    # Extract baseline parameters (last two values in params)

    # Calculate baseline
    y_model = np.zeros_like(x)

    # Add peaks
    n_peaks = (len(params)) // 2  # Subtract 2 for baseline params

    for i in range(n_peaks):
        fwhm = params[i * 2]
        mixing = params[i * 2 + 1]
        y_model += pseudo_voigt(x, fwhm, mixing)

    return y_model

def jacobian(params, x_data, y_data, weights=1):
    """Calculate the Jacobian matrix for the residuals function"""
    n_params = len(params)
    n_peaks = n_params // 2
    n_points = len(x_data)

    # Initialize Jacobian matrix
    jac = np.zeros((n_points, n_params))

    # Calculate derivatives for each peak
    for i in range(n_peaks):
        fwhm = params[i * 2 ]
        mixing = params[i * 2 + 1]

        # Get partial derivatives for this peak
        d_fwhm, d_mixing = pseudo_voigt_derivatives(
            x_data, fwhm, mixing
        )

        # Fill the Jacobian matrix for peak parameters
        jac[:, i * 2] = d_fwhm * weights
        jac[:, i * 2 + 1] = d_mixing * weights


    return jac

def residuals(params, x_data, y_data, weights=1):
    """Calculate residuals for least-squares fitting"""
    y_model = model_function(x_data, params)
    return (y_model - y_data) * weights


def main(transmission_xls, substrate_source_distance_cm):
    df = pd.read_excel(
        transmission_xls, sheet_name='Deposition cone', usecols=['x (mm)', 'Thickness (nm)', 'Thickness error (nm)']
    )
    df['r (mm)'] = df['x (mm)'] - df['x (mm)'].min()
    # df = df[df['r (mm)'] <= 25.]
    df = df.reset_index(drop=True)
    print(df)

    r = df['r (mm)'].values * 0.1
    d = df['Thickness (nm)'].values
    d_err = df['Thickness error (nm)'].values

    d0 = d[0]
    dn = d / d0
    yerr = dn*np.linalg.norm([np.ones_like(d)*d_err[0]/d[0], d_err/d], axis=0)

    weigths = 1. / ((yerr/dn) ** 2. + np.median(yerr/dn))
    # weigths /= weigths.max()
    # print('weights:', weigths)
    h0 = substrate_source_distance_cm
    all_tol = np.finfo(np.float64).eps
    nn = len(r)

    res_de: OptimizeResult = differential_evolution(
        func=fobj_de,
        args=(r, h0, dn, 1),
        x0=[10],
        bounds=[(-1000, 1000)],
        maxiter=nn * 1000000,
        tol=all_tol,
        atol=all_tol,
        workers=-1,
        updating='deferred',
        recombination=0.5,
        strategy='best1bin',
        mutation=(0.5, 1.5),
        init='sobol',
        polish=False,
        disp=True
    )

    res = least_squares(
        fobj,
        res_de.x,
        loss='soft_l1', f_scale=0.1,
        jac=jac,
        args=(r, h0, dn, 1),
        bounds=([-1000, ], [1000.]),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        diff_step=all_tol**0.5,
        max_nfev=1000 * nn,
        method='trf',
        x_scale='jac',
        tr_solver='exact',
        verbose=2
    )
    #
    # print("res_de.x:", res_de.x)
    # print("res_ls.x:", res.x)

    # Perform the least-squares fitting
    tol = float(np.finfo(np.float64).eps)
    initial_params = [r.max()*0.5, 1.]
    lower_bounds = [1E-3, 0.]
    upper_bounds = [2*r.max(), 1.]
    # try:
    #     res = least_squares(
    #         residuals,
    #         initial_params,
    #         args=(r, dn, weigths),
    #         bounds=(lower_bounds, upper_bounds),
    #         method='trf',
    #         jac=jacobian,
    #         ftol=tol,
    #         xtol=tol,
    #         gtol=tol,
    #         loss='soft_l1',
    #         f_scale=1.0,
    #         max_nfev=1000 * len(initial_params),
    #         verbose=2
    #     )
    # except ValueError as e:
    #     for lb, ip, ub in zip(lower_bounds, initial_params, upper_bounds):
    #         print(f"{lb:.3E}, {ip:.3E}, {ub:.3E}")
    #     raise (e)

    popt = res.x
    n_opt = popt[0]
    ci = cf.confidence_interval(res=res)
    dn_opt = np.max(np.abs(ci - n_opt), axis=1)[0]
    # dmixing_opt = np.max(np.abs(ci - popt[1]), axis=1)[1]
    print(f'n = {n_opt:.3f}±{dn_opt}')
    xp = np.linspace(r.min(), r.max(), 500)

    print(popt)
    print(ci)

    def model_restricted(x, b):
        return modified_knudsen(x, h0, b[0])

    # yp, lpb, upb = cf.predint(x=xp, xd=r, yd=dn, func=model_restricted, res=res, mode='observation', )
    yp, delta = cf.prediction_intervals(model=model_restricted, x_pred=xp, jac=jac, new_observation=True, ls_res=res)
    # yp, delta = cf.prediction_intervals(model=model_function, x_pred=xp, jac=jacobian, new_observation=True, ls_res=res)
    # yp = modified_knudsen(xp, h0, popt[0])

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)

    model_str = r"$\dfrac{d}{d_0} = \dfrac{h_0^2}{h^2}\cos^n(\theta)$"
    # model_str = r"Pseudo-voigt"


    # ax.fill_between(xp, lpb, upb, color=lighten_color('C0', 0.5), zorder=0)
    ax.fill_between(xp, yp-delta, yp+delta, color='C0', alpha=0.3, zorder=0)
    markers_p, caps_p, bars_p = ax.errorbar(
        r, dn, xerr=0.05, yerr=yerr, ms=7, ls='none', marker='o', color='C0', label='data', mew=1.5,
        fillstyle='none',
        capsize=2.75, elinewidth=1.25, zorder=2
    )
    ax.plot(xp, yp, ls='-', color='k', label=model_str, zorder=10)

    [bar.set_alpha(0.75) for bar in bars_p]
    [cap.set_alpha(0.75) for cap in caps_p]

    ax.set_xlabel('$r$ (cm)')
    ax.set_ylabel('$d/d_0$')
    ax.set_title(f'B deposition profile (h$_{{\mathregular{{0}}}}$ = {substrate_source_distance_cm:.1f} mm)')

    ax.set_xlim(0, 5)
    ax.set_ylim(0.0, 1.2)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    model_str = f"$n = {popt[0]:.3f}\pm{dn_opt:.3f}$"
    # model_str = (f"FWHM: ${popt[0]:.3f}\pm{dn_opt:.3f}$\n"
    #              f"MIXING: ${popt[1]:.3f}\pm{dmixing_opt:.3f}$")

    ax.text(
        0.95, 0.95,
        model_str,
        va='top',
        ha='right',
        transform=ax.transAxes,
        color='k',
        fontsize=11
    )

    ax.legend(
        loc='center right', fontsize=11, frameon=True
    )

    fig.savefig('boron_deposition_profile_cosine_law.png', dpi=600)
    # fig.savefig('deposition_profile_cosine_law.svg', dpi=600)
    fig.savefig('boron_deposition_profile_cosine_law.pdf', dpi=600)
    # fig.savefig('deposition_profile_cosine_law.eps', dpi=600)

    # save whole figure
    # pickle.dump(fig, open("deposition_profile_cosine_law.pickle", "wb"))

    # load figure from file
    # fig = pickle.load(open("figure.pickle", "rb"))

    # Estimate the total sublimated boron
    # rate =


    plt.show()


if __name__ == '__main__':
    load_plot_style()
    main(TRANSMISSION_XLS, SUBSTRATE_SOURCE_DISTANCE_CM)
