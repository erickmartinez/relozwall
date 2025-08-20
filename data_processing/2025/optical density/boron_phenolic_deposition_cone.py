import numpy as np
import pandas as pd
from scipy.optimize import least_squares, OptimizeResult, differential_evolution
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import os
import data_processing.confidence as cf
from data_processing.utils import latex_float, lighten_color
from data_processing.misc_utils.plot_style import load_plot_style

"""
Modified Knudsen model reference:
V. E. de Matos Loureiro da Silva Pereira, J. R. Nicholls, and R. Newton, Surf. Coat. Technol. 311, 307 (2017).
https://doi.org/10.1016/j.surfcoat.2016.12.054
DOI: 10.1016/j.surfcoat.2016.12.054

Boron density reference:
Mónica Fernández-Perea, Juan I. Larruquert, José A. Aznárez, José A. Méndez, Manuela Vidal-Dasilva, Eric Gullikson, 
Andy Aquila, Regina Soufli, and J. L. G. Fierro, "Optical constants of electron-beam evaporated boron films in the 
6.8-900 eV photon energy range," J. Opt. Soc. Am. A 24, 3800-3807 (2007)
DOI: 10.1364/JOSAA.24.003800

Density: 2.1 ± 0.1 g/cm³
"""


TRANSMISSION_XLS = r'./data/2025-S0803.xlsx'
SUBSTRATE_SOURCE_DISTANCE_CM = 3.8 # cm
EXPOSURE_TIME = 1.0 # In seconds
BORON_DENSITY, BORON_DENSITY_DELTA = 2.1, 0.1
BORON_MOLAR_MASS = 10.811 # g / mol
ROD_DIAMETER = 1.0 # cm




def modified_knudsen(r_, h0_, n_):
    cos_q = h0_ / np.sqrt(r_ ** 2. + h0_ ** 2.)
    return np.power(cos_q, n_ + 2.)

def deposit_rate_knudsen(
    density, h0, d0, n, molar_mass, sublimation_time, delta_density=0, delta_h0=0, delta_d0=0, delta_n=0
):
    rate_g_s = 2E-7 * np.pi * density * (h0 ** 2) * d0 / n / sublimation_time
    rate_mol_s = 2E-7 * np.pi * density * (h0 ** 2) * d0 / n / molar_mass / sublimation_time
    rate_mol_s_uncertainty = rate_mol_s * np.linalg.norm(np.array([
        delta_density / density, delta_h0 / h0, delta_d0 / d0, delta_n / n
    ]))
    rate_atom_s = 6.02214076e+23 * rate_mol_s
    rate_atom_s_uncertainty = rate_mol_s_uncertainty * 6.02214076e+23
    return rate_atom_s, rate_atom_s_uncertainty, rate_g_s



def fobj(b, r_, h0_, d_, w_=1.):
    return w_ * (modified_knudsen(r_, h0_, b[0]) - d_)


def fobj_de(b, r_, h0_, d_, w_=1.):
    return 0.5 * np.linalg.norm(fobj(b, r_, h0_, d_, w_))


def jac(b, r_, h0_, d_, w_=1.0):
    cos_q = h0_ / np.sqrt(r_ ** 2. + h0_ ** 2.)
    jj = np.empty((len(r_), 1))
    jj[:, 0] = np.log(cos_q) * modified_knudsen(r_, h0_, b[0]) * w_
    return jj

def knudsen_2mix(r, h0, n1, n2, mixing):
    """
    Calculate the mix of multiple `modified knudsen` models
    Parameters
    ----------
    r: np.ndarray
        The radius
    h0: float
        The value of h0
    n1: float
        The value of n for the first Knudsen model
    n2: float
        The value of n for the second Knudsen model
    mixing: float
        The mixing coefficient for the two models

    Returns
    -------
    np.ndarray

    """
    y1 = modified_knudsen(r, h0, n1)
    y2 = modified_knudsen(r, h0, n2)
    return mixing * y1 + (1. - mixing) * y2

def residuals_knudsen_2mix(b, r, h0, d, weights=1):
    r = weights * (knudsen_2mix(r, h0, b[0], b[1], b[2]) - d)
    return r

def residuals_knudsen_2mix_de(b, r, h0, d, weights=1):
    r = residuals_knudsen_2mix(b, r, h0, d, weights)
    return 0.5 * np.linalg.norm(r)

def jacobian_knudsen_2mix(b, r, h0, d, weights=1):
    jacobian = np.empty((len(r), 3))
    j1 = jac([b[0]], r, h0, d, weights)
    j2 = jac([b[1]], r, h0, d, weights)
    jacobian[:, 0] = j1[:, 0] * b[2]
    jacobian[:, 1] = j2[:, 0] * (1 - b[2])
    jacobian[:, 2] = modified_knudsen(r, h0, b[0]) - modified_knudsen(r, h0, b[1])
    return jacobian



def main(
    transmission_xls, substrate_source_distance_cm, boron_density, boron_density_error, boron_molar_mass, exposure_time,
    rod_diameter_cm
):
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

    # res_de: OptimizeResult = differential_evolution(
    #     func=fobj_de,
    #     args=(r, h0, dn, 1),
    #     x0=[10],
    #     bounds=[(-1000, 1000)],
    #     maxiter=nn * 1000000,
    #     tol=all_tol,
    #     atol=all_tol,
    #     workers=-1,
    #     updating='deferred',
    #     recombination=0.5,
    #     strategy='best1bin',
    #     mutation=(0.5, 1.5),
    #     init='sobol',
    #     polish=False,
    #     disp=True
    # )
    #
    # res = least_squares(
    #     fobj,
    #     res_de.x,
    #     loss='soft_l1', f_scale=0.1,
    #     jac=jac,
    #     args=(r, h0, dn, 1),
    #     bounds=([-1000, ], [1000.]),
    #     xtol=all_tol,
    #     ftol=all_tol,
    #     gtol=all_tol,
    #     diff_step=all_tol**0.5,
    #     max_nfev=1000 * nn,
    #     method='trf',
    #     x_scale='jac',
    #     tr_solver='exact',
    #     verbose=2
    # )

    # res_de: OptimizeResult = differential_evolution(
    #     func=residuals_knudsen_2mix_de,
    #     args=(r, h0, dn, 1.),
    #     x0=[10., 10., 0.5],
    #     bounds=[(-1000, 1000), (-1000, 1000), (0, 1)],
    #     maxiter=nn * 1000000,
    #     tol=all_tol,
    #     atol=all_tol,
    #     workers=-1,
    #     updating='deferred',
    #     recombination=0.5,
    #     strategy='best1bin',
    #     mutation=(0.5, 1.5),
    #     init='sobol',
    #     polish=False,
    #     disp=True
    # )

    res = least_squares(
        residuals_knudsen_2mix,
        # res_de.x,
        [10., 10., 0.5],
        loss='soft_l1', f_scale=0.1,
        jac=jacobian_knudsen_2mix,
        args=(r, h0, dn, weigths),
        bounds=([-1000, -1000, 0], [1000., 1000, 1]),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        diff_step=all_tol ** 0.5,
        max_nfev=1000 * nn,
        method='trf',
        x_scale='jac',
        tr_solver='exact',
        verbose=2
    )
    #
    # print("res_de.x:", res_de.x)
    # print("res_ls.x:", res.x)


    popt = res.x
    n_opt = popt[0]
    ci = cf.confidence_interval(res=res)

    delta_popt = ci[:,1] - popt
    dn_opt = np.max(np.abs(ci - n_opt), axis=1)[0]
    xp = np.linspace(r.min(), r.max(), 500)

    # print(popt)
    # print(ci)

    # def model_restricted(x, b):
    #     return modified_knudsen(x, h0, b[0])

    def model_restricted(x, b):
        return knudsen_2mix(r=x, h0=h0, n1=b[0], n2=b[1], mixing=b[2])

    # yp, lpb, upb = cf.predint(x=xp, xd=r, yd=dn, func=model_restricted, res=res, mode='observation', )
    # yp, delta = cf.prediction_intervals(model=model_restricted, x_pred=xp, jac=jac, new_observation=True, ls_res=res)
    yp, delta = cf.prediction_intervals(
        model=model_restricted, x_pred=xp, jac=jacobian_knudsen_2mix, new_observation=True, ls_res=res,
    )
    # yp = modified_knudsen(xp, h0, popt[0])

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)

    # model_str = r"$\dfrac{d}{d_0} = \dfrac{h_0^2}{h^2}\cos^n(\theta)$"
    model_str = r"$\dfrac{d}{d_0} = \eta\dfrac{h_0^2}{h^2}\cos^{n_1}\theta + (1-\eta)\dfrac{h_0^2}{h^2}\cos^{n_2}\theta$"
    ax.text(
        x=0.2, y=0.95,
        s=model_str, ha='left', va='top',
        transform=ax.transAxes,
        usetex=True,
        color='k'
    )


    # ax.fill_between(xp, lpb, upb, color=lighten_color('C0', 0.5), zorder=0)
    ax.fill_between(xp, yp-delta, yp+delta, color='C0', alpha=0.3, zorder=0)
    markers_p, caps_p, bars_p = ax.errorbar(
        r, dn, xerr=0.05, yerr=yerr, ms=7, ls='none', marker='o', color='C0', label='data', mew=1.5,
        fillstyle='none',
        capsize=2.75, elinewidth=1.25, zorder=2
    )
    ax.plot(xp, yp, ls='-', color='k', label='2-Source Knudsen', zorder=10)

    ax.plot(xp,  popt[2]*modified_knudsen(xp, h0, popt[0]), color='C3', lw=1.)
    ax.plot(xp, (1-popt[2]) * modified_knudsen(xp, h0, popt[1]), color='C4', lw=1.)

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

    # model_str = (f"$n_1 = {popt[0]:.1f}\pm{delta_popt[0]:.1f}$\n"
    #              f"$n_2 = {popt[1]:.1f}\pm{delta_popt[1]:.1f}$\n"
    #              f"$\eta = {popt[2]:.2f}\pm{delta_popt[2]:.2f}$\n")

    text1 = r'\begin{align*} '
    text2 = r'n_1 &= ' + f'{popt[0]:.1f} \pm {delta_popt[0]:.1f}' + '\\\\'
    text3 = r'n_1 &= ' + f'{popt[1]:.1f} \pm {delta_popt[1]:.1f}' + '\\\\'
    text4 = r'\eta &= ' + f'{popt[2]:.2f} \pm {delta_popt[2]:.2f}'
    text5 = r'\end{align*}'
    fit_results_str = text1 + text2 + text3 + text4 + text5
    # plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    ax.text(
        0.67, 0.07,
        fit_results_str,
        va='bottom',
        ha='left',
        transform=ax.transAxes,
        color='k',
        fontsize=11,
        usetex=True
    )

    ax.legend(
        loc='center right', fontsize=10, frameon=True
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
    rate1, rate1_delta, rate1_gs = deposit_rate_knudsen(
        density=boron_density, h0=h0, d0=d0, n=popt[0], molar_mass=boron_molar_mass, sublimation_time=exposure_time,
        delta_density=boron_density_error, delta_h0=0.1, delta_d0=d_err[0], delta_n=delta_popt[0],
    )

    rate2, rate2_delta, rate2_gs = deposit_rate_knudsen(
        density=boron_density, h0=h0, d0=d0, n=popt[0], molar_mass=boron_molar_mass, sublimation_time=exposure_time,
        delta_density=boron_density_error, delta_h0=0.1, delta_d0=d_err[0], delta_n=delta_popt[0]
    )

    area = 0.25E-4 * np.pi * rod_diameter_cm ** 2.
    print(f"SAMPLE AREA: {area*1E4:.2f} cm²")

    total_rate = (popt[2]*rate1 + (1-popt[2]) * rate2) / area
    total_rate_delta = np.linalg.norm([(rate1 - rate2)*delta_popt[2], popt[2]*rate1_delta, popt[2]*rate2_delta]) / area
    total_rate_gs = (popt[2]*rate1_gs + (1-popt[2]) * rate2_gs)

    print(f"TOTAL SUBLIMATED BORON: {total_rate:.3E} ± {total_rate_delta:.3E} B atoms/m²/s")
    print(f"TOTAL SUBLIMATED BORON (G/S): {total_rate_gs:.3E} g/s")


    plt.show()


if __name__ == '__main__':
    load_plot_style()
    main(
        transmission_xls=TRANSMISSION_XLS, substrate_source_distance_cm=SUBSTRATE_SOURCE_DISTANCE_CM,
        boron_density=BORON_DENSITY, boron_density_error=BORON_DENSITY_DELTA, boron_molar_mass=BORON_MOLAR_MASS,
        exposure_time=EXPOSURE_TIME, rod_diameter_cm=ROD_DIAMETER
    )
