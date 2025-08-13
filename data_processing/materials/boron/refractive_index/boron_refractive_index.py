import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from data_processing.utils import latex_float, latex_float_with_error
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf


"""
Optical and Electrical Properties of Boron
Nobuyoshi Morita and Akira Yamamoto 1975 Jpn. J. Appl. Phys. 14 825
doi: 10.1143/JJAP.14.825
"""

LASER_WL = 650.

def model_poly(x, b) -> np.ndarray:
    """
    A polynomial model

    Parameters
    ----------
    x: np.ndarray
        The x data points the polynomial is evaluated at
    b: np.ndarray
        The coefficients of the polynomial

    Returns
    -------

    """
    n = len(b)
    r = np.zeros(len(x))
    for i in range(n):
        r += b[i] * x ** i
    return r


def chi_poly(b, x, y, w=None):
    """
    A residual function for the polynomial model
    Parameters
    ----------
    b: np.ndarray
        The coefficients of the polynomial
    x: np.ndarray
        The x data points
    y: np.ndarray
        The observerved y values
    w: np.ndarray
        The weights of each data point

    Returns
    -------
    np.ndarray
    """
    if w is None:
        return model_poly(x, b) - y

    return (model_poly(x, b) - y) * w


def jac_poly(b, x, y, w=1):
    """
    The jacobian of the residual function for the polynomial model

    Parameters
    ----------
    b: np.ndarray
        The coefficients of the polynomial
    x: np.ndarray
        The x data points
    y: np.ndarray
        The observerved y values
    w: np.ndarray
        The weights of each data point

    Returns
    -------
    np.ndarray
    """
    n = len(b)
    r = np.zeros((len(x), n))
    for i in range(n):
        r[:, i] = w * x ** i
    return r

def fit_polynomial(
    x, y, weights=None, poly_order:int=10, loss:str= 'soft_l1', f_scale:float=1.0, tol:float=None, verbose=2
) -> OptimizeResult:
    """
    Fits the curve to a polynomial function
    Parameters
    ----------
    x: np.ndarray
        The x values
    y: np.ndarray
        The y values
    weights: float
        The weights for the residuals
    poly_order: int
        The degree of the polynomial to be used
    loss: str
        The type of loss to be used
    f_scale: float
        The scaling factor for the outliers
    tol: float
        The tolerance for the convergence
    verbose: int
        The verbose argument for the least_squares function (default 2)

    Returns:
    -------
    OptimizeResult:
        The least squares optimized result
    """

    if tol == None:
        tol = float(np.finfo(np.float64).eps)
    ls_res = least_squares(
        chi_poly,
        x0=[(0.01) ** (i-1) for i in range(poly_order+1)],
        args=(x, y, weights),
        loss=loss, f_scale=f_scale,
        jac=jac_poly,
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=verbose,
        x_scale='jac',
        method='trf',
        tr_solver='exact',
        max_nfev=1000 * poly_order,
    )
    return ls_res

def model_polylog(x, b):
    """
    A polylog model

    Parameters
    ----------
    x: np.ndarray
        The x data points the polynomial is evaluated at
    b: np.ndarray
        The coefficients of the polynomial

    Returns
    -------

    """
    n = len(b)
    r = np.ones(len(x))
    xi = np.ones(len(x))
    for i in range(n):
        r *= np.exp(b[i]*xi)
        xi *= x
    return r

def chi_polylog(b, x, y, w=None):
    """
    A residual function for the polylog model
    Parameters
    ----------
    b: np.ndarray
        The coefficients of the polynomial
    x: np.ndarray
        The x data points
    y: np.ndarray
        The observerved y values
    w: np.ndarray
        The weights of each data point

    Returns
    -------
    np.ndarray
    """
    if w is None:
        return model_polylog(x, b) - y

    return (model_polylog(x, b) - y) * w

def jac_polylog(b, x, y, w=1):
    """
    The jacobian of the residual function for the polynomial model

    Parameters
    ----------
    b: np.ndarray
        The coefficients of the polynomial
    x: np.ndarray
        The x data points
    y: np.ndarray
        The observerved y values
    w: np.ndarray
        The weights of each data point

    Returns
    -------
    np.ndarray
    """
    n = len(b)
    r = np.ones((len(x), n))
    xi = np.ones_like(x, dtype=np.float64)
    y_model = model_polylog(x, b)
    for i in range(n):
        r[:, i] = xi * y_model
        xi *= x
    return r

def fit_polylog(
    x, y, weights=None, poly_order:int=10, loss:str= 'soft_l1', f_scale:float=1.0, tol:float=None, verbose=2
) -> OptimizeResult:
    """
    Fits the curve to a polynomial function
    Parameters
    ----------
    x: np.ndarray
        The x values
    y: np.ndarray
        The y values
    weights: float
        The weights for the residuals
    poly_order: int
        The degree of the polynomial to be used
    loss: str
        The type of loss to be used
    f_scale: float
        The scaling factor for the outliers
    tol: float
        The tolerance for the convergence
    verbose: int
        The verbose argument for the least_squares function (default 2)

    Returns:
    -------
    OptimizeResult:
        The least squares optimized result
    """

    if tol == None:
        tol = float(np.finfo(np.float64).eps)

    ylog = np.log(y)
    slope = (ylog[2] - ylog[0]) / (x[2] - x[0])
    intercept = ylog[0]
    x0 = [intercept, slope, 1E-6, 1E-8]
    print(x0)
    ls_res = least_squares(
        chi_polylog,
        # x0=[(-1**i)*(0.01) ** (i) for i in range(poly_order+1)],
        x0=x0,
        args=(x, y, weights),
        loss=loss, f_scale=f_scale,
        jac=jac_polylog,
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=verbose,
        x_scale='jac',
        method='trf',
        tr_solver='exact',
        max_nfev=1000 * poly_order,
    )
    return ls_res

def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def main(laser_wavelength):
    n_df: pd.DataFrame = pd.read_csv(
        'Morita-Yakamoto1975_refractive_index_rhombohedral_boron.csv', comment='#'
    )
    k_df: pd.DataFrame = pd.read_csv(
        'Morita-Yakamoto1975_exctinction_coefficient_rhombohedral_boron.csv', comment='#'
    )
    n_df = n_df.apply(pd.to_numeric)
    k_df = k_df.apply(pd.to_numeric)
    wl_n_nm = n_df['Wavelength (um)'].values * 1000.
    wl_k_nm = n_df['Wavelength (um)'].values * 1000.
    refractive_index = n_df['n'].values
    extinction_coefficient = k_df['k'].values

    reflectance = (1.0003 - refractive_index) / (1.0003 + refractive_index)
    reflectance *= reflectance

    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.5)

    ax.plot(wl_n_nm, refractive_index, marker='o', ls='none', c='C0', label='n', mfc='none')
    ax.plot(wl_k_nm, extinction_coefficient, marker='s', ls='none', c='C1', label='k', mfc='none')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('n, k')

    ax.set_xlim(300, 1000.)
    ax.set_ylim(0, 3.5)

    ax.legend(
        loc='upper right', frameon=True
    )

    ax.set_title('Boron (Morita & Yakamoto 1975)')

    fig_r, ax_r = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig_r.set_size_inches(4.0, 3.5)

    ax_r.plot(
        wl_n_nm, reflectance * 100., marker='o', ls='none', c='C0',  mfc='none',
        label='Morita & Yakamoto (1975)'
    )

    # Get the abosrption coefficient
    alpha = 4. * np.pi * extinction_coefficient / wl_k_nm * 1E7


    fit_result = fit_polylog(x=wl_k_nm, y=alpha, poly_order=2, loss='soft_l1', f_scale=1.0)
    wl_k_interp = np.linspace(wl_k_nm.min(), wl_k_nm.max(), num=1000)

    y_pred, delta_pred = cf.prediction_intervals(
        model=model_polylog, x_pred=wl_k_interp, ls_res=fit_result, jac=jac_polylog, new_observation=True
    )

    f_interp = interp1d(x=wl_n_nm, y=alpha)
    y_interp = interp1d(x=wl_k_interp, y=y_pred)
    y_delta_interp = interp1d(x=wl_k_interp, y=delta_pred)


    ax_r.set_xlabel('Wavelength (nm)')
    ax_r.set_ylabel('R')

    ax_r.set_xlim(300, 1000.)
    ax_r.set_ylim(0, 100.)

    ax_r.legend(
        loc='upper right',
        frameon=True
    )

    fig_a, ax_a = plt.subplots(1, 1, constrained_layout=True)
    fig_a.set_size_inches(4.0, 3.5)
    ax_a.plot(
        wl_k_nm, alpha, marker='^', color='C4', mfc='none', ls='none',
        label='Morita & Yakamoto (1975)'
    )


    ax_a.plot(
        wl_k_interp, y_pred, ls='-',
        lw='1.25', color='C4', label='Fit'
    )
    ax_a.fill_between(wl_k_interp, y_pred-delta_pred, y_pred+delta_pred, color='C4', alpha=0.25)

    ax_a.set_xlabel('Wavelenght (nm)')
    ax_a.set_ylabel('Absorption coefficient (cm$^{\mathregular{-1}}$)')
    ax_a.set_yscale('log')

    ax_a.set_xlim(300, 1000.)
    ax_a.set_ylim(1E4, 1E6)
    ax_a.xaxis.set_minor_locator(ticker.MultipleLocator(50))

    ax_a.legend(
        loc='upper right',
        frameon=True
    )

    fig.savefig('Morita_Yakamoto1975_refractive_index.png', dpi=600)
    fig_r.savefig('Morita_Yakamoto1975_reflectance.png', dpi=600)
    fig_a.savefig('Morita_Yakamoto1975_absorption.png', dpi=600)
    print(f'R at {wl_n_nm[-1]:.0f}: {reflectance[-1]:.3f}')

    # print(f'R at {laser_wavelength:.1f} nm: {f_interp(laser_wavelength):.3E} cm^-1')
    print(
        f'Absorption coefficient at {laser_wavelength:.1f} nm: '
        f'{y_interp(laser_wavelength):.3E} ± {y_delta_interp(laser_wavelength):.3E} cm^-1'
    )

    plt.show()


if __name__ == '__main__':
    main(laser_wavelength=LASER_WL)
