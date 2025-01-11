import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf
import re
from decimal import Decimal

temperature_csv = r'./data/oes_black_body/20241031_bb_temp.xlsx'

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    plt.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}')

def model_poly(x, b) -> np.ndarray:
    n = len(b)
    r = np.zeros(len(x))
    for i in range(n):
        r += b[i] * x ** i
    return r


def res_poly(b, x, y, w=1.):
    return (model_poly(x, b) - y) * w


def jac_poly(b, x, y, w=1):
    n = len(b)
    r = np.zeros((len(x), n))
    for i in range(n):
        r[:, i] = w * x ** i
    return r

EPS = float(np.finfo(np.float64).eps)
def fit_polynomial(x, y, weights=1, poly_order:int=10, loss:str= 'soft_l1', f_scale:float=1.0, tol:float=EPS) -> OptimizeResult:
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

    Returns:
    -------
    OptimizeResult:
        The least squares optimized result
    """
    ls_res = least_squares(
        res_poly,
        x0=[(0.01) ** (i-1) for i in range(poly_order)],
        args=(x, y, weights),
        loss=loss, f_scale=f_scale,
        jac=jac_poly,
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=2,
        x_scale='jac',
        method='trf',
        tr_solver='exact',
        max_nfev=10000 * poly_order
    )
    return ls_res


def main():
    global temperature_csv
    temperature_df = pd.read_excel(temperature_csv, sheet_name=0)
    temp_cols = [
        'Elapsed time (s)',
        'Temperature (K)',
        'Temperature error (K)'
    ]
    temperature_df[temp_cols] = temperature_df[temp_cols].apply(pd.to_numeric)
    time_s = temperature_df['Elapsed time (s)'].values
    temperature_k = temperature_df['Temperature (K)'].values
    temperature_err_k = temperature_df['Temperature error (K)'].values

    weights = 1. / (temperature_err_k + np.median(temperature_err_k) / 10) # Include measurement uncertainties
    fit_result: OptimizeResult = fit_polynomial(x=time_s, y=temperature_k, weights=weights, loss='soft_l1', f_scale=0.5, poly_order=6)
    popt = fit_result.x

    x_pred = np.linspace(0, time_s.max(), 1000)
    y_pred, delta = cf.prediction_intervals(model_poly, x_pred=x_pred, ls_res=fit_result, jac=jac_poly, weights=weights)

    load_plot_style()

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(4., 2.5)

    ax.errorbar(
        x=time_s/60, y=temperature_k-273.15, yerr=temperature_err_k, label='Data',
        capsize=2.75, mew=1.25, marker='o', ms=8, elinewidth=1.25,
        color='C0', fillstyle='none',
        ls='none',  # lw=1.25,
    )

    ax.plot(x_pred/60, y_pred-273.15, color='C0', label='Polynomial fit')
    ax.fill_between(x_pred/60, y_pred-273.15-delta, y_pred-273.15+delta, color='C0', alpha=0.2)

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Surface temperature')

    ax.legend(
        loc='upper right', frameon=True
    )

    ax.set_xlim(0,100)
    ax.set_ylim(400, 900)

    model_df = pd.DataFrame(data={
        'Time (s)': x_pred,
        'Temperature (K)': y_pred,
        'Temperature error (K)': delta
    })

    model_df.to_csv(r'./data/oes_black_body/echelle_20241031/20241031_temperature_model.csv', index=False)
    fig.savefig(r'./figures/fig_20241031_surface_temperature.pdf', dpi=600)
    fig.savefig(r'./figures/fig_20241031_surface_temperature.svg', dpi=600)
    fig.savefig(r'./figures/fig_20241031_surface_temperature.png', dpi=600)

    plt.show()

if __name__ == '__main__':
    main()