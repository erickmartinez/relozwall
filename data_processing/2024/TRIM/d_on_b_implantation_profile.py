from typing import Callable, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gamma
from scipy.integrate import simpson
from Ono_sputtering_yield_model import load_plot_style
from scipy.optimize import least_squares, OptimizeResult

DATA_FILE = r'data/simulations/SDTrimSP/D_ON_B_42eV_depth_proj'

def gamma_dist(x, params):
    """Gamma distribution"""
    amp, shape, scale, loc = params
    return amp * gamma.pdf(x, shape, loc=loc, scale=scale)

def residual_gamma(params, x, y):
    return gamma_dist(x, params) - y

def fit_gamma_distribution(
    xdata:np.ndarray, ydata:np.ndarray, x0:List=None, loss:str='soft_l1', f_scale:float=1.0,
    tol:float=float(np.finfo(float).eps)
):
    """
    Fits
    Parameters
    ----------
    xdata: np.ndarray
        The x-data points
    ydata: np.ndarray
        The y-data points
    x0: List, np.ndarray
    loss: str
        The loss used by least_squares optimizer
    f_scale: float
        The f_scale parameter in least_squares minimizer
    tol: float
        The tolerance for the convergence (defaults to machine epsilon)

    Returns
    -------
    OptimizeResult
    """
    if x0 is None:
        # Find the peak implantation
        ymax = np.max(ydata)
        idx_peak = np.argmin(np.abs(ydata - ymax))

        x_peak = xdata[idx_peak]

        # Get the mean of y
        y_mean = np.mean(ydata)
        # Get the standard deviation of y
        std = np.std(ydata)
        shape = (y_mean / std) ** 2.
        scale = y_mean / shape

        print([ymax, shape, scale, x_peak])
        # x0 = np.array([ymax, 0.1, 2, 0.1])
        x0 = np.array([ymax, shape, scale, 0.])

    bounds = ([0, 0, 0, -np.max(xdata)], [np.inf, np.inf, np.inf, np.max(xdata)])

    result = least_squares(
        residual_gamma,
        x0=x0,
        bounds=bounds,
        args=(xdata, ydata),
        loss=loss,
        f_scale=f_scale,
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=2,
        x_scale='jac',
        jac='3-point',
        max_nfev=1000 * len(x0)
    )

    return result



def main(data_file):
    data_df = pd.read_csv(data_file, sep=r'\s+', usecols=[0,1], comment='#', names=['x (A)', 'stops']).apply(pd.to_numeric)
    x = data_df['x (A)'].values
    implantation = data_df['stops'].values / data_df['stops'].sum()
    print(data_df)

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3.)

    ax.plot(x, implantation, color='C0', label='D', marker='o', mfc='None', ls='None')
    # ax.plot(x, b_fraction, color='C1', label='B')

    # fit the distribution
    fit_result = fit_gamma_distribution(xdata=x, ydata=implantation, loss='soft_l1', f_scale=0.1)
    print(fit_result.x)

    x_pred = np.linspace(x.min(), x.max(), 1000)
    y_pred = gamma_dist(x_pred, fit_result.x)

    print(f"Integrated profile: {simpson(y=y_pred, x=x_pred)}")

    ax.plot(x_pred, y_pred, color='tab:red', label='Fit')

    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Atomic fraction')

    ax.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    main(DATA_FILE)
