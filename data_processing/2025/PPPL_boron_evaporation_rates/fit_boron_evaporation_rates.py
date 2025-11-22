import pandas as pd
import numpy as np
import data_processing.confidence as cf
from scipy.optimize import least_squares, OptimizeResult
import matplotlib.pyplot as plt
from data_processing.misc_utils.plot_style import load_plot_style
import matplotlib.ticker as ticker
from scipy.interpolate import CubicSpline, PPoly
import h5py
from typing import Tuple, Union, Callable
from pathlib import Path

"""
This fits boron evaporation data as a function of temperature from 

H.W. Kugel, Y. Hirooka, J. Timberlake et al., Initial boronization of PB-X using ablation of solid boronized probes.
PPL-2903 (1993) 

Figure 16
"""


SUBLIMATION_RATE_CSV_LIST = [
    {'path': r'./data/boron_evaporation_rate_type_a.csv', 'type': 'a', 'm': 'o', 'mfc': 'None'},
    {'path': r'./data/boron_evaporation_rate_type_b.csv', 'type': 'b', 'm': 'o', 'mfc': 'C0'},
    {'path': r'./data/boron_evaporation_rate_type_c.csv', 'type': 'c', 'm': 's', 'mfc': 'None'},
    {'path': r'./data/boron_evaporation_rate_type_d.csv', 'type': 'd', 'm': 'x', 'mfc': 'None'},
]
TARGET_TEMPERATURE = 2000.
PISCES_A_D_FLUX_MEAN = 4E17 # atoms/cm^2-s
PISCES_Y_D = 0.04

def load_csv(path) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(path).apply(pd.to_numeric)
    df.sort_values(by=['Temperature (K)'], ascending=True, inplace=True)
    return df



def plot_evaporation(ax: plt.axes, df:pd.DataFrame, lbl:str, marker:str, mfc:str):
    ax.plot(
        df['Temperature (K)'].values, df['Evaporation rate (atoms/cm^2/s)'].values,
        marker=marker, mfc=mfc, color='C0', label=lbl, ls='none'
    )

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

def format_latex_with_bounds(value, lower_bound, upper_bound, usetex=True):
    """
    Format a number with its uncertainty in LaTeX notation.

    Args:
        value (float): The main value (e.g., 1.234E16)
        lower_bound (float): The uncertainty value (e.g., 1.0E16)
        upper_bound (float): The uncertainty value (e.g., 1.5E16)
        usetex (str): If usetex, add \sffamily around the 95% CI

    Returns:
        str: LaTeX formatted string with uncertainty

    Example:
        #>>> format_latex_with_bounds(1.234e16, 1.0E16, 1.5E16)
        '1.2 \\times 10^{16} 95% CI: [1.0, 1.5] \\times 10^{16}'
    """
    from decimal import Decimal
    # Convert to Decimal for better precision
    val = Decimal(str(value))
    lb = Decimal(str(lower_bound))
    ub = Decimal(str(upper_bound))

    # Get scientific notation components
    val_exp = val.adjusted()
    lb_exp = lb.adjusted()
    ub_exp = ub.adjusted()

    # Normalize both numbers to the same exponent (using the larger one)
    target_exp = min(val_exp, lb_exp, ub_exp)

    # Normalize value
    val_coeff = val / Decimal(f'1e{val_exp}')
    lb_coeff = lb / Decimal(f'1e{lb_exp}')
    ub_coeff = ub / Decimal(f'1e{ub_exp}')

    ci_str = f"95% CI: [${lb_coeff:.2f} \\times 10^{{{lb_exp}}}$, ${ub_coeff:.2f} \\times 10^{{{ub_exp}}}$]"
    if usetex:
        ci_str = rf"{{\sffamily 95\% CI: [}} ${lb_coeff:.3f} \times 10^{{{lb_exp}}}$, ${ub_coeff:.3f} \times 10^{{{ub_exp}}}$ {{\sffamily ]}}"
    val_str = f"{val_coeff:.2f}"
    # Construct LaTeX string
    latex_str = f"{val_str} \\times 10^{{{val_exp}}}"
    return latex_str, ci_str




def main(sublimation_rate_csv_list, target_temperature, pisces_a_d_flux_mean, pisces_y_d):
    load_plot_style()
    fig, axes = plt.subplots(2, 1, sharex=False, constrained_layout=True, height_ratios=[1, 0.15])
    fig.set_size_inches(5.5, 6.5)

    axes[0].set_yscale('log')
    axes[0].set_xlabel('T (K)')
    axes[1].set_xlabel('T (K)')
    axes[0].set_ylabel(r'{\sffamily Boron evaporation rate (atoms/cm\textsuperscript{2}-s)', usetex=True)
    axes[0].set_xlim(600, 2400)
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(200))
    axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(25))

    axes[0].set_ylim(1E7, 1E19)

    combined_df = pd.DataFrame(data={'Temperature (K)': [], 'Evaporation rate (atoms/cm^2/s)': []}).apply(pd.to_numeric)

    for item in sublimation_rate_csv_list:
        path_to_csv = item['path']
        label = 'B (' + item['type'] + ')'
        marker = item['m']
        mfc = item['mfc']
        evaporation_df: pd.DataFrame = load_csv(path_to_csv)
        combined_df = pd.concat([combined_df, evaporation_df]).reset_index(drop=True)
        combined_df.sort_values(by=['Temperature (K)'], ascending=True, inplace=True)
        combined_df = combined_df.reset_index(drop=True)
        plot_evaporation(ax=axes[0], df=evaporation_df, lbl=label, marker=marker, mfc=mfc)

    x = combined_df['Temperature (K)'].values
    y = np.log(combined_df['Evaporation rate (atoms/cm^2/s)'].values)

    fit_result: OptimizeResult = fit_polynomial(x, y, weights=1, poly_order=1, loss='soft_l1', f_scale=0.1)
    popt = fit_result.x
    ci =cf.confidence_interval(res=fit_result)

    x_pred = np.linspace(x.min(), x.max(), num=2000)
    ypred_log, ydelta_log = cf.prediction_intervals(model=model_poly, x_pred=x_pred, ls_res=fit_result, jac=jac_poly, new_observation=True)
    ypred, ydelta = np.exp(ypred_log), np.exp(ydelta_log)
    x_extra = np.linspace(600, 2400, num=3000)
    y_extra_log, ydelta_log = cf.prediction_intervals(model=model_poly, x_pred=x_extra, ls_res=fit_result, jac=jac_poly,
                                                    new_observation=True)
    y_extra = np.exp(y_extra_log)
    lb, ub = np.exp(y_extra_log - ydelta_log), np.exp(y_extra_log + ydelta_log)


    cs_lb = CubicSpline(x_extra, y_extra_log - ydelta_log)
    cs_ub = CubicSpline(x_extra, y_extra_log + ydelta_log)

    pp_lb = PPoly(cs_lb.c, cs_ub.x)
    pp_ub = PPoly(cs_ub.c, cs_lb.x)


    with h5py.File('boron_evaporation_model.hdf5', 'w') as hf:
        model_poly_log_gp = hf.create_group('model')
        popt_ds = model_poly_log_gp.create_dataset('popt', data=popt)
        popt_ds.attrs['description'] = ('Coefficients of the polynomial fit (in logarithmic scale) for the boron'
                                        r'evaporation rate (atoms/cm\textsuperscript{2}-s)')
        popt_ci = model_poly_log_gp.create_dataset('popt_ci', data=ci)
        popt_ci.attrs['description'] = '95% confidence interval for popt'
        lb_model_gp = model_poly_log_gp.create_group('lb_ppoly')
        lb_model_gp.attrs['description'] = 'Coefficients of the piecewise polynomial for the lower bound'
        lb_model_pp_c_ds = lb_model_gp.create_dataset('c', data=cs_lb.c)
        lb_model_pp_x_ds = lb_model_gp.create_dataset('x', data=cs_lb.x)

        ub_model_gp = model_poly_log_gp.create_group('ub_ppoly')
        ub_model_gp.attrs['description'] = 'Coefficients of the piecewise polynomial for the upper bound'
        ub_model_pp_c_ds = ub_model_gp.create_dataset('c', data=cs_ub.c)
        ub_model_pp_x_ds = ub_model_gp.create_dataset('x', data=cs_ub.x)


    axes[0].plot(x_extra, y_extra, ls='--', color='r', label='Extrapolation')
    axes[0].plot(x_pred, ypred, ls='-', color='r', label='Fit')
    axes[0].fill_between(x_extra, lb, ub, color='r', alpha=0.2)
    axes[0].plot(x_extra, np.exp(pp_lb(x_extra)), ls='--', color='k', label='Cubic spline', lw=1)
    axes[0].plot(x_extra, np.exp(pp_ub(x_extra)), ls='--', color='k', lw=1)


    axes[1].plot(x, chi_poly(popt, x, y), color='0.5')
    axes[1].set_title('Log residuals')

    axes[0].legend(loc='lower right', frameon=True, fontsize=10)

    fig.savefig('boron_evaporation_rates.png', dpi=600)
    plt.show()


def load_model(path_to_pppl_fit) \
        -> Callable[[Union[float, np.ndarray]], Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Load the fit from

    H.W. Kugel, Y. Hirooka, J. Timberlake et al., Initial boronization of PB-X using ablation of solid boronized probes.
    PPL-2903 (1993)

    Figure 16

    to estimate the evaporation rate at each time

    Parameters
    ----------
    path_to_pppl_fit: str, pathlib.Path

    Returns
    -------
    callable:
        The evaporation model
    """
    path_to_pppl_fit = Path(path_to_pppl_fit)
    with h5py.File(str(path_to_pppl_fit), 'r') as hf:
        # Load the coefficients of the polynomial fit (in log scale) for the boron evaporation rate in
        # (atoms/cm^2/s)
        model_popt = np.array(hf['/model/popt'])

        lb_ppoly_c = np.array(hf['/model/lb_ppoly/c'])
        lb_ppoly_x = np.array(hf['/model/lb_ppoly/x'])
        ub_ppoly_c = np.array(hf['/model/ub_ppoly/c'])
        ub_ppoly_x = np.array(hf['/model/ub_ppoly/x'])

    ppoly_lb = PPoly(lb_ppoly_c, lb_ppoly_x)
    ppoly_ub = PPoly(ub_ppoly_c, ub_ppoly_x)
    def evaporation_rate_model(temperature: Union[np.ndarray, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The evaporation rate in atoms/cm^2/s

        Parameters
        ----------
        temperature: np.ndarray
            The temperature in Kelvin

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The evaporation rate in atoms/cm^2/s
            the lower and upper bounds of the evaporation rate
        """
        if isinstance(temperature, float) or isinstance(temperature, int):
            temperature = np.array([temperature])
        rate = np.exp(model_poly(temperature, model_popt))
        lb = np.exp(ppoly_lb(temperature))
        ub = np.exp(ppoly_ub(temperature))
        return rate, lb, ub

    return evaporation_rate_model

if __name__ == '__main__':
    main(
        sublimation_rate_csv_list=SUBLIMATION_RATE_CSV_LIST, target_temperature=TARGET_TEMPERATURE,
        pisces_a_d_flux_mean=PISCES_A_D_FLUX_MEAN, pisces_y_d=PISCES_Y_D
    )

    """
    The model can be later loaded from the hdf5
    """
    model_evaporation = load_model(path_to_pppl_fit='boron_evaporation_model.hdf5')
    evaporation_rate, upper_bound, lower_bound = model_evaporation(temperature=2000)
    print(f'Evaporation rate: {evaporation_rate} 95% confidence interval: [{upper_bound}, {lower_bound}] B/cm^2/s')


