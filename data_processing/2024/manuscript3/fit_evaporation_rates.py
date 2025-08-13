import pandas as pd
import numpy as np

import data_processing.confidence as cf
from scipy.optimize import least_squares, OptimizeResult
import matplotlib.pyplot as plt

from fig_tds20250228 import load_plot_style
import matplotlib.ticker as ticker

from scipy.interpolate import interp1d


SUBLIMATION_RATE_CSV_LIST = [
    {'path': r'./data/boron_sublimation_rates/boron_evaporation_rate_type_a.csv', 'type': 'a', 'm': 'o', 'mfc': 'None'},
    {'path': r'./data/boron_sublimation_rates/boron_evaporation_rate_type_b.csv', 'type': 'b', 'm': 'o', 'mfc': 'C0'},
    {'path': r'./data/boron_sublimation_rates/boron_evaporation_rate_type_c.csv', 'type': 'c', 'm': 's', 'mfc': 'None'},
    {'path': r'./data/boron_sublimation_rates/boron_evaporation_rate_type_d.csv', 'type': 'd', 'm': 'x', 'mfc': 'None'},
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
        >>> format_latex_with_bounds(1.234e16, 1.0E16, 1.5E16)
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
    fig.set_size_inches(4.5, 6.5)

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

    x_pred = np.linspace(x.min(), x.max(), num=2000)
    ypred_log, ydelta_log = cf.prediction_intervals(model=model_poly, x_pred=x_pred, ls_res=fit_result, jac=jac_poly, new_observation=True)
    ypred, ydelta = np.exp(ypred_log), np.exp(ydelta_log)
    x_extra = np.linspace(600, x.max(), num=3000)
    y_extra_log, ydelta_log = cf.prediction_intervals(model=model_poly, x_pred=x_extra, ls_res=fit_result, jac=jac_poly,
                                                    new_observation=True)
    y_extra = np.exp(y_extra_log)
    lb, ub = np.exp(y_extra_log - ydelta_log), np.exp(y_extra_log + ydelta_log)

    f_y_log = interp1d(x_extra, y_extra_log)
    f_lb_log = interp1d(x_extra, y_extra_log - ydelta_log)
    f_ub_log = interp1d(x_extra, y_extra_log + ydelta_log)
    f_y = interp1d(x_extra, y_extra)
    f_lb = interp1d(x_extra, lb)
    f_ub = interp1d(x_extra, ub)

    y_t = f_y(target_temperature)
    y_t_le = y_t - f_lb(target_temperature)
    y_t_ue = f_ub(target_temperature) - y_t

    print(f"Target_temperature: {target_temperature:.0f} K")
    # print(f"Log(extrapolated rate): {f_y_log(target_temperature):.3E} log(B/cm^2/s)")
    # print(f"Log(lower bond extrapolated rate): {f_lb_log(target_temperature):.3E} log(B/cm^2/s)")
    # print(f"Log(upper bond extrapolated rate): {f_ub_log(target_temperature):.3E} log(B/cm^2/s)")
    print(f"Extrapolated rate: {y_t* 1E4:.3E} log(B/m^2/s)")
    print(f"Lower bond extrapolated rate: {f_lb(target_temperature)* 1E4:.3E} log(B/m^2/s)")
    print(f"Upper bond extrapolated rate: {f_ub(target_temperature)* 1E4:.3E} log(B/m^2/s)")



    # Estimate the total sublimated boron during the whole experiment at PISCES
    experiment_time = 1.5 * 3600. # s
    boron_density = 2.35 # g / cm^3
    boron_molar_mass = 10.811 # g/mol
    rod_diameter = 0.4 * 2.54
    rod_front_area = 0.25 * np.pi * rod_diameter ** 2.
    rod_front_area = 1
    evaporation_rate_rod = y_t * boron_molar_mass / 6.02214076e+23  * 1E4
    evaporation_rate_rod_ul = f_ub(target_temperature) * boron_molar_mass / 6.02214076e+23 * 1E4
    evaporation_rate_rod_ll = f_lb(target_temperature) * boron_molar_mass / 6.02214076e+23 * 1E4
    physical_erosion_rate = pisces_a_d_flux_mean * pisces_y_d * boron_molar_mass / 6.02214076e+23
    print(f"Rod front area: {rod_front_area} cm^2")
    print(f"Evaporation rate boron in PISCES-A experiment: {evaporation_rate_rod:.2E} g/m^2/s")
    print(f"Evaporation rate boron in PISCES-A experiment (ul): {f_ub(target_temperature):.2E} atoms/m^2/s")
    print(f"Evaporation rate boron in PISCES-A experiment (ul): {f_ub(target_temperature)/3.22E19:.2E} Torr-L/m^2/s")

    print(f"Lower bound: {evaporation_rate_rod_ll:.2E} g/m^2/s")
    print(f"Upper bound: {evaporation_rate_rod_ul:.2E} g/m^2/s")

    print(f"Physical erosion rate boron in PISCES-A experiment: {physical_erosion_rate:.2E} g/cm/s")


    # Estimate boron erosion for a reactor according to Stangeby 2022
    # 0.025 P_heat = E_cx_T * Phi_cx_T
    # Erosion = Y_cx * Phi_cx_T
    E_cx_T = 300. # eV
    P_heat = 400E6 # J/s
    Y_cx = 0.056
    tau_annual = 2.5E7 # s/yr
    b_erosion_reactor_gs = Y_cx * 0.025 * P_heat / (E_cx_T * 1.602176634e-19) * boron_molar_mass / 6.02214076e+23
    b_erosion_reactor_kgyr = b_erosion_reactor_gs * tau_annual * 1E-3
    print(f"Stangeby 2022 - B erosion in fusion reactor: {b_erosion_reactor_gs:.2f} g/s")
    print(f"Stangeby 2022 - B erosion in fusion reactor: {b_erosion_reactor_kgyr:.0f} kg/yr")

    print(f"Evaporation rate boron in reactor: {evaporation_rate_rod*tau_annual*1E-3:.2f} kg/yr")





    axes[0].plot(x_extra, y_extra, ls='--', color='r', label='Extrapolation')
    axes[0].plot(x_pred, ypred, ls='-', color='r', label='Fit')
    axes[0].fill_between(x_extra, lb, ub, color='r', alpha=0.2)

    markers_p, caps_p, bars_p = axes[0].errorbar(
        [target_temperature], [y_t], yerr=([y_t_le], [y_t_ue]),
        marker='D', ms=7, mew=1.5, mfc='none', label=f'Target',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='k', ls='none'
    )

    [bar.set_alpha(0.25) for bar in bars_p]
    [cap.set_alpha(0.25) for cap in caps_p]

    axes[1].plot(x, chi_poly(popt, x, y), color='0.5')
    axes[1].set_title('Log residuals')

    axes[0].legend(loc='upper left', frameon=True, fontsize=10)

    extrapolated_txt = f"${target_temperature:.0f}\;\mathrm{{K}}$ \n"
    y_t_str, y_t_ci_str = format_latex_with_bounds(y_t, f_lb(target_temperature), f_ub(target_temperature))
    extrapolated_txt += r"{\sffamily Rate:} " +f"${y_t_str}$" + r" {\sffamily B/cm\textsuperscript{2}-s}" + "\n"
    extrapolated_txt += f"{y_t_ci_str}"
    connectionstyle = "angle,angleA=0,angleB=60,rad=0"
    bbox = dict(boxstyle="round", fc="wheat")
    arrowprops = dict(
        arrowstyle="->", color="k",
        shrinkA=5, shrinkB=5,
        patchA=None, patchB=None,
        connectionstyle=connectionstyle
    )
    axes[0].annotate(
        extrapolated_txt,
        xy=(target_temperature, y_t), xycoords='data',  # 'figure pixels', #data',
        # transform=axes[1].transAxes,
        xytext=(200, 60), textcoords='offset pixels',
        ha='left', va='bottom',
        fontsize=11,
        arrowprops=arrowprops,
        bbox=bbox,
        usetex=True
    )


    plt.show()


if __name__ == '__main__':
    main(
        sublimation_rate_csv_list=SUBLIMATION_RATE_CSV_LIST, target_temperature=TARGET_TEMPERATURE,
        pisces_a_d_flux_mean=PISCES_A_D_FLUX_MEAN, pisces_y_d=PISCES_Y_D
    )


