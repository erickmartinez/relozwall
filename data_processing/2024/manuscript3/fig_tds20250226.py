"""
This code tries to fit the approximate TDS profile according to

E. Tomkova/Surface Science 351 (1996) 309-318

"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import os
import json
import numpy as np
from scipy.optimize import least_squares, OptimizeResult, differential_evolution
import data_processing.confidence as cf
from scipy.stats.distributions import t
import matplotlib as mpl
from scipy.integrate import simpson
from data_processing.utils import latex_float, latex_float_with_error
from data_processing.utils import lighten_color
from WilsonBaskesv1 import ThermalDesorptionSimulator
from scipy.interpolate import interp1d

TDS_FILE_SBR = r'./data/TDS/20250213/Brod_mks_v3.txt'
TDS_FILE_ABPR = r'./data/TDS/20250213/Bpebble_srs_v3.txt'
TDS_FILE_PBPR = r'./data/TDS/20250213/Bpebble_crystalline_srs_v3.txt'

"""
+----------+-------------+------------+-------------+-----------+-----+
| Material | D0 (cm^2/s) | D0 (m^2/s) | Ed (kJ/mol) |  Ed (eV)  | Ref |
+----------+-------------+------------+-------------+-----------+-----+
| B4C      |      2.3E-4 |     2.3E-8 |     96 ± 10 | 1.0 ± 0.1 | [1] |
| B2O3     |      4.9E-5 |     4.9E-9 |    124 ± 15 | 2.3 ± 0.2 | [1] |
+----------+-------------+------------+-------------+-----------+-----+




Taking lambda parameter from beta-rhombohedral boron lattice parameter = 10 Å [2]

Assume that total retention is the integral of the trapped species over the distance.

[1] K. Schnarr and H. Münzel, "Release of tritium from boron carbide." (1990) J. Chem. Soc., Faraday Trans. 86, 651-656
doi: 10.1039/FT9908600651

[2] R.E. Hughes, C.H.L. Kennard, D.B. Sullenger, H.A. Weakliem, D.E. Sands, J.L. Hoard, "The Structure of β-Rhombohedral Boron"
(1963) Journal of the American Chemical Society, 85, 361
doi: https://doi.org/10.1021/ja00886a036
"""


MODEL_FIXED_PARAMS = dict(
    T0=300,  # Initial temperature (K)
    Tf=1300,  # Final temperature (K)
    beta=0.3,  # Heating rate (K/s)
    L=25e-6,  # Sample thickness (m)
    D0=2.3E-8,  # Diffusion pre-exponential (m²/s)
    Ed=1.0,  # Diffusion activation energy (eV)
    kr=3.2e-15,  # Recombination coefficient (m³/s)
    Er=1.16,
    lam=1E-9,  # The jump distance, taken from lattice parameter
    v0=1e13,  # attempt frequency for detrapping (1/s)
    density_host=2.31,  # Host density (g/cm³)
    atomic_mass_host=10.811,  # g/mol
)

k_b = 8.617333262E-5 # eV/K

def model(temperature: np.ndarray, params):
    global MODEL_FIXED_PARAMS
    T = temperature
    T0 = T.min()
    Tf = T.max()
    trap_filling, Et, L, D0, Ed = params
    simulator = ThermalDesorptionSimulator(
        Et=Et,
        trap_filling=trap_filling,
        T0=MODEL_FIXED_PARAMS['T0'],  # Initial temperature (K)
        Tf=MODEL_FIXED_PARAMS['Tf'],  # Final temperature (K)
        beta=MODEL_FIXED_PARAMS['beta'],  # Heating rate (K/s)
        L=L,  # Sample thickness (m)
        # D0=MODEL_FIXED_PARAMS['D0'],  # Diffusion pre-exponential (m²/s)
        # Ed=MODEL_FIXED_PARAMS['Ed'],  # Diffusion activation energy (eV)
        D0=D0,
        Ed=Ed,
        kr=MODEL_FIXED_PARAMS['kr'],  # Recombination coefficient (m³/s)
        Er=MODEL_FIXED_PARAMS['Er'],
        lam=MODEL_FIXED_PARAMS['lam'],  # The jump distance, taken from lattice parameter
        v0=MODEL_FIXED_PARAMS['v0'],  # attempt frequency for detrapping (1/s)
        density_host=MODEL_FIXED_PARAMS['density_host'],  # Host density (g/cm³)
        atomic_mass_host=MODEL_FIXED_PARAMS['atomic_mass_host'],  # g/mol
        nx=51,
        adapt_mesh=True
    )

    t_max = (simulator.Tf - simulator.T0) / simulator.beta
    solution = simulator.simulate(t_max)

    # Interpolate the solution at the selected points
    f = interp1d(x=solution['temperature'], y=solution['flux'])
    try:
        y = f(T)
    except ValueError as e:
        print(e)
        y = np.zeros_like(T)
        print(params)
        raise e
    return y

def residual(params, temperature, effusion):
    effusion_sim = model(temperature, params)
    eps = float(np.finfo(float).eps)
    # return np.log(np.abs(effusion)+eps) - np.log(np.abs(effusion_sim)+eps)
    return effusion_sim - effusion


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    rcParams.update(plot_style)
    rcParams['pgf.texsystem'] = 'pdflatex'
    rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{color}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')


def fit_tds(xdata, ydata, x0, bounds, loss='soft_l1', f_scale=0.1, tol=None) -> OptimizeResult:
    if tol is None:
        # tol = np.finfo(np.float64).eps
        tol = 1E-8

    lower_bounds, upper_bounds = bounds
    bounds_de = []
    for lb, ub in zip(lower_bounds, upper_bounds):
        bounds_de.append((lb, ub))

    # try:
    #     result_de: OptimizeResult = differential_evolution(
    #             func=residual,
    #             args=(xdata, ydata),
    #             x0=x0,
    #             bounds=bounds_de,
    #             maxiter=1000 * len(x0),
    #             # tol=tol,
    #             # atol=tol,
    #             workers=-1,
    #             updating='deferred',
    #             recombination=0.2,
    #             strategy='best1bin',
    #             mutation=(0.5, 1.5),
    #             init='sobol',
    #             polish=False,
    #             disp=True
    #         )
    # except ValueError as e:
    #     for lb, xi, ub in zip(lower_bounds, x0, upper_bounds):
    #         print(f"lb: {lb:.3g}, x0: {xi:.3g}, ub: {ub:.3g}")
    #         raise e

    result: OptimizeResult = least_squares(
        residual,
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
        # diff_step=1E-12,
        max_nfev=1000 * len(x0)
    )

    return result


def load_tds_data(path_to_csv):
    df = pd.read_csv(path_to_csv, comment = '#', delimiter = r'\s+').apply(pd.to_numeric)
    # df['[D/m^2/s]'] = np.where(df['[D/m^2/s]'] < 0, 0, df['[D/m^2/s]'])
    # df['[HD/m^2/s]'] = np.where(df['[HD/m^2/s]'] < 0, 0, df['[HD/m^2/s]'])
    # df['[D2/m^2/s]'] = np.where(df['[D2/m^2/s]'] < 0, 0, df['[D2/m^2/s]'])
    mean_heating_rate = np.mean(np.gradient(df['Temp[K]'].values, df['Time[s]'].values))
    print(f"file: {path_to_csv}, mean heating rate (K/s) {mean_heating_rate:.3E}")
    return df, mean_heating_rate

def plot_d_retention(axes, row, tds_df, color, title=None):
    # axes[row,0].plot(
    #     tds_df['Temp[K]'].values, tds_df['[D/m^2/s]'].values, marker='o',
    #     color=color, mew=1., mfc='none', ls='none', ms=7,
    #     label='Total D'
    # )
    axes[row].plot(
        tds_df['Temp[K]'].values, tds_df['[D2/m^2/s]'].values, marker='^',
        color=color, mew=1., mfc='none', ls='none', ms=7,
        label=r'D$_{\mathregular{2}}$'
    )
    if not title is None:
        axes[row].set_title(title)
        # axes[row, 1].set_title(title)

def plot_fit_model(axes, row, fit_result: OptimizeResult, x_pred: np.ndarray):
    popt = fit_result.x
    axes[row].plot(x_pred, model(x_pred, popt), lw=2., color='k', label='Model')


def get_deuterium_retention(tds_df: pd.DataFrame):
    time_s = tds_df['Time[s]'].values
    d_total = tds_df['[D/m^2/s]'].values
    dh = tds_df['[HD/m^2/s]'].values
    d2 = tds_df['[D2/m^2/s]'].values

    integrated_d = simpson(y=d_total, x=time_s)
    integrated_dh = simpson(y=dh, x=time_s)
    integrated_d2 = simpson(y=d2, x=time_s)
    # print(f'Retained D: {integrated_d:.3E}')
    return integrated_d, integrated_dh, integrated_d2

def save_fit_results(fit_result: OptimizeResult, path_to_tds_file, d_retention):
    popt = fit_result.x
    ci = cf.confidence_interval(fit_result)
    delta = ci[:,1] - popt[:]
    fit_result_df = pd.DataFrame(data={
        'Total desorption (1/m^2)': [popt[0]],
        'Total desorption error (1/m^2)': [delta[0]],
        'Et (eV)': [popt[1]],
        'Et _err (eV)': [delta[1]],
        'L (m)': [popt[2]],
        'L error (m)': [delta[2]],
        'D0 (m^2/s)': [popt[3]],
        'D0 error (m^2/s)': [delta[3]],
        'Ed (eV)': [popt[4]],
        'Ed error (eV)': [delta[4]]
    })

    print(fit_result_df)

    (integrated_d, integrated_dh, integrated_d2) = d_retention

    filename_without_extension = os.path.splitext(path_to_tds_file)[0]
    with open(filename_without_extension + '_fit_results.csv', 'w') as f:
        f.write(f'# Total D retention: {integrated_d:.3E} 1/m^2\n')
        f.write(f'# DH retention: {integrated_dh:.3E} 1/m^2\n')
        f.write(f'# D2 retention: {integrated_d2:.3E} 1/m^2\n')
        fit_result_df.to_csv(f, index=False)


def main(tds_file_sbr, tds_file_abpr, tds_file_pbpr, model_fixed_params):
    sintered_boron_rod_df, beta_sbr = load_tds_data(tds_file_sbr)
    abpr_df, beta_apbr = load_tds_data(tds_file_abpr)
    pbpr_df, beta_pbpr = load_tds_data(tds_file_pbpr)

    load_plot_style()

    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True, sharex=True)
    fig.set_size_inches(4.5, 5.5)

    plot_d_retention(axes, 0, sintered_boron_rod_df, color='C0', title='Solid boron')
    plot_d_retention(axes, 1, abpr_df, color='C1', title='Sample A')
    plot_d_retention(axes, 2, pbpr_df, color='C2', title='Sample B')

    """
    Fit the data for boron rod total d2  with single trap 
    """
    x_data_sbr = sintered_boron_rod_df['Temp[K]'].values
    y_data_sbr = sintered_boron_rod_df['[D2/m^2/s]'].values
    integrated_d, integrated_dh, integrated_d2 = get_deuterium_retention(tds_df=sintered_boron_rod_df)
    msk_fit = x_data_sbr >= 600
    MODEL_FIXED_PARAMS['beta'] = beta_sbr

    x0 = np.array([
        5.2E+22, 2.2, 2.3E-06, 5E-9, 1.15
    ])
    bounds = (
        [
            1E10, 1.5, 1E-10, 1E-9, 0.1
        ],
        [
            1E40, 10.0, 25E-6, 1E-6, 2.0
        ]
    )

    fit_result_sbr: OptimizeResult = fit_tds(
        xdata=x_data_sbr[msk_fit], ydata=y_data_sbr[msk_fit], x0=x0, bounds=bounds, loss='linear', f_scale=0.1
    )

    x_pred = np.linspace(300, 1200, 2000)


    save_fit_results(
        fit_result=fit_result_sbr, path_to_tds_file=tds_file_sbr, d_retention=get_deuterium_retention(sintered_boron_rod_df)
    )

    # axes[0].plot(x_pred, y_pred_sbr, color='k', label='Model')
    plot_fit_model(axes=axes, row=0, fit_result=fit_result_sbr, x_pred=x_pred)

    """
    Fit the data for boron pebble rod d-h with multiple desorption peaks
    The initial guess for the least_squares
    """
    x_data_abpr = abpr_df['Temp[K]'].values
    y_data_abpr = abpr_df['[D2/m^2/s]'].values
    MODEL_FIXED_PARAMS['beta'] = beta_apbr
    msk_fit = x_data_abpr >= 600
    integrated_d, integrated_dh, integrated_d2 = get_deuterium_retention(tds_df=abpr_df)

    x0 = np.array([
        1.24E+20, 2.8, 4E-06, 6E-9, 1.15
    ])
    bounds = (
        [
            1E10, 1.5, 1E-10, 1E-9, 0.1
        ],
        [
            1E40, 10.0, 25E-6, 1E-6, 2.0
        ]
    )

    fit_result_abpr: OptimizeResult = fit_tds(
        xdata=x_data_abpr, ydata=y_data_abpr, x0=x0, bounds=bounds, loss='linear', f_scale=0.1
    )


    plot_fit_model(axes=axes, row=1, fit_result=fit_result_abpr, x_pred=x_pred)

    save_fit_results(
        fit_result=fit_result_abpr, path_to_tds_file=tds_file_abpr, d_retention=get_deuterium_retention(abpr_df)
    )

    """
    Fit the data for boron rod (PBPR) total d2 with single desorption peak of order 1
    """
    x_data_pbpr = pbpr_df['Temp[K]'].values
    y_data_pbpr = pbpr_df['[D2/m^2/s]'].values
    msk_fit = x_data_pbpr >= 600
    MODEL_FIXED_PARAMS['beta'] = beta_pbpr
    integrated_d, integrated_dh, integrated_d2 = get_deuterium_retention(tds_df=pbpr_df)

    x0 = np.array([
        8E+19, 2.9, 2E-06, 6E-9, 1.
    ])
    bounds = (
        [
            1E10, 1.5, 1E-10, 1E-9, 0.1
        ],
        [
            1E40, 10.0, 25E-6, 1E-6, 2.0
        ]
    )

    fit_result_pbpr: OptimizeResult = fit_tds(
        xdata=x_data_pbpr[msk_fit], ydata=y_data_pbpr[msk_fit], x0=x0, bounds=bounds, loss='linear', f_scale=0.1
    )


    save_fit_results(
        fit_result=fit_result_pbpr, path_to_tds_file=tds_file_pbpr, d_retention=get_deuterium_retention(pbpr_df)
    )


    plot_fit_model(axes=axes, row=2, fit_result=fit_result_pbpr, x_pred=x_pred)

    for ax in axes.flatten():
        ax.ticklabel_format(axis='y', useMathText=True)
        ax.tick_params(axis='y', which='both', right=True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        ax.legend(loc='upper left', frameon=True, fontsize=10)

    # for ax in axes[-1]:
    axes[-1].set_xlim(300, 1200)
    fig.supxlabel('Temperature (K)')

    axes[0].set_ylim(0, 8E19)
    axes[1].set_ylim(0, 2E17)
    axes[2].set_ylim(0, 2E17)
    #
    # axes[0].yaxis.set_major_locator(ticker.MultipleLocator(1E20))
    # axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(5E19))
    #
    # axes[1].yaxis.set_major_locator(ticker.MultipleLocator(2E17))
    # axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(1E17))
    #
    # axes[2].yaxis.set_major_locator(ticker.MultipleLocator(2E17))
    # axes[2].yaxis.set_minor_locator(ticker.MultipleLocator(1E17))

    fig.supylabel(r'Desorption flux (m$^{\mathregular{-2}}$ s$^{\mathregular{-1}}$)')

    for i, axi in enumerate(axes.flatten()):
        panel_label = chr(ord('`') + i + 1) # starts from a
        # panel_label = chr(ord('`') + i + 3)
        axi.text(
            -0.12, 1.1, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )


    fig.savefig(r'./figures/fig_tds_plots_20250226-2svg', dpi=600)
    plt.show()



if __name__ == '__main__':
    main(
        tds_file_sbr=TDS_FILE_SBR, tds_file_abpr=TDS_FILE_ABPR, tds_file_pbpr=TDS_FILE_PBPR,
        model_fixed_params=MODEL_FIXED_PARAMS
    )