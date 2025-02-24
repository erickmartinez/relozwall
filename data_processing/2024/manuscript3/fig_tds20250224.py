"""
This code tries to fit the approximate TDS profile according to

E. Tomkova/Surface Science 351 (1996) 309-318

"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from docutils.nodes import label
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
from matplotlib.font_manager import FontProperties
from data_processing.utils import lighten_color
from mcnabb_foster import ThermalDesorptionSimulator
from scipy.interpolate import interp1d

TDS_FILE_SBR = r'./data/TDS/20250213/Brod_mks_v3.txt'
TDS_FILE_ABPR = r'./data/TDS/20250213/Bpebble_srs_v3.txt'
TDS_FILE_PBPR = r'./data/TDS/20250213/Bpebble_crystalline_srs_v3.txt'

"""
Reference diffusion coefficient of hydrogen in boron
+----------+-------------+------------+-------------+-----------+-----+
| Material | D0 (cm^2/s) | D0 (m^2/s) | Ed (kJ/mol) |  Ed (eV)  | Ref |
+----------+-------------+------------+-------------+-----------+-----+
| B4C      |      2.3E-4 |     2.3E-8 |     96 ± 10 | 1.0 ± 0.1 | [1] |
| B2O3     |      4.9E-5 |     4.9E-9 |    124 ± 15 | 2.3 ± 0.2 | [1] |
+----------+-------------+------------+-------------+-----------+-----+


[1] K. Schnarr and H. Münzel, "Release of tritium from boron carbide." (1990) J. Chem. Soc., Faraday Trans. 86, 651-656
doi: 10.1039/FT9908600651
"""


MODEL_FIXED_PARAMS = dict(
    beta=0.3,  # Heating rate (K/s)
    C0=5e25,  # Peak concentration (1/m³)
    L=1e-6,  # Sample thickness (m)
    k0=1e-15,  # Trapping coefficient pre-exponential (m³/s)
    Et=1.2,  # Trapping energy (eV)
    p0=1e13,  # Detrapping frequency factor (1/s)
    Eb=1.2,  # Binding energy (eV) for detrapping
    N=1e27,  # Trap density (1/m³)
    nx=1001  # Number of spatial points
)

k_b = 8.617333262E-5 # eV/K

def model(temperature: np.ndarray, params):
    global MODEL_FIXED_PARAMS
    T = temperature
    T0 = T.min()
    Tf = T.max()
    D0, Ed, C0 = params
    simulator = ThermalDesorptionSimulator(
        T0=T0-20,  # Initial temperature (K)
        Tf=Tf+20,  # Final temperature (K)
        beta=MODEL_FIXED_PARAMS['beta'],  # Heating rate (K/s)
        C0=C0,  # Peak concentration (1/m³)
        L=MODEL_FIXED_PARAMS['L'],  # Sample thickness (m)
        D0=D0,  # Diffusion pre-exponential (m²/s)
        Ed=Ed,  # Diffusion activation energy (eV)
        k0=MODEL_FIXED_PARAMS['k0'],  # Trapping coefficient pre-exponential (m³/s)
        Et=MODEL_FIXED_PARAMS['Et'],  # Trapping energy (eV)
        p0=MODEL_FIXED_PARAMS['p0'],  # Detrapping frequency factor (1/s)
        Eb=MODEL_FIXED_PARAMS['Eb'],  # Binding energy (eV) for detrapping
        N=MODEL_FIXED_PARAMS['N'],  # Trap density (1/m³)
        nx=MODEL_FIXED_PARAMS['nx']  # Number of spatial points
    )

    t_max = (simulator.Tf - simulator.T0) / simulator.beta
    solution = simulator.simulate(t_max)

    # Interpolate the solution at the selected points
    f = interp1d(x=solution['temperature'], y=solution['flux'])

    return f(T)

def residual(params, temperature, effusion):
    effusion_sim = model(temperature, params)
    return effusion - effusion_sim


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

def draw_retention_table(ax, retained_dh, retained_d2):
    col_labels = [r'Retention m$^{\mathregular{-2}}$']
    row_labels = [r'DH ', r'D$_{\mathregular{2}}$ ', r'Total D ']
    table1_vals = [
        [fr'${latex_float(retained_dh, significant_digits=1)}$'],
        [fr'${latex_float(retained_d2, significant_digits=1)}$'],
        [fr'${latex_float(retained_dh + 2.*retained_d2, significant_digits=1)}$']
    ]
    row_colors = ['C1', 'C2', 'C0']
    table = ax.table(
        cellText=table1_vals,
        colLabels=col_labels,
        rowLabels=row_labels,
        rowColours=row_colors,
        loc='upper left',
        bbox=[0.28, 0.54, 0.48, 0.44],
        edges='closed',
        fontsize=12,
        rowLoc='left',
    )
    # table_1.PAD=-0.1

    cells = table.get_celld()

    for (row, col), cell in cells.items():
        if col == -1:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
            cell.set_text_props(color='w')
    table.set_fontsize(12)


def fit_tds(xdata, ydata, x0, bounds, loss='soft_l1', f_scale=0.1, tol=None) -> OptimizeResult:
    if tol is None:
        # tol = np.finfo(np.float64).eps
        tol = 1E-4

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
        diff_step=1E-15,
        max_nfev=1000 * len(x0)
    )

    return result


def load_tds_data(path_to_csv) -> pd.DataFrame:
    df = pd.read_csv(path_to_csv, comment = '#', delimiter = r'\s+').apply(pd.to_numeric)
    df['[D/m^2/s]'] = np.where(df['[D/m^2/s]'] < 0, 0, df['[D/m^2/s]'])
    df['[HD/m^2/s]'] = np.where(df['[HD/m^2/s]'] < 0, 0, df['[HD/m^2/s]'])
    df['[D2/m^2/s]'] = np.where(df['[D2/m^2/s]'] < 0, 0, df['[D2/m^2/s]'])
    mean_heating_rate = np.mean(np.gradient(df['Temp[K]'].values, df['Time[s]'].values))
    print(f"file: {path_to_csv}, mean heating rate (K/s) {mean_heating_rate:.3E}")
    return df

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
        'D0 (m^2/s)': [popt[0]],
        'D0_error (m^2/s)': [delta[0]],
        'Ed (eV)': [popt[1]],
        'Ed _err (eV)': [delta[1]],
        'C0 (/m^3)': [popt[2]],
        'C0_err (/m^3)': [delta[2]],
    })

    print(fit_result_df)

    (integrated_d, integrated_dh, integrated_d2) = d_retention

    filename_without_extension = os.path.splitext(path_to_tds_file)[0]
    with open(filename_without_extension + '_fit_results.csv', 'w') as f:
        f.write(f'# Total D retention: {integrated_d:.3E} 1/m^2\n')
        f.write(f'# DH retention: {integrated_dh:.3E} 1/m^2\n')
        f.write(f'# D2 retention: {integrated_d2:.3E} 1/m^2\n')
        fit_result_df.to_csv(f, index=False)


def main(tds_file_sbr, tds_file_abpr, tds_file_pbpr):
    sintered_boron_rod_df = load_tds_data(tds_file_sbr)
    abpr_df = load_tds_data(tds_file_abpr)
    pbpr_df = load_tds_data(tds_file_pbpr)

    load_plot_style()

    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True, sharex=True)
    fig.set_size_inches(4.5, 5.5)

    plot_d_retention(axes, 0, sintered_boron_rod_df, color='C0', title='Boron rod (SBR)')
    plot_d_retention(axes, 1, abpr_df, color='C1', title='Boron pebble rod (ABPR)')
    plot_d_retention(axes, 2, pbpr_df, color='C2', title='Boron pebble rod (PBPR)')

    """
    Fit the data for boron rod total d2  with single desorption peak of order 1
    """
    x_data_sbr = sintered_boron_rod_df['Temp[K]'].values
    y_data_sbr = sintered_boron_rod_df['[D2/m^2/s]'].values
    msk_fit = x_data_sbr >= 600
    x0 = np.array([
        9.602984112810235e-09, 1.0707792071844882, 5.501928944902321e+25
    ])
    bounds = (
        [
            1E-9, 0.9, 1E23
        ],
        [
            1E-7, 1.5, 1E27
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
    msk_fit = x_data_abpr >= 600

    x0 = np.array([
        9.602984112810235e-09, 1.0707792071844882, 5.501928944902321e+23
    ])
    bounds = (
        [
            1E-9, 0.9, 1E20
        ],
        [
            9E-7, 1.2, 1E27
        ]
    )


    fit_result_abpr: OptimizeResult = fit_tds(
        xdata=x_data_abpr[msk_fit], ydata=y_data_abpr[msk_fit], x0=x0, bounds=bounds, loss='linear', f_scale=0.1
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

    x0 = np.array([
        9.602984112810235e-09, 1.0707792071844882, 5.501928944902321e+23
    ])
    bounds = (
        [
            1E-9, 0.9, 1E20
        ],
        [
            9E-7, 1.2, 1E27
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

    axes[0].set_ylim(0, 7E19)
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


    fig.savefig(r'./figures/fig_tds_plots_20250226.svg', dpi=600)
    plt.show()



if __name__ == '__main__':
    main(tds_file_sbr=TDS_FILE_SBR, tds_file_abpr=TDS_FILE_ABPR, tds_file_pbpr=TDS_FILE_PBPR)