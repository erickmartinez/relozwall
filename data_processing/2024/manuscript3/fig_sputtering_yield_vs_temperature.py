import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpmath import degree
from scipy.stats.distributions import t

from edx_calibration import load_plot_style
from scipy.optimize import least_squares, OptimizeResult

BI_SPUTTERING_FILE = r"data/boron_physical_sputtering_yields.csv"
BD_SPUTTERING_FILE = r"./data/bd_sputtering_yields.csv"

SDTRIMSP_SPUTTERING_W_ERROR = [0.016705392, 0.007924338]

def load_sputtering_file(path_to_csv):
    df: pd.DataFrame = pd.read_csv(path_to_csv)
    numeric_cols = [
        'Elapsed time (s)',
        'Temperature (K)',
        'Gamma_B (1/cm^2/s)',
        'Gamma_B error (1/cm^2/s)',
        'Sputtering yield',
        'Sputtering yield error'
    ]
    df = df[df['Folder'] == 'echelle_20241031'].sort_values(by=['Temperature (K)'], ascending=True).reset_index(drop=True)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    df['Timestamp'] = df['Timestamp'].apply(pd.to_datetime)
    return df


PATH_TO_TRIM_XLS = r'./data/trim_results.xlsx'

def series_se(x: pd.Series):
    n = len(x)
    x = x.to_numpy()
    ddof = 1 if n > 1 else 0
    std = np.std(x, ddof=ddof)
    tval = t.ppf(1. - 0.5*0.05, n-1)
    se = std * tval / np.sqrt(n)
    return se

def series_mean_err(x: pd.Series):
    x = x.to_numpy()
    n = len(x)
    return np.linalg.norm(x) / n

def load_sputtering_data(path_to_csv, folder):
    df: pd.DataFrame = pd.read_csv(path_to_csv)
    numeric_cols = [
        'Elapsed time (s)',
        'Temperature (K)',
        'Gamma_B (1/cm^2/s)',
        'Gamma_B error (1/cm^2/s)',
        'Sputtering yield',
        'Sputtering yield error'
    ]
    df = df[df['Folder'] == folder].sort_values(by=['Temperature (K)'], ascending=True).reset_index(
        drop=True)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    df['Timestamp'] = df['Timestamp'].apply(pd.to_datetime)
    df['Temperature_rounded (K)'] = (df['Temperature (K)'] / 50).round().astype(int) * 50
    df_agg = df.groupby('Temperature_rounded (K)').agg({
        'Temperature_rounded (K)': ['mean', series_se], 'Temperature error (K)': [series_mean_err],
        'Sputtering yield': ['mean', series_se], 'Sputtering yield error': [series_mean_err]
    })
    out_df = pd.DataFrame(data={
        'Temperature_rounded (K)': df_agg['Temperature_rounded (K)']['mean'].values,
        'Temperature error (K)': np.linalg.norm(
            np.column_stack(
                [df_agg['Temperature_rounded (K)']['series_se'].values, df_agg['Temperature error (K)']['series_mean_err'].values]
            ),
            axis=1
        ),
        'Sputtering yield': df_agg['Sputtering yield']['mean'].values,
        'Sputtering yield error': df_agg['Sputtering yield error']['series_mean_err'].values
    })
    temperature_mean_error = out_df['Temperature error (K)'].mean()
    sputtering_mean_error = out_df['Sputtering yield error'].mean()
    out_df['Temperature error (K)'] =  out_df['Temperature error (K)'].fillna(value=temperature_mean_error)
    out_df['Sputtering yield error'] = out_df['Sputtering yield error'].fillna(value=sputtering_mean_error)
    # print(out_df)
    return out_df

def estimated_trim_weighted_sputtering_yield(trimsp_df: pd.DataFrame, main_energy):
    df = trimsp_df[trimsp_df['D+ energy (eV)'] == main_energy]
    sputtering_yield = df['sputtering  yield'].values
    ion_composition = df['ion composition'].values
    yield_mean = np.dot(sputtering_yield, ion_composition)
    yield_squared_mean = np.dot(sputtering_yield*sputtering_yield, ion_composition)

    # Estimate the t-val for a confidence level 0f 95%
    alpha = 1 - 0.95
    n = len(trimsp_df)
    tval = t.ppf(1. - alpha / 2, n - 1)
    yield_std = np.sqrt(np.abs(yield_squared_mean - yield_mean * yield_mean)) #* np.sqrt(n / (n - 1))
    yield_se = yield_std * tval / np.sqrt(n)
    return yield_mean, yield_std

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


eps = float(np.finfo(np.float64).eps)
def fit_polylog(xdata, ydata, xerror=None, yerror=None, weights=None, degree=5, loss='soft_l1', f_scale=1.0, tol=eps):
    if yerror is None:
        yerror = np.ones_like(xdata)
    if xerror is None:
        xerror = np.ones_like(xdata)
    if weights is None:
        weights = np.log(1 / (xerror +  yerror + 0.1 * np.median(yerror)))
    fit_result_g = least_squares(
        res_poly, x0=[0.1 ** i for i in range(degree)], args=(xdata, np.log(ydata), weights),
        loss=loss, f_scale=f_scale,
        jac=jac_poly,
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=2,
        x_scale='jac',
        max_nfev=1000 * degree
    )
    return fit_result_g

def main(bi_sputtering_file, bd_sputtering_file, sdtrim_yield):
    bi_sputtering_df = load_sputtering_file(bi_sputtering_file)
    bd_sputtering_df = load_sputtering_file(bd_sputtering_file)

    # # Get TRIMSP simulated sputtering yields
    # trimsp_df = pd.read_excel(path_to_trim_xls, sheet_name=0)
    # trimsp_sy, trimsp_sy_se = estimated_trim_weighted_sputtering_yield(trimsp_df, main_energy=42.)
    #
    # trimsp_sy_lb, _ = estimated_trim_weighted_sputtering_yield(trimsp_df, main_energy=35.)
    # trimsp_sy_ub, _ = estimated_trim_weighted_sputtering_yield(trimsp_df, main_energy=50.)

    trimsp_sy, trimsp_sy_delta = sdtrim_yield

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, constrained_layout=True)
    fig.set_size_inches(4., 5.)

    # ax.set_title('Sputtering yield')

    ax.set_xlim(400, 1050)
    ax.set_ylim(1E-7, 1)
    ax.set_yscale('log')

    xpred = np.linspace(bi_sputtering_df['Temperature (K)'].min(), bi_sputtering_df['Temperature (K)'].max(), num=500)

    """
    Plot data from boron pebble rods
    """
    apr_bi_sputtering_df = load_sputtering_data(path_to_csv=bi_sputtering_file, folder='echelle_20240827')
    markers_b, caps_b, bars_b = ax.errorbar(
        apr_bi_sputtering_df['Temperature_rounded (K)'], apr_bi_sputtering_df['Sputtering yield'],
        xerr=apr_bi_sputtering_df['Temperature error (K)'],
        yerr=apr_bi_sputtering_df['Sputtering yield error'],
        capsize=2.75, mew=1.25, marker='o', ms=8, elinewidth=1.25,
        color='C1', fillstyle='none',
        ls='none',
        label='Sample A (B-I)',
    )

    [bar.set_alpha(0.35) for bar in bars_b]

    # BD from boron pebble rods
    apr_bd_sputtering_df = load_sputtering_data(path_to_csv=bd_sputtering_file, folder='echelle_20240827')
    markers_b, caps_b, bars_b = ax.errorbar(
        apr_bd_sputtering_df['Temperature_rounded (K)'], apr_bd_sputtering_df['Sputtering yield'],
        xerr=100,
        yerr=apr_bd_sputtering_df['Sputtering yield error'],
        capsize=2.75, mew=1.25, marker='o', ms=8, elinewidth=1.25,
        color='C1', fillstyle='full',
        ls='none',
        label='Sample A (BD)',
    )

    [bar.set_alpha(0.35) for bar in bars_b]

    pcpr_bi_sputtering_df = load_sputtering_data(path_to_csv=bi_sputtering_file, folder='echelle_20241003')
    markers_b, caps_b, bars_b = ax.errorbar(
        pcpr_bi_sputtering_df['Temperature_rounded (K)'], pcpr_bi_sputtering_df['Sputtering yield'],
        xerr=100,
        yerr=pcpr_bi_sputtering_df['Sputtering yield error'],
        capsize=2.75, mew=1.25, marker='D', ms=8, elinewidth=1.25,
        color='C2', fillstyle='none',
        ls='none',
        label='Sample B (B-I)',
    )

    [bar.set_alpha(0.35) for bar in bars_b]



    pcpr_bd_sputtering_df = load_sputtering_data(path_to_csv=bd_sputtering_file, folder='echelle_20241003')
    markers_b, caps_b, bars_b = ax.errorbar(
        pcpr_bd_sputtering_df['Temperature_rounded (K)'], pcpr_bd_sputtering_df['Sputtering yield'],
        xerr=100,
        yerr=pcpr_bd_sputtering_df['Sputtering yield error'],
        capsize=2.75, mew=1.25, marker='D', ms=8, elinewidth=1.25,
        color='C2', fillstyle='full',
        ls='none',
        label='Sample B (BD)',
    )

    [bar.set_alpha(0.35) for bar in bars_b]

    chemical_sputtering = np.concatenate([
        bd_sputtering_df['Sputtering yield'].values,
        apr_bd_sputtering_df['Sputtering yield'].values,
        pcpr_bd_sputtering_df['Sputtering yield'].values
    ])

    chemical_sputtering_mean = chemical_sputtering.mean()
    chemical_sputtering_std = np.std(chemical_sputtering, ddof=1)
    confidence_level = 0.95
    alpha = 1 - confidence_level
    n_points_chemical_erosion = len(chemical_sputtering)
    t_val = t.ppf(1 - alpha/2, n_points_chemical_erosion - 1)
    chemical_sputtering_se = chemical_sputtering_std * t_val / np.sqrt(n_points_chemical_erosion)

    """
    Solid rod
    """

    markers_b, caps_b, bars_b = ax.errorbar(
        bi_sputtering_df['Temperature (K)'], bi_sputtering_df['Sputtering yield'],
        xerr=bi_sputtering_df['Temperature error (K)'],
        yerr=bi_sputtering_df['Sputtering yield error'],
        capsize=2.75, mew=1.25, marker='^', ms=8, elinewidth=1.25,
        color='tab:red', fillstyle='none',
        ls='none',
        label='Sample C (B-I)',
    )

    [bar.set_alpha(0.35) for bar in bars_b]

    sy = bi_sputtering_df['Sputtering yield'].values
    sy_err = bi_sputtering_df['Sputtering yield error'].values
    temp_err = bi_sputtering_df['Temperature error (K)']
    weights = np.abs((sy - np.mean(sy))  / (sy_err + temp_err + np.median(sy_err)))

    bi_sy_mean = bi_sputtering_df['Sputtering yield'].mean()
    bi_sy_std = bi_sputtering_df['Sputtering yield'].std(ddof=1)
    n = len(bi_sputtering_df)
    conf_level = 0.95
    alpha = 1 - conf_level
    tval = t.ppf(1 - alpha/2, n-1)
    bi_sy_se = bi_sy_std * tval / np.sqrt(n)

    # fit_result_bi: OptimizeResult = fit_polylog(
    #     xdata=bi_sputtering_df['Temperature (K)'],
    #     ydata=bi_sputtering_df['Sputtering yield'],
    #     xerror=bi_sputtering_df['Temperature error (K)'],
    #     yerror=bi_sputtering_df['Sputtering yield error'],
    #     weights=weights,
    #     f_scale=0.01,
    #     loss='cauchy',
    #     degree=1
    # )

    # ax.plot(
    #     xpred, np.exp(model_poly(xpred, fit_result_bi.x)),
    #     ls='--', c='tab:red'
    # )
    ax.plot(
        bi_sputtering_df['Temperature (K)'],
        np.ones(n)*bi_sy_mean, ls='--', c='tab:red', lw=1.25
    )
    ax.axhspan(ymin=bi_sy_mean - bi_sy_se, ymax=bi_sy_mean + bi_sy_se, color='tab:red', alpha=0.2)

    markers_b, caps_b, bars_b = ax.errorbar(
        bd_sputtering_df['Temperature (K)'], bd_sputtering_df['Sputtering yield'],
        xerr=bd_sputtering_df['Temperature error (K)'],
        yerr=bd_sputtering_df['Sputtering yield error'],
        capsize=2.75, mew=1.25, marker='^', ms=8, elinewidth=1.25,
        color='tab:red',
        ls='none',
        label='Sample C (BD)',
    )

    [bar.set_alpha(0.35) for bar in bars_b]

    fit_result_bd: OptimizeResult = fit_polylog(
        xdata=bd_sputtering_df['Temperature (K)'],
        ydata=bd_sputtering_df['Sputtering yield'],
        xerror=bd_sputtering_df['Temperature error (K)'],
        yerror=bd_sputtering_df['Sputtering yield error'],
        f_scale=0.1,
        loss='soft_l1',
        degree=5
    )

    ax.plot(
        xpred, np.exp(model_poly(xpred, fit_result_bd.x)),
        ls='--', c='tab:red'
    )





    ax.axhline(y=trimsp_sy, ls='-.', lw=1.5, color='tab:blue')
    # ax.axhspan(ymin=trimsp_sy-trimsp_sy_delta, ymax=trimsp_sy+trimsp_sy_delta, color='k', alpha=0.2)

    print(f"Y_BI: {bi_sy_mean:.3E} -/+ {bi_sy_se:.3E}")
    print(f"TRIM sputtering yield: {trimsp_sy:.4E} [{trimsp_sy-trimsp_sy_delta:.4E}, [{trimsp_sy+trimsp_sy_delta:.4E}]")
    print(f"Y_BD: {chemical_sputtering_mean:.3E} -/+ {chemical_sputtering_se:.3E}")

    ax.legend(bbox_to_anchor=(0., -.28, 1., .102), loc='upper left',
                      ncols=2, mode="expand", borderaxespad=0., frameon=True)

    ax.set_xlabel('Surface temperature (K)')
    ax.set_ylabel(r'Sputtering yield')

    fig.savefig(r"./figures/fig_sputtering_yield_vs_temperature.png", dpi=600)
    fig.savefig(r"./figures/fig_sputtering_yield_vs_temperature.svg", dpi=600)
    fig.savefig(r"./figures/fig_sputtering_yield_vs_temperature.pdf", dpi=600)
    plt.show()


if __name__ == '__main__':
    main(
        bi_sputtering_file=BI_SPUTTERING_FILE,
        bd_sputtering_file=BD_SPUTTERING_FILE,
       sdtrim_yield=SDTRIMSP_SPUTTERING_W_ERROR
    )