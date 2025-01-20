import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from edx_calibration import load_plot_style
from scipy.optimize import least_squares, OptimizeResult

BI_SPUTTERING_FILE = r"./data/boron_physical_sputtering_yields.csv"
BD_SPUTTERING_FILE = r"./data/bd_sputtering_yields.csv"

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
def fit_polylog(xdata, ydata, xerror=None, yerror=None, degree=5, loss='soft_l1', f_scale=1.0, tol=eps):
    if yerror is None:
        yerror = np.ones_like(xdata)
    if xerror is None:
        xerror = np.ones_like(xdata)
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

def main(bi_sputtering_file, bd_sputtering_file):
    bi_sputtering_df = load_sputtering_file(bi_sputtering_file)
    bd_sputtering_df = load_sputtering_file(bd_sputtering_file)

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, constrained_layout=True)
    fig.set_size_inches(4., 4.5)

    # ax.set_title('Sputtering yield')

    ax.set_xlim(750, 1050)
    ax.set_ylim(1E-7, 1E-1)
    ax.set_yscale('log')

    xpred = np.linspace(bi_sputtering_df['Temperature (K)'].min(), bi_sputtering_df['Temperature (K)'].max(), num=500)

    markers_b, caps_b, bars_b = ax.errorbar(
        bi_sputtering_df['Temperature (K)'], bi_sputtering_df['Sputtering yield'],
        xerr=bi_sputtering_df['Temperature error (K)'],
        yerr=bi_sputtering_df['Sputtering yield error'],
        capsize=2.75, mew=1.25, marker='^', ms=8, elinewidth=1.25,
        color='tab:red', fillstyle='none',
        ls='none',
        label='B (neutral)',
    )

    [bar.set_alpha(0.35) for bar in bars_b]

    fit_result_bi: OptimizeResult = fit_polylog(
        xdata=bi_sputtering_df['Temperature (K)'],
        ydata=bi_sputtering_df['Sputtering yield'],
        xerror=bi_sputtering_df['Temperature error (K)'],
        yerror=bi_sputtering_df['Sputtering yield error'],
        f_scale=0.1
    )

    ax.plot(
        xpred, np.exp(model_poly(xpred, fit_result_bi.x)),
        ls='--', c='tab:red'
    )

    markers_b, caps_b, bars_b = ax.errorbar(
        bd_sputtering_df['Temperature (K)'], bd_sputtering_df['Sputtering yield'],
        xerr=bd_sputtering_df['Temperature error (K)'],
        yerr=bd_sputtering_df['Sputtering yield error'],
        capsize=2.75, mew=1.25, marker='^', ms=8, elinewidth=1.25,
        color='tab:red',
        ls='none',
        label='BD',
    )

    [bar.set_alpha(0.35) for bar in bars_b]

    fit_result_bd: OptimizeResult = fit_polylog(
        xdata=bd_sputtering_df['Temperature (K)'],
        ydata=bd_sputtering_df['Sputtering yield'],
        xerror=bd_sputtering_df['Temperature error (K)'],
        yerror=bd_sputtering_df['Sputtering yield error'],
        f_scale=0.1
    )

    ax.plot(
        xpred, np.exp(model_poly(xpred, fit_result_bd.x)),
        ls='--', c='tab:red'
    )

    ax.legend(loc='center left', frameon=True)

    fig.supxlabel('Surface temperature (K)')
    fig.supylabel(r'Sputtering yield')

    fig.savefig(r"./figures/fig_sputtering_rage_vs_temperature.png", dpi=600)
    fig.savefig(r"./figures/fig_sputtering_rage_vs_temperature.svg", dpi=600)
    fig.savefig(r"./figures/fig_sputtering_rage_vs_temperature.pdf", dpi=600)
    plt.show()


if __name__ == '__main__':
    main(
        bi_sputtering_file=BI_SPUTTERING_FILE,
        bd_sputtering_file=BD_SPUTTERING_FILE
    )