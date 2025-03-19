import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json
import matplotlib.ticker as ticker
from scipy.optimize import least_squares, OptimizeResult
from pathlib import Path

PATH_TO_DATA = r'./data/20250513.xlsx'


def load_data(path):
    dome_df = pd.read_excel(path, sheet_name='TLD1')
    reference_df = pd.read_excel(path, sheet_name='TLD1-back')
    reference_id2ig = reference_df.loc[0, 'ID/IG']
    return dome_df, reference_id2ig

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
        x0=[(0.01) ** (i-1) for i in range(poly_order+1)],
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

def main(path_to_data):
    peak_df, ref_ratio = load_data(path_to_data)
    path_to_xlsx = Path(path_to_data)
    base_name = path_to_xlsx.name.split()
    print(base_name)

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3.)

    x = (peak_df['x (um)'].values - peak_df['x (um)'].min()) * 1E-3

    raman_ratio = peak_df['ID/IG'].values

    fit_result: OptimizeResult = fit_polynomial(
        x=x, y=raman_ratio, loss='soft_l1', f_scale=0.1, poly_order=2
    )

    print('popt:', fit_result.x)

    x_pred = np.linspace(x.min(), x.max(), 1000)
    y_pred = model_poly(x_pred, fit_result.x)

    ax.plot(x, raman_ratio, marker='o', ls='none', mfc='None', label='Dome')
    ax.plot([x.min(), x.max()], [ref_ratio, ref_ratio], ls='--', color='tab:red', label='Back of dome')
    ax.plot(x_pred, y_pred, color='k', label='Fit')


    ax.legend(loc='best', frameon=True)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel(r'{\sffamily I\textsubscript{D}/I\textsubscript{G}}', usetex=True)
    ax.set_title('DiMES dome #1')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.5)

    # fig.savefig(r'./figures/tld1_20250224_id2ig.png', dpi=600)

    plt.show()


if __name__ == '__main__':
    main(PATH_TO_DATA)


