from reliability.Fitters import Fit_Weibull_2P
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import platform

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import platform
from scipy.stats.distributions import t
from data_processing.utils import get_experiment_params
import json

platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'

base_path = r'Documents/ucsd/Postdoc/research/data/bending_tests'

bending_xlsx = 'bending_strength_vs_matrix_content.xlsx'

"""
Reliability Weibul 2P

alpha = scale parameter (alpha > 0)
beta = shape parameter (beta > 0)
"""


def normalize_path(the_path):
    global platform_system, drive_path
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    return os.path.join(drive_path, the_path)


def bending_to_tensile(bending_strength: np.ndarray, shape_parameter: float, dbs: np.ndarray, dsp: float):
    """
    Estimate the tensile strength from the 3 point bending strength using the shape parameter from the
    fitted Weibull distribution of failures. See

    Whitney, J.M., Knight, M. The relationship between tensile strength and flexure strength in
    fiber-reinforced composites. Experimental Mechanics 20, 211–216 (1980).
    https://doi.org/10.1007/BF02327601

    Parameters
    ----------
    bending_strength: float
        The bending strength
    shape_parameter: float
        The fitted shape parameter
    dbs: np.ndarray
        The error in the bending strength
    dsp: float
        The error in the fitted shape parameter

    Returns
    -------
    np.ndarray, np.ndarray
        An array with the tensile strength and the corresponding error
    """
    n = len(bending_strength)
    a = shape_parameter * np.ones(n, dtype=np.float64)
    x = a + 1.
    x2 = x ** 2.
    bya = 1. / a
    r = np.power(2. * x2, -bya)
    ft = r * bending_strength
    dfdsp = bending_strength * np.power(2., -bya) * np.power(x, -2. * bya) * (a * (np.log(2.) - 2.) + x * np.log(x2) + np.log(2.))
    dfdsp /= ((a ** 2.) * x)
    dfdbs = r
    dft = np.linalg.norm([dfdsp * dsp, dfdbs*dbs], axis=0)
    return r, ft, dft


def main():
    global base_path
    global bending_xlsx
    base_path = normalize_path(base_path)
    weibull_path = os.path.join(base_path, 'weibull')
    if not os.path.exists(weibull_path):
        os.makedirs(weibull_path)

    bending_xlsx = os.path.join(base_path, bending_xlsx)

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

    """
    Load bending data
    """
    bending_df = pd.read_excel(os.path.join(base_path, bending_xlsx), sheet_name=0)
    bending_df.drop(columns='Sample ID', inplace=True)
    bending_df = bending_df.apply(pd.to_numeric)

    matrix_wt_pct = list(bending_df['Matrix wt %'].unique())
    matrix_wt_pct.sort()
    weibull_type = np.dtype([
        ('Matrix wt %', 'd'), ('# samples', 'i'),
        ('Scale param', 'd'), ('Scale param SE', 'd'),
        ('Shape param', 'd'), ('Shape param SE', 'd'),
        ('R_b2t', 'd'), ('R_t2b', 'd'), ('dR_b2t', 'd')

    ])
    weibull_fits = np.empty(len(matrix_wt_pct), dtype=weibull_type)
    for i, w in enumerate(matrix_wt_pct):
        fig = plt.figure()
        bs = bending_df[bending_df['Matrix wt %'] == w]['Flexural strength (KPa)'].values
        wb = Fit_Weibull_2P(failures=bs, optimizer='best')
        ax_w = wb.probability_plot
        fig.axes.append(ax_w)
        fig.set_size_inches(6.5, 6.5)
        ax_w.set_xlabel('Bending strength (KPa)')
        ax_w.set_ylabel('CDF')
        ax_title = ax_w.get_title() + f' Matrix wt % {w:02.0f}'
        ax_w.set_title(ax_title)
        aa = wb.beta
        bya = 1. / aa
        xx = aa + 1.
        xx2 = xx ** 2.
        rr = np.power(2. * (1. + aa) ** 2., -1./aa)
        drda = np.power(2., -bya) * np.power(xx, -2. * bya) * (aa * (np.log(2.) - 2.) + xx * np.log(xx2) + np.log(2.))
        drda /= ((aa ** 2.) * xx)
        drr = np.abs(drda*wb.beta_SE)
        n = len(bs)
        weibull_fits[i] = (w, n, wb.alpha, wb.alpha_SE, wb.beta, wb.beta_SE, rr, 1/rr, drr)
        fig_name = f'weibull_fit_{w:02.0f}'
        fig.savefig(os.path.join(weibull_path, fig_name + '.png'), dpi=600)

    weibull_df = pd.DataFrame(weibull_fits)
    weibull_df.to_csv(os.path.join(weibull_path, 'weibull_fit_glassy_carbon.csv'), index=False)

    """
    Use the weibull shape parameter to estimate the tensile strength from the bending
    strength in bending df.
    """
    bending_df['B2T'] = np.nan
    bending_df['Tensile strength (KPa)'] = np.nan
    bending_df['Tensile strength error (KPa)'] = np.nan
    for i, row in weibull_df.iterrows():
        idx = bending_df['Matrix wt %'] == row['Matrix wt %']
        sp = row['Shape param']
        dsp = row['Shape param SE']
        bs = bending_df.loc[idx, 'Flexural strength (KPa)'].values
        dbs = bending_df.loc[idx, 'Flexural strength err (KPa)'].values
        ratio, tensile_strength, tensile_strength_error = bending_to_tensile(
            bending_strength=bs, shape_parameter=sp, dbs=dbs, dsp=dsp
        )
        bending_df.loc[idx, 'B2T'] = ratio
        bending_df.loc[idx, 'Tensile strength (KPa)'] = np.round(tensile_strength,0)
        bending_df.loc[idx, 'Tensile strength error (KPa)'] = np.round(tensile_strength_error,0)

    tensile_strength_df = bending_df
    tensile_strength_df.to_csv(os.path.join(base_path, 'tensile_strength_vs_matrix_content.csv'), index=False)



    plt.show()


if __name__ == '__main__':
    main()
