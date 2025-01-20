import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import least_squares, OptimizeResult
import matplotlib.ticker as ticker
import json
import os
from scipy.integrate import simpson
from scipy import sparse
from scipy.linalg import cholesky
import data_processing.confidence as cf
import re
from data_processing.utils import latex_float_with_error
from data_processing.utils import lighten_color


brightness_csv = r'./data/brightness_data_fitspy_wl-calibrated/echelle_20240815/MechelleSpect_007.csv'
FOLDER_MAP_XLS = r'./PISCES-A_folder_mapping.xlsx'
output_folder = r'./figures/Echelle_plots/B-I'
echelle_xlsx = r'./data/echelle_db.xlsx'
output_xls = r'./data/cd_bd_lorentzian.xlsx'

wl_range = (429., 434.5)
fit_range = (430, 433.5)

lorentz_peaks_guess = [
    433.13, 432.6, 431.65, 431.52, 431.25, 431.06, 430.93, 430.7, 430.5, 430.3, 430.1
]

def lorentzian(x, h, mu, gamma):
    return 2.*gamma*h/(np.pi)/(4*(x-mu)**2. + gamma**2.)

def res_sum_lorentzians(b, x, y):
    return sum_lorentzians(x, b) - y

def sum_lorentzians(x, b):
    m = len(x)
    n3 = len(b)
    selector = np.arange(0, n3) % 3
    h = b[selector == 0]
    mu = b[selector == 1]
    gamma = b[selector == 2]
    n = len(h)
    res = np.zeros(m)
    for i in range(n):
        res += h[i]*gamma[i] / ( ((x-mu[i])**2.) + (0.25* gamma[i] ** 2.) )
    return 0.5 * res / np.pi

def jac_sum_lorentzians(b, x, y):
    m = len(x)
    n3 = len(b)
    selector = np.arange(0, n3) % 3
    h = b[selector == 0]
    mu = b[selector == 1]
    gamma = b[selector == 2]
    n = len(h)
    res = np.empty((m, n3), dtype=np.float64)
    for i in range(n):
        g = gamma[i]
        g2 = g ** 2.
        xm = x - mu[i]
        xm2 = xm ** 2.
        den = (4. * xm2 + g2) ** 2.
        res[:, 3*i] = 0.5 * g / (xm2 + 0.25*g2)
        res[:, 3*i+1] = 16. * g * h[i] * xm / den
        res[:, 3*i+2] = h[i] * (8. * xm2 - 2. * g2) / den
    return res / np.pi

def load_folder_mapping():
    global FOLDER_MAP_XLS
    df = pd.read_excel(folder_map_xls, sheet_name=0)
    mapping = {}
    for i, row in df.iterrows():
        mapping[row['Echelle folder']] = row['Data label']
    return mapping

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')


def arpls(y, coef=4, ratio=0.05, itermax=10):
    r"""
    Asymmetrically Reweighted Penalized Least Squares smoothing extracted from:
    https://irfpy.irf.se/projects/ica/_modules/irfpy/ica/baseline.html#arpls

    Original article
    ----------------

    Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
    Analyst, 2015, 140, 250 (2015), https://doi.org/10.1039/C4AN01061B

    Parameters
    ----------
    y: numpy.ndarray(n)
        input data (i.e. spectrum intensity)
    coef: float, optional
        parameter that can be adjusted by user.
        The larger coef is, the smoother the resulting background, y_smooth
    ratio: float, optional
        wheighting deviations: 0 < ratio < 1, smaller values allow less negative
        values
    itermax: int, optional
        number of iterations to perform

    Returns
    -------
    y_smooth: numpy.ndarray(n)
        the fitted background
    """
    # pylint:disable=invalid-name, unused-variable

    N = len(y)
    D = sparse.eye(N, format='csc')
    # workaround: numpy.diff( ,2) does not work with sparse matrix
    D = D[1:] - D[:-1]
    D = D[1:] - D[:-1]

    H = 10 ** coef * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        WH = sparse.csc_matrix(W + H)
        C = sparse.csc_matrix(cholesky(WH.todense()))
        y_smooth = sparse.linalg.spsolve(C, sparse.linalg.spsolve(C.T, w * y))
        d = y - y_smooth
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt

    return y_smooth

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

temp_cols = [
    'Folder',
    'File',
    'CD H (photons/cm^2/s)',
    'CD H delta (photons/cm^2/s)',
    'CD mu (nm)',
    'CD mu delta (nm)',
    'CD gamma (nm)',
    'CD gamma delta (nm)',
    'BD H (photons/cm^2/s)',
    'BD H delta (photons/cm^2/s)',
    'BD mu (nm)',
    'BD mu delta (nm)',
    'BD gamma (nm)',
    'BD gamma delta (nm)',
    'Timestamp'
]

def load_output_db(xlsx_source):
    global temp_cols
    try:
        out_df: pd.DataFrame = pd.read_excel(xlsx_source, sheet_name=0)
    except Exception as e:
        out_df = pd.DataFrame(data={
            col: [] for col in cd_bd_columns
        })
    return out_df

def update_df(db_df:pd.DataFrame, row_data):
    row = pd.DataFrame(data={key: [val] for key, val in row_data.items()})
    if len(db_df) == 0:
        return row
    folder = row_data['Folder']
    file = row_data['File']
    # Try finding the folder and file in db_df
    row_index = (db_df['Folder'] == folder) & (db_df['File'] == file)
    previous_row = db_df[row_index]
    if len(previous_row) == 0:
        return pd.concat([db_df, row], ignore_index=True).reset_index(drop=True)
    row_index = db_df.loc[row_index].index[0]
    for col, val in row_data.items():
        db_df.loc[row_index, col] = val
    return db_df

def estimate_lorentz_param(x, y, mu, d=0.1, fwhm=None):
    msk = ((mu - d) <= x) & (x <= (mu + d))
    x_search = x[msk]
    y_search = y[msk]
    y_max = y_search.max()
    idx_max = np.argmin(np.abs(y_search - y_max))
    x_peak = x_search[idx_max]
    if not fwhm is None:
        g = fwhm
        h = 0.5 * np.pi * g * y_max
        return h, x_peak, g
    hm = y_max * 0.5
    idx_hm = np.argmin(np.abs(y_search - hm))
    x_hm = x_search[idx_hm]
    g = 2. * np.abs(x_hm - x_peak)
    h = 0.5 * np.pi * g * y_max
    return h, x_peak, g

def main():
    global brightness_csv, d_pattern, wl_range, output_folder, echelle_xlsx
    global temp_cols, lorentz_peaks_guess
    # Get the relative path to the echelle file
    relative_path = os.path.dirname(brightness_csv)
    # Define the file tag as the echelle file without the extension
    file_tag = os.path.splitext(os.path.basename(brightness_csv))[0]
    # df, params = load_brightness_file(path_to_file=brightness_csv)
    full_df = pd.read_csv(brightness_csv, comment='#').apply(pd.to_numeric)
    # Focus only on the wavelength region defined in wl_range
    df = full_df[full_df['Wavelength (nm)'].between(wl_range[0], wl_range[1])].reset_index(drop=True)
    wavelength = df['Wavelength (nm)'].values
    photon_flux = df['Brightness (photons/cm^2/s/nm)'].values

    fit_df =  full_df[full_df['Wavelength (nm)'].between(fit_range[0], fit_range[1])].reset_index(drop=True)
    wavelength_fit = fit_df['Wavelength (nm)'].values
    photon_flux_fit = fit_df['Brightness (photons/cm^2/s/nm)'].values
    baseline_fit = arpls(photon_flux_fit, coef=5, itermax=1000)
    baselined_photon_flux_fit = photon_flux_fit# - baseline_fit

    # Fit the baseline with a polynomial
    eps = float(np.finfo(np.float64).eps)
    # res_ls_poly = least_squares(
    #     res_poly, x0=[0.01**i for i in range(5)], args=(wavelength_fit, baselined_photon_flux_fit),
    #     loss='linear', f_scale=0.1,
    #     jac=jac_poly,
    #     xtol=eps,
    #     ftol=eps,
    #     gtol=eps,
    #     verbose=2,
    #     max_nfev=10000 * len(wavelength_fit)
    # )


    params_df: pd.DataFrame = pd.read_excel(echelle_xlsx, sheet_name=0)
    # Get the elapased time since the first spectrum for each spectrum in the folder
    params_df['Timestamp'] = params_df['Timestamp'].apply(pd.to_datetime)
    params_df['Elapsed time (s)'] = (params_df['Timestamp'] - params_df[
        'Timestamp'].min()).dt.total_seconds()  # Arbitrary value for now, different t0 for every folder
    unique_folders = params_df['Folder'].unique()
    for folder in unique_folders:
        row_indexes = params_df['Folder'] == folder
        ts = params_df.loc[row_indexes, 'Timestamp'].reset_index(drop=True)
        params_df.loc[row_indexes, 'Elapsed time (s)'] = (params_df.loc[row_indexes, 'Timestamp'] - ts[0]).dt.seconds

    # Find the timestamp for the selected plot
    folder = os.path.basename(os.path.dirname(brightness_csv))
    file = file_tag + '.asc'
    try:
        selected_folder_df = params_df[(params_df['Folder'] == folder) & (params_df['File'] == file)].reset_index(
            drop=True)
        elapsed_time = selected_folder_df['Elapsed time (s)'][0]
        timestamp = selected_folder_df['Timestamp'][0]
        print(f"Elapsed time: {elapsed_time}")
    except KeyError:
        print(f"Could not find Folder '{folder}', File '{file}' in the echelle_db")
        print(params_df[['Folder', 'File', 'Elapsed time (s)']])
        exit(-1)

    # Use folder_map_xls to map the dated folder to the corresponding sample
    folder_mapping = load_folder_mapping()
    dated_folder = os.path.basename(relative_path)
    sample_label = folder_mapping[dated_folder]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    path_to_figures = os.path.join(output_folder, dated_folder)
    if not os.path.exists(path_to_figures):
        os.makedirs(path_to_figures)

    # Try to fit the spectrum
    n_peaks = len(lorentz_peaks_guess)
    x0 = np.empty(3*n_peaks)
    lb = np.empty(3*n_peaks)
    ub = np.empty(3*n_peaks)
    msk_gtz = baselined_photon_flux_fit >= 0.

    print(f"****** GUESS ESTIMATION ********")
    for i, pc in enumerate(lorentz_peaks_guess):
        hi, mui, gi = estimate_lorentz_param(wavelength_fit, baselined_photon_flux_fit, pc, d=0.2, fwhm=0.1)
        print(f"h[{i:>2d}]: {hi:.3E}, mu[{i:>2d}]: {mui:.3E}, g[{i:>2d}]: {gi:.3E}")
        idx_h = 3*i
        idx_m = idx_h+1
        idx_g = idx_m+1
        x0[idx_h] = hi * 0.1
        x0[idx_m] = mui
        x0[idx_g] = gi

        lb[idx_h] = eps
        lb[idx_m] = mui - 0.2
        lb[idx_g] = 1E-2

        ub[idx_h] = hi * 2
        ub[idx_m] = mui + 0.2
        ub[idx_g] = 1E20


    res_lsq = least_squares(
        res_sum_lorentzians, x0=x0, args=(wavelength_fit[msk_gtz], baselined_photon_flux_fit[msk_gtz]),
        loss='linear', f_scale=0.1,
        jac=jac_sum_lorentzians,
        bounds=(
            lb,
            ub
        ),
        xtol=eps,
        ftol=eps,
        gtol=eps,
        x_scale='jac',
        verbose=2,
        tr_solver='exact',
        max_nfev=10000 * len(wavelength_fit)
    )
    popt = res_lsq.x
    ci = cf.confidence_interval(res=res_lsq)
    popt_delta = ci[:, 1] - popt
    # print("popt_delta:", popt_delta)

    xpred = np.linspace(wavelength_fit.min(), wavelength_fit.max(), 500)
    ypred = sum_lorentzians(xpred, popt) #+ model_poly(xpred, res_ls_poly.x)


    load_plot_style()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True)
    fig.set_size_inches(6.5, 5.5)
    # fig.subplots_adjust(left=0.15, right=0.98, top=0.9, bottom=0.1, hspace=0)

    ax1.plot(wavelength, photon_flux, ls='-', marker='o', ms=4, color='C0', lw=1., mfc='none')
    ax2.plot(wavelength, photon_flux, ls='-', color=lighten_color('C0', 0.5), lw=1.)
    ax2.plot(wavelength_fit, photon_flux_fit, ls='-', marker='o', ms=4, color='C0', lw=1., mfc='none', label='Data', mew=1.5)
    # ax2.plot(wavelength_fit, baseline_fit, ls='-', color='tab:green', lw=1., label='Baseline')
    ax2.plot(xpred, ypred, ls='-',  color='red', lw=1., label='Fit')
    ax2.set_ylim(bottom=-0.1E12, top=photon_flux_fit.max()*1.5)

    peak_data_df = pd.DataFrame(data={
        'Peak': [],
        'x0': [], 'x0_err': [], 'x0_err_pct': [],
        'A': [], 'A_err': [], 'A_err_pct': [],
        'G': [], 'G_err': [], 'G_err_pct': []
    })
    peak_id = 1
    for i in range(n_peaks):
        idx_h = 3 * i
        idx_m = idx_h + 1
        idx_g = idx_m + 1
        hi, mui, gi = popt[idx_h], popt[idx_m], popt[idx_g]
        ypred_i = lorentzian(xpred, hi, mui, gi)
        ax2.fill_between(xpred, 0, ypred_i, ls='-',  lw=1., alpha=0.3)
        peak_data = {
            'Peak': f"L{i}",
            'x0': popt[idx_m],
            'x0_err': popt_delta[idx_m],
            'x0_err_pct': np.round(100.*popt_delta[idx_m]/popt[idx_m],3),
            'A': np.round(popt[idx_h], decimals=3),
            'A_err': np.round(popt_delta[idx_h], decimals=3),
            'A_err_pct': np.round(100.*popt_delta[idx_h]/popt[idx_h],2),
            'G': popt[idx_g],
            'G_err': popt_delta[idx_g],
            'G_err_pct': np.round(100.*popt_delta[idx_g]/popt[idx_g],3),
        }
        peak_id += 1
        row = pd.DataFrame(data={key:[val] for key, val in peak_data.items()})
        peak_data_df = pd.concat([peak_data_df, row], ignore_index=True).reset_index(drop=True)
    ax2.set_xlabel(r"$\lambda$ {\sffamily (nm)}", usetex=True)
    fig.supylabel(r"$B_{\lambda}$ {\sffamily (photons/cm\textsuperscript{2}/s/nm)}", usetex=True)
    print(peak_data_df[['x0', 'x0_err_pct', 'A', 'A_err_pct', 'G_err', 'G_err_pct']])

    for ax in (ax1, ax2):
        ax.set_xlim(wl_range)
        mf = ticker.ScalarFormatter(useMathText=True)
        mf.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(mf)
        ax.ticklabel_format(useMathText=True)


    plt.show()


if __name__ == '__main__':
    main()