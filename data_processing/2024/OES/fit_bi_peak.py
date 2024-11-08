import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import least_squares, OptimizeResult
import matplotlib.ticker as ticker
import json
import os
from datetime import timedelta, datetime
import warnings
from scipy.integrate import simpson, trapezoid
import data_processing.confidence as cf
import re
from data_processing.utils import latex_float_with_error


brightness_csv = r'./data/brightness_data/echelle_20240815/MechelleSpect_003.csv'
folder_map_xls = r'./PISCES-A_folder_mapping.xlsx'
output_folder = r'./figures/Echelle_plots/B-I'
subtract_background = True

# calibration_line = {'center_wl': 433.93, 'label': r'D$_{\gamma}$'}
calibration_line = {'center_wl': 821.2, 'label': r'B I'}

wl_range = (818, 824)

d_pattern = '%a %b %d %H:%M:%S %Y'
window_coefficients = np.array([12.783, 0.13065, -8.467e-5])

f_pattern = re.compile(r'\s*\#\s*(.*)?\:\s+(.*)')

over_sqrt_pi = 1. / np.sqrt(2. * np.pi)
def gaussian(x, c, sigma, mu):
    global over_sqrt_pi
    p = c * over_sqrt_pi / sigma
    arg = 0.5 * np.power((x - mu) / sigma, 2)
    return p * np.exp(-arg)

def sum_gaussians(x, b):
    global over_sqrt_pi
    m = len(x)
    nn = len(b)
    # Assume that the new b contains a list ordered like
    # (c1, sigma_1, mu1, c_2, sigma_2, mu_2, ..., c_n, sigma_n, mu_n)
    selector = np.arange(0, nn) % 3
    msk_c = selector == 0
    msk_sigma = selector == 1
    msk_mu = selector == 2
    cs = b[msk_c]
    sigmas = b[msk_sigma]
    mus = b[msk_mu]
    n = len(mus)
    u = over_sqrt_pi * np.power(sigmas, -1) * cs
    u = u.reshape((1, len(u)))
    v = np.zeros((n, m), dtype=np.float64)
    for i in range(len(sigmas)):
        arg = 0.5*np.power((x-mus[i])/sigmas[i], 2.)
        v[i, :] = np.exp(-arg)
    res = np.dot(u, v)
    return res.flatten()


def res_sum_gauss(b, x, y):
    return sum_gaussians(x, b) - y

def jac_sum_gauss(b, x, y):
    m, nn  = len(x), len(b)
    # Assume that the new b contains a list ordered like
    # (c1, sigma_1, mu1, c_2, sigma_2, mu_2, ..., c_n, sigma_n, mu_n)
    selector = np.arange(0, nn) % 3
    msk_c = selector == 0
    msk_sigma = selector == 1
    msk_mu = selector == 2
    c = b[msk_c]
    s = b[msk_sigma]
    mu = b[msk_mu]
    r = np.zeros((m, nn), dtype=np.float64)
    for i in range(len(s)):
        k = 3 * i
        g = gaussian(x, c[i], s[i], mu[i])
        r[:, k] = g / c[i]
        r[:, k+1] = np.power(s[i], -1) * ( np.power( (x - mu[i]) / s[i], 2) - 1.) * g
        r[:, k+2] = np.power(s[i], -2) * ( x - mu[i]) * g

    return r

def load_brightness_file(path_to_file):
    params = {}
    with open(path_to_file, 'r') as f:
        for line in f:
            matches = f_pattern.match(line)
            if matches:
                params[matches.group(1)] = matches.group(2)
            if not line.startswith('#'):
                break
    df: pd.DataFrame = pd.read_csv(path_to_file, comment='#').apply(pd.to_numeric)
    return df, params


def load_folder_mapping():
    global folder_map_xls
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




def main():
    global brightness_csv, d_pattern, wl_range, output_folder
    # Get the relative path to the echelle file
    relative_path = os.path.dirname(brightness_csv)
    # Define the file tag as the echelle file without the extension
    file_tag = os.path.splitext(os.path.basename(brightness_csv))[0]
    df, params = load_brightness_file(path_to_file=brightness_csv)
    # Focus only on the wavelength region defined in wl_rangew
    df = df[df['Wavelength (nm)'].between(wl_range[0], wl_range[1])].reset_index(drop=True)
    wavelength = df['Wavelength (nm)'].values
    photon_flux = df['Brightness (photons/cm^2/s/nm)'].values

    timestamp = datetime.strptime(params['Date and Time'], d_pattern)

    # Use folder_map_xls to map the dated folder to the corresponding sample
    folder_mapping = load_folder_mapping()
    dated_folder = os.path.basename(relative_path)
    sample_label = folder_mapping[dated_folder]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    path_to_figures = os.path.join(output_folder, dated_folder)
    if not os.path.exists(path_to_figures):
        os.makedirs(path_to_figures)

    # Fit a gaussian to the main line
    n_peaks = 1
    line_center = calibration_line['center_wl']
    delta_wl = 0.25
    msk_window = ((line_center - delta_wl) <= wavelength) & (wavelength <= (line_center + delta_wl))
    wl_window = wavelength[msk_window]
    intensity_window = photon_flux[msk_window]
    peak_height = intensity_window.max()
    idx_peak = np.argmin(np.abs(peak_height - intensity_window))
    wl_peak = wl_window[idx_peak]
    # recalculate the window around the peak from the data
    msk_window = ((wl_peak - delta_wl) <= wavelength) & (wavelength <= (wl_peak + delta_wl))
    wl_window = wavelength[msk_window]
    intensity_window = photon_flux[msk_window]
    area_window = simpson(y=intensity_window, x=wl_window)
    c_window = area_window * over_sqrt_pi

    x0 = [c_window, (wl_window.max() - wl_window.min()), wl_peak]
    all_tol = float(np.finfo(np.float64).eps)
    res_lsq = least_squares(
        res_sum_gauss, x0=x0, args=(wl_window, intensity_window), loss='linear', f_scale=0.1,
        jac=jac_sum_gauss,
        bounds=(
            [0., 1E-5, wl_peak-0.15],
            [3.*c_window, np.inf, wl_peak+0.15]
        ),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        verbose=2,
        max_nfev=10000 * len(wl_window)
    )
    popt = res_lsq.x
    ci = cf.confidence_interval(res=res_lsq)
    popt_delta = ci[:, 1] - popt

    cmap = mpl.colormaps.get_cmap('rainbow')
    norm1 = mpl.colors.Normalize(vmin=0, vmax=n_peaks - 1)
    colors1 = [cmap(norm1(i)) for i in range(n_peaks)]

    table1 = r"\setcellgapes{1.5pt}\makegapedcells" # increase vertical padding in cells to avoid superscript overlaps
    table1 += r'''\begin{tabular}{ | r | c | c | c |} ''' + '\n'
    table1 += r'''\hline''' + '\n'
    # table1 += r''' n &  $c$ (photons/cm\textsuperscript{2}/s) & $\sigma$ (nm) & $\mu$ (nm) \\[0.4ex] \hline''' + '\n'
    table1 += r''' n &  $c$ (photons/cm\textsuperscript{2}/s) & $\sigma$ (nm) & $\mu$ (nm) \\ \hline''' + '\n'

    for i in range(n_peaks):
        color_i = mpl.colors.to_rgba(colors1[i]) if n_peaks > 1 else  mpl.colors.to_rgba('tab:red')
        c_txt = r"\textcolor[rgb]{%.3f, %.3f, %.3f}" % (color_i[0], color_i[1], color_i[2])
        # print(c_txt)
        table1 += r'''{0}{{ {1} }} & ${2}$ & ${3}$ & ${4}$ '''.format(
            c_txt, i + 1,
            latex_float_with_error(popt[i * 3], popt_delta[i * 3], digits_err=2),
            latex_float_with_error(popt[i * 3 + 1], popt_delta[i * 3 + 1], digits_err=2),
            latex_float_with_error(popt[i * 3 + 2], popt_delta[i * 3 + 2], digits_err=2),
        )
        # table1 += r"\\[0.4ex] \hline" + '\n'
        table1 += r"\\ \hline" + '\n'
    table1 += r'''\end{tabular}'''
    # print(table1)
    table1 = table1.replace("\n", "")

    timestamp_str = timestamp.strftime('%Y/%m/%d %H:%M:%S')

    print(f'FOLDER: {dated_folder}')
    print(f'FILE:   {file_tag}')
    print(f'C:      {popt[0]:.3E} ± {popt_delta[0]:.3E} photons/cm^2/s')
    print(f'SIGMA:  {popt[1]:.3f} ± {popt_delta[1]:.3f} nm')
    print(f'MU:     {popt[2]:.3f} ± {popt_delta[2]:.3f} nm')
    print(f'TIME:   {timestamp_str}')


    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.75, 3.)

    wl_fit = np.linspace(wl_window.min(), wl_window.max(), num=500)

    ax.plot(wavelength, photon_flux)
    ax.plot(wl_fit, gaussian(x=wl_fit, c=popt[0], sigma=popt[1], mu=popt[2]), c='tab:red', lw='1.25', ls='-')

    wl_extrapolate = np.linspace(wavelength.min(), wavelength.max(), num=5000)
    ax.plot(wl_extrapolate, gaussian(x=wl_extrapolate, c=popt[0], sigma=popt[1], mu=popt[2]), c='tab:red', lw='1.25', ls='--')

    ax.set_xlabel(r'$\lambda$ {\sffamily (nm)}', usetex=True)
    ax.set_ylabel(r"$B_{\lambda}$ {\sffamily (photons/cm\textsuperscript{2}/nm)}", usetex=True)

    ax.set_xlim(wl_range)

    ax.text(
        0.99, 0.98, table1,
        transform=ax.transAxes,
        fontsize=9,
        ha='right', va='top',
        usetex=True
    )

    mf = ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(mf)
    ax.ticklabel_format(useMathText=True)

    ax.set_ylim(top=photon_flux.max()*1.3)
    plot_title = fr'{sample_label} - {timestamp_str}'
    ax.set_title(plot_title)

    # ax.axvspan(wl_peak-delta_wl, wl_peak+delta_wl, color='tab:red', alpha=0.15)

    # fig.savefig(os.path.join(path_to_figures, file_tag + '.png'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()

