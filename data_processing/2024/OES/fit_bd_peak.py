import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.proj3d import transform
from scipy.optimize import least_squares, OptimizeResult
import matplotlib.ticker as ticker
import json
import os
from datetime import timedelta, datetime
import warnings
from scipy.integrate import simpson
import data_processing.confidence as cf
import re
from data_processing.utils import latex_float_with_error


brightness_csv = r'./data/brightness_data/echelle_20240827/MechelleSpect_032.csv'
FOLDER_MAP_XLS = r'./PISCES-A_folder_mapping.xlsx'
output_folder = r'./figures/Echelle_plots/B-D'
ouput_xls = r'./data/b-d_gaussian_peak.xlsx'
subtract_background = True

calibration_line = {'center_wl': 433.93, 'label': r'D$_{\gamma}$'}
known_peaks = [
    {'center_wl': 431.0, 'label': r"CD"},
    {'center_wl': 432.7, 'label': r"BD"},
    {'center_wl': 433.93, 'label': r'$D_{\mathrm{\gamma}}$'}
]

peaks_guess = [
    {'center_wl': 431.0, 'sigma':0.02},
    {'center_wl': 432.7, 'sigma':0.012},

]

wl_range = (428., 435)
fit_range = (430, 433.3)

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




def main():
    global brightness_csv, d_pattern, wl_range, output_folder, peaks_guess, ouput_xls
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

    # Calibrate the spectrum to Dgamma
    delta_wl = 0.15
    line_center: float = calibration_line['center_wl']
    msk_window = ((line_center - delta_wl) <= wavelength) & (wavelength <= (line_center + delta_wl))
    wl_window = wavelength[msk_window]
    intensity_window = photon_flux[msk_window]
    peak_height = intensity_window.max()
    idx_peak_window = np.argmin(np.abs(peak_height - intensity_window))
    # wl_peak = wl_window[idx_peak_window]
    # wl_delta = wl_peak - line_center
    # wavelength -= wl_delta


    msk_fit = (fit_range[0] <= wavelength) & (wavelength <= fit_range[1])
    wl_fit = wavelength[msk_fit]
    flux_fit = photon_flux[msk_fit]

    all_tol = float(np.finfo(np.float64).eps)#**(1/2)

    # First fit the Dgamma and shift the spectrum to match the calibrated value
    cal_wl = calibration_line['center_wl']
    msk_cal = ((cal_wl - 0.1) <= wavelength ) & (wavelength <= (cal_wl + 0.1))
    x_cal = wavelength[msk_cal]
    y_cal = photon_flux[msk_cal]

    res_lsq = least_squares(
        res_sum_gauss, x0=[1.5*simpson(y=y_cal, x=x_cal), 0.1, cal_wl],
        args=(wl_fit, flux_fit),
        loss='soft_l1', f_scale=0.1,
        jac=jac_sum_gauss,
        bounds=(
            [0., 1E-3, cal_wl-0.2],
            [np.inf, np.inf, cal_wl+0.2],
        ),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        verbose=2,
        x_scale='jac',
        max_nfev=10000 * len(flux_fit)
    )
    popt = res_lsq.x
    wl_dgamma = popt[2]
    wl_delta = wl_dgamma - calibration_line['center_wl']
    wavelength -= wl_delta
    wl_fit = wavelength[msk_fit]
    flux_fit = photon_flux[msk_fit]

    print(f'Fitted D_gamma center wavelength {wl_dgamma:.3f} nm')

    # Fit a gaussian to the band heads individually
    n_peaks = len(peaks_guess)
    popt = np.empty(n_peaks*3, dtype=np.float64)
    popt_delta = np.empty(n_peaks*3, dtype=np.float64)
    popt_ci = np.empty((n_peaks*3, 2), dtype=np.float64)
    delta_wl = 0.25
    total_area = simpson(y=photon_flux, x=wavelength)
    for i in range(n_peaks):
        pg = peaks_guess[i]
        x0 = np.empty(3, dtype=np.float64)
        flb = np.empty(3, dtype=np.float64)
        fub = np.empty(3, dtype=np.float64)

        line_center: float = pg['center_wl']
        msk_window = ((line_center - delta_wl) <= wavelength) & (wavelength <= (line_center + delta_wl))
        wl_window = wavelength[msk_window]
        intensity_window = photon_flux[msk_window]

        area_window = simpson(y=intensity_window, x=wl_window)
        c_window = area_window * over_sqrt_pi
        x0[0] = 1.5 * c_window
        x0[1] = pg['sigma']
        x0[2] = line_center

        flb[0] = 0.
        flb[1] = 1E-10
        flb[2] = line_center - 0.15

        fub[0] = total_area * 0.08
        fub[1] = pg['sigma'] * 6.
        fub[2] = line_center + 0.15


        res_lsq: OptimizeResult = least_squares(
            res_sum_gauss, x0=x0, args=(wl_fit, flux_fit),
            # loss='soft_l1', f_scale=0.1,
            jac=jac_sum_gauss,
            bounds=(
                flb,
                fub
            ),
            xtol=all_tol,
            ftol=all_tol,
            gtol=all_tol,
            verbose=2,
            x_scale='jac',
            max_nfev=10000 * len(flux_fit)
        )
        popt_i = res_lsq.x
        ci = cf.confidence_interval(res=res_lsq)
        popt_i_delta = ci[:, 1] - popt_i
        for j in range(len(popt_i)):
            popt[3*i + j] = popt_i[j]
            popt_delta[3*i + j] = popt_i_delta[j]
            popt_ci[3*i +j, :] = ci[j, :]


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
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True, height_ratios=[0.35, 1])
    fig.set_size_inches(4.75, 6.)


    wl_fit_pred = np.linspace(wl_fit.min(), wl_fit.max(), num=5000)
    wl_extrapolate = np.linspace(wavelength.min(), wavelength.max(), num=5000)
    for ax in axes:
        ax.plot(wavelength, photon_flux)
        ax.set_xlabel(r'$\lambda$ {\sffamily (nm)}', usetex=True)

        ax.set_xlim(wl_range)
        mf = ticker.ScalarFormatter(useMathText=True)
        mf.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(mf)
        ax.ticklabel_format(useMathText=True)

    # axes[1].plot(wl_fit_pred, sum_gaussians(x=wl_fit_pred, b=popt), c='tab:red', lw='1.25', ls='-')
    axes[1].plot(wl_extrapolate, sum_gaussians(x=wl_extrapolate, b=popt), c='tab:red', lw='1.25', ls='--')

    axes[1].text(
        0.01, 0.99, table1,
        transform=axes[1].transAxes,
        fontsize=9,
        ha='left', va='top',
        usetex=True
    )

    for i in range(n_peaks):
        idx_c = 3 * i
        idx_s = idx_c + 1
        idx_m = idx_s + 1
        a = popt[idx_c]
        s = popt[idx_s]
        m = popt[idx_m]
        x_pred_i = np.linspace(m-delta_wl, m + delta_wl, num=500)
        y_pred_i = gaussian(x_pred_i, a, s, m)
        axes[1].fill_between(x_pred_i, 0, y_pred_i, color=colors1[i], alpha=0.2)
        axes[1].plot(x_pred_i, y_pred_i, color=colors1[i], lw=1.25)

        kp = known_peaks[i]

        lc = kp['center_wl']
        lbl = kp['label']
        y1 = sum_gaussians([m], popt)
        # xdisplay, ydisplay = axes[1].transData.transform_point((lc, y1))

        offset = -20
        # connectionstyle = "angle3,angleA=0,angleB=90"
        connectionstyle = "angle,angleA=180,angleB=90,rad=0"
        bbox = dict(boxstyle="round", fc="wheat")
        arrowprops = dict(
            arrowstyle="->", color="0.5",
            shrinkA=5, shrinkB=5,
            patchA=None, patchB=None,
            connectionstyle=connectionstyle
        )
        axes[1].annotate(
            f"{lbl} ({lc:.1f}) nm",
            xy=(m, y1), xycoords='data',  # 'figure pixels', #data',
            # transform=axes[1].transAxes,
            xytext=(3.5 * offset, -1.5 * offset), textcoords='offset points',  # 'data',
            arrowprops=arrowprops,
            bbox=bbox,
            ha='left',
            fontsize=9
        )



    fig.supylabel(r"$B_{\lambda}$ {\sffamily (photons/cm\textsuperscript{2}/nm)}", usetex=True)

    axes[0].set_ylim(top=photon_flux.max()*1.1)
    axes[1].set_ylim(bottom=-photon_flux.max()*1E-3, top=gaussian([popt[2]], popt[0], popt[1], popt[2])*1.65)
    plot_title = fr'{sample_label} - {timestamp_str}'
    axes[0].set_title(plot_title)



    # ax.axvspan(wl_peak-delta_wl, wl_peak+delta_wl, color='tab:red', alpha=0.15)

    # fig.savefig(os.path.join(path_to_figures, file_tag + '.png'), dpi=600)
    # try opening the output excel file
    try:
        out_df: pd.DataFrame = pd.read_excel(ouput_xls, sheet_name=0)
    except Exception as e:
        print(e)
        out_df = pd.DataFrame(data={
            'Folder': [],
            'File': [],
            'C-D C (photons/cm^2/s)': [],
            'C-D C lb (photons/cm^2/s)': [],
            'C-D C ub (photons/cm^2/s)': [],
            'C-D Sigma (nm)': [],
            'C-D Sigma lb (nm)': [],
            'C-D Sigma ub (nm)': [],
            'C-D Mu (nm)': [],
            'C-D Mu lb (nm)': [],
            'C-D Mu ub (nm)': [],
            'B-D C (photons/cm^2/s)': [],
            'B-D C lb (photons/cm^2/s)': [],
            'B-D C ub (photons/cm^2/s)': [],
            'B-D Sigma (nm)': [],
            'B-D Sigma lb (nm)': [],
            'B-D Sigma ub (nm)': [],
            'B-D Mu (nm)': [],
            'B-D Mu lb (nm)': [],
            'B-D Mu ub (nm)': [],
            'Timestamp': []
        })

    row_data = {'Folder': [dated_folder],
        'File': [file_tag],
        'C-D C (photons/cm^2/s)': [popt[0]],
        'C-D C lb (photons/cm^2/s)': [popt_ci[0, 0]],
        'C-D C ub (photons/cm^2/s)': [popt_ci[0, 1]],
        'C-D Sigma (nm)': [popt[1]],
        'C-D Sigma lb (nm)': [popt_ci[1, 0]],
        'C-D Sigma ub (nm)': [popt_ci[1, 1]],
        'C-D Mu (nm)': [popt[2]],
        'C-D Mu lb (nm)': [popt_ci[2, 0]],
        'C-D Mu ub (nm)': [popt_ci[2, 1]],
        'B-D C (photons/cm^2/s)': [popt[3]],
        'B-D C lb (photons/cm^2/s)': [popt_ci[3, 0]],
        'B-D C ub (photons/cm^2/s)': [popt_ci[3, 1]],
        'B-D Sigma (nm)': [popt[4]],
        'B-D Sigma lb (nm)': [popt_ci[4, 0]],
        'B-D Sigma ub (nm)': [popt_ci[4, 1]],
        'B-D Mu (nm)': [popt[5]],
        'B-D Mu lb (nm)': [popt_ci[5, 0]],
        'B-D Mu ub (nm)': [popt_ci[5, 1]],
        'Timestamp': [timestamp]
    }
    row_df = pd.DataFrame(data=row_data)
    print(row_df[['B-D C lb (photons/cm^2/s)', 'B-D C (photons/cm^2/s)','B-D C ub (photons/cm^2/s)']])

    if len(out_df) == 0:
        out_df = row_df
    else:
        # find the entry (if it exists) for the folder and file define in the header
        previous_df = out_df[(out_df['Folder'] == dated_folder) & (out_df['File'] == file_tag)]
        if len(previous_df) == 0:
            out_df = pd.concat([out_df, row_df], ignore_index=True).reset_index(drop=True)
        else:
            row_index = out_df.loc[(out_df['Folder'] == dated_folder) & (out_df['File'] == file_tag)].index[0]
            print(row_index)
            for col, val in row_data.items():
                out_df.loc[row_index, col] = val[0]

    out_df.sort_values(by=['Folder', 'File'], ascending=True, inplace=True)
    out_df.to_excel(excel_writer=ouput_xls, index=False)

    fig.savefig(os.path.join(path_to_figures, file_tag + '.png'), dpi=600)
    # fig.savefig(os.path.join(path_to_figures, file_tag + '.pdf'), dpi=600)
    plt.show()


if __name__ == '__main__':
    main()

