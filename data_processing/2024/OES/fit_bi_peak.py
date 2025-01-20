import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import least_squares, OptimizeResult
import matplotlib.ticker as ticker
import json
import os
from scipy.integrate import simpson
import data_processing.confidence as cf
import re
from data_processing.utils import latex_float_with_error


brightness_csv = r'./data/brightness_data_fitspy_wl-calibrated/echelle_20241031/MechelleSpect_029.csv'
FOLDER_MAP_XLS = r'./PISCES-A_folder_mapping.xlsx'
output_folder = r'./figures/Echelle_plots/B-I'
echelle_xlsx = r'./data/echelle_db.xlsx'
output_xls = r'./data/bi_lorentzian.xlsx'

cold_baseline_fit_csv = r'./data/baseline_echelle_20240815_MechelleSpect_007.csv'
subtract_baseline = True


# calibration_line = {'center_wl': 433.93, 'label': r'D$_{\gamma}$'}
calibration_line = {'center_wl': 821.2, 'label': r'B I'}
# calibration_line = {'center_wl': 434.0, 'label': r'D$_{\gamma}$'}


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


def model_poly(x, b) -> np.ndarray:
    n = len(b)
    r = np.zeros(len(x))
    for i in range(n):
        r += b[i] * x ** i
    return r


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


bi_df_columns = [
    'Folder',
    'File',
    'BI H (photons/cm^2/s)',
    'BI H lb (photons/cm^2/s)',
    'BI H ub (photons/cm^2/s)',
    'BI mu (nm)',
    'BI mu lb (nm)',
    'BI mu ub (nm)',
    'BI gamma (nm)',
    'BI gamma lb (nm)',
    'BI gamma ub (nm)',
    'Elapsed time (s)',
    'Timestamp'
]

def load_output_db(xlsx_source):
    global bi_df_columns
    try:
        out_df: pd.DataFrame = pd.read_excel(xlsx_source, sheet_name=0)
    except Exception as e:
        out_df = pd.DataFrame(data={
            col: [] for col in bi_df_columns
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
    db_df.sort_values(by=['Folder', 'File'], ascending=(True, True), inplace=True)
    return db_df

def main():
    global brightness_csv, d_pattern, wl_range, output_folder, echelle_xlsx
    global bi_df_columns, cold_baseline_fit_csv, subtract_baseline
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

    baseline = None
    if subtract_baseline:
        baseline_df = pd.read_csv(cold_baseline_fit_csv, comment='#').apply(pd.to_numeric)
        # Get the coefficients of the polynomial fit to the continuum
        # Also get the intensity of the calibration wavelength (D_gamma) from the spectrum used to get the continuum
        # baseline.
        # Note the baseline spectrum was fit in units of brightness of (photons/cm^2/s/nm) x1E12
        #
        # D_gamma: 433.933 -/+ 0.0000 nm, Intensity: 3.308E+01 (photons/cm^2/s/nm)
        p = re.compile(r"\#\s+D\_gamma\:\s+(\d+\.\d*)\s+.*Intensity\:\s+(\d+\.?\d*[eE][\-\+]\d+).*")
        reference_wl = 433.93  # nm
        reference_intenstiy = 3.3E1  # photons/cm^2/s
        with open(cold_baseline_fit_csv, 'r') as f:
            for line in f:
                m = p.match(line)
                if m:
                    reference_wl = float(m.group(1))
                    reference_intenstiy = float(m.group(2))
                    break
        popt_bl = np.array([xi for xi in baseline_df.iloc[0]])
        wavelength_full = full_df['Wavelength (nm)'].values
        photon_flux_full = full_df['Brightness (photons/cm^2/s/nm)'].values
        # Find the intensity of the spectrum at the reference wavelength
        wl_dg = 434.0
        msk_ref = ((wl_dg - 0.3) <= wavelength_full) & (wavelength_full <= (wl_dg + 0.3))
        wl_win = wavelength_full[msk_ref]
        b_win = photon_flux_full[msk_ref]
        rad_peak = b_win.max()
        idx_peak = np.argmin(np.abs(rad_peak - b_win))
        wl_peak = wl_win[idx_peak]
        # print(f"Reference wl:\t{wl_dg:.3f} nm")
        # print(f"Reference intensity:\t{reference_intenstiy:.3E} (photons/cm^2/s/nm)")
        # print(f"Wl @ ref peak :\t{wl_peak:.3f} nm")
        # print(f"Intensity @ ref peak:\t{rad_peak:.3E} (photons/cm^2/s/nm)")
        scaling_factor = rad_peak / reference_intenstiy
        baseline = model_poly(wavelength, popt_bl)
        baseline *= scaling_factor
        photon_flux -= baseline
        photon_flux -= photon_flux[0]


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
        selected_folder_df = params_df[(params_df['Folder'] == folder) & (params_df['File'] == file)].reset_index(drop=True)
        elapsed_time = selected_folder_df['Elapsed time (s)'][0]
        timestamp = selected_folder_df['Timestamp'][0]
        print(f"Elapsed time: {elapsed_time}")
    except KeyError:
        print(f"Could not find Folder '{folder}', File '{file}' in the echelle_db")
        print(params_df[['Folder','File', 'Elapsed time (s)']])
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

    x0 = [c_window*0.75, wl_peak, (wl_window.max() - wl_window.min())]
    all_tol = float(np.finfo(np.float64).eps)
    res_lsq = least_squares(
        res_sum_lorentzians, x0=x0, args=(wl_window, intensity_window), loss='linear', f_scale=0.1,
        jac=jac_sum_lorentzians,
        bounds=(
            [0., wl_peak-0.15, 1E-5],
            [3.*c_window, wl_peak+0.15, np.inf]
        ),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        verbose=2,
        # x_scale='jac',
        max_nfev=10000 * len(wl_window)
    )
    popt = res_lsq.x
    ci = cf.confidence_interval(res=res_lsq)
    popt_delta = ci[:, 1] - popt


    row_data = {
        'Folder': folder,
        'File': file,
        'BI H (photons/cm^2/s)': popt[0],
        'BI H lb (photons/cm^2/s)': ci[0, 0],
        'BI H ub (photons/cm^2/s)': ci[0, 1],
        'BI mu (nm)': popt[1],
        'BI mu lb (nm)': ci[1, 0],
        'BI mu ub (nm)': ci[1, 1],
        'BI gamma (nm)': popt[2],
        'BI gamma lb (nm)': ci[2, 0],
        'BI gamma ub (nm)': ci[2, 1],
        'Elapsed time (s)': elapsed_time,
        'Timestamp': timestamp
    }
    out_df = load_output_db(output_xls)
    out_df = update_df(out_df, row_data)

    out_df.to_excel(excel_writer=output_xls, sheet_name='Lorentzian peaks', index=False)


    cmap = mpl.colormaps.get_cmap('rainbow')
    norm1 = mpl.colors.Normalize(vmin=0, vmax=n_peaks - 1)
    colors1 = [cmap(norm1(i)) for i in range(n_peaks)]

    table1 = r"\setcellgapes{1.5pt}\makegapedcells" # increase vertical padding in cells to avoid superscript overlaps
    table1 += r'''\begin{tabular}{ | r | c | c | c |} ''' + '\n'
    table1 += r'''\hline''' + '\n'
    # table1 += r''' n &  $c$ (photons/cm\textsuperscript{2}/s) & $\sigma$ (nm) & $\mu$ (nm) \\[0.4ex] \hline''' + '\n'
    table1 += r''' n &  $c$ (photons/cm\textsuperscript{2}/s) & $\mu$ (nm) & $\gamma$ (nm) \\ \hline''' + '\n'

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
    print(f'H:      {popt[0]:.3E} ± {popt_delta[0]:.3E} photons/cm^2/s')
    print(f'MU:  {popt[1]:.3f} ± {popt_delta[1]:.3f} nm')
    print(f'GAMMA:     {popt[2]:.3f} ± {popt_delta[2]:.3f} nm')
    print(f'TIME:   {timestamp_str}')

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.75, 3.)

    wl_fit = np.linspace(wl_window.min(), wl_window.max(), num=500)

    ax.plot(wavelength, photon_flux)
    ax.plot(wl_fit, lorentzian(x=wl_fit, h=popt[0], mu=popt[1], gamma=popt[2]), color='tab:red', lw='1.25', ls='-')

    wl_extrapolate = np.linspace(wavelength.min(), wavelength.max(), num=5000)
    ax.plot(wl_extrapolate, lorentzian(x=wl_extrapolate, h=popt[0], mu=popt[1], gamma=popt[2]), color='tab:red', lw='1.25', ls='--')

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

    ax.set_ylim(top=photon_flux.max()*1.5, bottom=-photon_flux.max()*0.1)
    plot_title = fr'{sample_label} - {timestamp_str}'
    ax.set_title(plot_title)

    # ax.axvspan(wl_peak-delta_wl, wl_peak+delta_wl, color='tab:red', alpha=0.15)

    fig.savefig(os.path.join(path_to_figures, file_tag + '.png'), dpi=600)


    plt.show()


if __name__ == '__main__':
    main()

