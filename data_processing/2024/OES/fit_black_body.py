import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf
import re
from data_processing.utils import lighten_color
from typing import List, Dict


echelle_spectrum = r'./data/brightness_data_fitspy_wl-calibrated/echelle_20241031/MechelleSpect_006.csv'
baseline_fit_csv = r'./data/baseline_echelle_20240815_MechelleSpect_010.csv'
temperature_excel = r'./data/20241031_bb_temp.xlsx'
echelle_xlsx = r'./data/echelle_db.xlsx'


save_data_for_figure = True
fit_local_minima = True
f_scale = 0.1
loss = 'soft_l1'
lm_window_size = 150

lookup_lines = [
    {'center_wl': 410.06, 'label': r'D$_{\delta}$'},
    {'center_wl': 434.0, 'label': r'D$_{\gamma}$'},
    {'center_wl': 486.00, 'label': r'D$_{\beta}$'},
    {'center_wl': 656.10, 'label': r'D$_{\alpha}$'}
]

calibration_wl = lookup_lines[1]

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

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    plt.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}')


def radiance_at_temperature(temperature: float, wavelength_nm: float, A=1.) -> float:
    hc = 6.62607015 * 2.99792458  # x 1E -34 (J s) x 1E8 (m/s) = 1E-26 (J m)
    hc2 = hc * 2.99792458  # x 1E -34 (J s) x 1E16 (m/s)^2 = 1E-18 (J m^2 s^{-1})
    factor = 2. * 1E14 * hc2 * np.power(wavelength_nm, -5.0)  # W / cm^2 / nm
    arg = 1E6 * hc / wavelength_nm / 1.380649 / temperature  # 26 (J m) / 1E-9 m / 1E-23 J/K
    return A * factor / (np.exp(arg) - 1.)


def model_bb(wavelength_nm: np.ndarray, b):
    temperature, factor = b[0], b[1]
    return radiance_at_temperature(temperature=temperature, wavelength_nm=wavelength_nm, A=factor)


def res_bb(b, x, y, w=1):
    return (model_bb(wavelength_nm=x, b=b) - y)*w

all_tol = float(np.finfo(np.float64).eps)


def fit_black_body(
    wavelength: np.ndarray, radiance:np.ndarray, temperature_guess:float, scaling_factor_guess:float, tol=all_tol,
    f_scale=1., loss='soft_l1', fit_local_minima=False
) -> OptimizeResult:
    """
    Tries to fit the spectrum to a black body spectrum
    Parameters
    ----------
    wavelength: np.ndarray
        The wavelength in nm
    radiance: np.ndarray
        The spectral radiance in W/m^2/s/nm
    temperature_guess: float
        The initial guess for the temperature in K
    scaling_factor_guess: float
        The initial guess of the scaling factor for the black body spectrum
    tol: float
        The tolerance used for the convergence of the least_squares
    f_scale: float
        The scaling factor for the outliers
    loss: str
        The loss function for the least_squares
    fit_local_minima: bool
        If true, fit the local minima of the spectra

    Returns
    -------
    OptimizeResult:
        The results from scipy.optimize.least_squares optimization
    """

    b0 = np.array([temperature_guess, scaling_factor_guess])
    global lm_window_size

    if fit_local_minima:
        # Find minima
        window_size = len(wavelength) // lm_window_size
        if window_size % 2 == 0:
            window_size += 1
        minima_data = find_local_minima(radiance.tolist(), window_size=window_size)
        indices = minima_data['minima_indices']
        wavelength = np.array([wavelength[i] for i in indices])
        radiance = np.array(minima_data['minima_values'])
        n = len(wavelength)
        result: OptimizeResult = least_squares(
            res_bb,
            b0,
            loss=loss,
            f_scale=f_scale,
            args=(wavelength, radiance),
            bounds=([all_tol, all_tol], [np.inf, np.inf]),
            xtol=tol,
            ftol=tol,
            gtol=tol,
            max_nfev=10000 * n,
            x_scale='jac',
            verbose=2
        )

    else:
    # Try fitting the "baseline" of the spectrum
        percentile_threshold = 20
        window_size = len(wavelength) // 20  # 5% of data length
        rolling_min = pd.Series(radiance).rolling(window=window_size, center=True).quantile(percentile_threshold / 100)
        baseline_points = ~pd.isna(rolling_min)
        # Set weights (higher weight for lower intensity points)
        weights = 1 / (radiance + np.median(radiance) / 10)  # Add small value to prevent division by zero
        weights[~baseline_points] = 0  # Zero weight for non-baseline points

        n = len(wavelength)
        result: OptimizeResult = least_squares(
            res_bb,
            b0,
            loss=loss,
            f_scale=f_scale,
            args=(wavelength[baseline_points], radiance[baseline_points], weights[baseline_points]),
            bounds=([all_tol, all_tol], [np.inf, np.inf]),
            xtol=tol,
            ftol=tol,
            gtol=tol,
            max_nfev=10000 * n,
            x_scale='jac',
            verbose=2
        )

    return result


temp_cols = [
    'Folder', 'File',
    'Elapsed time (s)',
    'Temperature (K)',
    'Temperature error (K)'
]

def load_output_db(xlsx_source):
    global temp_cols
    try:
        out_df: pd.DataFrame = pd.read_excel(xlsx_source, sheet_name=0)
    except Exception as e:
        out_df = pd.DataFrame(data={
            col: [] for col in temp_cols
        })
        out_df.to_excel(xlsx_source, index=False)
    return out_df

def update_out_df(db_df:pd.DataFrame, row_data):
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

def load_echelle_xlsx(xlsx_file):
    echelle_df: pd.DataFrame = pd.read_excel(xlsx_file, sheet_name=0)
    echelle_df['Timestamp'] = echelle_df['Timestamp'].apply(pd.to_datetime)
    echelle_df['Elapsed time (s)'] = (echelle_df['Timestamp'] - echelle_df[
        'Timestamp'].min()).dt.total_seconds()  # Arbitrary value for now, different t0 for every folder
    unique_folders = echelle_df['Folder'].unique()
    for folder in unique_folders:
        row_indexes = echelle_df['Folder'] == folder
        ts = echelle_df.loc[row_indexes, 'Timestamp'].reset_index(drop=True)
        echelle_df.loc[row_indexes, 'Elapsed time (s)'] = (echelle_df.loc[row_indexes, 'Timestamp'] - ts[0]).dt.seconds
    return  echelle_df

def get_spectrum_timestamp(folder, file, echelle_df):
    # Get the elapased time since the first spectrum for each spectrum in the folder
    try:
        selected_folder_df = echelle_df[(echelle_df['Folder'] == folder) & (echelle_df['File'] == file)].reset_index(
            drop=True)
        elapsed_time = selected_folder_df['Elapsed time (s)'][0]
        print(f"Elapsed time: {elapsed_time}")
    except KeyError:
        print(f"Could not find Folder '{folder}', File '{file}' in the echelle_db")
        raise KeyError
    return elapsed_time


def find_local_minima(signal: List[float], window_size: int = 3) -> Dict[str, List]:
    """
    Find local minima in a signal using a sliding window approach.

    Parameters:
    -----------
    signal : List[float]
        Input signal as a list of numerical values
    window_size : int, optional
        Size of the window to consider for finding local minima (default: 3)
        Must be odd number

    Returns:
    --------
    Dict with three keys:
        'minima_indices': List[int] - Indices where local minima occur in original signal
        'minima_values': List[float] - Values of the local minima
        'minima_pairs': List[Tuple[int, float]] - List of (index, value) pairs for compatibility

    Raises:
    -------
    ValueError
        If window_size is not odd or is larger than signal length
    """
    if len(signal) < window_size:
        raise ValueError("Signal length must be greater than window size")
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")

    # Convert to numpy array for easier processing
    signal_array = np.array(signal)
    half_window = window_size // 2

    # Initialize separate lists for indices and values
    minima_indices = []
    minima_values = []

    # Iterate through signal excluding edges
    for i in range(half_window, len(signal_array) - half_window):
        window = signal_array[i - half_window:i + half_window + 1]
        center_value = window[half_window]

        # Check if center point is minimum in window
        if center_value == np.min(window):
            # Ensure it's strictly less than at least one neighbor
            if np.sum(window == center_value) == 1:
                minima_indices.append(i)
                minima_values.append(center_value)

    # Return results in a dictionary for clarity
    return {
        'minima_indices': minima_indices,
        'minima_values': minima_values,
        'minima_pairs': list(zip(minima_indices, minima_values))
    }

def main():
    global echelle_spectrum, baseline_fit_csv, temperature_excel, fit_local_minima
    global f_scale, lm_window_size
    echelle_df: pd.DataFrame = load_echelle_xlsx(echelle_xlsx)
    file = os.path.basename(echelle_spectrum)
    file_tag = os.path.splitext(file)[0]
    folder = os.path.basename(os.path.dirname(echelle_spectrum))
    elapsed_time = get_spectrum_timestamp(folder, file_tag + '.asc', echelle_df)
    spectrum_df:pd.DataFrame = pd.read_csv(
        echelle_spectrum, comment='#',
    ).apply(pd.to_numeric).sort_values(by=['Wavelength (nm)'])

    spectrum_df = spectrum_df[spectrum_df['Brightness (photons/cm^2/s/nm)'] > 0.].reset_index(drop=True)

    baseline_df = pd.read_csv(baseline_fit_csv, comment='#').apply(pd.to_numeric)
    # Get the coefficients of the polynomial fit to the continuum
    # Also get the intensity of the calibration wavelength (D_delta at 410 nm) from the spectrum used to get the continuum
    # baseline.
    # Note the baseline spectrum was fit in units of brightness of (photons/cm^2/s/nm)
    #
    p = re.compile(r"\#\s+REF\:\s+(\d+\.\d*)\s+.*Intensity\:\s+(\d+\.?\d*[eE][\-\+]\d+).*")
    reference_wl = 410.1293# nm
    reference_intenstiy = 2.018E+12 # photons/cm^2/s
    with open(baseline_fit_csv, 'r') as f:
        for line in f:
            m = p.match(line)
            if m:
                reference_wl = float(m.group(1))
                reference_intenstiy = float(m.group(2))
                break

    print(baseline_df)
    popt_bl =np.array([xi for xi in baseline_df.iloc[0]])

    brightness_complete = spectrum_df['Brightness (photons/cm^2/s/nm)'].values
    wl_complete = spectrum_df['Wavelength (nm)'].values
    h = 6.62607015  # E-34
    c = 2.99792458  # E8
    xhc_by_lambda_c = h * c / wl_complete * 1E-17



    # Find the intensity of the spectrum at the reference wavelength
    msk_ref = ((reference_wl - 0.3) <= wl_complete) & (wl_complete <= (reference_wl + 0.3))
    wl_win = wl_complete[msk_ref]
    b_win = brightness_complete[msk_ref]
    # rad_peak = b_win.mean()
    rad_peak = b_win.max()
    idx_peak = np.argmin(np.abs(rad_peak - b_win))
    wl_peak = wl_win[idx_peak]
    print(f"Reference wl:\t{reference_wl:.3f} nm")
    print(f"Reference intensity:\t{reference_intenstiy:.3E} (photons/cm^2/s/nm)")
    print(f"Wl @ ref peak :\t{wl_peak:.3f} nm")
    print(f"Intensity @ ref peak:\t{rad_peak:.3E} (photons/cm^2/s/nm)")
    scaling_factor = rad_peak / reference_intenstiy

    spectrum_df = spectrum_df[spectrum_df['Wavelength (nm)'].between(400, 850, inclusive='both')].reset_index(drop=True)
    brightness = spectrum_df['Brightness (photons/cm^2/s/nm)'].values
    wl = spectrum_df['Wavelength (nm)'].values
    baseline = model_poly(wl, popt_bl) * 1E12 # The polynomial was fitted in units of x1E12 photons/s/cm^2
    baseline *= scaling_factor
    brightness_baselined = brightness - baseline
    brightness_complete_baselined = brightness_complete - scaling_factor * model_poly(wl_complete, popt_bl) * 1E12

    # radiance = brightness * 1240. / wl * 1.6019E-19 / 4. / np.pi
    xhc_by_lambda = h * c / wl * 1E-17
    radiance = brightness_baselined * xhc_by_lambda / (4. * np.pi)
    radiance_complete = brightness_complete_baselined * xhc_by_lambda_c / (4. * np.pi)
    msk_fit = radiance > 0.


    n = len(wl)
    print(f'wl.min: {wl.min():.0f}, wl.max(): {wl.max():.0f}')


    ls_res = fit_black_body(
        wavelength=wl[msk_fit], radiance=radiance[msk_fit], temperature_guess=800, scaling_factor_guess=1.2,
        f_scale=f_scale, loss='cauchy', fit_local_minima=fit_local_minima
    )

    popt = ls_res.x
    ci = cf.confidence_interval(ls_res)
    popt_err = np.abs(ci[:, 1] - popt)
    print(f"I0: {popt[1]:.3E} ± {popt_err[1]:.3E}, 95% CI: [{ci[1, 0]:.5E}, {ci[1, 1]:.5E}]")
    nn = wl.max() - 200.
    x_pred = np.linspace(wl.min(), wl.max(), num=2000)
    y_pred, delta = cf.prediction_intervals(
        model=model_bb, x_pred=x_pred, ls_res=ls_res
    )

    temp_data = {
        'Folder': folder, 'File': file_tag, 'Elapsed time (s)': elapsed_time,
        'Temperature (K)': popt[0], 'Temperature error (K)': popt_err[0]
    }
    output_df = load_output_db(xlsx_source=temperature_excel)
    output_df = update_out_df(output_df, temp_data)
    output_df.sort_values(by=['Folder', 'File'], inplace=True)
    output_df.to_excel(excel_writer=temperature_excel, index=False)


    load_plot_style()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 5.)

    ax1.plot(wl_complete, brightness_complete, color='C1', label='Data')
    ax1.plot(wl_complete, model_poly(wl_complete, popt_bl)*1E12*scaling_factor, color='tab:green', label='\"Cold\" baseline')

    ax2.plot(wl_complete, radiance_complete, color=lighten_color('C0', amount=.25), label='Data')
    ax2.plot(wl, radiance, label='Fit range', color='C0')
    ax2.plot(x_pred, y_pred, color='tab:red', label='Fit')
    ax2.fill_between(x_pred, y_pred - delta, y_pred + delta, color='tab:red', alpha=0.25)

    if fit_local_minima:
        window_size = len(wl) // lm_window_size
        if window_size % 2 == 0:
            window_size += 1
        minima_data = find_local_minima(radiance.tolist(), window_size=window_size)
        indices = minima_data['minima_indices']
        wavelength_lm = np.array([wl[i] for i in indices])
        radiance_lm = np.array(minima_data['minima_values'])
        ax2.plot(wavelength_lm, radiance_lm, 'ro', ms=4, mfc='none')

    ax2.set_xlabel(r'$\lambda$ {\sffamily (nm)}', usetex=True)
    ax1.set_ylabel(r'B (photons/cm$^{\mathregular{2}}$/nm)', usetex=False)
    ax2.set_ylabel(r'Radiance (W/cm$^{\mathregular{2}}$/ster/nm)', usetex=False)

    # (W/cm^{2}/ster/nm
    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='center left', fontsize=10)

    results_txt = f'$T = {popt[0]:.0f}\pm{popt_err[0]:.0f}~\mathrm{{K}}$ ~({popt[0]-273.15:.0f} °C)\n'

    ax2.text(
        0.03, 0.95, results_txt, transform=ax2.transAxes, va='top', ha='left', c='r',

    )
    ax2.ticklabel_format(axis='y', useMathText=True)
    ax2.set_title(os.path.basename(echelle_spectrum))
    # ax.set_xlim(350, wl.max())
    ax1.set_ylim(bottom=0.)#, top=rad_peak * 5.)
    ax2.set_ylim(0, radiance_complete.max()*0.25)
    for ax in (ax1, ax2):
        mf = ticker.ScalarFormatter(useMathText=True)
        mf.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(mf)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))

    folder = os.path.basename(os.path.dirname(echelle_spectrum))
    bbody_folder = './figures/echelle_blackbody'
    if not os.path.exists(bbody_folder):
        os.makedirs(bbody_folder)
    output_folder = os.path.join(bbody_folder, folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = os.path.splitext(os.path.basename(echelle_spectrum))[0]
    path_to_output = os.path.join(output_folder, filename)

    if save_data_for_figure:
        save_df = pd.DataFrame(data={
            'Wavelength (nm)': wl,
            'Radiance (W/cm^2/ster/nm)':radiance,
        })
        path_to_csv = path_to_output + '_data.csv'
        with open(path_to_csv, 'w') as f:
            f.write("# Results from black body fit \n")
            f.write("# =========================== \n")
            f.write(f"# Temperature [K]: {popt[0]:.1f} -/+ {popt_err[0]:.1f}\n")
            f.write(f"# Scaling factor: {popt[1]:.2E} -/+ {popt_err[1]:.2E}\n")
            f.write(f"# Fit local minima: {fit_local_minima}\n")
            f.write(f"# f_scale: {f_scale}\n")
            f.write(f"# lm_window_size: {lm_window_size}\n")
            save_df.to_csv(f, index=False)





    fig.savefig(path_to_output + '.png', dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
