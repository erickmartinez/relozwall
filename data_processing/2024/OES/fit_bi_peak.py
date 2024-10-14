import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
from boltons.timeutils import total_seconds
from scipy.interpolate import interp1d
import data_processing.secrets as my_secrets
import data_processing.echelle as ech
import os
from datetime import timedelta, datetime
from specutils import Spectrum1D, SpectralRegion
import astropy.units as u
from specutils.fitting import find_lines_threshold, find_lines_derivative
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import fit_generic_continuum
from specutils.manipulation import box_smooth, gaussian_smooth, trapezoid_smooth
from astroquery.nist import Nist
import warnings
from scipy.signal import savgol_filter


echelle_file = r'./data/Echelle_data/echelle_20241003/MechelleSpect_001.asc'
echelle_file_mapping_xls = r'./data/'
folder_map_xls = r'./PISCES-A_folder_mapping.xlsx'
sample_label = 'Boron rod'
subtract_background = True

wl_range = (818, 824)

d_pattern = '%a %b %d %H:%M:%S %Y'
window_coefficients = np.array([12.783, 0.13065, -8.467e-5])

def load_echelle_calibration(preamp_gain):
    csv_file = r'./data/echelle_calibration_20240910.csv'
    if preamp_gain not in [1, 4]:
        msg = f'Error loading echelle calibration: {preamp_gain} not found in calibration.'
        print(msg)
        raise ValueError(msg)
    col_cal = fr'Radiance @pregain {preamp_gain:d} (W/sr/cm^2/nm)'
    col_err = fr'Radiance @pregain {preamp_gain:d} error (W/sr/cm^2/nm)'
    df = pd.read_csv(csv_file, usecols=[
        'Wavelength (nm)', col_cal, col_err
    ]).apply(pd.to_numeric)
    return df

def get_interpolated_calibration(preamp_gain:int) -> tuple[callable, callable]:
    cal_df = load_echelle_calibration(preamp_gain=preamp_gain)
    if preamp_gain not in [1, 4]:
        msg = f'Error loading echelle calibration: {preamp_gain} not found in calibration.'
        print(msg)
        raise ValueError(msg)
    col_cal = fr'Radiance @pregain {preamp_gain:d} (W/sr/cm^2/nm)'
    col_err = fr'Radiance @pregain {preamp_gain:d} error (W/sr/cm^2/nm)'
    wl = cal_df['Wavelength (nm)'].values
    cal_factor = cal_df[col_cal].values
    cal_error = cal_df[col_err].values

    fc = interp1d(x=wl, y=cal_factor)
    fe = interp1d(x=wl, y=cal_error)
    return fc, fe

def transmission_dirty_window(wavelength: np.ndarray) -> np.ndarray:
    global window_coefficients
    wavelength = np.array(wavelength)
    n = len(window_coefficients)
    m = len(wavelength)
    x = np.ones(m, dtype=np.float64)
    transmission = np.zeros(m, dtype=np.float64)
    for i in range(n):
        transmission += window_coefficients[i] * x
        x = x * wavelength
    return transmission

def load_folder_mapping():
    global echelle_file_mapping_xls
    df = pd.read_excel(echelle_file_mapping_xls, sheet_name=0)
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
                                           r'\usepackage{siunitx}')




def main():
    global echelle_file, d_pattern, sample_label, wl_range, echelle_file_mapping_xls
    # Get the relative path to the echelle file
    relative_path = os.path.dirname(echelle_file)
    # Define the file tag as the echelle file without the extension
    file_tag = os.path.splitext(os.path.basename(echelle_file))[0]
    # Load the spectrometer calibration for preamp gain 1
    cal_pag_1, cal_pag_1_err = get_interpolated_calibration(preamp_gain=1)
    # load the spectrometer calibration for preamp gain 4
    cal_pag_4, cal_pag_4_err = get_interpolated_calibration(preamp_gain=4)
    # Load the data, and spectrometer conditions in the echelle .asc file
    df, params = ech.load_echelle_file(path_to_file=echelle_file)
    # Focus only on the wavelength region defined in wl_rangew
    df = df[df['wl (nm)'].between(wl_range[0], wl_range[1])].reset_index(drop=True)
    # Get the last file from the folder and subtract it from the spectrum
    bgnd_asc = [f for f in os.listdir(relative_path) if f.endswith('.asc')][-1]
    bgnd_df, bgnd_params = ech.load_echelle_file(path_to_file=os.path.join(relative_path, bgnd_asc))
    bgnd_df = bgnd_df[bgnd_df['wl (nm)'].between(wl_range[0], wl_range[1])].reset_index(drop=True)

    # load the folder to data label mapping
    folder_label_mapping = load_folder_mapping()

    # Try to get the number of accumulations in the asc file
    try:
        accumulations = int(params['Number of Accumulations'])
        accumulations_bgnd = int(bgnd_params['Number of Accumulations'])
    except KeyError as ke:
        print(f"Number of accumulations not found. Assuming single scan")
        accumulations = 1
        accumulations_bgnd = 1

    preamp_gain = float(params['Pre-Amplifier Gain'])
    exposure_s = float(params['Exposure Time (secs)'])
    time_stamp = datetime.strptime(params['Date and Time'], d_pattern)


    preamp_gain_bgnd = float(bgnd_params['Pre-Amplifier Gain'])
    exposure_bgnd_s = float(bgnd_params['Exposure Time (secs)'])

    wavelength = df['wl (nm)'].values
    counts = df['counts'].values
    counts[counts < 0.] = 0.
    counts_ps = counts / exposure_s / accumulations

    # Add dirty window correction (if not examining calibration spectra)
    transmission = 1.

    if os.path.basename(relative_path) != 'echelle_20240910':
        transmission = transmission_dirty_window(wavelength)

    counts_ps /= transmission

    wavelength_bgnd = bgnd_df['wl (nm)'].values
    counts_bg = bgnd_df['counts'].values
    # Smooth the background
    cps_bg = savgol_filter(
        cps_bg,
        window_length=5,
        polyorder=3
    )
    cps_bg[cps_bg < 0.] = 0.
    counts_ps_bgnd = cps_bg / exposure_bgnd_s / accumulations_bgnd


    f_counts_ps = interp1d(x=wavelength_bgnd, y=counts_ps_bgnd)
    # Subtract the background
    counts_ps -= f_counts_ps(wavelength)

    if preamp_gain == 1:
        cal_factor = cal_pag_1(wavelength)
    elif preamp_gain == 4:
        cal_factor = cal_pag_4(wavelength)
    else:
        raise ValueError(f'Calibration not performed for preamplifier gain {preamp_gain:d}.')

    # counts_ps /= transmission
    flux_b = cal_factor * counts_ps * 1E4

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3.5)

    ax.plot(wavelength, flux_b)

