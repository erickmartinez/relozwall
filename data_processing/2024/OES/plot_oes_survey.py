import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
from boltons.timeutils import total_seconds
from scipy.interpolate import interp1d
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




# brightness_csv = r'./data/Echelle_data/echelle_20241003/MechelleSpect_024.asc'
brightness_csv = r'./data/brightness_data_fitspy_wl-calibrated/echelle_20240827/MechelleSpect_016.csv'
echelle_xlsx = r'./data/echelle_db.xlsx'
sample_label = 'Boron rod'
subtract_background = True

wl_range = (350, 850)

d_pattern = '%a %b %d %H:%M:%S %Y'
window_coefficients = np.array([12.783, 0.13065, -8.467e-5])


# https://en.wikipedia.org/wiki/Balmer_series
# D transitions
# https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=D&output_type=0&low_w=409&upp_w=800&unit=1&submit=Retrieve+Data&de=0&plot_out=2&I_scale_type=2&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on
# https://physics.nist.gov/PhysRefData/Handbook/Tables/borontable2.htm
# https://physics.nist.gov/PhysRefData/Handbook/Tables/carbontable2.htm
lookup_lines = [
    {'center_wl': 410.06, 'label': r'D$_{\delta}$'},
    {'center_wl': 433.93, 'label': r'D$_{\gamma}$'},
    {'center_wl': 486.00, 'label': r'D$_{\beta}$'},
    {'center_wl': 656.10, 'label': r'D$_{\alpha}$'},
    {'center_wl': 587.56, 'label': r'He I'},
    {'center_wl': 412.19, 'label': r'B II'},
    {'center_wl': 419.48, 'label': r'B II'},
    {'center_wl': 447.21, 'label': r'B II'},
    {'center_wl': 494.04, 'label': r'B II'},
    {'center_wl': 608.44, 'label': r'B II'},
    # {'center_wl': 635.70, 'label': r'B II'}, # https://iopscience.iop.org/article/10.1088/0031-8949/1/5-6/013
    {'center_wl': 563.3, 'label': r'B I'},
    {'center_wl': 703.2, 'label': r'B II'},
    {'center_wl': 821.2, 'label': r'B I'}, # https://iopscience.iop.org/article/10.1088/0031-8949/76/5/024
    {'center_wl': 391.89, 'label': r'C II'},
    {'center_wl': 426.73, 'label': r'C II'},
    {'center_wl': 514.51, 'label': r'C II'},
    {'center_wl': 515.10, 'label': r'C II'},
    {'center_wl': 538.03, 'label': r'C I'},
    # {'center_wl': 588.97, 'label': r'C II'},
    {'center_wl': 601.32, 'label': r'C I'},
    {'center_wl': 657.80, 'label': r'C II'},
    {'center_wl': 723.64, 'label': r'C II'},
    {'center_wl': 433.2, 'label': r'B-H'},
    # {'center_wl': 635.59, 'label': r'B II$^*$'},
    # {'center_wl': 678.17, 'label': r'O IV$^*$'},
    # {'center_wl': 678.17, 'label': r'C II'},
]

def find_line(wl, intensity, line_center: float, delta_wl:float = 0.1) -> tuple:
    # Look into a smaller window:
    msk_window = ((line_center - delta_wl) <= wl) & (wl <= (line_center + delta_wl))
    wl_window = wl[msk_window]
    intensity_window = intensity[msk_window]
    intensity_peak = intensity_window.max()
    idx_peak = np.argmin(np.abs(intensity_window - intensity_peak))
    wl_peak = wl_window[idx_peak]
    return wl_peak, intensity_peak

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
    global brightness_csv, lookup_lines, d_pattern, sample_label
    global echelle_xlsx
    relative_path = os.path.dirname(brightness_csv)

    file_tag = os.path.splitext(os.path.basename(brightness_csv))[0]

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

    folder = os.path.basename(relative_path)
    try:
        selected_folder_df = params_df[(params_df['Folder'] == folder) & (params_df['File'] == file_tag+'.asc')].reset_index(drop=True)
        elapsed_time = selected_folder_df['Elapsed time (s)'][0]
        timestamp = selected_folder_df['Timestamp'][0]
        print(f"Elapsed time: {elapsed_time}")
    except KeyError:
        print(f"Could not find Folder '{folder}', File '{file_tag}' in the echelle_db")
        print(params_df[['Folder','File', 'Elapsed time (s)']])
        exit(-1)

    # cal_pag_1, cal_pag_1_err = get_interpolated_calibration(preamp_gain=1)
    # cal_pag_4, cal_pag_4_err = get_interpolated_calibration(preamp_gain=4)
    # df, params = ech.load_echelle_file(path_to_file=brightness_csv)
    df = pd.read_csv(brightness_csv).apply(pd.to_numeric)
    df = df[df['Wavelength (nm)'].between(wl_range[0], wl_range[1])].reset_index(drop=True)
    # Get the last file from the folder and subtract it from the spectrum
    # bgnd_asc = [f for f in os.listdir(relative_path) if f.endswith('.asc')][-1]
    # bgnd_df, bgnd_params = ech.load_echelle_file(path_to_file=os.path.join(relative_path, bgnd_asc))
    # bgnd_df = bgnd_df[bgnd_df['wl (nm)'].between(wl_range[0], wl_range[1])].reset_index(drop=True)
    #
    #
    # try:
    #     accumulations = int(params['Number of Accumulations'])
    #     accumulations_bgnd = int(bgnd_params['Number of Accumulations'])
    # except KeyError as ke:
    #     print(f"Number of accumulations not found. Assuming single scan")
    #     accumulations = 1
    #     accumulations_bgnd = 1
    #
    # preamp_gain = float(params['Pre-Amplifier Gain'])
    # exposure_s = float(params['Exposure Time (secs)'])
    # time_stamp = datetime.strptime(params['Date and Time'], d_pattern)
    #
    # preamp_gain_bgnd = float(bgnd_params['Pre-Amplifier Gain'])
    # exposure_bgnd_s = float(bgnd_params['Exposure Time (secs)'])
    #
    # wavelength = df['wl (nm)'].values
    # counts = df['counts'].values
    # counts[counts < 0.] = 0.
    # counts_ps = counts / exposure_s / accumulations
    # transmission = 1.
    #
    # if os.path.basename(relative_path) != 'echelle_20240910':
    #     transmission = transmission_dirty_window(wavelength)
    #
    # counts_ps /= transmission
    #
    # wavelength_bgnd = bgnd_df['wl (nm)'].values
    # counts_bgnd = bgnd_df['counts'].values
    # counts_bgnd[counts_bgnd < 0.] = 0.
    # counts_ps_bgnd = counts_bgnd / exposure_bgnd_s / accumulations_bgnd
    #
    # f_counts_ps = interp1d(x=wavelength_bgnd, y=counts_ps_bgnd)
    #
    # if subtract_background:
    #     counts_ps -= f_counts_ps(wavelength)
    #
    # if preamp_gain == 1:
    #     cal_factor = cal_pag_1(wavelength)
    # elif preamp_gain == 4:
    #     cal_factor = cal_pag_4(wavelength)
    # else:
    #     raise ValueError(f'Calibration not performed for preamplifier gain {preamp_gain:d}.')
    #
    #
    # if preamp_gain_bgnd == 1:
    #     cal_factor_bgnd = cal_pag_1(wavelength_bgnd)
    # elif preamp_gain_bgnd == 4:
    #     cal_factor_bgnd = cal_pag_4(wavelength_bgnd)
    # else:
    #     raise ValueError(f'Calibration not performed for preamplifier gain {preamp_gain:d}.')

    # counts_ps /= transmission
    wavelength = df['Wavelength (nm)'].values
    brightness = df['Brightness (photons/cm^2/s/nm)'].values
    spectrum = Spectrum1D(flux=brightness*1E10 * u.W / u.cm / u.cm / u.nm / u.sr, spectral_axis=wavelength * u.nm)
    # with warnings.catch_warnings():  # Ignore warnings
    #     warnings.simplefilter('ignore')
    #     g1_fit = fit_generic_continuum(spectrum)
    # spectrum -= g1_fit(wavelength * u.nm)

    noise_region = SpectralRegion(350. * u.nm, 850. * u.nm)
    spectrum = noise_region_uncertainty(spectrum, noise_region)
    # spec1_gsmooth = gaussian_smooth(spectrum, stddev=3)

    lines = find_lines_threshold(spectrum, noise_factor=0.5)
    # lines = find_lines_derivative(spectrum, flux_threshold=0.02)
    # lines[lines['line_type'] == 'emission']
    print(lines)
    line_centers = np.array(lines['line_center'])
    msk_line_center = [False if lc not in line_centers else True for lc in wavelength]

    # nist_lines_df = pd.read_csv('./data/NIST/boron_rod_nist_lines.txt', delimiter=r'\s+')
    # nist_num_cols = ['obs_wl_vac(nm)', 'unc_obs_wl', 'intens', 'Aki(s^-1)', 'Ei(eV)', 'Ek(eV)']
    # nist_lines_df[nist_num_cols] = nist_lines_df[nist_num_cols].apply(pd.to_numeric)
    nist_data: pd.DataFrame = Nist.query(
        wl_range[0] * u.nm, wl_range[1] * u.nm, linename='D;B;C;N', energy_level_unit='eV',
        output_order='wavelength', wavelength_type='vacuum'
    ).to_pandas()
    # nist_data.dropna(inplace=True)
    nist_data = nist_data[~nist_data['Observed'].isna()].reset_index(drop=True)
    # nist_data = nist_data[~nist_data['Rel.'].isna()].reset_index(drop=True)
    print(nist_data[['Spectrum','Observed', 'Rel.']])

    identified_lines = pd.DataFrame(data={
        'nist_wl_nm': [], 'line': [], 'nist_intensity': [], 'spec_wl_nm': [], 'spec_intensity': []
    })
    for i, row in nist_data.iterrows():
        nist_wl = row['Observed']
        msk_close = np.isclose(nist_wl, line_centers, atol=0.075)
        if msk_close.any():
            spec_wl = line_centers[msk_close][0]
            spec_intensity_idx = np.argmin(np.abs(spec_wl - wavelength))
            spec_intensity = brightness[spec_intensity_idx]
            id_df = pd.DataFrame(data={
                'nist_wl_nm': [nist_wl], 'line': [row['Spectrum']],
                'nist_intensity': [row['Rel.']],
                'spec_wl_nm': [spec_wl], 'spec_intensity': [spec_intensity],
            })
            identified_lines = pd.concat([identified_lines, id_df], ignore_index=True).reset_index(drop=True)

    folder_name = os.path.basename(relative_path)
    path_to_lines_folder = os.path.join('./data/identified_echelle_lines', folder_name)
    if not os.path.exists(path_to_lines_folder):
        os.makedirs(path_to_lines_folder)
    path_to_lines_csv = os.path.join(path_to_lines_folder, file_tag + '.csv')
    identified_lines.to_csv(path_to_lines_csv, index=False)

    # if subtract_background:
    #     radiance -= cal_factor_bgnd * f_counts_ps(wavelength)
    brightness[brightness<0] = brightness[brightness>0].min()
    # nu = 299792458.0 / wavelength
    # hnu = 6.62607015e-34 * nu
    # radiance = radiance * hnu * 1E9

    # peaks, _ = find_peaks(photon_flux, threshold=photon_flux.max()*0.005, distance=350)

    bh_range = (426, 435)
    bi_range = (818, 825)

    msk_bh = (bh_range[0] <= wavelength) & (wavelength <= bh_range[1])
    msk_bi = (bi_range[0] <= wavelength) & (wavelength <= bi_range[1])


    load_plot_style()
    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
    fig.set_size_inches(6., 6.5)

    # fig_bh, axes_bh = plt.subplots(nrows=2, ncols=1)
    # fig_bh.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0)
    # fig_bh.set_size_inches(6., 4.5)

    fig.supylabel(r"$B_{\lambda}$ {\sffamily(W/sr/m\textsuperscript{2}/nm)}", usetex=True)
    # ax.set_ylabel(r"$B$ {\sffamily(W/s/cm\textsuperscript{2}/nm)}", usetex=True)

    axes[0].plot(wavelength, brightness)
    # axes[0].plot(wavelength[peaks], photon_flux[peaks], "x")

    axes[1].plot(wavelength[msk_bh], brightness[msk_bh])
    axes[2].plot(wavelength[msk_bi], brightness[msk_bi])

    # axes_bh[0].plot(wavelength[msk_bh], photon_flux[msk_bh])
    # axes_bh[1].plot(wavelength[msk_bh], photon_flux[msk_bh])

    axes[0].xaxis.set_ticks_position('bottom')
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(100.))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(50.))
    axes[0].set_xlim(wl_range)
    axes[0].set_ylim(top=brightness.max()*1.25)


    peak_positions = []
    for i, peak in enumerate(lookup_lines):
        pc, pi = find_line(wavelength, brightness, peak['center_wl'], delta_wl=0.65)
        peak_positions.append({
            'peak_center': pc,
            'intensity': pi,
            'label': peak['label']
        })
        if (wl_range[0] < pc) & (pc < wl_range[1]) and pi >= brightness.max()*0.045:
            axes[0].text(
                pc, pi*1.075,
                r"\qquad \; " + peak['label'],
                ha='center',
                va='bottom',
                fontsize=10,
                usetex=True,
                rotation=90
            )

    # axes[0].plot(wavelength[msk_line_center], radiance[msk_line_center], marker='x', color='r', ls='none')

    axes[0].set_title(fr"{sample_label} - {file_tag}.asc")
    axes[0].text(
        0.99, 0.98, timestamp,
        transform=axes[0].transAxes,
        ha='right', va='top',
        fontsize=10
    )

    axes[1].set_xlim(bh_range)
    axes[2].set_xlim(bi_range)

    axes[1].set_ylim(bottom=0, top=brightness[msk_bh].max()*0.095)

    for ax in axes:
        ax.set_xlabel(r"$\lambda$ {\sffamily (nm)}", usetex=True)
        mf = ticker.ScalarFormatter(useMathText=True)
        mf.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(mf)
        ax.ticklabel_format(useMathText=True)

    # axes_bh[-1].set_xlabel(r"$\lambda$ {\sffamily (nm)}", usetex=True)
    # fig_bh.supylabel(r"$\Phi$ {\sffamily(photons/s/cm\textsuperscript{2}/nm)}", usetex=True)
    # for ax in axes_bh:
    #     ax.set_xlim(bh_range)
    #     mf = ticker.ScalarFormatter(useMathText=True)
    #     mf.set_powerlimits((-2, 2))
    #     ax.yaxis.set_major_formatter(mf)
    #     ax.ticklabel_format(useMathText=True)

    # axes_bh[0].set_ylim(top=photon_flux[msk_bh].max()*1.2, bottom=0.)
    # axes_bh[1].set_ylim(top=photon_flux[msk_bh].max() * 0.1, bottom=0.)

    plt.show()



if __name__ == '__main__':
    main()





