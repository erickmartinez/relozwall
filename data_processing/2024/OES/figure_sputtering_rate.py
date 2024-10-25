import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
from scipy.interpolate import interp1d, RegularGridInterpolator
import data_processing.echelle as ech
import os
from datetime import timedelta, datetime
import re
from scipy.signal import savgol_filter
from scipy.optimize import least_squares, OptimizeResult



path_to_pec = r'./data/ADAS/PEC/B/pec93#b_llu#b0.dat'
db_excel = r'./data/echelle_db.xlsx'


plasma_params = pd.DataFrame(data={
    'Sample': ['Boron rod', 'Boron pebble rod'],
    'Date': ['2024/08/15', '2024/08/27'],
    'T_e (eV)': [5.85, 5.3],
    'n_e 10^18 (/m^3)': [0.212, 0.2475],
    'phi_i 10^22 (/m^3/s)': [0.2125, 0.2375],
    'E_i (eV)': [62.6, 61.25],
    'Temp (K)': [458, 472]
})


d_pattern = '%a %b %d %H:%M:%S %Y'
window_coefficients = np.array([12.783, 0.13065, -8.467e-5])

def load_db():
    global db_excel
    df = pd.read_excel(db_excel, sheet_name='Spectrometer parameters')
    return df

def find_line(wl, intensity, line_center: float, delta_wl:float = 0.3, n_gaussians=1) -> tuple:
    # Look into a smaller window:
    msk_window = ((line_center - delta_wl) <= wl) & (wl <= (line_center + delta_wl))
    delta_fit = np.abs(delta_wl - 0.1)
    msk_fit = ((line_center - delta_fit) <= wl) & (wl <= (line_center + delta_fit))
    wl_window = wl[msk_window]
    intensity_window = intensity[msk_window]
    peak_height = intensity_window.max()

    all_tol = float(np.finfo(np.float64).eps)
    if n_gaussians == 1:
        idx_peak = np.argmin(np.abs(intensity_window - peak_height))
        wl_peak = wl_window[idx_peak]
        x0 = [peak_height, 0.01, wl_peak]
        res_lsq = least_squares(
            res_sum_gauss, x0=x0, args=(wl[msk_window], intensity[msk_window]), loss='linear', f_scale=0.1,
            jac=jac_sum_gauss,
            bounds=(
                [0., 1E-5, line_center - delta_wl],
                [intensity_window.max()*2., np.inf, line_center + delta_wl]
            ),
            xtol=all_tol,
            ftol=all_tol,
            gtol=all_tol,
            verbose=0,
            max_nfev=10000 * len(wl_window)
        )
        popt = res_lsq.x
        return wl_peak, peak_height, popt
    else:
        centers = np.linspace(line_center - delta_fit, line_center + delta_fit, num=n_gaussians)
        x0 = np.zeros(3 * n_gaussians, dtype=np.float64)
        lb = np.empty(n_gaussians * 3, dtype=np.float64)
        ub = np.empty(n_gaussians * 3, dtype=np.float64)
        for i in range(n_gaussians):
            idx_c = 3 * i
            idx_s = idx_c + 1
            idx_m = idx_s + 1

            x0[idx_c] = peak_height
            x0[idx_s] = 1E-1
            x0[idx_m] = centers[i]

            lb[idx_c] = 0.
            lb[idx_s] = 1E-4
            lb[idx_m] = wl[msk_window].min()

            ub[idx_c] = intensity_window.max()
            ub[idx_s] = np.inf
            ub[idx_m] = wl[msk_window].max()

            # print(f'C{i}: {lb[idx_c]:.4E} <= {x0[idx_c]:.4E} <= {ub[idx_c]:.4E}')
            # print(f'S{i}: {lb[idx_s]:.4E} <= {x0[idx_s]:.4E} <= {ub[idx_s]:.4E}')
            # print(f'M{i}: {lb[idx_m]:.4E} <= {x0[idx_m]:.4E} <= {ub[idx_m]:.4E}')


        res_lsq = least_squares(
            res_sum_gauss, x0=x0, args=(wl_window, intensity_window), loss='linear',
            f_scale=0.1,
            jac=jac_sum_gauss,
            bounds=(
                lb,
                ub
            ),
            xtol=all_tol,
            ftol=all_tol,
            gtol=all_tol,
            verbose=0,
            x_scale='jac',
            max_nfev=10000 * len(wl_window)
        )
        popt = res_lsq.x
        return popt[2::3], popt[0:3], popt

def get_wavelengths(path_to_file):
    p = re.compile(r'^\s*(\d+\.?\d*)\s+A')
    wl = []
    with open(file=path_to_file, mode='r') as f:
        for line in f.readlines():
            m = p.match(line)
            if m:
                wl.append(float(m.group(1)))
    wl.sort()
    return np.array([wl])

def get_pec(path_to_file, wl) -> pd.DataFrame:
    valid_wl = get_wavelengths(path_to_file)
    if wl not in valid_wl:
        raise(KeyError(f"The wavelength: {wl:.1f} Å was not found in {path_to_file}."))
    p_str = rf'^\s*{wl:.1f}\s+A\s+(\d+)\s+(\d+)'
    p_wl = re.compile(p_str)
    i0 = 100000000000
    n_e = []
    T_e = []
    pec_i = []
    found_wl = False
    done_with_ne = False
    done_with_Te = False
    done_with_pec = False
    p_n = re.compile(r'(\d+\.?\d+[Ee][\+\-]\d+)', re.IGNORECASE)
    pec_df = pd.DataFrame(data={
        'n_e (1/cm^3)': [],
        'T_e (eV)': [],
        'PEC (photons/cm^3/s)': []
    })
    k = 0 # count the number of
    with open(path_to_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if not found_wl:
                m_wl = p_wl.match(line)
                if m_wl:
                    i0 = i
                    n = int(m_wl.group(1)) # the number of electron densities
                    m = int(m_wl.group(2)) # the number of electron temperatures
            if i > i0:
                # find the first n numbers that correspond to n_e
                if not done_with_ne:
                    m_n = p_n.findall(line)
                    for v in m_n:
                        n_e.append(float(v))
                    if len(n_e) >= n:
                        done_with_ne = True
                elif not done_with_Te:
                    m_n = p_n.findall(line)
                    for v in m_n:
                        T_e.append(float(v))
                    if len(T_e) >= m:
                        done_with_Te = True
                elif not done_with_pec:
                    m_n = p_n.findall(line)
                    for v in m_n:
                        pec_i.append(float(v))
                        k += 1
                        if k >= m*n:
                            done_with_pec = True
            if done_with_pec:
                break


    pec = np.array(pec_i).reshape(n, m)

    n_e = np.array(n_e)
    T_e = np.array(T_e)
    for i in range(m):
        for j in range(n):
            row = pd.DataFrame(data={
                'n_e (1/cm^3)': [n_e[j]],
                'T_e (eV)': [T_e[i]],
                'PEC (photons/cm^3/s)': [pec[j][i]]
            })
            pec_df = pd.concat([pec_df, row], ignore_index=True)

    return pec_df

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


def bh_x_rate(T_e: np.ndarray) -> np.ndarray:
    """
    Estimates the excitation rate coefficient from the
    ground state of B-H for the transition:

    .. math::\Chi^1 \Sigma^+ \to \mathrm{A}^1\Pi

    as a function of the electron temperature.

    This relationship corresponds to the modified Arrhenius function
    .. math:: k = A T_e^n\exp\left(-\frac{E_{\mathrm{act}}{T_e}\right)

    described in Kawate et al. Plasma Sources Sci. Technol. 32, 085006 (2023)
    doi: 10.1088/1361-6595/acec0c


    Parameters
    ----------
    T_e: np.ndarray
        The electron temperature in eV

    Returns
    -------
    np.ndarray:
        The excitation rate coefficient in cm^3/s

    """
    return 5.62E-8 * np.power(T_e, 0.021) * np.exp(-3.06 / T_e)

def bh_s_rate(T_e: np.ndarray) -> np.ndarray:
    """
        Estimates the ionization rate coefficient from the
        ground state of B-H for the transition:

        .. math::\Chi^1 \Sigma^+ \to \mathrm{A}^1\Pi

        as a function of the electron temperature.

        This relationship corresponds to the modified Arrhenius function
        .. math:: k = A T_e^n\exp\left(-\frac{E_{\mathrm{act}}{T_e}\right)

        described in Kawate et al. Plasma Sources Sci. Technol. 32, 085006 (2023)
        doi: 10.1088/1361-6595/acec0c


        Parameters
        ----------
        T_e: np.ndarray
            The electron temperature in eV

        Returns
        -------
        np.ndarray:
            The ionization rate coefficient in cm^3/s

        """
    return 1.46E-8 * np.power(T_e, 0.690) * np.exp(-9.38 / T_e)

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}')


# Constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
m_B = 10.811 * 1.66053906660e-27  # Mass of boron atom (kg)
m_e = 9.10938356e-31  # Mass of electron (kg)
m_D = 2.014 * 1.66053906660e-27  # Mass of deuterium (kg)
def thermal_velocity(T, m):
    """
    Calculate thermal velocity
    T: temperature (K)
    m: mass of particle (kg)
    """
    global k_B
    # return np.sqrt(8 * k_B * T / (np.pi * m))
    return np.sqrt(2. * k_B * T / m)

def pec2flux(vth, intensity, n_e, pec, intensity_error):
    L = 1. # cm
    fb = vth * intensity / n_e / L / pec
    fb_err = np.abs(fb) * np.sqrt(np.power(intensity_error / intensity, 2.) + (0.05/L)**2. + (0.2*vth/vth)**2.)
    return fb, fb_err


sample_diameter = 1.016
sample_area = 0.25 * np.pi * sample_diameter ** 2.
flux_d = 0.23E18 # /cm^3/s
def flux2yield(flux_b):
    global flux_d
    return flux_b / flux_d

def yield2flux(x):
    global flux_d
    return x * flux_d




def plot_spectra(path_to_echelle, label, ax, line_center, wl_range, color, bg_df, n_gauss=1, center_del=0.1):

    wl_b, radiance_b, radiance_b_err = baselined_spectrum(
        path_to_echelle=path_to_echelle, background_df=bg_df,
        wl_min=wl_range[0], wl_max=wl_range[1]
    )

    ax.plot(wl_b, radiance_b, color=color, label=label)
    if n_gauss == 1:
        line_b, peak_height_b, popt_b = find_line(wl=wl_b, intensity=radiance_b, line_center=line_center, delta_wl=center_del)

        ax.plot([line_b], [peak_height_b], color='lime', marker='|', ms=12, ls='none', mew=2.5, mfc='lime')#, mec='lime')
        ax.plot(wl_b, gaussian(x=wl_b, c=popt_b[0], sigma=popt_b[1], mu=popt_b[2]), c='k', lw='0.5', ls=':')
    else:
        _, _, popt_b = find_line(wl=wl_b, intensity=radiance_b, line_center=line_center,
                                                     delta_wl=center_del, n_gaussians=n_gauss)
        cs = popt_b[0::3]
        sigmas = popt_b[1::3]
        mus = popt_b[2::3]
        total_c = np.sum(cs)
        hmax = sum_gaussians(x=wl_b, b=popt_b).max()
        for i in range(len(cs)):
            s = sigmas[i]
            ax.plot([mus[i]], [hmax], color='lime', marker='|', ms=12, ls='none', mew=2.5,
                    mfc='lime')
        ax.plot(wl_b, sum_gaussians(x=wl_b, b=popt_b), c='k', lw='0.5', ls=':')




def baselined_spectrum(path_to_echelle, background_df, wl_min=350, wl_max=850):
    h = 6.62607015  # E-34
    c = 2.99792458  # E8

    folder = os.path.basename(os.path.dirname(path_to_echelle))
    # load the data
    e_df, params = ech.load_echelle_file(path_to_file=path_to_echelle)
    e_df = e_df[e_df['wl (nm)'].between(wl_min, wl_max)].reset_index(drop=True)

    preamp_gain = int(params['Pre-Amplifier Gain'])
    exposure_s =float(params['Exposure Time (secs)'])
    try:
        accumulations = int(params['Number of Accumulations'])
    except KeyError as ke:
        accumulations = 1

    # select the background file based on the folder name
    background_df = background_df[background_df['Folder'] == folder].reset_index(drop=True)
    # Check that the pre-amp gain is the same for the sample and background spectra
    background_df = background_df[background_df['Pre-Amplifier Gain'] == preamp_gain].reset_index(drop=True)
    nn = len(background_df)
    file_bgnd = str(background_df.loc[nn - 1, 'File'])
    path_to_bg_echelle = os.path.join(r'./data', 'Echelle_data', folder, file_bgnd)
    # Load the background
    bg_df, bg_params = ech.load_echelle_file(path_to_file=path_to_bg_echelle)
    bg_df = bg_df[bg_df['wl (nm)'].between(wl_min, wl_max)].reset_index(drop=True)


    wl_sample = e_df['wl (nm)'].values
    counts_sample = e_df['counts'].values
    counts_sample[counts_sample < 0.] = 0.
    cps_sample = counts_sample / exposure_s / accumulations
    transmission = transmission_dirty_window(wl_sample)
    cps_sample /= transmission

    # preamp_gain_bg = int(bg_params['Pre-Amplifier Gain'])
    exposure_s_bg = float(bg_params['Exposure Time (secs)'])
    try:
        accumulations_bg = int(bg_params['Number of Accumulations'])
    except KeyError as ke:
        accumulations_bg = 1

    wl_bg = bg_df['wl (nm)'].values
    counts_bg = bg_df['counts'].values
    counts_bg[counts_bg < 0.] = 0.
    cps_bg = counts_bg / exposure_s_bg / accumulations_bg

    # Smooth the background
    cps_bg = savgol_filter(
        cps_bg,
        window_length=5,
        polyorder=3
    )

    # interpolate the background to the wavelengths of the sample
    f_bg = interp1d(x=wl_bg, y=cps_bg)
    cps_bg_interp = f_bg(wl_sample)

    cps_sample -= cps_bg_interp
    cps_sample[cps_sample < 0.] = 0.

    # Construct interpolations of the calibration for preamp gain
    cal, cal_err = get_interpolated_calibration(preamp_gain=preamp_gain)
    radiance = cal(wl_sample) * cps_sample  # W / cm^2 / sr /nm
    radiance_err = cal_err(wl_sample) * cps_sample

    byhnu = wl_sample / c / h * 1E17  * 4. * np.pi# 1 / J

    return wl_sample, radiance*byhnu, radiance_err*byhnu # photons/m^2/s/nm/sr


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


def main():
    global path_to_pec, k_B, m_B, m_e, m_D
    # wl = get_wavelengths(path_to_file=path_to_pec)
    pec_df: pd.DataFrame = get_pec(path_to_file=path_to_pec, wl=8148.2)
    # pec_df = pec_df[pec_df['T_e (eV)'] <= 100]
    bi_pec_ne = pec_df['n_e (1/cm^3)'].unique()
    bi_pec_te = pec_df['T_e (eV)'].unique()
    n_te, n_ne = len(bi_pec_te), len(bi_pec_ne)
    z_pec = np.empty((n_te, n_ne), dtype=np.float64)
    for i in range(n_te):
        for j in range(n_ne):
            dfi = pec_df[(pec_df['T_e (eV)'] == bi_pec_te[i]) & (pec_df['n_e (1/cm^3)'] == bi_pec_ne[j])].reset_index(
                drop=True)
            z_pec[i, j] = dfi['PEC (photons/cm^3/s)'][0]

    # interpolate the pec coefficient for b_i
    f_pec = RegularGridInterpolator((bi_pec_ne, bi_pec_te), z_pec.T)


    db_df = load_db()
    db_df.sort_values(by=['Folder', 'File'], ascending=[True, True])
    db_df = db_df[~(db_df['Label'] == 'Labsphere')]
    full_db_df = db_df.reset_index(drop=True)
    # db_df = db_df[~(db_df['Folder'] == 'echelle_20240910')]
    db_background_df = db_df[db_df['Is dark'] == 1].reset_index(drop=True)
    db_df = db_df[~(db_df['Is dark'] == '1')]
    db_df.reset_index(inplace=True, drop=True)
    sample_labels = db_df['Label'].unique()

    load_plot_style()
    wl_range_bi = [818, 823]
    wl_range_bh = [432, 434.4]

    load_plot_style()
    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True, width_ratios=[1., 1.1])
    fig.set_size_inches(7., 6.)

    for ax in axes[:, 0]:
        ax.set_xlabel(r'$\lambda$ {\sffamily (nm)}', usetex=True)
        ax.set_ylabel(r"$B_{\lambda}$ {\sffamily (photons/cm\textsuperscript{2}/nm)}", usetex=True)

    for ax in axes[:, 1]:
        ax.set_xlabel(r"$t$ {\sffamily (min)}", usetex=True)

    axes[0, 1].set_ylabel(r"$\Phi_{\mathrm{B}}$ {\sffamily (cm\textsuperscript{-2} s\textsuperscript{-1})}", usetex=True)
    axes[1, 1].set_ylabel(r"$\Phi_{\mathrm{BD}}$ {\sffamily (cm\textsuperscript{-2} s\textsuperscript{-1})}",
                          usetex=True)


    for i, axi in enumerate(axes.flatten()):
        panel_label = chr(ord('`') + i + 1)
        axi.text(
            -0.175, 1.05, f'({panel_label})', transform=axi.transAxes, fontsize=12, fontweight='bold',
            va='top', ha='right'
        )


    fbi_r = []
    fbh_r = []
    fbi_r_err = []
    fbh_r_err = []
    fbi_p = []
    fbh_p = []
    fbi_p_err = []
    fbh_p_err = []
    time_br = []
    time_bp = []

    # plot boron rod spectra
    path_to_echelle = os.path.join('./data/Echelle_data', 'echelle_20240815', 'MechelleSpect_009.asc')
    plot_spectra(
        path_to_echelle=path_to_echelle, label='Boron rod',
        ax=axes[0, 0], line_center=821.2, wl_range=wl_range_bi,
        color='C0', bg_df=db_background_df
    )

    # plot boron pebble rod spectra
    path_to_echelle = os.path.join('./data/Echelle_data', 'echelle_20240827', 'MechelleSpect_022.asc')
    plot_spectra(
        path_to_echelle=path_to_echelle, label='Boron pebble rod',
        ax=axes[0, 0], line_center=821.2, wl_range=wl_range_bi,
        color='C1', bg_df=db_background_df
    )

    path_to_echelle = os.path.join('./data/Echelle_data', 'echelle_20240815', 'MechelleSpect_012.asc')
    plot_spectra(
        path_to_echelle=path_to_echelle, label='Boron rod',
        ax=axes[1, 0], line_center=432.89, wl_range=wl_range_bh,
        color='C0', bg_df=db_background_df, n_gauss=4, center_del=0.4
    )

    path_to_echelle = os.path.join('./data/Echelle_data', 'echelle_20240827', 'MechelleSpect_026.asc')
    plot_spectra(
        path_to_echelle=path_to_echelle, label='Boron pebble rod',
        ax=axes[1, 0], line_center=432.89, wl_range=wl_range_bh,
        color='C1', bg_df=db_background_df, n_gauss=4, center_del=0.4
    )


    for sl in sample_labels:
        dfi = db_df[db_df['Label'] == sl].reset_index(drop=True)
        # dfi = dfi[dfi['Is dark'] == 0].reset_index(drop=True)
        # dfi = dfi[dfi['Plot B-I']==True].reset_index(drop=True)
        # Find the corresponding background
        # bgnd_df = db_background_df[db_background_df['Label'] == sl].reset_index(drop=True)
        n_samples = len(dfi)
        norm = mpl.colors.Normalize(vmin=0, vmax=(n_samples-1))
        t0 = 0.

        for i, row in dfi.iterrows():
            folder = row['Folder']
            path_to_echelle = os.path.join('./data/Echelle_data', folder, row['File'])
            # file_tag = os.path.splitext(row['File'])[0]
            # e_df, params = ech.load_echelle_file(path_to_file=path_to_echelle)
            timestamp = row['Timestamp']
            if i == 0:
                t0 = timestamp
            elapsed_time: timedelta = timestamp - t0


            wl_bi, radiance_bi, radiance_bi_err = baselined_spectrum(
                path_to_echelle=path_to_echelle, background_df=db_background_df,
                wl_min=wl_range_bi[0], wl_max=wl_range_bi[1]
            )

            wl_bh, radiance_bh, radiance_bh_err = baselined_spectrum(
                path_to_echelle=path_to_echelle, background_df=db_background_df,
                wl_min=wl_range_bh[0], wl_max=wl_range_bh[1]
            )


            _, _, popt_bi = find_line(wl=wl_bi, intensity=radiance_bi, line_center=821.2, delta_wl=0.15, n_gaussians=2)
            # idx_at_line = np.argmin(np.abs(line_bi - wl_bi))
            peak_height_bi = sum_gaussians(wl_bi, popt_bi).max()
            idx_at_line = np.argmin(np.abs(wl_bi - 821.2))
            intensity_bi = np.sum(popt_bi[0::3])
            # intensity_bi = peak_height_bi

            intensity_bi_err = intensity_bi * radiance_bi_err[idx_at_line] / peak_height_bi
            # intensity_bi_err = radiance_bi_err[idx_at_line]

            # line_bh, peak_height_bh, popt_bh = find_line(wl=wl_bh, intensity=radiance_bh, line_center=432.57, delta_wl=0.1)
            # idx_at_line = np.argmin(np.abs(line_bh - wl_bh))
            #
            # intensity_bh = popt_bh[0]
            # intensity_bh_err = np.abs(intensity_bh * radiance_bh_err[idx_at_line] / peak_height_bh)

            _, _, popt_bh = find_line(wl=wl_bh, intensity=radiance_bh, line_center=432.57,
                                                         delta_wl=0.4, n_gaussians=2)

            intensity_bh = np.sum(popt_bh[0::3])
            # print(f'Intensity_bd: {intensity_bh:.1E}')
            peak_height_bh = sum_gaussians(wl_bh, popt_bh).max()
            idx_at_line = np.argmin(np.abs(radiance_bh - peak_height_bh))
            # intensity_bh = peak_height_bh
            intensity_bh_err = np.abs(intensity_bh * radiance_bh_err[idx_at_line] / peak_height_bh)
            # line_bh = np.mean(popt_bh[2::3])
            line_bh = 432.57 # radiance_bh[idx_at_line]
            # intensity_bh_err = radiance_bh_err[idx_at_line]

            vth_B = thermal_velocity(T=500, m=m_B) * 100.
            vth_BH = thermal_velocity(T=500, m=(m_B+m_D)) * 100.
            vth_B = 1E4
            vth_BH = 1E4
            pec_bi = 5.1E-11
            if sl == 'Boron rod':
                t_e = 5.85 # eV
                n_e = 0.212E12 # 1/cm^3

                # pec_bi = f_pec((n_e, t_e))
                pec_bh = bh_x_rate(T_e=t_e)
                # print(f'v_th_B: {vth_B:.3E} cm/s, v_th_BH: {vth_BH:.3E} cm/s, pec_bi: {pec_bi:.3E} cm^3/s, radiance_bi: {intensity_bi:.3E}, wl_bi: {line_bi:.1f}')

                fb, fb_err = pec2flux(vth=vth_B, intensity=intensity_bi, n_e=n_e, pec=pec_bi,
                                      intensity_error=intensity_bi_err)
                fbi_r.append(fb)
                fbi_r_err.append(fb_err)

                fb, fb_err = pec2flux(vth=vth_BH, intensity=intensity_bh, n_e=n_e, pec=pec_bh,
                                      intensity_error=intensity_bh_err)
                fbh_r.append(fb)
                fbh_r_err.append(fb_err)
                time_br.append(elapsed_time.total_seconds()/60)


            else:
                t_e = 5.3  # eV
                n_e = 0.2475E12  # 1/cm^3

                # pec_bi = f_pec((n_e, t_e))
                pec_bh = bh_x_rate(T_e=t_e)
                time_bp.append(elapsed_time.total_seconds() / 60.)

                fb, fb_err = pec2flux(vth=vth_B, intensity=intensity_bi, n_e=n_e, pec=pec_bi,
                                      intensity_error=intensity_bi_err)
                fbi_p.append(fb)
                fbi_p_err.append(fb_err)
                fb, fb_err = pec2flux(vth=vth_BH, intensity=intensity_bh, n_e=n_e, pec=pec_bh,
                                      intensity_error=intensity_bh_err)
                fbh_p.append(fb)
                fbh_p_err.append(fb_err)



    time_br = np.array(time_br)
    time_bp = np.array(time_bp)
    fbi_r = np.array(fbi_r)
    fbi_r_err = np.array(fbi_r_err)
    fbi_p = np.array(fbi_p)
    fbi_p_err = np.array(fbi_p_err)
    fbh_r = np.array(fbh_r)
    fbh_r_err = np.array(fbh_r_err)
    fbh_p = np.array(fbh_p)
    fbh_p_err = np.array(fbh_p_err)

    fbi_r_mean = fbi_r.mean()
    markers_bir, caps_bir, bars_bir = axes[0, 1].errorbar(
        time_br, fbi_r, yerr=fbi_r_err, capsize=2.75, mew=1.25, marker='o', ms=8, elinewidth=1.25,
        color='C0', fillstyle='none',
        ls='none',
        label='Boron rod',
    )

    axes[0, 1].axhline(y=fbi_r_mean, ls=':', c='C0', lw=1.5)
    print(f'Boron rod, Mean sputtering yield BI: {100*flux2yield(fbi_r_mean):.1f} ')
    [bar.set_alpha(0.35) for bar in bars_bir]
    [cap.set_alpha(0.35) for cap in caps_bir]
    axes[0, 1].text(
        100, fbi_r_mean, fr"$\langle Y_{{\mathrm{{B}} }}\rangle = \SI{{{100*flux2yield(fbi_r_mean):.1f}}}{{\percent}} \\$", ha='right', va='bottom',
        c='C0', usetex=True
    )


    markers_bip, caps_bip, bars_bip = axes[0, 1].errorbar(
        time_bp, fbi_p, yerr=fbi_p_err, capsize=2.75, mew=1.25, marker='^', ms=8, elinewidth=1.25,
        color='C1', fillstyle='none',
        ls='none',
        label='Boron pebble rod',
    )

    fbi_p_mean = fbi_p.mean()
    axes[0, 1].axhline(y=fbi_p_mean, ls=':', c='C1', lw=1.5)
    print(f'Boron pebble rod, Mean sputtering yield BI: {100*flux2yield(fbi_p_mean):.1f} ')
    [bar.set_alpha(0.35) for bar in bars_bip]
    [cap.set_alpha(0.35) for cap in caps_bip]
    axes[0, 1].text(
        100, fbi_p_mean, fr"$\langle Y_{{\mathrm{{B}} }} \rangle = \SI{{{100 * flux2yield(fbi_p_mean):.1f}}}{{\percent}} \\$",
        ha='right', va='bottom',
        c='C1', usetex=True
    )

    fbh_r_mean = fbh_r.mean()
    markers_bhr, caps_bhr, bars_bhr = axes[1, 1].errorbar(
        time_br, fbh_r, yerr=fbh_r_err, capsize=2.75, mew=1.25, marker='s', ms=8, elinewidth=1.25,
        color='C0', fillstyle='none',
        ls='none',
        label='Boron rod',
    )

    axes[1, 1].axhline(y=fbh_r_mean, ls=':', c='C0', lw=1.5)
    print(f'Boron rod, Mean sputtering yield BD: {100*flux2yield(fbh_r_mean):.0E} %')
    [bar.set_alpha(0.35) for bar in bars_bhr]
    [cap.set_alpha(0.35) for cap in caps_bhr]

    axes[1, 1].text(
        100, fbh_r_mean, fr"$\langle Y_{{\mathrm{{BD}} }} \rangle = \SI{{{100 * flux2yield(fbh_r_mean):.0E}}}{{\percent}} \\$",
        ha='right', va='bottom',
        c='C0', usetex=True
    )

    fbh_p_mean = fbh_p.mean()
    markers_bhp, caps_bhp, bars_bhp = axes[1, 1].errorbar(
        time_bp, fbh_p, yerr=fbh_p_err, capsize=2.75, mew=1.25, marker='v', ms=8, elinewidth=1.25,
        color='C1', fillstyle='none',
        ls='none',
        label='Boron pebble rod',
    )

    axes[1, 1].axhline(y=fbh_p_mean, ls=':', c='C1', lw=1.5)
    print(f'Boron pebble rod, Mean sputtering yield BD: {100*flux2yield(fbh_p_mean):.1E} ')
    [bar.set_alpha(0.35) for bar in bars_bhp]
    [cap.set_alpha(0.35) for cap in caps_bhp]

    axes[1, 1].text(
        100, fbh_p_mean, fr"$\langle Y_{{\mathrm{{BD}} }}\rangle = \SI{{{100 * flux2yield(fbh_p_mean):.0E}}}{{\percent}} \\$",
        ha='right', va='bottom',
        c='C1', usetex=True
    )

    axes[0, 0].set_xlim(wl_range_bi)
    axes[1, 0].set_xlim(wl_range_bh)

    axes[0, 0].set_ylim(0, 1.75E13)
    axes[1, 0].set_ylim(0, 1.2E12)
    axes[1, 0].axvline(x=433.98, ls='--', lw=1., c='k')
    axes[1, 0].text(
        0.85, 0.15, r"D$_{\gamma}$", fontsize=11, color='k',
        ha='left', va='bottom',
        transform=axes[1, 0].transAxes
    )

    eckstein_flux = yield2flux(1.17E-3)
    axes[0, 1].plot([0, 100], [eckstein_flux, eckstein_flux], ls='--', lw=1., c='k', label= 'Eckstein 2002\n'+ r"$T_e = 30~\mathrm{eV}$",)


    axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(1.))
    axes[0, 0].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    axes[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(1.))
    axes[1, 0].xaxis.set_minor_locator(ticker.MultipleLocator(0.2))

    axes[0, 1].set_ylim(1E14, 1.0E17)
    axes[1, 1].set_ylim(1E10, 1E13)



    for ax in axes.flatten():
        mf = ticker.ScalarFormatter(useMathText=True)
        mf.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(mf)
        ax.ticklabel_format(useMathText=True)

    for ax in axes[:, 0]:
        ax.legend(loc='upper left', fontsize=9, ncols=1)
    for ax in axes[:, 1]:
        ax.legend(loc='upper right', fontsize=9, ncols=1)
        ax.set_xlim(0, 100.)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.set_yscale('log')

    secax1 = axes[0, 1].secondary_yaxis('right', functions=(flux2yield, yield2flux))
    secax2 = axes[1, 1].secondary_yaxis('right', functions=(flux2yield, yield2flux))
    secax1.set_ylabel(r'$Y_{\mathrm{B/D^+}}$', usetex=True)
    secax2.set_ylabel(r'$Y_{\mathrm{BD/D^+}}$', usetex=True)

    for ax in [secax1, secax2]:
        # mf = ticker.ScalarFormatter(useMathText=True)
        # mf.set_powerlimits((-2, 2))
        # ax.yaxis.set_major_formatter(mf)
        # ax.ticklabel_format(useMathText=True)
        ax.set_yscale('log')

    for ax in axes[0, :]:
        ax.set_title('B I')
    for ax in axes[1, :]:
        ax.set_title('B-D')

    fig.savefig(r'./figures/fig_sputtering_rate.png', dpi=600)
    fig.savefig(r'./figures/fig_sputtering_rate.pdf', dpi=600)
    fig.savefig(r'./figures/fig_sputtering_rate.svg', dpi=600)
    plt.show()




if __name__ == '__main__':
    main()