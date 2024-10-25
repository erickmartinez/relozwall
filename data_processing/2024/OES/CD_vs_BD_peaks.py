import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from scipy.optimize import least_squares, OptimizeResult

from httplib2.auth import params

bd_peak_db = r'./data/b-d_gaussian_peak.xlsx'
echelle_db = r'./data/echelle_db.xlsx'
folder_map_xls = r'./PISCES-A_folder_mapping.xlsx'


plasma_params = pd.DataFrame(data={
    'Sample': ['Boron rod', 'Boron pebble rod'],
    'Date': ['2024/08/15', '2024/08/27'],
    'T_e (eV)': [5.85, 5.3],
    'n_e 10^18 (/m^3)': [0.212, 0.2475],
    'phi_i 10^22 (/m^3/s)': [0.2125, 0.2375],
    'E_i (eV)': [62.6, 61.25],
    'Temp (K)': [458, 472]
})

def load_db():
    global echelle_db
    df = pd.read_excel(echelle_db, sheet_name='Spectrometer parameters')
    return df

def load_folder_mapping():
    global folder_map_xls
    df = pd.read_excel(folder_map_xls, sheet_name=0)
    mapping = {}
    for i, row in df.iterrows():
        mapping[row['Echelle folder']] = row['Data label']
    return mapping

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


def pec2flux(vth, intensity, n_e, pec, intensity_error):
    L = 1. # cm
    fb = vth * intensity / n_e / L / pec
    fb_err = np.abs(fb) * np.sqrt(np.power(intensity_error / intensity, 2.) + (0.05/L)**2. + (0.2*vth/vth)**2.)
    return fb, fb_err

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

sample_diameter = 1.016
sample_area = 0.25 * np.pi * sample_diameter ** 2.
flux_d = 0.23E18 # /cm^3/s
def flux2yield(flux_b):
    global flux_d
    return flux_b / flux_d

def yield2flux(x):
    global flux_d
    return x * flux_d

def ci2yerr(y, y_lb, y_ub):
    return y_ub - y, y - y_lb

def main():
    peaks_df: pd.DataFrame = pd.read_excel(bd_peak_db, sheet_name=0)
    peaks_cols = peaks_df.columns
    peaks_df[peaks_cols[2:-2]] = peaks_df[peaks_cols[2:-2]].apply(pd.to_numeric)
    peaks_df['Timestamp'] = peaks_df['Timestamp'].apply(pd.to_datetime)
    peaks_df['File'] = peaks_df['File'] + '.asc'
    params_df = load_db()
    # Exclude labsphere measurements
    params_df = params_df[~(params_df['Label'] == 'Labsphere')]
    # Exclude measurements labeled as 'dark'
    params_df = params_df[~(params_df['Is dark'] == 1)]
    # params_num_cols = [
    #     'Exposure Time (secs)', 'Number of Accumulations', 'Horizontal binning', 'Vertical Shift Speed (usecs)',
    #     'Pixel Readout Rate (MHz)', 'Pre-Amplifier Gain', 'Gain level', 'Gate Width (nsecs)', 'Gate Delay (nsecs)',
    #     'Current Temperature (C)', 'Is dark'
    # ]
    # Get the elapased time since the first spectrum for each spectrum in the folder
    params_df['Timestamp'] = params_df['Timestamp'] .apply(pd.to_datetime)
    params_df['Elapsed time (s)'] = (params_df['Timestamp'] -  params_df['Timestamp'].min()).dt.total_seconds()# Arbitrary value for now, different t0 for every folder
    unique_folders = params_df['Folder'].unique()
    for folder in unique_folders:
        row_indexes = params_df['Folder'] == folder
        ts = params_df.loc[row_indexes, 'Timestamp'].reset_index(drop=True)
        params_df.loc[row_indexes, 'Elapsed time (s)'] = (params_df.loc[row_indexes,'Timestamp'] - ts[0]).dt.seconds

    params_df = params_df[params_df['Number of Accumulations']>=20]
    good_peaks_df = pd.merge(params_df, peaks_df, how='right', on=['Folder', 'File']).reset_index(drop=True)

    # Use folder_map_xls to map the dated folder to the corresponding sample
    folder_mapping = load_folder_mapping()
    sample_labels = [folder_mapping[folder] for folder in good_peaks_df['Folder']]
    good_peaks_df['Sample label'] = sample_labels

    # print(good_peaks_df[['File', 'Folder', 'Number of Accumulations', 'Elapsed time (s)', 'C-D C (photons/cm^2/s)']])

    load_plot_style()
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.2, 6.)

    vth_B = 1E4
    vth_BH = 1E4
    pec_bi = 5.1E-11
    n_e = 0.212E12  # 1/cm^3

    pec_bh = bh_x_rate(T_e=5.85)
    all_tol = float(np.finfo(np.float64).eps)

    colors = [f'C{i}' for i in range(len(unique_folders))]
    for i, folder in enumerate(unique_folders):
        lbl = folder_mapping[folder]
        sample_df = good_peaks_df[good_peaks_df['Folder'] == folder].reset_index(drop=True)
        # print(sample_df['B-D C lb (photons/cm^2/s)'].describe())
        # sample_df.loc[(sample_df['B-D C lb (photons/cm^2/s)'] < 0.5*sample_df['B-D C lb (photons/cm^2/s)'].mean()), 'B-D C lb (photons/cm^2/s)'] = 5*sample_df['B-D C lb (photons/cm^2/s)'].min()
        # # print(sample_df[['C-D C (photons/cm^2/s)', 'B-D C (photons/cm^2/s)']])
        # print(sample_df['B-D C lb (photons/cm^2/s)'].describe())
        xx = sample_df['C-D C (photons/cm^2/s)'].values
        yy = sample_df['B-D C (photons/cm^2/s)'].values
        xerr = ci2yerr(
                xx,
                sample_df['C-D C lb (photons/cm^2/s)'].values,
                sample_df['C-D C ub (photons/cm^2/s)'].values
        )
        yerr = ci2yerr(
                yy,
                sample_df['B-D C lb (photons/cm^2/s)'].values,
                sample_df['B-D C ub (photons/cm^2/s)'].values
            )

        xerr_m = np.hstack([xerr]).T
        yerr_m = np.hstack([yerr]).T

        xymaxerr = np.stack([xerr_m[:,1], yerr_m[:,1]]).T
        # print(xymaxerr)

        ww = np.linalg.norm(xymaxerr, axis=1)
        ww = np.power(ww, -1.)


        ls_res = least_squares(
            res_poly,
            x0=[0.1, 1.],
            args=(xx, yy, ww),
            # loss='soft_l1', f_scale=0.1,
            jac=jac_poly,
            xtol=all_tol,
            ftol=all_tol,
            gtol=all_tol,
            verbose=2,
            # x_scale='jac',
            max_nfev=10000 * len(xx)
        )

        popt = ls_res.x
        xpred = np.linspace(xx.min(), xx.max(), 500)
        axes[0].plot(
            xpred, model_poly(xpred, popt), color=colors[i], ls='--', lw=1.25
        )
        markers_cb, caps_cb, bars_cb = axes[0].errorbar(
            sample_df['C-D C (photons/cm^2/s)'].values,
            sample_df['B-D C (photons/cm^2/s)'].values,
            xerr=xerr,
            yerr=yerr,
            capsize=2.75, mew=1.25, marker='o', ms=8, elinewidth=1.25,
            color=colors[i], fillstyle='none',
            ls='none',
            label=lbl,
        )

        [bar.set_alpha(0.25) for bar in bars_cb]
        [cap.set_alpha(0.25) for cap in caps_cb]

        time_min = sample_df['Elapsed time (s)'].values/60

        fb, fb_err = pec2flux(vth=vth_B, intensity=yy, n_e=n_e, pec=pec_bh,
                              intensity_error=np.min(yerr_m,axis=1))

        # ls_fb = least_squares(
        #     model_poly,
        #     x0=[0.1, 1.],
        #     args=(time_min, np.log(fb)),
        #     # loss='soft_l1', f_scale=0.1,
        #     jac=jac_poly,
        #     xtol=all_tol,
        #     ftol=all_tol,
        #     gtol=all_tol,
        #     verbose=2,
        #     x_scale='jac',
        #     max_nfev=10000 * len(xx)
        # )
        #
        # t_pred = np.linspace()

        axes[1].errorbar(
            time_min,
            fb,
            yerr=fb_err,
            capsize=2.75, mew=1.25, marker='o', ms=8, elinewidth=1.25,
            color=colors[i], fillstyle='none',
            ls='none',
            label=lbl,
        )

    axes[0].set_xlabel('C-D peak (photons/cm$^{\mathregular{2}}$/s)')
    axes[0].set_ylabel('B-D peak (photons/cm$^{\mathregular{2}}$/s)')
    axes[1].set_xlabel('Time (min)')
    axes[1].set_ylabel(r"$\Phi_{\mathrm{BD}}$ {\sffamily (cm\textsuperscript{-2} s\textsuperscript{-1})}", usetex=True)
    axes[1].set_yscale('log')

    mf = ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    axes[0].xaxis.set_major_formatter(mf)
    axes[0].yaxis.set_major_formatter(mf)
    axes[0].ticklabel_format(useMathText=True)
    axes[0].legend(loc='upper left', frameon=True, fontsize=10)

    axes[1].set_ylim(1E9, 1E13)

    secax1 = axes[1].secondary_yaxis('right', functions=(flux2yield, yield2flux))
    secax1.set_ylabel(r'$Y_{\mathrm{B/D^+}}$', usetex=True)
    fig.savefig(r'./figures/CD_vs_BD.png', dpi=600)

    plt.show()



if __name__ == '__main__':
    main()


