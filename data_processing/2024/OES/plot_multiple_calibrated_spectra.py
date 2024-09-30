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
from scipy.stats.distributions import t

db_excel = r'./data/echelle_db.xlsx'
wl_range = (821-3, 821+3)

window_coefficients = np.array([12.783, 0.13065, -8.467e-5])


def load_db():
    global db_excel
    df = pd.read_excel(db_excel, sheet_name='Spectrometer parameters')
    # df = df[df['Folder'] == 'echelle_20240910'].reset_index(drop=True)
    return df

def load_echelle_calibration(preamp_gain):
    csv_file = r'./data/echelle_calibration_20240910.csv'
    if preamp_gain not in [1, 4]:
        msg = f'Error loading echelle calibration: {preamp_gain} not found in calibration.'
        print(msg)
        raise ValueError(msg)
    col_cal = fr'Flux @pregain {preamp_gain:d} (Photons/s/sr/cm^2/nm)'
    col_err = fr'Flux @pregain {preamp_gain:d} error (Photons/s/sr/cm^2/nm)'
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
    col_cal = fr'Flux @pregain {preamp_gain:d} (Photons/s/sr/cm^2/nm)'
    col_err = fr'Flux @pregain {preamp_gain:d} error (Photons/s/sr/cm^2/nm)'
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

def load_labsphere_calibration():
    df = pd.read_csv(
        './data/PALabsphere_2014.txt', sep=' ', comment='#',
        usecols=[0], names=['Radiance (W/cm2/ster/nm)']
    ).apply(pd.to_numeric)
    # radiance = df['Radiance (W/cm2/ster/nm)']
    n = len(df)
    wl = 350. + np.arange(n) * 10.
    df['Wavelength (nm)'] = wl
    return df[['Wavelength (nm)', 'Radiance (W/cm2/ster/nm)']]


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
    db_df = load_db()
    db_df.sort_values(by=['Folder', 'File'], ascending=[True, True])
    # Construct interpolations of the calibration for preamp gain 1 and 4
    cal_pag_1, cal_pag_1_err = get_interpolated_calibration(preamp_gain=1)
    cal_pag_4, cal_pag_4_err = get_interpolated_calibration(preamp_gain=4)
    # labsphere_df = load_labsphere_calibration()
    # wl_ls = labsphere_df['Wavelength (nm)'].values
    wl_min, wl_max = 350, 900
    load_plot_style()
    prev_folder = ''
    t0 = 0.
    for i, row in db_df.iterrows():
        folder = row['Folder']
        path_to_echelle = os.path.join('./data/Echelle_data', folder, row['File'])
        file_tag = os.path.splitext(row['File'])[0]
        path_to_figures = os.path.join('./figures/Echelle_plots/calibrated', row['Folder'], f'{wl_range[0]}-{wl_range[1]}_nm')
        if not os.path.exists(path_to_figures):
            os.makedirs(path_to_figures)
        df, params = ech.load_echelle_file(path_to_file=path_to_echelle)
        timestamp = row['Timestamp']
        if folder != prev_folder:
            t0 = timestamp
        elapsed_time: timedelta = timestamp - t0
        df = df[df['wl (nm)'].between(wl_range[0], wl_range[1])].reset_index(drop=True)
        preamp_gain = int(row['Pre-Amplifier Gain'])
        exposure_s = row['Exposure Time (secs)']
        accumulations = row['Number of Accumulations']
        wl_i = df['wl (nm)'].values
        counts = df['counts'].values
        counts[counts < 0.] = 0.
        counts_ps = counts / exposure_s / accumulations
        transmission = 1.
        if row['Folder'] != 'echelle_20240910':
            transmission = transmission_dirty_window(wl_i)
        if preamp_gain == 1:
            cal_factor = cal_pag_1(wl_i)
        elif preamp_gain == 4:
            cal_factor = cal_pag_4(wl_i)
        else:
            raise ValueError(f'Calibration not performed for preamplifier gain {preamp_gain:d}.')

        counts_ps /= transmission
        photon_flux = cal_factor * counts_ps

        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        fig.set_size_inches(4.5, 3.)

        ax.set_xlabel(r"$\lambda$ {\sffamily (nm)}", usetex=True)
        ax.set_ylabel(r"$\Phi$ {\sffamily(photons/s/cm\textsuperscript{2}/nm)}", usetex=True)

        ax.plot(wl_i, photon_flux)

        ax.xaxis.set_ticks_position('bottom')
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(100.))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(50.))
        mf = ticker.ScalarFormatter(useMathText=True)
        mf.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(mf)
        ax.ticklabel_format(useMathText=True)
        ax.set_xlim(wl_range)

        elapsed_total_sec = elapsed_time.total_seconds()
        delta_hours = int(elapsed_total_sec/3600.)
        rem = elapsed_total_sec - delta_hours * 3600.
        delta_min = int(rem / 60.)
        rem -= delta_min * 60.
        delta_sec = int(rem)
        ax.text(
            0.01, 0.98, f"{elapsed_time.__str__()[-8::]}", #\n{delta_hours:02d}:{delta_min:02d}:{delta_sec:02d}",
            transform=ax.transAxes, ha='left', va='top', fontsize=10
        )

        ax.set_title(fr"{file_tag}.asc")
        prev_folder = folder
        fig.savefig(os.path.join(path_to_figures, file_tag + f'_preamp_gain_{preamp_gain}.png'), dpi=600)
        plt.close()





if __name__ == '__main__':
    main()

