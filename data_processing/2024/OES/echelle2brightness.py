import os
import pandas as pd
from scipy.interpolate import interp1d
import data_processing.echelle as ech
import os
from scipy.stats.distributions import t
from scipy.signal import savgol_filter
import numpy as np


output_path = r'./data/brightness_data'

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

def load_db():
    df = pd.read_excel(r'./data/echelle_db.xlsx', sheet_name='Spectrometer parameters')
    # df = df[df['Folder'] == 'echelle_20240910'].reset_index(drop=True)
    return df

def baselined_spectrum(path_to_echelle, background_df, wl_min=350, wl_max=850):
    base_name = os.path.basename(path_to_echelle)
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
    h = 6.62607015  # E-34
    c = 2.99792458  # E8
    byhnu = wl_sample / c / h * 1E17  * 4. * np.pi # 1 / J / s

    return wl_sample, radiance*byhnu, radiance_err*byhnu # photons/cm^2/s/nm/sr


def main():
    db_df = load_db()
    db_df.sort_values(by=['Folder', 'File'], ascending=[True, True])
    # Do not plot calibration files
    db_df = db_df[~(db_df['Label'] == 'Labsphere')]
    db_background_df = db_df[db_df['Is dark'] == 1].reset_index(drop=True)
    # Do not process files spectra under dark conditions (used as background correction)
    db_df = db_df[~(db_df['Is dark'] == '1')]
    db_df.reset_index(inplace=True, drop=True)
    # The calibration ranges from 350 to 900 nm
    wl_min, wl_max = 350, 900

    for i, row in db_df.iterrows():
        folder = row['Folder']
        path_to_echelle = os.path.join('./data/Echelle_data', folder, row['File'])
        file_tag = os.path.splitext(row['File'])[0]
        output_path = os.path.join('./data/brightness_data', row['Folder'])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        wl_i, brightness_i, brightness_i_err = baselined_spectrum(
            path_to_echelle=path_to_echelle, background_df=db_background_df,
            wl_min=wl_min, wl_max=wl_max
        )

        output_df = pd.DataFrame(data={
            'Wavelength (nm)': wl_i,
            'Brightness (photons/cm^2/s/nm)': brightness_i,
            'Brightness error (photons/cm^2/s/nm)': brightness_i_err,
        })
        output_df.to_csv(path_or_buf=os.path.join(output_path, os.path.splitext(row['File'])[0] + '.csv'), index=False)





if __name__ == '__main__':
    main()




