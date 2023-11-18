import pandas as pd
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import os
import json
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf
from data_processing.utils import lighten_color, latex_float_with_error
import cv2
from scipy.stats.distributions import t

camera_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\BFS-U3-16S2M_QE.csv'
filter_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\86117_Transmission.csv'
labsphere_csv = r'../../ir_thermography/PISCES labsphere.csv'
save_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\CALIBRATION_20231010'

TEMP_CAL = 3310. # K
TEMP_CAL_ERR = 10. #K
I0 = 5.0E-04
I0_CI = np.array([5.172E-04, 5.379E-04])

TRANSMISSION_WINDOW = 0.912
TRANSMISSION_SLIDE = 0.934

TRANMISSION_ND2 = 10. ** -2.


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def radiance_at_temperature(temperature: float, wavelength_nm: float, A=1.) -> float:
    hc = 6.62607015 * 2.99792458  # x 1E -34 (J s) x 1E8 (m/s) = 1E-26 (J m)
    hc2 = hc * 2.99792458  # x 1E -34 (J s) x 1E16 (m/s)^2 = 1E-18 (J m^2 s^{-1})
    factor = 2. * 1E14 * hc2 * np.power(wavelength_nm, -5.0)  # W / cm^2 / nm
    arg = 1E6 * hc / wavelength_nm / 1.380649 / temperature  # 26 (J m) / 1E-9 m / 1E-23 J/K
    return A * factor / (np.exp(arg) - 1.)


def get_phi_s(optical_factor_df: pd.DataFrame, temperature: float, i0=1.):
    wl = optical_factor_df['Wavelength [nm]'].values
    tsp_qe = optical_factor_df['TSPxQE'].values
    b = radiance_at_temperature(temperature=temperature, wavelength_nm=wl)
    return 2.* np.pi * i0 * simps(y=b * tsp_qe, x=wl)


TRANSMISSION_FACTOR = 1. / (TRANSMISSION_SLIDE * TRANSMISSION_WINDOW * TRANMISSION_ND2)
def get_calibrated_g_at_temperature(
        temperature: float, g_temp_cal: float, t_cal: float, t_m: float, phi_s_temp_cal: float,
        optical_factor_df: pd.DataFrame,
        i0: float
):
    phi_s_temp_m = get_phi_s(optical_factor_df=optical_factor_df, temperature=temperature, i0=i0)
    g =  TRANSMISSION_FACTOR * (t_m / t_cal) * (phi_s_temp_m / phi_s_temp_cal) * g_temp_cal
    return g


def model_bb(wavelength_nm: np.ndarray, b):
    temperature, factor = b[0], b[1]
    return factor * radiance_at_temperature(temperature=temperature, wavelength_nm=wavelength_nm)


def fobj(b, x, y):
    return model_bb(wavelength_nm=x, b=b) - y


def main():
    load_plot_style()
    qe_df = pd.read_csv(camera_csv, comment='#').apply(pd.to_numeric)
    filter_df = pd.read_csv(filter_csv, comment='#').apply(pd.to_numeric)
    qe_df.sort_values(by=['Wavelength (nm)'], inplace=True)
    filter_df.sort_values(by=['Wavelength (nm)'], inplace=True)
    wl_qe = qe_df['Wavelength (nm)'].values
    wl_trans = filter_df['Wavelength (nm)'].values
    qe = qe_df['Quantum efficiency (%)'].values / 100.
    transmission = filter_df['Transmission (%)'].values / 100.
    labsphere_df = pd.read_csv(labsphere_csv, comment='#').apply(pd.to_numeric)
    wl_ls = labsphere_df['Wavelength [nm]'].values
    sr_ls = labsphere_df['Spectral radiance [W/(sr cm^2 nm)]'].values

    # interpolate the quantum efficiency spectrum in steps of 0.125 nm
    dl = 0.125
    # N = int((1100. - 300.) / dl) + 1
    N = int((2500. - 300.) / dl) + 1
    f1 = interp1d(wl_qe, qe, kind='linear', bounds_error=False, fill_value=0.0)
    f2 = interp1d(
        wl_trans, transmission, kind='linear', bounds_error=False, fill_value=0.0  # transmission[0]
    )
    wl_qe_interp = dl * np.arange(0, N) + 300.
    qe_interp = f1(wl_qe_interp)
    trans_interp = f2(wl_qe_interp)

    tsp_qe = qe_interp * trans_interp
    system_factor_df = pd.DataFrame(data={
        'Wavelength [nm]': wl_qe_interp,
        'QE [%]': qe_interp,
        'Transmission [%]': trans_interp,
        'TSPxQE': tsp_qe
    })

    all_tol = np.finfo(np.float64).eps
    b0 = np.array([2900., 1E-5])
    res = least_squares(
        fobj,
        b0,
        # loss='cauchy', f_scale=0.001,
        loss='soft_l1', f_scale=0.1,
        # jac=jac_poly,
        args=(wl_ls, sr_ls),
        bounds=([0., 0.], [np.inf, np.inf]),
        xtol=all_tol,  # ** 0.5,
        ftol=all_tol,  # ** 0.5,
        gtol=all_tol,  # ** 0.5,
        max_nfev=10000 * N,
        x_scale='jac',
        verbose=2
    )

    """
    Measure the stats for the capture images
    """
    img_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\CALIBRATION_20231010'
    img_files_df = pd.DataFrame(data={
        'File': [
            'PISCES_LABSPHERE_501_us_5dB_6.147A_8902_ft-L.tiff',
            'PISCES_LABSPHERE_1002_us_5dB_6.147A_8902_ft-L.tiff'
        ],
        'Gain [dB]': [5., 5.],
        'Exposure time [us]': [501., 1002.]
    })

    gray_scale_stats = np.empty(
        len(img_files_df), dtype=np.dtype([
            ('Exposure [us]', 'd'), ('Mean gray', 'd'), ('Gray std', 'd'), ('Points', 'd'),
            ('T-value', 'd'), ('95% Error', 'd')
        ])
    )
    confidence = 0.95
    alpha = 1.0 - confidence

    for i, r in img_files_df.iterrows():
        fn = r['File']
        exposure = r['Exposure time [us]']
        image = cv2.imread(os.path.join(img_dir, fn), cv2.IMREAD_GRAYSCALE)
        width = image.shape[1]
        height = image.shape[0]
        number_of_pixels = width * height
        img_array = np.array(image)
        gray_mean = np.mean(image)
        gray_std = np.std(img_array, ddof=1)
        tval = t.ppf(1.0 - alpha / 2.0, number_of_pixels - 1)
        error = gray_std * tval / np.sqrt(number_of_pixels)
        # print(f'{exposure:>5.0f} us, {gray_mean:>4.1f}, {error:>4.1f}')
        gray_scale_stats[i] = (exposure, gray_mean, gray_std, number_of_pixels, tval, error)

    gray_scale_df = pd.DataFrame(data=gray_scale_stats)
    print(gray_scale_df)

    p1, p2 = gray_scale_stats[0], gray_scale_stats[1]
    g1, g2 = p1['Mean gray'], p2['Mean gray']
    a1, a2 = p1['Exposure [us]'], p2['Exposure [us]']
    e1, e2 = p1['95% Error'], p2['95% Error']

    slope = (g2 - g1) / (a2 - a1)
    intercept = g1 - slope * a1
    slope_err = slope * np.linalg.norm([e1 / (g2 - g1), e2 / (g2 - g1)])
    intercept_err = np.linalg.norm([e1, a1 * slope_err])

    g_at_4us = 4. * slope + intercept
    g_at_4us_err = np.linalg.norm([4. * slope_err, intercept_err])

    print(f'Slope:{slope:.3E}, intercept: {intercept_err:.3E}')
    print(f'ADC(4 us): {g_at_4us:.3f}±{g_at_4us_err:.4f}')

    """
    Use the calibration to create a table for temperatures in the range between 1000 and 4000 K
    """
    # Get the calibrated radiated power per unit area as perceived by the NIR camer with the short pass filter
    phi_s_cal = get_phi_s(optical_factor_df=system_factor_df, temperature=TEMP_CAL, i0=I0)
    print(f'Phi_s({TEMP_CAL:.0f} K) = {phi_s_cal:.3E}')
    temperature_calibration = 1000. + np.arange(0, 3010, 10.)
    exposure_times = np.array([4., 5., 20., 50, 100., 500])
    n_temps = len(temperature_calibration)
    table_rows = n_temps * len(exposure_times)
    error_pct = TEMP_CAL_ERR / TEMP_CAL
    calibration_table = np.empty(shape=table_rows, dtype=np.dtype([
        ('Exposure time [us]', 'd'), ('Gray value', 'd'), ('Temperature [K]', 'd'), ('Temperature error [K]', 'd')
    ]))

    k = 0
    for i in range(table_rows):
        e = exposure_times[k]
        temp = temperature_calibration[i%n_temps]
        g = get_calibrated_g_at_temperature(
            temperature=temp,
            g_temp_cal=g1,
            t_cal=501.,
            t_m=e,
            phi_s_temp_cal=phi_s_cal,
            optical_factor_df=system_factor_df,
            i0=I0
        )
        print(f'Exposure: {e:>5.3f} us, T: {temp:>4.0f} K, g: {g:>6.3f}')
        calibration_table[i] = (e, g, temp, temp*error_pct)
        if (i+1) % (n_temps) == 0:
            k += 1

    calibration_table_df = pd.DataFrame(data=calibration_table)
    print(calibration_table_df)
    calibration_table_df.to_csv(os.path.join(save_path, 'calibration_table_20231010.csv'), index=False)

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.0)

    ax.plot(wl_ls, sr_ls, color='C0', label='OL-455', zorder=4)
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel(r'W/(sr cm$^{\mathregular{2}}$ nm)')
    ax.set_title('Spectral Radiance OL-455')

    popt = res.x
    pcov = cf.get_pcov(res)
    ci = cf.confint(n=N, pars=popt, pcov=pcov, confidence=0.95)
    popt_err = np.array([max(abs(ci[i, :] - popt[i])) for i in range(len(popt))])

    ypred, lpb, upb = cf.predint(x=wl_ls, xd=wl_ls, yd=sr_ls, func=model_bb, res=res)

    ax.plot(wl_ls, ypred, color='r', label='Black body', zorder=5)
    ax.fill_between(wl_ls, lpb, upb, color=lighten_color('r', 0.25), zorder=1)

    mf = ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(mf)

    ax.ticklabel_format(axis='y', useMathText=True)
    ax.set_xlim(wl_ls.min(), wl_ls.max())
    ax.set_ylim(-0.2E-4, 1.2E-4)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2E-4))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1E-4))

    results_txt = f'T = {popt[0]:.0f}±{popt_err[0]:.0f} K\n'
    results_txt += f'I$_{{\mathregular{{0}}}}$ = $\mathregular{{{latex_float_with_error(value=popt[1], error=popt_err[1], digits=0)}}}$'

    for i in range(len(popt)):
        print(f'popt[{i}]: {popt[i]:.3E}, 95% CI: ({ci[i, 0]:.3E}, {ci[i, 1]:.3E})')

    ax.text(
        0.25, 0.05, results_txt, transform=ax.transAxes, va='bottom', ha='left', c='r'
    )

    ax.legend(loc='upper right', frameon=True, fontsize=11)

    """
    Plot the calibration for 4 us
    """
    fig_cal, ax_cal = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig_cal.set_size_inches(4.5, 3.0)

    calibration_4us = calibration_table_df[calibration_table_df['Exposure time [us]'] == 4.]
    g_4us_f = calibration_4us['Gray value'].values
    temp_4us_f = calibration_4us['Temperature [K]'].values

    ff = interp1d(g_4us_f, temp_4us_f, fill_value='extrapolate')

    calibration_4us_int = np.empty(256, dtype=np.dtype([
        ('Gray value', 'i8'), ('Temperature [K]', 'd'), ('Temperature error [K]', 'd')
    ]))

    confidence = 0.95
    alpha = 1.0 - confidence
    for i in range(256):
        temp_i = ff(float(i))
        calibration_4us_int[i] = (i, temp_i, temp_i * error_pct)

    calibration_4us_df = pd.DataFrame(data=calibration_4us_int)
    print(calibration_4us_df)
    calibration_4us_df.to_csv(os.path.join(save_path, 'calibration_20231010_4us.csv'), index=False)


    x_plot = calibration_4us_int['Gray value']
    y_plot = calibration_4us_int['Temperature [K]']
    y_delta = calibration_4us_int['Temperature error [K]']
    lpb1, upb1 = y_plot - y_delta, y_plot + y_delta
    ax_cal.fill_between(x_plot, lpb1, upb1, color=lighten_color('C3', 0.25))
    ax_cal.plot(x_plot, y_plot, color='C3')

    ax_cal.set_xlabel('Gray value')
    ax_cal.set_ylabel('Temperature [K]')
    ax_cal.set_title(r'Calibration 4 $\mathregular{\mu}$s, 5dB')

    ax_cal.set_xlim(0, 255)
    ax_cal.set_ylim(1600, 4000)
    ax_cal.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax_cal.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax_cal.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax_cal.yaxis.set_minor_locator(ticker.MultipleLocator(100))

    fig.savefig(os.path.join(save_path, 'PISCES_LAB_spectrum_20231010.png'), dpi=600)
    fig_cal.savefig(os.path.join(save_path, 'calibration_figure_20231010.png'), dpi=600)

    plt.show()
    # integrated_bb = integrate_blackbody(optical_factor_df=system_factor_df, temperature=2900)


if __name__ == '__main__':
    main()
