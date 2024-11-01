import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json
from scipy.optimize import least_squares, OptimizeResult
from scipy.interpolate import interp1d
import data_processing.confidence as cf
from scipy.integrate import simpson
from data_processing.utils import lighten_color, latex_float_with_error, latex_float
from scipy.signal import savgol_filter
from data_processing.echelle import load_echelle_file

echelle_spectrum = r'./data/brightness_data_fitspy/echelle_20241031/MechelleSpect_006.csv'

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
    return factor * radiance_at_temperature(temperature=temperature, wavelength_nm=wavelength_nm)


def res_bb(b, x, y):
    return model_bb(wavelength_nm=x, b=b) - y


def main():
    global echelle_spectrum
    # spectrum_df, params = load_echelle_file('./data/Echelle_data/echelle_20241031/MechelleSpect_012.asc')
    spectrum_df = pd.read_csv(
        echelle_spectrum, comment='#',
    ).apply(pd.to_numeric)
    spectrum_df = spectrum_df[spectrum_df['Wavelength (nm)']>=400.]
    brightness = spectrum_df['Brightness (photons/cm^2/s/nm)'].values


    wl = spectrum_df['Wavelength (nm)'].values
    radiance = brightness * 1240. / wl * 1.6019E-19 / 4. / np.pi
    # wl = spectrum_df['wl (nm)'].values
    # radiance = spectrum_df['counts'].values

    # radiance = savgol_filter(
    #     radiance,
    #     window_length=53,
    #     polyorder=3
    # )

    n = len(wl)
    print(f'wl.min: {wl.min():.0f}, wl.max(): {wl.max():.0f}')

    radiated_power = simpson(y=radiance, x=wl)

    radiance_interp = interp1d(x=wl, y=radiance)

    all_tol = float(np.finfo(np.float64).eps)
    b0 = np.array([3425., 1E-3])
    ls_res = least_squares(
        res_bb,
        b0,
        loss='cauchy', f_scale=0.1,
        # loss='soft_l1', f_scale=0.1,
        args=(wl, radiance),
        bounds=([all_tol, all_tol], [np.inf, np.inf]),
        xtol=all_tol,  # ** 0.5,
        ftol=all_tol,  # ** 0.5,
        gtol=all_tol,  # ** 0.5,
        # diff_step=all_tol,
        max_nfev=10000 * n,
        # x_scale='jac',
        verbose=2
    )

    popt = ls_res.x
    ci = cf.confidence_interval(ls_res)
    popt_err = np.abs(ci[:, 1] - popt)
    print(f"I0: {popt[1]:.3E} ± {popt_err[1]:.3E}, 95% CI: [{ci[1, 0]:.5E}, {ci[1, 1]:.5E}]")
    nn = wl.max() - 200.
    x_pred = np.linspace(wl.min(), 1000, num=2000)
    y_pred, delta = cf.prediction_intervals(
        model=model_bb, x_pred=x_pred, ls_res=ls_res
    )

    # msk_extrapolated = x_pred <= 350.
    # extrapolated_df = pd.DataFrame(data={
    #     'Wavelength (nm)': x_pred[msk_extrapolated],
    #     'Radiance (W/cm^2/ster/nm)': y_pred[msk_extrapolated],
    #     'Radiance error (W/cm^2/ster/nm)': delta[msk_extrapolated],
    # })

    # with open(r'./data/PALabsphere_2014_extrapolated.txt', 'w') as f:
    #     f.write(f'# T_fit {popt[0]:.0f} -/+ {popt_err[0]:.0f} K\n')
    #     f.write(f'# I_fit {popt[1]:.3E} -/+ {popt_err[1]:.3E} \n')
    #     extrapolated_df.to_csv(f, index=False, line_terminator='\n')

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3.)

    ax.plot(wl, radiance, label='Data')
    ax.plot(x_pred, y_pred, color='tab:red', label='Fit')
    ax.plot(x_pred, radiance_at_temperature(3400, x_pred, 7.2E-6), color='tab:purple', label='3400 K')
    # ax.fill_between(x_pred, y_pred - delta, y_pred + delta, color='tab:red', alpha=0.25)

    ax.set_xlabel(r'$\lambda$ {\sffamily (nm)}', usetex=True)
    ax.set_ylabel(r'Radiance (W/cm$^{\mathregular{2}}$/ster/nm)', usetex=False)

    # (W/cm^{2}/ster/nm
    ax.legend(loc='upper right')
    radiated_power_txt = f"$P = {latex_float(radiated_power, significant_digits=3)}~ \mathrm{{(W/cm^2/ster)}}$"
    ax.text(
        0.03, 0.95, radiated_power_txt,
        transform=ax.transAxes,
        ha='left', va='top', fontsize=11, color='tab:blue', usetex=True
    )

    results_txt = f'$T = {popt[0]:.0f}\pm{popt_err[0]:.0f}~\mathrm{{K}}$\n'
    results_txt += rf'$\xi = {latex_float_with_error(value=popt[1], error=popt_err[1], digits=0, digits_err=0)}$'

    ax.text(
        0.03, 0.85, results_txt, transform=ax.transAxes, va='top', ha='left', c='r',

    )
    ax.ticklabel_format(axis='y', useMathText=True)
    ax.set_title(os.path.basename(echelle_spectrum))
    # ax.set_xlim(350, wl.max())
    ax.set_ylim(0, y_pred.max())
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

    fig.savefig(path_to_output + '.png', dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
