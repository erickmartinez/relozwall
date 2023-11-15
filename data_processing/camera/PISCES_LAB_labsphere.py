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

camera_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\BFS-U3-16S2M_QE.csv'
filter_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\86117_Transmission.csv'
labsphere_csv = r'../../ir_thermography/PISCES labsphere.csv'


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

def radiance_at_temperature(temperature: float, wavelength_nm: float) -> float:
    hc = 6.62607015 * 2.99792458  # x 1E -34 (J s) x 1E8 (m/s) = 1E-26 (J m)
    hc2 = hc * 2.99792458  # x 1E -34 (J s) x 1E16 (m/s)^2 = 1E-18 (J m^2 s^{-1})
    factor = 2. * 1E14 * hc2 * np.power(wavelength_nm, -5.0)  # W / cm^2 / nm
    arg = 1E6 * hc / wavelength_nm / 1.380649 / temperature  # 26 (J m) / 1E-9 m / 1E-23 J/K
    return factor / (np.exp(arg) - 1.)


def integrate_blackbody(optical_factor_df: pd.DataFrame, temperature: float):
    wl = optical_factor_df['Wavelength [nm]'].values
    b = radiance_at_temperature(temperature=temperature, wavelength_nm=wl)
    return simps(y=b, x=wl)


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

    output_radiance_integrated = simps(y=sr_ls, x=wl_ls)
    print(f'Labsphere output radiance: {output_radiance_integrated:.3E} W/sr/cm^2 x 2pi = {output_radiance_integrated*2.*np.pi:.3E} W/cm^2')

    # interpolate the quantum efficiency spectrum in steps of 0.125 nm
    dl = 0.125
    # N = int((1100. - 300.) / dl) + 1
    N = int((2500. - 300.) / dl) + 1
    f1 = interp1d(wl_qe, qe, kind='linear', bounds_error=False)
    f2 = interp1d(
        wl_trans, transmission, kind='linear', bounds_error=False, fill_value=0.0  # transmission[0]
    )
    wl_qe_interp = dl * np.arange(0, N) + 300.
    qe_interp = f1(wl_qe_interp)
    trans_interp = f2(wl_qe_interp)

    system_factor = qe_interp * trans_interp
    system_factor_df = pd.DataFrame(data={
        'Wavelength [nm]': wl_qe_interp,
        'QE [%]': qe_interp,
        'Transmission [%]': trans_interp,
        'Factor': system_factor
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

    results_txt = f'T = {popt[0]:.0f}±{popt_err[0]:.0f} K\n'
    results_txt += f'I$_{{\\mathregular{{0}}}}$ = $\mathregular{{{latex_float_with_error(value=popt[1], error=popt_err[1], digits=0)}}}$'

    for i in range(len(popt)):
        print(f'popt[{i}]: {popt[i]:.3E}, 95% CI: ({ci[i,0]:.3E}, {ci[i,1]:.3E})')

    ax.text(
        0.25, 0.05, results_txt, transform=ax.transAxes, va='bottom', ha='left', c='r'
    )

    ax.legend(loc='upper right', frameon=True, fontsize=11)

    fig.savefig('PISCES_LAB_spectrum_20231010.png', dpi=600)

    plt.show()
    # integrated_bb = integrate_blackbody(optical_factor_df=system_factor_df, temperature=2900)


if __name__ == '__main__':
    main()