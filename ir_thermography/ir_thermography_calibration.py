import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from matplotlib.ticker import ScalarFormatter
from scipy import integrate

import numpy as np
from scipy import constants

window_transmission_at_900 = 0.912
slide_transmission_at_900 = 0.934

calibration_temperature = 2900 # K
calibration_wavelength = 900.0 # nm
labsphere_brightness_at_900 = 5.16E-5 # W/s/cm^2/ster/nm


def spectral_radiance(wavelength_nm, temperature: float):
    """
    Estimates the blackbody radiation as a function of wavelength and temperature
    .. math::
        B_{\nu}(T) = \frac{2 h c^2 / \lambda^5}{e^{h c/\lambda kT} - 1}
    where :math:`\nu = c/\lambda`

    Parameters
    ----------
    wavelength_nm: float
        The wavelength in nanometers
    temperature: float
        The temperature in Kelvin

    Returns
    -------
    float:
        The spectral radiance in W / (cm^2 nm sr)
    """

    hc = 6.62607015 * 2.99792458  # x 1E -34 (J s) x 1E8 (m/s) = 1E-26 (J m)
    hc2 = hc * 2.99792458  # x 1E -34 (J s) x 1E16 (m/s)^2 = 1E-18 (J m^2 s^{-1})
    hc_by_kt_lambda = 1E6 * hc / 1.380649 / temperature / wavelength_nm
    radiance = 2.0 * hc2 * np.power(wavelength_nm, -5.0)  # 1E-18 (J m^2 s^{-1}) / 1E-45 (m^5) = 1E27 J / (m^3 s)
    radiance *= 1E14 / (np.exp(hc_by_kt_lambda) - 1.0)  # 1E27 J/m^3 s = 1E18 J / (m^2 nm s) = 1E14 J / cm^2 nm
    return radiance

def latex_float(f):
    float_str = f"{f:5.4g}".lower()
    if "e" in float_str:
        base, exponent = float_str.split("e")
        # return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
        return f"{base} \\times 10^{{{int(exponent):2d}}}"
    else:
        return float_str

def temperature_at_radiance(radiance: float, wavelength_nm: float):
    """
    Estimates the blackbody temperature in Kelvin for a source at a given wavelength in nm and a known spectral radiance
    ..math::
        T(B, \lambda) = \frac{h c}{\lambda k} \left\{ \ln\left[\frac{hc^2}{\lambda^5 B} + 1\right]\right\}^{-1}

    Parameters
    ----------
    radiance: float
        The spectral radiance in W / cm^2 / nm / s / sr
    wavelength_nm: float
        The wavelength in nm


    Returns
    -------
    float:
        The estimated temperature in K
    """
    hc = 6.62607015 * 2.99792458  # x 1E -34 (J s) x 1E8 (m/s) = 1E-26 (J m)
    hc2 = hc * 2.99792458  # x 1E -34 (J s) x 1E16 (m/s)^2 = 1E-18 (J m^2 s^{-1})
    arg = 2.0 * hc2 * np.power(wavelength_nm, -5.0)  # 1E18 J / (m^2 nm s)
    arg = 1.0E14 * arg /radiance  #  1E14 [J / (cm^2 nm s)] / [J / cm^2 nm s] = 1
    arg += 1.0
    temperature = 1E6 * hc / wavelength_nm / 1.380649  # 1E-26 (J m) / 1E-9 m / 1E-23 J/K
    temperature = temperature / np.log(arg)
    return temperature


if __name__ == "__main__":
    calibration_df = pd.read_csv('https://raw.githubusercontent.com/erickmartinez/relozwall/main/ir_thermography/photodiode_brightness_calibration_202202.csv')
    calibration_df[:] = calibration_df[:].apply(pd.to_numeric)
    print(calibration_df.columns)
    brightness = calibration_df['Labsphere Brightness at 900 nm (W/ster/cm^2/nm)'].values #/ window_transmission_at_900 / slide_transmission_at_900
    wavelength_nm = 900.0 * np.ones_like(brightness)

    calibration_brightness_900 = spectral_radiance(900.0, calibration_temperature)
    print(f"Expected Brightness at {calibration_wavelength:.1f} nm and {calibration_temperature:.0f} K: {calibration_brightness_900:5.3E} W/cm^2/ster/s/nm")

    # calibration_df['Temperature at radiance (K)'] = temperature_at_radiance(
    #     brightness,
    #     wavelength_nm
    # )

    labsphere_aperture = calibration_brightness_900 / labsphere_brightness_at_900
    print(f"Labsphere aperture: {labsphere_aperture:6.3E}")
    # calibration_df['Calibration Factor (V/W/ster/cm^2/nm) at 900 nm'] = calibration_df['Signal out (V)'] / calibration_df['Labsphere Brightness at 900 nm (W/ster/cm^2/nm)']
    calibration_df['Aperture Factor'] = brightness / calibration_brightness_900

    calibration_df['Calibration Factor (W/ster/cm^2/nm/V) at 900 nm and 2900 K'] = brightness / calibration_df['Signal out (V)']

    print(calibration_df)

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(4.5, 3.0)

    color_brightness = 'C0'
    color_temperature = 'C1'

    ax1.set_yscale('log')

    ax1.tick_params(axis='y', labelcolor=color_brightness)

    ax1.plot(
        calibration_df['Photodiode Gain (dB)'].values,
        brightness, color=color_brightness,
        # ls='none',
        marker='o', fillstyle='none'
    )

    ax1.set_xlabel('Photodiode Gain (dB)')
    ax1.set_ylabel('$B_{\lambda = 900\; \mathregular{nm}}$ (W/ster/cm$^{\mathregular{2}}$/nm)', color=color_brightness)

    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor=color_temperature)
    # ax2.set_yscale('log')

    ax2.plot(
        calibration_df['Photodiode Gain (dB)'].values,
        temperature_at_radiance(brightness, calibration_wavelength),
        color=color_temperature,
        # ls='none',
        marker='s', fillstyle='none'
    )

    ax2.set_ylabel('Temperature (K)', color=color_temperature)

    fig1.tight_layout()

    fig2, ax1 = plt.subplots()
    fig2.set_size_inches(4.75, 3.5)

    color_cf = 'C2'
    color_cb = 'C3'

    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color_cf)
    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color_cb)

    ax1.plot(
        calibration_df['Photodiode Gain (dB)'].values,
        calibration_df['Aperture Factor'],
        color=color_cf,
        marker='o', fillstyle='none'
    )

    ax2.plot(
        calibration_df['Photodiode Gain (dB)'].values,
        calibration_df['Calibration Factor (W/ster/cm^2/nm/V) at 900 nm and 2900 K'],
        color=color_cb,
        marker='o', fillstyle='none'
    )

    ax1.set_xlabel('Photodiode Gain (dB)')
    ax1.set_ylabel(
        'Aperture Factor',
        color=color_cf
    )
    ax2.set_ylabel(
        'C$_{\lambda = 900\;\mathregular{nm}, T = 2900\;\mathregular{K}}$ (W/ster/cm$^{\mathregular{2}}$/nm/V)',
        color=color_cb
    )

    fig2.tight_layout()

    fig3, ax1 = plt.subplots()
    fig3.set_size_inches(4.75, 3.5)

    cf = calibration_df['Signal out (V)'].values / calibration_df['Labsphere Brightness at 900 nm (W/ster/cm^2/nm)'].values
    ax1.set_yscale('log')
    ax1.plot(
        calibration_df['Photodiode Gain (dB)'].values,
        cf,
        color=color_cf,
        marker='o', fillstyle='none'
    )
    ax1.tick_params(axis='y', labelright=True, which='both', right=True)
    ax1.set_xlabel('Photodiode Gain (dB)')
    ax1.set_ylabel(
        'CF$_{\lambda = 900\;\mathregular{nm}}$ (V/W/ster/cm$^{\mathregular{2}}$/nm)',
    )

    fig3.tight_layout()

    fig1.savefig('gain_vs_brightness.png', dpi=600)
    fig2.savefig('gain_vs_cf.png', dpi=600)
    fig3.savefig('gain_vs_cf_eh.png', dpi=600)

    calibration_df.to_csv('pd_brightness_processed.csv', index=False, float_format='%8.4g')

    calibration_df = pd.read_csv(
        'https://raw.githubusercontent.com/erickmartinez/relozwall/main/ir_thermography/optronics_OL_455-12-2_SN_96203007_calibration_table.csv'
    )

    calibration_wl = calibration_df['Wavelength (nm)'].values
    calibration_radiance = calibration_df['Spectral Radiance (W/(sr cm^2 nm))'].values
    wl_sim = np.linspace(calibration_wl.min(), calibration_wl.max(), 500)
    B = spectral_radiance(wl_sim, calibration_temperature)

    radiated_power_bb = integrate.simps(B, wl_sim)
    radiated_power_cal = integrate.simps(calibration_radiance, calibration_wl)

    aperture_factor = radiated_power_bb / radiated_power_cal

    fig4, ax1 = plt.subplots()
    fig4.set_size_inches(4.75, 3.5)

    ax1.plot(
        calibration_wl / 1000.0, calibration_radiance * labsphere_aperture, color='C0',
        label=rf'${latex_float(labsphere_aperture)} \times $ LabSphere'
    )

    ax1.plot(
        wl_sim / 1000.0, B, color='C1', ls='--', lw=1.25,
        label=f'Planck'
    )

    # ax2 = ax1.twinx()

    ax1.set_xlabel('Wavelength ($\mathregular{\mu}$m)')
    ax1.set_ylabel('$B_{\lambda}$ (W /sr cm$^2$ nm)')#, color='C0')
    # ax2.set_ylabel('$B_{\lambda}$ (Lab sphere) (W /sr cm$^2$ nm)', color='C1')
    ax1.ticklabel_format(useMathText=True, axis='y')
    # ax2.ticklabel_format(useMathText=True, axis='y')

    # ax2.plot(
    #     calibration_wl / 1000.0, calibration_radiance , color='C1',
    #     label=f'LabSphere'
    # )
    # ax2.legend(loc='lower right', frameon=False)

    ymin, ymax = ax1.get_ylim()
    idx_sim = (np.abs(wl_sim - calibration_wavelength)).argmin()
    ax1.set_ylim(ymin, ymax)
    ax1.legend(loc='upper left', frameon=False)

    wl_sim_ref = wl_sim[idx_sim] / 1000.0
    B_sim_ref = B[idx_sim]

    x1 = calibration_wavelength / 1000.0
    y1 = calibration_brightness_900
    xdisplay, ydisplay = ax1.transData.transform_point((x1, y1))

    offset = 50
    connectionstyle = "angle3,angleA=0,angleB=90"
    bbox = dict(boxstyle="round", fc="0.8")
    arrowprops = dict(
        arrowstyle="->", color="0.5",
        shrinkA=5, shrinkB=5,
        patchA=None, patchB=None,
        connectionstyle=connectionstyle
    )
    ax1.annotate(
        f"$P_{{\mathrm{{cal}}}} = {latex_float(radiated_power_cal)}$\n$P_{{\mathrm{{Planck}}}} = {latex_float(radiated_power_bb)}$",
        xy=(x1, y1), xycoords='data',  # 'figure pixels', #data',
        xytext=(offset, -2 * offset), textcoords='offset points',  # 'data',
        arrowprops=arrowprops,
        bbox=bbox,
        ha='left'
    )

    fig4.tight_layout()

    plt.show()
