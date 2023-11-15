import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import simps
import os
import json

labsphere_csv = 'PISCES labsphere.csv'

def load_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

def radiance_at_temperature(temperature: float, wavelength_nm: float) -> float:
    hc = 6.62607015 * 2.99792458  # x 1E -34 (J s) x 1E8 (m/s) = 1E-26 (J m)
    hc2 = hc * 2.99792458  # x 1E -34 (J s) x 1E16 (m/s)^2 = 1E-18 (J m^2 s^{-1})
    factor = 2. * 1E14 * hc2 * np.power(wavelength_nm, -5.0) # W / cm^2 / nm
    arg = 1E6 * hc / wavelength_nm / 1.380649 / temperature  # 26 (J m) / 1E-9 m / 1E-23 J/K
    return factor / (np.exp(arg) - 1.)

def main():
    df = pd.read_csv(labsphere_csv, comment='#').apply(pd.to_numeric)
    wl = df['Wavelength [nm]'].values
    sr = df['Spectral radiance [W/(sr cm^2 nm)]'].values
    bb = radiance_at_temperature(temperature=2900, wavelength_nm=wl)

    rad_power = simps(y=sr, x=wl) * 2. * np.pi

    load_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.0)

    ax_t = ax.twinx()

    colors = ['C0', 'C1']

    ax.plot(wl, sr, color='C0')
    ax.tick_params(axis='y', labelcolor=colors[0])
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel(r'$B_{\lambda, \mathrm{OL}}$ [W/(sr cm$^{\mathregular{2}}$ nm)]', color=colors[0])
    ax.ticklabel_format(axis='y', useMathText=True)

    ax_t.plot(wl, bb, color=colors[1])
    ax_t.tick_params(axis='y', labelcolor=colors[1])
    ax_t.set_ylabel(r'$B_{\lambda, \mathrm{BB}}$ [W/(sr cm$^{\mathregular{2}}$ nm)]', color=colors[1])

    rad_power_txt = f'{rad_power:.1f}'

    ax.text(
        0.95, 0.95, rf'$P={rad_power_txt}~\mathrm{{W/cm^2}}$',
        transform=ax.transAxes,
        fontsize=11,
        va='top', ha='right',
        color='b'
    )
    plt.show()

if __name__ == '__main__':
    main()