import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import platform
import json
from functools import partial


data_path = 'Documents/ucsd/Postdoc/research/data/DIIID carbon deposition preruns'
csv = 'Selective matrix removal - Laser test DB - atomic_masses.csv'

"""
1 Torr-L = 133.322 Pa*1 L = 0.133322 J
1 g of C / (12.011 g / mol) * 8.314462 J / mol / K = 0.6922372825 J / K = 5.192220957 Torr-L/K
1 g of C * T[K] = K = 5.192220957 Torr-L * T 

Deposition cone d=d_0(h_0^2/h^2)cos^n(q)

V = 2*pi*d_0*h_0^2/n

m_c = rho_c * V = rho * 2 * pi * d_0 * h_0^2 / n

"""

c_molecular_weight = 12.011
b_atomic_weight = 10.811
rho_c = 2.2 # g /cm^3
r_constant = 8.314462
temperature = 300. # K
n_cos, dn_cos = 7., 2.
h_0 = 10.5 * 2.54 # cm
sample_diameter = 1.
# 1 Torr-L = (133.322 N / m²) x (1000 mL * 1 cm³ / 1 mL * [1 m / 10² cm]³ ) / ( 1.38065E-23 N * m / K) / (300 K)
# 1 Torr-L = (133.322 N * m * 1E-3) / (1.38065E-23 * 300 N * m)
# 1 Torr-L = 3.218824383798247e+19 atoms
TORR_L2MOL = 5.344983639668773e-05 # moles
sample_area = 0.25 * np.pi * sample_diameter ** 2. # cm2

def nmps2tlpspm2(rate, atomic_mass, density):
    global h_0, n_cos, sample_area, TORR_L2MOL
    # Get the atomic mass deposition rate (mol/s)
    rate_moles = rate * 1E-7 * 2. * np.pi * density * np.power(h_0, 2.) / n_cos / atomic_mass
    # Convert moles to Torr-L/s
    rate_tl = rate_moles / TORR_L2MOL
    # divide by sample area
    return rate_tl / sample_area * 1E4 # 1 / cm² x (10² cm / 1 m)² = 1E4 m^{-2}

def tlpspm2nmps(rate, atomic_mass, density):
    global h_0, n_cos, c_molecular_weight, r_constant, rho_c, sample_diameter
    # Multiply by area to get Torr-L/s
    rate_tl = rate * sample_area * 1E-4 # Area of sample given in cm², must change to m²
    rate_moles = rate_tl * TORR_L2MOL
    rate_nmps = rate_moles * 1E7 * 0.5 / (np.pi * density * np.power(h_0,  2.)) * n_cos * atomic_mass
    return rate_nmps


platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'


def normalize_path(the_path):
    global platform_system, drive_path
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    else:
        the_path = the_path.replace('/', '\\')
    return os.path.join(drive_path, the_path)


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def main():
    global data_path, csv
    base_path = normalize_path(base_path)
    df = pd.read_csv(os.path.join(base_path, csv))
    columns = df.columns
    num_cols = list(set(columns) - set(['Sample ID', 'Filler', 'Atomic weight (u)', 'Film density (g/cm3)']))
    df[num_cols] = df[num_cols].apply(pd.to_numeric)

    x = df.index
    r_d = df['Deposition rate (nm/s)'].values

    r_d_lb = df['Deposition rate lb (nm/s)'].values
    r_d_ub = df['Deposition rate ub (nm/s)'].values
    atomic_masses = df['Atomic weight (u)'].values
    film_densities = df['Film density (g/cm3)'].values

    # sample_area = 0.25 * np.pi * sample_diameter ** 2.
    r_o = nmps2tlpspm2(rate=r_d, atomic_mass=atomic_masses, density=film_densities)
    r_o_lb = nmps2tlpspm2(rate=r_d_lb, atomic_mass=atomic_masses, density=film_densities)
    r_o_ub = nmps2tlpspm2(rate=r_d_ub, atomic_mass=atomic_masses, density=film_densities)

    log_ro = np.log(r_o)
    log_ro_ub = np.log(r_o_ub)
    log_delta = log_ro_ub - log_ro
    log_ro_lb = log_ro - log_delta
    exp_log_ro_lb = np.exp(log_ro_lb)

    load_plot_style()
    fig_all, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True, sharex=True)
    fig_all.set_size_inches(4.5, 7.5)

    # Create partial functions
    # def forward_partial(xx):
    #     print(xx, atomic_masses)
    #     return tlpspm2nmps(xx, atomic_mass=atomic_masses, density=film_densities)
    # def inverse_partial(xx):
    #     return nmps2tlpspm2(xx, atomic_mass=atomic_masses, density=film_densities)
    #
    # secax = ax.secondary_yaxis('right', functions=(inverse_partial, forward_partial))
    # secax.set_ylabel(r'nm/s')

    labels = []

    for i, row in df.iterrows():
        labels.append(
            row['Sample ID'] + f" HT{row['Heat treatment ID']}"
            + f" US {row['Ultrasonic treatment ID']}" + f"  {row['Filler']}")

    lb, ub = r_o - exp_log_ro_lb, r_o_ub - r_o
    axes[1].errorbar(
        x, r_o*1E-3, yerr=(lb*1E-3, ub*1E-3),
        marker='o', color='C0',
        ms=9, mew=1.25, mfc='none', ls='none',
        capsize=2.75, elinewidth=1.25, lw=1.5,
    )

    axes[0].errorbar(
        x, r_d, yerr=(r_d_lb, r_d_ub),
        marker='o', color='C0',
        ms=9, mew=1.25, mfc='none', ls='none',
        capsize=2.75, elinewidth=1.25, lw=1.5,
    )

    # ax.set_yscale('log')
    axes[1].set_xticks(x, labels)
    # Rotating X-axis labels
    axes[1].tick_params(axis='x', labelrotation=90)
    axes[1].set_ylabel(r'$\mathregular{\times}$10$^{\mathregular{3}}$ Torr-L/s/m$^{\mathregular{2}}$')
    axes[0].set_ylabel(r'nm/s')
    axes[0].set_ylim(0, 150)

    # fig_all.savefig(os.path.join(base_path, 'outgassing_rate_20240516.png'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()



