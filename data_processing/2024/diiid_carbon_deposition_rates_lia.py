import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import platform
import json

"""
1 Torr-L = 133.322 Pa*1 L = 0.133322 J
1 g of C / (12.011 g / mol) * 8.314462 J / mol / K = 0.6922372825 J / K = 5.192220957 Torr-L/K
1 g of C * T[K] = K = 5.192220957 Torr-L * T 

Deposition cone d=d_0(h_0^2/h^2)cos^n(q)

V = 2*pi*d_0*h_0^2/n

m_c = rho_c * V = rho * 2 * pi * d_0 * h_0^2 / n

"""

c_molecular_weight = 12.011
rho_c = 2.2  # g /cm^3
r_constant = 8.314462
temperature = 300.  # K
n_cos, dn_cos = 7., 2.
h_0 = 10.5 * 2.54  # cm
sample_diameter = 1.

max_idx = 18
beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707


def map_laser_power_settings():
    df: pd.DataFrame = pd.read_csv('data/laser_power_mapping.csv').apply(pd.to_numeric)
    print(df)
    power_setting = df['Laser power setting (%)'].values
    power = df['Laser power (W)'].values
    mapping = {power_setting[i]: power[i] for i in df.index}
    return mapping

def nmps2tlpspm2(deposit_rate):
    global h_0, n_cos, temperature, c_molecular_weight, r_constant, rho_c, sample_diameter
    return 5.192220957 * temperature * 1E-7 * 1E4 * 8. * rho_c * (h_0 ** 2.) * deposit_rate / n_cos / (
                sample_diameter ** 2.)  # Torr-L/s/m^2


def tlpspm2nmps(rate):
    global h_0, n_cos, temperature, c_molecular_weight, r_constant, rho_c, sample_diameter
    f = 5.192220957 * temperature * 1E-7 * 1E4 * 8. * rho_c * (h_0 ** 2.) / n_cos / (sample_diameter ** 2.)
    return 1E3 * rate / f  # Torr-L/s/m^2


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


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)


def main():
    global beam_radius
    df: pd.DataFrame = pd.read_excel('data/transmission_measurements_carbon.xlsx', sheet_name='measurements')
    columns = df.columns
    num_cols = list(set(columns) - set(['Sample ID', 'Filler']))
    df[num_cols] = df[num_cols].apply(pd.to_numeric)

    df = df[df['in_report'] == 1]
    df.reset_index(inplace=True, drop=True)

    x = df.index
    r_d = df['Deposition rate (nm/s)'].values
    r_d_error = df['Deposition rate error (nm/s)'].values

    r_d_lb = r_d - r_d_error
    r_d_ub = r_d + r_d_error

    sample_area = 0.25 * np.pi * sample_diameter ** 2.
    r_o = nmps2tlpspm2(deposit_rate=r_d)
    r_o_error = nmps2tlpspm2(deposit_rate=r_d_error)
    r_o_lb = nmps2tlpspm2(deposit_rate=r_d_lb)
    r_o_ub = nmps2tlpspm2(deposit_rate=r_d_ub)

    log_ro = np.log(r_o)
    log_ro_ub = np.log(r_o_ub)
    log_delta = log_ro_ub - log_ro
    log_ro_lb = log_ro - log_delta
    exp_log_ro_lb = np.exp(log_ro_lb)
    min_delta = np.zeros_like(r_o)

    load_plot_style()
    fig_all, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig_all.set_size_inches(6.0, 6.)

    secax = ax.secondary_yaxis('right', functions=(tlpspm2nmps, nmps2tlpspm2))
    secax.set_ylabel(r'nm/s')

    labels = []

    for i, row in df.iterrows():
        labels.append(
            row['Sample ID'] + f" ROW{row['Laser test ID']}" + f"  {row['Filler']}")

    # ld, ud = r_o - exp_log_ro_lb, np.max(np.stack([r_o_ub - r_o, min_delta]).T, axis=1)
    ld, ud = r_o - r_o_lb, r_o_ub - r_o
    # print(np.stack([ld, ud]).T)

    ax.errorbar(
        x, r_o * 1E-3, yerr=(ld * 1E-3, ud * 1E-3), # yerr=r_o_error*1E-3,
        marker='o', color='C0',
        ms=9, mew=1.25, mfc='none', ls='none',
        capsize=2.75, elinewidth=1.25, lw=1.5,
    )

    # ax.set_yscale('log')
    ax.set_xticks(x, labels)
    # Rotating X-axis labels
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel(r'$\mathregular{\times}$10$^{\mathregular{3}}$ Torr-L/s/m$^{\mathregular{2}}$')
    ax.set_ylim(0, 300)

    fig_all.savefig('outgassing_rate_20240606.png', dpi=600)

    laser_power_mapping = map_laser_power_settings()
    # Estimate the laser power for each row in the dataframe
    lp = np.array([laser_power_mapping[int(x)] for x in df['Laser power setting (%)'].values])
    # Estimate the area of the sample
    sd = df['Sample diameter (cm)'].values
    sa = 0.25 * np.pi * np.power(sd, 2.)
    av = gaussian_beam_aperture_factor(beam_radius, 0.5 * sd)
    heat_load = lp * av / sa / 100.

    out_df = pd.DataFrame(data={
        'Sample ID': df['Sample ID'].values,
        'Laser test ID': df['Laser test ID'].values,
        'Filler': df['Filler'].values,
        'Heat load (MW/m^2)': heat_load,
        'Recession rate (cm/s)': df['Recession rate (cm/s)'].values,
        'Recession rate error (cm/s)': df['Recession rate error (cm/s)'].values,
        'Outgassing rate (Torr-L/s/m^2': r_o,
        'Outgassing rate error (Torr-L/s/m^2': r_o_error,
    })

    out_df.to_csv('data/outgassing_rates20240606.csv', index=False)

    plt.show()


if __name__ == '__main__':
    main()
