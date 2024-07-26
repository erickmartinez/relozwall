import numpy as np
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from data_processing.utils import get_experiment_params
from scipy.stats.distributions import t
import json
import re
import pandas as pd

excel_file = 'low_power_recession_rate_amb_2024.xlsx'
path_to_laser_data = './data/laser_tests'
beam_radius = 0.5 * 0.8165  # * 1.5 # 0.707

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def gaussian_beam_aperture_factor(r_beam, r_sample):
    return 1.0 - np.exp(-2.0 * (r_sample / r_beam) ** 2.0)


def get_mean_power(path_to_laser_test_csv: str) -> tuple:
    df: pd.DataFrame = pd.read_csv(path_to_laser_test_csv, comment='#').apply(pd.to_numeric)
    peak_power = df['Laser output peak power (W)'].values
    peak_power_max = peak_power.max()
    msk_on = peak_power > peak_power_max * 0.5
    peak_power = peak_power[msk_on]
    n = peak_power.size
    mean = peak_power.mean()
    std = peak_power.std(ddof=1)
    confidence_level = 0.95
    alpha = 1. - confidence_level
    tval = t.ppf(1. - 0.5 * alpha, n - 1)
    se = tval * std / np.sqrt(n)
    return mean, std


def main():
    global excel_file, path_to_laser_data, beam_radius
    recession_df: pd.DataFrame = pd.read_excel(excel_file, sheet_name=0)
    columns = recession_df.columns
    recession_df[columns[1::]] = recession_df[columns[1::]].apply(pd.to_numeric)
    recession_df = recession_df[recession_df['Quality'] > 0].reset_index(drop=True)
    n_rows = len(recession_df)
    e_laser = np.zeros(n_rows, dtype=np.float64)
    e_laser_err = np.zeros(n_rows, dtype=np.float64)
    # Construct a mapping from the laser test id to the corresponding csv file
    data_file_list = os.listdir(path_to_laser_data)
    # Match ROWXXX in the file name
    p = re.compile(r'.*?ROW(\d+).*?')
    data_file_map = {}
    for i, fn in enumerate(data_file_list):
        m = p.match(fn)
        row_id = int(m.group(1))
        data_file_map[row_id] = fn

    keys = list(data_file_map.keys())
    keys.sort()
    sorted_data_file_map = {i: data_file_map[i] for i in keys}
    data_file_map = sorted_data_file_map
    del sorted_data_file_map

    for i, row in recession_df.iterrows():
        test_id = row['TEST ID']
        csv = data_file_map[test_id]
        path_to_csv = os.path.join(path_to_laser_data, csv)
        e_laser[i], e_laser_err[i] = get_mean_power(path_to_laser_test_csv=path_to_csv)
        # print(test_id, round(e_laser[i]), round(e_laser_err[i] * 10.) / 10.)

    recession_df['Laser power (W)'] = np.round(e_laser)
    recession_df['Laser power err (W)'] = np.round(e_laser_err * 10.) / 10.
    sample_diameter_cm = recession_df['Sample diameter (cm)'].values
    sample_area_cm = 0.25 * np.pi * sample_diameter_cm ** 2.
    af = gaussian_beam_aperture_factor(r_beam=beam_radius, r_sample=sample_diameter_cm*0.5)
    # output before correcting for BN reflectivity
    heat_load = e_laser * af / sample_area_cm / 100.
    heat_load_err = e_laser_err * af / sample_area_cm / 100.
    # correction factor with respect to graphite
    graphite_abs = 0.55
    coated_graphite_abs = 0.5
    correction_abs = coated_graphite_abs / graphite_abs
    heat_load *= correction_abs
    recession_df['Heat load (MW/m^2)'] = heat_load
    recession_df['Heat load err (MW/m^2)'] = heat_load_err

    # Plot different power settings
    power_settings = recession_df['Power percent setting (%)'].unique()

    load_plot_style()
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 5.)

    markers = ['o', 's', '^']
    colors = ['C0', 'C1', 'C2']

    for i, ps in enumerate(power_settings):
        recession_at_ps_df = recession_df[recession_df['Power percent setting (%)'] == ps]
        n_i = len(recession_at_ps_df)
        t_val = t.ppf(1. - 0.5*0.05, n_i - 1)
        heat_load = recession_at_ps_df['Heat load (MW/m^2)'].values
        heat_load_err = recession_at_ps_df['Heat load err (MW/m^2)'].values
        heat_load_mean = recession_at_ps_df['Heat load (MW/m^2)'].mean()
        heat_load_std = recession_at_ps_df['Heat load (MW/m^2)'].std(ddof=1)
        heat_load_se = heat_load_std * t_val / np.sqrt(n_i) + np.linalg.norm(heat_load_err) / n_i
        lbl = rf'{round(heat_load_mean):.0f} MW/m$^{{\mathregular{{2}}}}$'
        # the recession rate
        time_s = recession_at_ps_df['Irradiation time (s)'].values
        energy = heat_load * sample_area_cm.mean() * 100. * 0.95 * time_s #recession_at_ps_df['Laser power (W)'].values * time_s
        energy_err = heat_load_se * sample_area_cm.mean() * 100. * 0.95 * time_s #recession_at_ps_df['Laser power err (W)'].values * time_s
        nu = recession_at_ps_df['Recession rate (cm/s)'].values
        nu_err = recession_at_ps_df['Recession rate error (cm/s)'].values
        nu_mean = nu.mean()
        nu_std = nu.std(ddof=1)
        nu_se = nu_std * t_val / np.sqrt(n_i) + np.linalg.norm(nu_err) / n_i
        ebc = mpl.colors.to_rgba(colors[i], 0.25)
        axes[0].errorbar(
            time_s, nu*10, yerr=nu_err*10,
            marker=markers[i], mfc='none', capsize=2.5,
            ls='none', lw=1.,
            ms=9, mew=1.25, ecolor=ebc,
            elinewidth=1.0, c=colors[i],
            label=lbl
        )

        axes[i].set_ylim(-0.01, 0.1)

        mean_txt = fr'$\langle \nu \rangle = {nu_mean*10.:.2f} \pm {nu_se*10.:.2f}~'
        mean_txt += r'\mathrm{mm/s}$'
        axes[1].text(
            2., (nu_mean)*10.,
            mean_txt,
            # transform=axes[0].transAxes,
            ha='left', va='bottom', fontsize=11, color=colors[i]
        )

        axes[1].axhline(
            y=nu_mean*10, ls=':', lw=1.25, c=colors[i]
        )

        axes[1].errorbar(
            energy*1E-3, nu*10, yerr=nu_err*10, xerr=1E-3*energy_err,
            marker=markers[i], mfc='none', capsize=2.5,
            ls='none', lw=1.,
            ms=9, mew=1.25, ecolor=ebc,
            elinewidth=1.0, c=colors[i],
            label=lbl
        )

    axes[0].set_xlabel('Heating time (s)')
    axes[1].set_xlabel('Laser energy (kJ)')
    axes[0].set_ylabel(r'$\nu$ (mm/s)')
    axes[1].set_ylabel(r'$\nu$ (mm/s)')

    axes[0].legend(
        loc='upper left', frameon=True, fontsize=9
    )

    axes[0].set_xlim(0, 18)
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))

    axes[1].set_xlim(0, 8)
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))


    fig.savefig('bpr_approach_to_equilibrium.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
