import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_processing.misc_utils.plot_style import load_plot_style
from pathlib import Path
from scipy.stats.distributions import t

DATA_XLS = r'./data/Laser heating tests 2025.xlsx'
SAMPLE_ID = 'R5N16'
SAMPLE_DIAMETER = 1 # cm
BEAM_RADIUS = 0.5 * 0.8165  # * 1.5 # 0.707

def load_data(data_xls, sample_id):
    df = pd.read_excel(data_xls, sheet_name=0)
    df = df[(df['Sample ID'] == sample_id) & (df['Quality'] == 1)]
    return df

def load_laser_power_mapping():
    df = pd.read_csv(r'./data/laser_power_mapping.csv').apply(pd.to_numeric, errors='coerce')
    df.sort_values(by=['Laser power setting (%)'], ascending=True, inplace=True)
    # df.set_index(keys=['Laser power setting (%)'], inplace=True)

    mapping = {}
    for index, row in df.iterrows():
        mapping[int(row['Laser power setting (%)'])] = row['Laser power (W)']

    return mapping

def mean_error(x):
    return np.linalg.norm(x)

def standard_error(x):
    confidence = 0.95
    alpha = 1 - confidence
    n = len(x)
    n_t = max(n-1, 2)
    return np.std(x) / np.sqrt(n_t) * t.ppf(1 - alpha/2, n_t)


def gaussian_beam_aperture_factor(beam_radius, sample_radius):
    return 1.0 - np.exp(-2.0 * (sample_radius / beam_radius) ** 2.0)

def main(data_xls, sample_id, sample_diameter, beam_radius):
    df = load_data(data_xls, sample_id)
    laser_power_setting = df['Power percent setting (%)'].values
    power_mapping = load_laser_power_mapping()
    laser_power = np.array([power_mapping[int(ps)] for ps in laser_power_setting])
    # for i in range(len(laser_power)):
    #     print(f'Power setting: {laser_power_setting[i]}%: {laser_power[i]:.1f}')
    sample_radius = 0.5 * sample_diameter
    sample_area = 0.25 * np.pi * sample_diameter * sample_diameter
    aperture_factor = gaussian_beam_aperture_factor(beam_radius, sample_radius)
    heat_load_mw_m2 = aperture_factor * laser_power / sample_area / 100.0
    df['Heat load (MW/m^2)'] = heat_load_mw_m2


    agg_df = df.groupby('Power percent setting (%)').agg({
        'Heat load (MW/m^2)': ['mean'],
        'Recession rate (cm/s)': ['mean', standard_error],
        'Recession rate error (cm/s)': [mean_error]
    })


    heat_load_mean = agg_df['Heat load (MW/m^2)']['mean'].values
    recession_rate = agg_df['Recession rate (cm/s)']['mean'].values
    recession_se = agg_df['Recession rate (cm/s)']['standard_error'].values
    recession_rate_error = agg_df['Recession rate error (cm/s)']['mean_error'].values
    # print(f'Heat load (mW/m^2): {heat_load_mw_m2}')
    # print(f'Recession rate: {recession_rate}')
    # print(f'Recession error: {recession_rate_error}')
    # for i in range(len(recession_rate)):
    #     print(f'Recession rate: {recession_rate[i]:.2f} -/+ {recession_rate_error[i]:.2f}')

    recession_rate_error_vector = np.stack([recession_se, recession_rate_error]).T
    print(f'recession_rate_error_vector.shape: {recession_rate_error_vector.shape}')
    recession_rate_total_error = np.linalg.norm(recession_rate_error_vector, axis=1)

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3)

    markers_p, caps_p, bars_p = ax.errorbar(
        heat_load_mean, recession_rate*10, yerr=recession_rate_total_error*10,
        marker='o', ms=9, mew=1.25, mfc='none',  # label=f'{lbl}',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='C0', ls='none'
    )

    [bar.set_alpha(0.35) for bar in bars_p]
    [cap.set_alpha(0.35) for cap in caps_p]

    ax.set_xlabel(r'{\sffamily Heat load (MW/m\textsuperscript{2})', usetex=True)
    ax.set_ylabel('Recession rate (mm/s)')
    ax.set_ylim(0, 1)

    path_to_figures = Path('./figures')
    path_to_figures.mkdir(parents=True, exist_ok=True)
    path_to_figure = path_to_figures / 'laser_recession_rate.png'

    fig.savefig(path_to_figure, dpi=600, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    main(data_xls=DATA_XLS, sample_id=SAMPLE_ID, sample_diameter=SAMPLE_DIAMETER, beam_radius=BEAM_RADIUS)
