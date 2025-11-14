import pandas as pd
from scipy.stats.distributions import t
from tkinter.filedialog import askdirectory
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from data_processing.misc_utils.plot_style import load_plot_style
from pathlib import Path

BASE_FOLDER = r'D:\ARPA-E\data\lock-in\2025\R5N16-0903'

PATTERN_CLEAN = re.compile(r'((.*)?_CLEAN\.txt)')
PATTERN_COATED = re.compile(r'((.*)?_X(\d+)_(\d+)MM_COATED\.txt)')
PATTERN_AIR = re.compile(r'((.*)?_AIR\.txt)')
PATTERN_BGD = re.compile(r'((.*)?_BGND\.txt)')

# Morita&Yamamoto
# https://iopscience.iop.org/article/10.1143/JJAP.14.825
# Optical and Electrical Properties of Boron
# Nobuyoshi Morita and Akira Yamamoto 1975 Jpn. J. Appl. Phys. 14 825
# doi: 10.1143/JJAP.14.825"
ABSORPTION_COEFFICIENT = 3.36E4 # 1/cm
ABSORPTION_COEFFICIENT_UNCERTAINTY = 1.19E4


def mean_err(x) -> float:
    x = np.array(x)
    n = len(x)
    return np.linalg.norm(x) / n

def standard_error(x):
    n = len(x)
    if n == 1:
        return np.inf
    std = np.std(x, ddof=1)
    confidence = 0.95
    alpha = 1 - confidence
    tval = t.ppf(1 - alpha/2, n-1)
    return std * tval / np.sqrt(n)

def main(
    pattern_clean, pattern_coated, pattern_air, pattern_bgd, folder=None, absorption_coefficient=ABSORPTION_COEFFICIENT,
    absorption_coefficient_delta=ABSORPTION_COEFFICIENT_UNCERTAINTY
):
    if folder is None:
        folder = askdirectory()
    file_list = [fn for fn in os.listdir(folder) if fn.endswith('.txt')]
    confidence = 0.95
    alpha = 1. - confidence

    out_df = pd.DataFrame(data={'FILE': [], 'SIGNAL': [], 'N': [], 'R_MEAN (V)': [], 'R_STD (V)': [], 'T.INV': [], 'R_SE (V)': [], 'X (mm)': []})
    print(os.path.basename(folder))
    print('       FILE                   N   R_MEAN      R_STD    T.INV       R_SE  X (MM)')

    for fn in file_list:
        m_clean = pattern_clean.match(fn)
        m_coated = pattern_coated.match(fn)
        m_bgnd = pattern_bgd.match(fn)
        m_air = pattern_air.match(fn)
        x = 0.0

        if m_clean:
            signal_type = 'CLEAN'

        if m_coated:
            signal_type = 'COATED'
            x = int(m_coated.group(4))

        if not m_bgnd is None:
            signal_type = 'BGND'

        if m_air:
            signal_type = 'AIR'

        df = pd.read_csv(os.path.join(folder, fn), delimiter='\t').apply(pd.to_numeric)
        r_val = df['R[V]'].values
        n = len(r_val)
        tval = t.ppf(1. - alpha / 2., n - 1)
        r_mean = r_val.mean()
        r_std = np.std(r_val, ddof=1)
        r_se = r_std * tval / np.sqrt(n)
        data = {
            'FILE': [fn],
            'SIGNAL': [signal_type],
            'N': [n],
            'R_MEAN (V)': [r_mean],
            'R_STD (V)': [r_std],
            'T.INV': [tval],
            'R_SE (V)': [r_se],
            'X (mm)': [x]
        }
        file = os.path.splitext(fn)[0]
        row = pd.DataFrame(data=data)
        out_df = pd.concat([out_df, row]).reset_index(drop=True)
        print(f'{file[-15:]:>10s} {signal_type:>7s} {n:>8d} {r_mean:>8.6f} {r_std:>6.4E} {tval:>8.6f} {r_se:>6.4E} {x:>6.0f}')


    file_tag = os.path.basename(folder) + '_averages.csv'
    clean_df = out_df[out_df['SIGNAL'] == 'CLEAN']
    r_mean_clean = clean_df['R_MEAN (V)'].values
    r_n_clean = clean_df['N'].values
    r_se_clean = clean_df['R_SE (V)'].values
    r_mean_clean_repetitions = r_mean_clean.mean()
    r_se_clean_repetitions = standard_error(r_mean_clean)
    r_se_mean = mean_err(r_se_clean)
    total_error = np.linalg.norm([r_se_clean_repetitions, r_se_mean])

    aggregated_df =   out_df[~(out_df['SIGNAL'] == 'CLEAN')]
    clean_row = pd.DataFrame(data={
        'FILE': ['AGGREGATED'],
        'SIGNAL': ['CLEAN'],
        'N': [r_n_clean.sum()],
        'R_MEAN (V)': [r_mean_clean_repetitions],
        'R_STD (V)': [r_mean_clean.std(ddof=1)],
        'T.INV': [t.ppf(1. - alpha / 2., r_n_clean.sum() - 1)],
        'R_SE (V)': [total_error],
        'X (mm)': [out_df['X (mm)'].max()]
    })
    aggregated_df = pd.concat([aggregated_df, clean_row],ignore_index=True).reset_index(drop=True)


    print(aggregated_df)
    out_df.to_csv(os.path.join(folder, file_tag), index=False)
    aggregated_df.to_csv(os.path.join(folder, os.path.basename(folder) + '_aggregated.csv'), index=False)


    coated_df = aggregated_df[aggregated_df['SIGNAL'] == 'COATED']
    x = coated_df['X (mm)'].values - coated_df['X (mm)'].min()

    mask_bgnd = aggregated_df['SIGNAL'] == 'BGND'
    # print(aggregated_df.loc[mask_bgnd, 'R_MEAN (V)'])
    r_bgnd, r_bgnd_error = aggregated_df.loc[mask_bgnd, 'R_MEAN (V)'].values[0], aggregated_df.loc[mask_bgnd, 'R_SE (V)'].values[0]

    intensity_coated = coated_df['R_MEAN (V)'].values - r_bgnd
    intensity_glass = clean_df['R_MEAN (V)'].mean() - r_bgnd

    intensity_coated_error = np.linalg.norm(np.column_stack([coated_df['R_SE (V)'].values, np.full(len(coated_df), fill_value=r_bgnd_error)]), axis=1)
    intensity_glass_error = np.linalg.norm([clean_df['R_SE (V)'].mean(), r_bgnd_error])

    transmission = intensity_coated / intensity_glass
    absorbance = -np.log10(transmission)
    absorbance_error = np.linalg.norm(
        np.column_stack([
            intensity_coated_error / intensity_coated,
            np.full(len(intensity_coated), fill_value=intensity_glass_error / intensity_glass)
        ]), axis=1
    ) / np.log(10)

    thickness_nm = absorbance * 1E7 / absorption_coefficient * np.log(10)
    thickness_nm_error = np.abs(thickness_nm) * np.linalg.norm(
        np.column_stack([absorbance_error / absorbance, np.full(len(absorbance), fill_value=absorption_coefficient_delta / absorption_coefficient)]), axis=1
    )

    load_plot_style()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, constrained_layout=True, sharex=True)
    fig.set_size_inches(4.5, 4)

    markers_p, caps_p, bars_p = ax1.errorbar(
        x, coated_df['R_MEAN (V)'].values, xerr=np.full(len(coated_df), fill_value=0.5), yerr=coated_df['R_SE (V)'].values,
        marker='o', ms=9, mew=1.25, mfc='none',  # label=f'{lbl}',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='C0', ls='none'
    )

    [bar.set_alpha(0.5) for bar in bars_p]
    [cap.set_alpha(0.5) for cap in caps_p]

    markers_p, caps_p, bars_p = ax2.errorbar(
        x, thickness_nm, xerr=np.full(len(coated_df), fill_value=1),
        yerr=thickness_nm_error,
        marker='s', ms=9, mew=1.25, mfc='none',  # label=f'{lbl}',
        capsize=2.75, elinewidth=1.25, lw=1.5, c='C1', ls='none'
    )

    [bar.set_alpha(0.5) for bar in bars_p]
    [cap.set_alpha(0.5) for cap in caps_p]

    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('Mean R (V)')
    ax1.set_xlim(0, 50)

    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('Thickness (nm)')

    main_path = Path(folder)
    path_to_figures = main_path / 'figures'
    path_to_figures.mkdir(parents=True, exist_ok=True)

    path_to_thickness_data = main_path / 'thickness_data'
    path_to_thickness_data.mkdir(parents=True, exist_ok=True)

    thickness_df = pd.DataFrame(data={
        'x (mm)': x, 'Thickness (nm)': thickness_nm, 'Thickness error (nm)': thickness_nm_error
    })

    file_tag = main_path.name
    path_to_thickness_csv = path_to_thickness_data / f'{file_tag}_thickness_data.csv'
    with open(str(path_to_thickness_csv), mode='w') as thickness_file:
        thickness_file.write('#'*60 + '\n')
        thickness_file.write(f'# Experiment ID: {file_tag}' + '\n')
        thickness_file.write(f'# Absorption coefficient: {absorption_coefficient:.3E} -/+ {absorption_coefficient_delta:.4E} (1/cm)' + '\n')
        thickness_file.write('#'*60 + '\n')
        thickness_df.to_csv(thickness_file, index=False)

    path_to_plot = path_to_figures / f'{file_tag}.png'
    fig.savefig(path_to_plot, dpi=600)

    plt.show()

if __name__ == '__main__':
    main(folder=BASE_FOLDER, pattern_clean=PATTERN_CLEAN, pattern_coated=PATTERN_COATED, pattern_air=PATTERN_AIR, pattern_bgd=PATTERN_BGD)
