import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from scipy.optimize import least_squares, OptimizeResult

cd_bd_excel = r'./data/cd_bd_db.xlsx'
echelle_db = r'./data/echelle_db.xlsx'
folder_map_xls = r'./PISCES-A_folder_mapping.xlsx'  # Folder name to plot label database


def load_folder_mapping():
    global folder_map_xls
    df = pd.read_excel(folder_map_xls, sheet_name=0)
    mapping = {}
    for i, row in df.iterrows():
        mapping[row['Echelle folder']] = row['Data label']
    return mapping


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}')


def model_poly(x, b) -> np.ndarray:
    n = len(b)
    r = np.zeros(len(x))
    for i in range(n):
        r += b[i] * x ** i
    return r


def res_poly(b, x, y, w=1.):
    return (model_poly(x, b) - y) * w


def jac_poly(b, x, y, w=1):
    n = len(b)
    r = np.zeros((len(x), n))
    for i in range(n):
        r[:, i] = w * x ** i
    return r


def main():
    global cd_bd_excel, echelle_db
    peaks_df: pd.DataFrame = pd.read_excel(cd_bd_excel, sheet_name=0)
    params_df: pd.DataFrame = pd.read_excel(echelle_db, sheet_name=0)
    peaks_df['Fit_file'] = peaks_df['File']
    peaks_df['File'] = [f.replace('_stats.txt', '.asc') for f in peaks_df['File']]

    # Get the elapased time since the first spectrum for each spectrum in the folder
    params_df['Timestamp'] = params_df['Timestamp'].apply(pd.to_datetime)
    params_df['Elapsed time (s)'] = (params_df['Timestamp'] - params_df[
        'Timestamp'].min()).dt.total_seconds()  # Arbitrary value for now, different t0 for every folder
    unique_folders = params_df['Folder'].unique()
    for folder in unique_folders:
        row_indexes = params_df['Folder'] == folder
        ts = params_df.loc[row_indexes, 'Timestamp'].reset_index(drop=True)
        params_df.loc[row_indexes, 'Elapsed time (s)'] = (params_df.loc[row_indexes, 'Timestamp'] - ts[0]).dt.seconds

    # Filter only the peaks with 20 accumulations
    params_df = params_df[params_df['Number of Accumulations'] >= 20]
    params_df = params_df[params_df['Is dark'] == 0]
    params_df = params_df[~(params_df['Label'] == 'Labsphere')].reset_index(drop=True)
    unique_folders = params_df['Folder'].unique()
    # We only care about the 'Elapsed time (s)', Folder, and File, drop all other columns
    params_df = params_df[['Folder', 'File', 'Elapsed time (s)']]
    peaks_df = pd.merge(peaks_df, params_df, how='left', on=['Folder', 'File']).reset_index(drop=True)

    folder_mapping = load_folder_mapping()
    sample_labels = [folder_mapping[folder] for folder in peaks_df['Folder']]

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(3.75, 3.75)

    all_tol = float(np.finfo(np.float64).eps)

    colors = [f'C{i}' for i in range(len(unique_folders))]
    for i, folder in enumerate(unique_folders):
        lbl = folder_mapping[folder]
        sample_df = peaks_df[peaks_df['Folder'] == folder].reset_index(drop=True)
        x = sample_df['CD_area'].values
        y = sample_df['BD_area'].values
        xerr = sample_df['CD_area_delta'].values
        yerr = sample_df['BD_area_delta'].values

        wyerr = np.stack([xerr, yerr]).T
        print(sample_df)

        w = np.linalg.norm(wyerr, axis=1)
        w = np.power(w, -1.)

        ls_res = least_squares(
            res_poly,
            x0=[1., 1.],
            args=(x, y, w),
            # loss='soft_l1', f_scale=0.1,
            jac=jac_poly,
            xtol=all_tol,
            ftol=all_tol,
            gtol=all_tol,
            verbose=2,
            # x_scale='jac',
            max_nfev=10000 * len(x)
        )

        popt = ls_res.x
        xpred = np.linspace(x.min(), x.max(), 500)
        ax.plot(
            xpred, model_poly(xpred, popt), color=colors[i], ls='--', lw=1.25
        )
        markers_cb, caps_cb, bars_cb = ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            capsize=2.75, mew=1.25, marker='o', ms=8, elinewidth=1.25,
            color=colors[i], fillstyle='none',
            ls='none',
            label=lbl,
        )

        [bar.set_alpha(0.25) for bar in bars_cb]
        [cap.set_alpha(0.25) for cap in caps_cb]

    ax.set_xlabel('C-D peak (photons/cm$^{\mathregular{2}}$/s)')
    ax.set_ylabel('B-D peak (photons/cm$^{\mathregular{2}}$/s)')

    mf = ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(mf)
    ax.yaxis.set_major_formatter(mf)
    ax.ticklabel_format(useMathText=True)
    ax.legend(loc='upper right', frameon=True, fontsize=10, ncols=1)
    ax.set_xlim(0, 5E11)
    ax.set_ylim(0, 5E11)
    ax.set_aspect('equal')
    fig.savefig(r'./figures/CD_vs_BD.png', dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
