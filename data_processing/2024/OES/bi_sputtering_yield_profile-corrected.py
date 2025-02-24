import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
import os
from scipy.optimize import least_squares, OptimizeResult
from scipy.stats.distributions import t

SPUTTERING_RATES_CSV = r'./data/boron_physical_sputtering_yields.old.csv'
FOLDER_MAP_XLS = r'./PISCES-A_folder_mapping.xlsx'  # Folder name to plot label database

AXES_MAPPING = {
    'echelle_20240815': 0, 'echelle_20240827': 1, 'echelle_20241003': 1, 'echelle_20241031': 0
}

LBL_MAPPING = {
    'echelle_20240815': 'SBR (High thermal contact)',
    'echelle_20240827': 'ABPR',
    'echelle_20241003': 'PBPR',
    'echelle_20241031': 'SBR (Low thermal contact)'
}

COLOR_MAPPING = {
    'echelle_20240815': 'C0', 'echelle_20240827': 'C1', 'echelle_20241003': 'C2', 'echelle_20241031': 'tab:red'
}

MARKER_MAPPING = {
    'echelle_20240815': 's', 'echelle_20240827': 'o', 'echelle_20241003': 'D', 'echelle_20241031': '^'
}

TRIMSP_DATA_DF = pd.DataFrame(data={
    'ion': ['D+', 'D2+', 'D3+'],
    'ion_composition': [0.41, 0.22, 0.37],
    'sputtering_yield': [0.016705392, 0.005855387, 0.0],
    'sputtered_energy_eV': [1.952335105, 0.0, 0.0]
})

def estimated_trim_weighted_sputtering_yield(trimsp_df: pd.DataFrame):
    sputtering_yield = trimsp_df['sputtering_yield'].values
    ion_composition = trimsp_df['ion_composition'].values
    yield_mean = np.dot(sputtering_yield, ion_composition)
    yield_squared_mean = np.dot(sputtering_yield*sputtering_yield, ion_composition)

    # Estimate the t-val for a confidence level 0f 95%
    alpha = 1 - 0.95
    n = len(trimsp_df)
    tval = t.ppf(1. - alpha / 2, n - 1)
    yield_std = np.sqrt(np.abs(yield_squared_mean - yield_mean * yield_mean)) #* np.sqrt(n / (n - 1))
    yield_se = yield_std * tval / np.sqrt(n)
    return yield_mean, yield_std


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


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}')


def load_folder_mapping():
    global FOLDER_MAP_XLS
    df = pd.read_excel(FOLDER_MAP_XLS, sheet_name=0)
    mapping = {}
    for i, row in df.iterrows():
        mapping[row['Echelle folder']] = row['Data label']
    return mapping


def main(sputtering_yield_file, color_mapping, axes_mapping, label_mapping, marker_mapping, trimsp_df):
    # load the fitted lorentzian peaks
    bi_df = pd.read_csv(sputtering_yield_file)
    numeric_cols = [
        'Elapsed time (s)', 'Temperature (K)', 'Gamma_B (1/cm^2/s)',
        'Gamma_B error (1/cm^2/s)',	'Sputtering yield',	'Sputtering yield error'''
    ]
    bi_df[numeric_cols] = bi_df[numeric_cols].apply(pd.to_numeric)
    bi_df['Timestamp'] = bi_df['Timestamp'].apply(pd.to_datetime)
    folder_mapping = load_folder_mapping()
    folders = bi_df['Folder'].unique()
    n_plots = len(folders)

    # Get TRIMSP simulated sputtering yields
    trimsp_sy, trimsp_sy_se = estimated_trim_weighted_sputtering_yield(trimsp_df)

    print(f'TRIM.SP sputtering yield: {trimsp_sy:.3E} -/+ {trimsp_sy_se:.4E} ({trimsp_sy-trimsp_sy_se:.3E}, {trimsp_sy+trimsp_sy_se:.3E})')

    # fig_cols = max(int(n_plots * 0.5), 1)
    # fig_rows = max(int(n_plots / fig_cols), 1)

    load_plot_style()
    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True, sharex=True)
    fig.set_size_inches(7., 5.5)



    markers = ['^', 's', 'o', 'v']
    colors = ['C0', 'C1', 'C2', 'C3']
    for i, folder in enumerate(folders):
        print(f"Processing folder: {folder}")
        ax_g = axes[axes_mapping[folder], 0]
        ax_y = axes[axes_mapping[folder], 1]
        lbl = label_mapping[folder]
        marker = marker_mapping[folder]
        idx_folder = bi_df['Folder'] == folder
        folder_df = bi_df[idx_folder].sort_values(by=['Elapsed time (s)'])
        time_s = folder_df['Elapsed time (s)'].values


        fb = folder_df['Gamma_B (1/cm^2/s)'].values
        fb_err = folder_df['Gamma_B error (1/cm^2/s)'].values
        weights = np.log(1 / (fb_err + 0.1 * np.median(fb_err)))

        sputtering_yield = folder_df['Sputtering yield'].values
        sputtering_yield_error = folder_df['Sputtering yield error'].values

        eps = float(np.finfo(np.float64).eps)
        degree = 3
        fit_result_g = least_squares(
            res_poly, x0=[0.1**i for i in range(degree)], args=(time_s, np.log(fb), weights),
            loss='linear', f_scale=1.,
            jac=jac_poly,
            xtol=eps,
            ftol=eps,
            gtol=eps,
            verbose=2,
            x_scale='jac',
            max_nfev=10000 * degree
        )

        fit_result_y = least_squares(
            res_poly, x0=[0.1 ** i for i in range(degree)], args=(time_s, sputtering_yield),
            loss='linear', f_scale=1.,
            jac=jac_poly,
            xtol=eps,
            ftol=eps,
            gtol=eps,
            verbose=2,
            x_scale='jac',
            max_nfev=10000 * degree
        )

        markers_b, caps_b, bars_b = ax_g.errorbar(
            time_s/60., fb, yerr=fb_err, capsize=2.75, mew=1.25, marker=marker, ms=8, elinewidth=1.25,
            color=color_mapping[folder], fillstyle='none',
            ls='none',# lw=1.25,
            label=lbl,
        )

        [bar.set_alpha(0.35) for bar in bars_b]

        markers_b, caps_b, bars_b = ax_y.errorbar(
            time_s / 60., sputtering_yield, yerr=sputtering_yield_error, capsize=2.75, mew=1.25, marker=marker, ms=8, elinewidth=1.25,
            color=color_mapping[folder], fillstyle='none',
            ls='none',  # lw=1.25,
            label=lbl,
        )

        [bar.set_alpha(0.35) for bar in bars_b]

        xpred = np.linspace(time_s.min(), time_s.max(), 500)
        ax_g.plot(xpred/60., np.exp(model_poly(xpred, fit_result_g.x)), color=colors[i], ls='--', lw=1)
        ax_y.plot(xpred / 60., model_poly(xpred, fit_result_y.x), color=colors[i], ls='--', lw=1)
        ax_g.set_yscale('log')
        # ax_g.legend(loc='upper right', frameon=True, fontsize=10)
        ax_g.set_ylim(1E15, 1E17)

        ax_y.set_yscale('log')
        # ax_y.legend(loc='lower left', frameon=True, fontsize=10)
        ax_y.set_ylim(1E-3, 1)

        title = 'Boron rod' if axes_mapping[folder] % 2 == 0 else 'Boron pebble rod'
        ax_g.set_title(title)
        ax_y.set_title(title)



    for i, ax in enumerate(axes.ravel()):
        ax.set_xlim(0, 90)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))


    for ax in axes[:,0]:
        ax.set_ylabel(r"$\Gamma_{\mathrm{B}}$ {\sffamily (cm\textsuperscript{-2} s\textsuperscript{-1})}", usetex=True)
        ax.axhline(y=trimsp_sy, ls='--', lw=1., color='0.75')
        # ax.axhspan(ymin=trimsp_sy-trimsp_sy_se, ymax=trimsp_sy+trimsp_sy_se, color='k')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper right')
    for ax in axes[:,1]:
        ax.set_ylabel(r'$Y_{\mathrm{B/D^+}}$', usetex=True)
        ax.axhline(y=trimsp_sy, ls='--', lw=1., color='0.5')

    print(f"TRIM weighted yield: {trimsp_sy:.3E}")

    fig.supxlabel(r'Time (min)', usetex=False)

    for i, axi in enumerate(axes.flatten()):
        panel_label = chr(ord('`') + i + 1) # starts from a
        # panel_label = chr(ord('`') + i + 3)
        axi.text(
            -0.2, 1.05, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )

    fig.savefig('./figures/bi_sputtering_yield.png', dpi=600)
    fig.savefig('./figures/bi_sputtering_yield.svg', dpi=600)
    fig.savefig('./figures/bi_sputtering_yield.pdf', dpi=600)
    plt.show()



if __name__ == '__main__':
    main(
        sputtering_yield_file=SPUTTERING_RATES_CSV, color_mapping=COLOR_MAPPING, axes_mapping=AXES_MAPPING,
        label_mapping=LBL_MAPPING, marker_mapping=MARKER_MAPPING, trimsp_df=TRIMSP_DATA_DF
    )


