import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import least_squares, OptimizeResult
from scipy.integrate import simpson
from scipy.interpolate import interp1d

brightness_folder = r'./data/brightness_data_fitspy_wl-calibrated'
FOLDER_MAP_XLS = r'./PISCES-A_folder_mapping.xlsx'  # Folder name to plot label database
echelle_xlsx = r'./data/echelle_db.xlsx'

calibration_line = {'center_wl': 433.93, 'label': r'D$_{\gamma}$'}


def load_folder_mapping(folder_map_xls):
    global FOLDER_MAP_XLS
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


over_sqrt_pi = 1. / np.sqrt(2. * np.pi)


def gaussian(x, c, sigma, mu):
    global over_sqrt_pi
    p = c * over_sqrt_pi / sigma
    arg = 0.5 * np.power((x - mu) / sigma, 2)
    return p * np.exp(-arg)


def sum_gaussians(x, b):
    global over_sqrt_pi
    m = len(x)
    nn = len(b)
    # Assume that the new b contains a list ordered like
    # (c1, sigma_1, mu1, c_2, sigma_2, mu_2, ..., c_n, sigma_n, mu_n)
    selector = np.arange(0, nn) % 3
    msk_c = selector == 0
    msk_sigma = selector == 1
    msk_mu = selector == 2
    cs = b[msk_c]
    sigmas = b[msk_sigma]
    mus = b[msk_mu]
    n = len(mus)
    u = over_sqrt_pi * np.power(sigmas, -1) * cs
    u = u.reshape((1, len(u)))
    v = np.zeros((n, m), dtype=np.float64)
    for i in range(len(sigmas)):
        arg = 0.5 * np.power((x - mus[i]) / sigmas[i], 2.)
        v[i, :] = np.exp(-arg)
    res = np.dot(u, v)
    return res.flatten()


def res_sum_gauss(b, x, y):
    return sum_gaussians(x, b) - y


def jac_sum_gauss(b, x, y):
    m, nn = len(x), len(b)
    # Assume that the new b contains a list ordered like
    # (c1, sigma_1, mu1, c_2, sigma_2, mu_2, ..., c_n, sigma_n, mu_n)
    selector = np.arange(0, nn) % 3
    msk_c = selector == 0
    msk_sigma = selector == 1
    msk_mu = selector == 2
    c = b[msk_c]
    s = b[msk_sigma]
    mu = b[msk_mu]
    r = np.zeros((m, nn), dtype=np.float64)
    for i in range(len(s)):
        k = 3 * i
        g = gaussian(x, c[i], s[i], mu[i])
        r[:, k] = g / c[i]
        r[:, k + 1] = np.power(s[i], -1) * (np.power((x - mu[i]) / s[i], 2) - 1.) * g
        r[:, k + 2] = np.power(s[i], -2) * (x - mu[i]) * g

    return r


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


def main(folder_map_xls):
    global brightness_folder, calibration_line, echelle_xlsx
    params_df: pd.DataFrame = pd.read_excel(echelle_xlsx, sheet_name=0)
    # Get the elapased time since the first spectrum for each spectrum in the folder
    params_df['Timestamp'] = params_df['Timestamp'].apply(pd.to_datetime)
    params_df['Elapsed time (s)'] = (params_df['Timestamp'] - params_df[
        'Timestamp'].min()).dt.total_seconds()  # Arbitrary value for now, different t0 for every folder
    unique_folders = params_df['Folder'].unique()
    for folder in unique_folders:
        row_indexes = params_df['Folder'] == folder
        ts = params_df.loc[row_indexes, 'Timestamp'].reset_index(drop=True)
        params_df.loc[row_indexes, 'Elapsed time (s)'] = (params_df.loc[row_indexes, 'Timestamp'] - ts[0]).dt.seconds

    folder_mapping = load_folder_mapping(folder_map_xls)
    folders = os.listdir(brightness_folder)
    n = len(folders)
    load_plot_style()
    cmap_names = ["Blues", "Oranges", "Greens", "Reds", "Purples"]
    cmaps = [mpl.colormaps.get_cmap(cmapi) for cmapi in cmap_names]
    # dx = 430.68 - 431.0
    # dx = 0
    wl_range = (429.5, 433.5)
    d_alpha_range = (433.5, 434.5)
    tab10_colors = [f'C{i}' for i in range(n)]

    fig, axes = plt.subplots(nrows=n, ncols=3, constrained_layout=True, width_ratios=[1., 0.35, 0.35])
    fig.set_size_inches(6.75, 5.5)
    # fig.subplots_adjust(hspace=0, left=0.15, right=0.98, top=0.95, bottom=0.1)

    all_tol = float(np.finfo(np.float64).eps)
    cal_wl = calibration_line['center_wl']

    for i, folder in enumerate(folders):
        files = [f for f in os.listdir(os.path.join(brightness_folder, folder)) if f.endswith('.csv')]
        path_to_json_model = os.path.join(r'./data/cd_bd_fit_data', folder, 'cd_bd_model.json')
        with open(path_to_json_model, 'r') as json_file:
            fit_model = json.load(json_file)
        n_files = len(files)
        norm = mpl.colors.Normalize(vmin=-1, vmax=(n_files - 1))
        cmap = cmaps[i]
        colors = [cmap(norm(j)) for j in range(n_files)]
        colors = colors[::-1]
        alphas = [norm(j) for j in range(n_files)]
        alphas = alphas[::-1]
        sample_lbl = folder_mapping[folder]
        elapsed_time_s = np.zeros(n_files)
        d_gamma_center = np.zeros(n_files)
        for j, file in enumerate(files):
            print(fit_model[f"{j}"]["fname"])
            echelle_file = file.replace('.csv', '.asc')
            # Find the elapsed time since the first spectrum using the params_df
            elapsed_time_s[j] = params_df.loc[(params_df['Folder'] == folder) & (params_df['File'] == echelle_file), 'Elapsed time (s)']
            # elapsed_time_s[j] = row['Elapsed time (s)']
            baseline_order = int(fit_model[f"{j}"]["baseline"]["order_max"])
            baseline_points = np.array(fit_model[f"{j}"]["baseline"]["points"])
            baseline_x = baseline_points[0]
            lbl = sample_lbl if j == 0 else None
            df = pd.read_csv(os.path.join(brightness_folder, folder, file), comment='#').apply(pd.to_numeric)
            # Fit the d_gamma peak to check for spectrometer calibration shifts
            d_alpha_df = df[df['Wavelength (nm)'].between(d_alpha_range[0], d_alpha_range[1])]
            x_cal = d_alpha_df['Wavelength (nm)'].values
            y_cal = d_alpha_df['Brightness (photons/cm^2/s/nm)'].values
            area_window = simpson(y=y_cal, x=x_cal)
            res_lsq = least_squares(
                res_sum_gauss, x0=[0.5 * area_window, 0.05, cal_wl],
                args=(x_cal, y_cal),
                # loss='soft_l1', f_scale=0.1,
                jac=jac_sum_gauss,
                bounds=(
                    [0., 1E-5, cal_wl - 0.1],
                    [area_window * 5., np.inf, cal_wl + 0.1],
                ),
                xtol=all_tol,
                ftol=all_tol,
                gtol=all_tol,
                verbose=0,
                x_scale='jac',
                max_nfev=10000 * len(y_cal)
            )
            popt = res_lsq.x
            wl_dgamma = popt[2]
            d_gamma_center[j] = wl_dgamma
            dx = wl_dgamma - cal_wl

            x_pred = np.linspace(x_cal.min(), x_cal.max(), num=200)
            y_pred = sum_gaussians(x_pred, popt)
            axes[i, 1].plot(x_cal, y_cal * 1E-12, marker='o', ms=5, mfc='none', color=colors[j], alpha=alphas[j],
                            ls='none')
            axes[i, 1].plot(x_pred, y_pred * 1E-12, marker='none', color=colors[j], alpha=alphas[j], ls='-')

            df = df[df['Wavelength (nm)'].between(wl_range[0]-dx, wl_range[1]+dx)]
            wl = df['Wavelength (nm)'].values
            wl -= dx
            sr = df['Brightness (photons/cm^2/s/nm)'].values

            f_interp = interp1d(x=wl, y=sr, fill_value='extrapolate', bounds_error=False)
            baseline_y = f_interp(baseline_x-dx)
            # sr = savgol_filter(
            #     sr,
            #     window_length=5,
            #     polyorder=3
            # )
            # Get the baseline
            # baseline_res = least_squares(
            #     res_poly,
            #     x0=np.ones(baseline_order) * 0.5,
            #     args=(baseline_x, baseline_y),
            #     # loss='soft_l1', f_scale=0.1,
            #     jac=jac_poly,
            #     xtol=all_tol,
            #     ftol=all_tol,
            #     gtol=all_tol,
            #     verbose=0,
            #     # x_scale='jac',
            #     max_nfev=10000 * len(baseline_x)
            # )

            # popt_baseline = baseline_res.x
            # sr -= model_poly(wl, popt_baseline)
            sr -= sr.min()
            axes[i, 0].plot(wl, sr * 1E-12, color=colors[j], label=lbl, lw=1., alpha=alphas[j])
            # axes[i, 0].plot(wl, model_poly(wl, popt_baseline)*1E-12, color='grey', lw=1., alpha=alphas[j])
        axes[i, 0].set_ylim(0, 4.)
        axes[i, 0].axvline(x=430.613, ls='--', color='grey', lw=1.)
        axes[i, 0].axvline(x=432.6, ls='--', color='grey', lw=1.)
        xlim = [round((ri - dx) * 10) / 10 for ri in wl_range]
        axes[i, 0].set_xlim(xlim)
        axes[i, 1].set_xlim(d_alpha_range)
        axes[i, 1].set_ylim(-5, 90)
        axes[i, 0].xaxis.set_major_locator(ticker.MultipleLocator(1))
        axes[i, 0].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        # axes[i].yaxis.set_major_locator(ticker.MultipleLocator(2))
        # axes[i].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        axes[i, 0].legend(loc='upper right', fontsize=10, frameon=True)

        axes[i, 2].plot(elapsed_time_s/60, d_gamma_center-cal_wl, c=colors[0], marker='o', ls='none', mfc='none', ms=5)
        axes[i, 2].set_ylabel(r'$\lambda_{\mathrm{p}} - \mathrm{D}_{\gamma}$ {\sffamily (nm)}', usetex=True)


    axes[0, 0].set_title('CD and BD peaks')
    axes[0, 1].set_title(r'D$_{\mathregular{\gamma}}$')
    axes[0, 2].set_title(r'D$_{\mathregular{\gamma}}$ shift')
    fig.supylabel('Spectral radiance (W/cm$^{\mathregular{2}}$/ster/nm) x10$^{\mathregular{12}}$')
    for ax in axes[-1, 0:2].flatten():
        ax.set_xlabel('$\lambda$ {\sffamily (nm)}', usetex=True)
    axes[0, 0].text(
        430.75, 2.5, r'C-D (430.61 nm)', ha='left', va='bottom', fontsize=9
    )

    axes[1, 0].text(
        432.5, 2., r'B-D (432.6 nm)', ha='right', va='bottom', fontsize=9
    )

    axes[-1, 2].set_xlabel(r'Time (min)')
    fig.savefig('./figures/c-d_band_plot.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    main(FOLDER_MAP_XLS)
