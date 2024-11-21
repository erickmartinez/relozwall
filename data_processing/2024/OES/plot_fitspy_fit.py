import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
import numpy as np
import re

path_to_fitspy_results = r'./data/fitspy_results_cd_bd/echelle_20241031/MechelleSpect_029.csv'
output_xls = r'./data/cd_bd_lorentzian.xlsx'
echelle_xlsx = r'./data/echelle_db.xlsx'
folder_map_xls = r'./PISCES-A_folder_mapping.xlsx'
output_folder = r'./figures/Echelle_plots/CD-BD'


wl_range = (429., 434.5)
fit_range = (430, 433.5)

calibration_line = {'center_wl': 434.0, 'label': r'D$_{\gamma}$'}

peaks_of_interest = [
    {'center_wl': 430.9, 'label': "C-D"},
    {'center_wl': 432.6, 'label': 'B-D (Q-branch)'}
]


def lorentzian(x, h, mu, gamma):
    return 2.*gamma*h/(np.pi)/(4*(x-mu)**2. + gamma**2.)

def res_sum_lorentzians(b, x, y):
    return sum_lorentzians(x, b) - y

def sum_lorentzians(x, b):
    m = len(x)
    n3 = len(b)
    selector = np.arange(0, n3) % 3
    h = b[selector == 0]
    mu = b[selector == 1]
    gamma = b[selector == 2]
    n = len(h)
    res = np.zeros(m)
    for i in range(n):
        res += h[i]*gamma[i] / ( ((x-mu[i])**2.) + (0.25* gamma[i] ** 2.) )
    return 0.5 * res / np.pi


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')


def get_peak_stats(line_number, path_to_stats_file):
    amount_pm_err_str = r"\s+(\d+\.?\d*[eE]?[\-\+]?\d*)\s+\+\/\-\s+(\d+\.?\d*[eE]?[\-\+]?\d*)"
    p_x0 = re.compile(rf"\s+m{line_number:02d}_x0.\s+{amount_pm_err_str}.*$")
    p_ampli = re.compile(rf"\s+m{line_number:02d}_ampli.\s+{amount_pm_err_str}.*$")
    p_gamma = re.compile(rf"\s+m{line_number:02d}_fwhm.\s+{amount_pm_err_str}.*$")
    x0_found = False
    ampli_found = False
    gamma_found = False
    output = {}
    with open(path_to_stats_file, 'r') as f:
        for line in f:
            if not x0_found:
                m = p_x0.match(line)
                if m:
                    output['x0'] = float(m.group(1))
                    output['x0_error'] = float(m.group(2))
                    ampli_found = True
            if not ampli_found:
                m = p_ampli.match(line)
                if m:
                    output['ampli'] = float(m.group(1))
                    output['ampli_error'] = float(m.group(2))
                    ampli_found = True
            if not gamma_found:
                m = p_gamma.match(line)
                if m:
                    output['fwhm'] = float(m.group(1))
                    output['fwhm_error'] = float(m.group(2))
                    ampli_found = True
            if ampli_found and gamma_found and x0_found:
                break
    area = 0.5 * output['ampli'] * np.pi * output['fwhm']
    error = area * np.linalg.norm([output['fwhm_error']/output['fwhm'], output['ampli_error']/output['ampli']])
    output['area'] = area
    output['area_error'] = error
    return output

def get_spectrum_timestamp(folder, file, echelle_df):
    # Get the elapased time since the first spectrum for each spectrum in the folder
    try:
        selected_folder_df = echelle_df[(echelle_df['Folder'] == folder) & (echelle_df['File'] == file)].reset_index(
            drop=True)
        elapsed_time = selected_folder_df['Elapsed time (s)'][0]
        timestamp = selected_folder_df['Timestamp'][0]
        print(f"Elapsed time: {elapsed_time}")
    except KeyError:
        print(f"Could not find Folder '{folder}', File '{file}' in the echelle_db")
        raise KeyError
    return elapsed_time

def load_echelle_xlsx(xlsx_file):
    echelle_df: pd.DataFrame = pd.read_excel(xlsx_file, sheet_name=0)
    echelle_df['Timestamp'] = echelle_df['Timestamp'].apply(pd.to_datetime)
    echelle_df['Elapsed time (s)'] = (echelle_df['Timestamp'] - echelle_df[
        'Timestamp'].min()).dt.total_seconds()  # Arbitrary value for now, different t0 for every folder
    unique_folders = echelle_df['Folder'].unique()
    for folder in unique_folders:
        row_indexes = echelle_df['Folder'] == folder
        ts = echelle_df.loc[row_indexes, 'Timestamp'].reset_index(drop=True)
        echelle_df.loc[row_indexes, 'Elapsed time (s)'] = (echelle_df.loc[row_indexes, 'Timestamp'] - ts[0]).dt.seconds
    return  echelle_df

cd_bd_columns = [
    'Folder', 'File',
    'Model',
    'x0_cd (nm)', 'ampli_cd (photons/cm^2/s/nm)', 'ampli_err_cd  (photons/cm^2/s/nm)',
    'fwhm_cd (nm)', 'fwhm_err_cd (nm)', 'area_cd (photons/cm^2/s)', 'area_err_cd (photons/cm^2/s)',
    'x0_bd (nm)', 'ampli_bd (photons/cm^2/s/nm)', 'ampli_err_bd  (photons/cm^2/s/nm)',
    'fwhm_bd (nm)', 'fwhm_err_bd (nm)', 'area_bd (photons/cm^2/s)', 'area_err_bd (photons/cm^2/s)',
    'Elapsed time (s)'
]

def load_output_db(xlsx_source):
    global cd_bd_columns
    try:
        out_df: pd.DataFrame = pd.read_excel(xlsx_source, sheet_name=0)
    except Exception as e:
        out_df = pd.DataFrame(data={
            col: [] for col in cd_bd_columns
        })
    return out_df

def update_out_df(db_df:pd.DataFrame, row_data):
    row = pd.DataFrame(data={key: [val] for key, val in row_data.items()})
    if len(db_df) == 0:
        return row
    folder = row_data['Folder']
    file = row_data['File']
    # Try finding the folder and file in db_df
    row_index = (db_df['Folder'] == folder) & (db_df['File'] == file)
    previous_row = db_df[row_index]
    if len(previous_row) == 0:
        return pd.concat([db_df, row], ignore_index=True).reset_index(drop=True)
    row_index = db_df.loc[row_index].index[0]
    for col, val in row_data.items():
        db_df.loc[row_index, col] = val
    return db_df

def load_folder_mapping():
    global folder_map_xls
    df = pd.read_excel(folder_map_xls, sheet_name=0)
    mapping = {}
    for i, row in df.iterrows():
        mapping[row['Echelle folder']] = row['Data label']
    return mapping

def main():
    global path_to_fitspy_results, calibration_line, peaks_of_interest, echelle_xlsx
    global cd_bd_columns, output_xls
    echelle_df = load_echelle_xlsx(echelle_xlsx)
    file = os.path.basename(path_to_fitspy_results)
    file_tag = os.path.splitext(file)[0]
    folder = os.path.basename(os.path.dirname(path_to_fitspy_results))
    elapsed_time = get_spectrum_timestamp(folder, file_tag + '.asc', echelle_df)
    path_to_data = os.path.join('./data/brightness_data_fitspy_wl-calibrated', folder, file)
    model_df = pd.read_csv(path_to_fitspy_results, sep=';',  usecols=np.arange(0,5)).set_index(['label'])
    num_cols = ['x0','ampli','fwhm']
    model_df[num_cols] =  model_df[num_cols].apply(pd.to_numeric)
    model_df = model_df.reset_index(drop=True)
    n_peaks = len(model_df)


    full_df = pd.read_csv(path_to_data, comment='#').apply(pd.to_numeric)
    # Focus only on the wavelength region defined in wl_range
    df = full_df[full_df['Wavelength (nm)'].between(wl_range[0], wl_range[1])].reset_index(drop=True)
    wavelength = df['Wavelength (nm)'].values
    photon_flux = df['Brightness (photons/cm^2/s/nm)'].values
    b_peak = photon_flux.max()
    idx_dg = np.argmin(np.abs(photon_flux - b_peak))
    w_dg = wavelength[idx_dg]
    dx = 434.0 - w_dg
    full_df['Wavelength (nm)'] = full_df['Wavelength (nm)'] + dx
    df = full_df[full_df['Wavelength (nm)'].between(wl_range[0], wl_range[1])].reset_index(drop=True)
    wavelength = df['Wavelength (nm)'].values
    photon_flux = df['Brightness (photons/cm^2/s/nm)'].values

    # fit_df = full_df[full_df['Wavelength (nm)'].between(fit_range[0], fit_range[1])].reset_index(drop=True)
    # wavelength_fit = fit_df['Wavelength (nm)'].values
    # photon_flux_fit = fit_df['Brightness (photons/cm^2/s/nm)'].values
    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(w=4.5, h=3.5)

    ax.plot(
        wavelength, photon_flux, color='C0', marker='o', ms=3, mfc='none', mew=1., label='Data'
    )

    popt = np.empty(n_peaks*3, dtype=np.float64)
    for i, row in model_df.iterrows():
        x0 = row['x0']
        ampli = row['ampli']
        g = row['fwhm']
        h = 0.5 * ampli * np.pi * g
        ax.fill_between(
            wavelength, 0, lorentzian(
                x=wavelength, h=h, mu=x0, gamma=g
            ),
            alpha=0.25
        )
        idx_h = 3 * i
        idx_m = idx_h + 1
        idx_g = idx_m + 1
        popt[idx_h] = h
        popt[idx_m] = x0
        popt[idx_g] = g

    yfit = sum_lorentzians(wavelength, popt)
    ax.plot(
        wavelength, yfit,
        color='red', lw=1.5, label='Fit'
    )

    ax.axvline(x=calibration_line['center_wl'], ls='--', lw=1., color='grey')

    connectionstyle = "angle,angleA=-90,angleB=180,rad=0"
    # connectionstyle = "arc3,rad=0."
    bbox = dict(boxstyle="round", fc="wheat")
    arrowprops = dict(
        arrowstyle="->", color="k",
        shrinkA=5, shrinkB=0,
        patchA=None, patchB=None,
        connectionstyle=connectionstyle
    )
    ax.annotate(
            f"{calibration_line['label']} ({calibration_line['center_wl']:.2f}) nm",
            xy=(calibration_line['center_wl'], yfit.max()*2.75), xycoords='data',  # 'figure pixels', #data',
            # transform=axes[1].transAxes,
            xytext=(0.75, 0.90), textcoords='axes fraction',
            ha='right', va='top',
            arrowprops=arrowprops,
            bbox=bbox,
            fontsize=10
        )

    lorentzians_x0 = model_df['x0'].values + dx
    print(lorentzians_x0)
    peak_data = {}
    peak_data['Folder'] = folder
    peak_data['File'] = file
    peak_data['Model'] = 'Lorentzian'
    path_to_stats_file = os.path.join(os.path.dirname(path_to_fitspy_results), file_tag + '_stats.txt')
    connectionstyle = "angle,angleA=0,angleB=-90,rad=0"
    # connectionstyle = "arc3,rad=0."
    bbox = dict(boxstyle="round", fc="honeydew")
    arrowprops = dict(
        arrowstyle="->", color="k",
        shrinkA=5, shrinkB=0,
        patchA=None, patchB=None,
        connectionstyle=connectionstyle
    )
    for peak in peaks_of_interest:
        pc = peak['center_wl']
        idx = np.argmin(np.abs(pc - lorentzians_x0))
        center_wl = lorentzians_x0[idx]
        lbl = peak['label']
        peak_stats = get_peak_stats(idx+1, path_to_stats_file)
        suffix = 'cd' if lbl == 'C-D' else 'bd'

        peak_data[f'x0_{suffix} (nm)'] = peak_stats['x0']
        peak_data[f'ampli_{suffix} (photons/cm^2/s/nm)'] = peak_stats['ampli']
        peak_data[f'ampli_err_{suffix}  (photons/cm^2/s/nm)'] = peak_stats['ampli_error']
        peak_data[f'fwhm_{suffix} (nm)'] = peak_stats['fwhm']
        peak_data[f'fwhm_err_{suffix} (nm)'] = peak_stats['fwhm_error']
        peak_data[f'area_{suffix} (photons/cm^2/s)'] = peak_stats['area']
        peak_data[f'area_err_{suffix} (photons/cm^2/s)'] = peak_stats['area_error']

        # ax.axvline(x=center_wl, color='0.5', ls='-.', lw=1.)
        ax.annotate(
            f"{lbl}\n({center_wl:.2f}) nm",
            xy=(center_wl, sum_lorentzians(np.array([center_wl]), popt)*1.1), xycoords='data',  # 'figure pixels', #data',
            # transform=axes[1].transAxes,
            xytext=(-100, 80), textcoords='offset pixels',
            ha='center', va='bottom',
            arrowprops=arrowprops,
            bbox=bbox,
            fontsize=10
        )

    peak_data['Elapsed time (s)'] = elapsed_time
    output_df = load_output_db(xlsx_source=output_xls)
    output_df = update_out_df(output_df, peak_data)
    output_df.sort_values(by=['Folder', 'File'], inplace=True)
    output_df.to_excel(excel_writer=output_xls, index=False)

    ax.set_ylim(bottom=0, top=2.E13)
    ax.set_xlim(wl_range)
    mf = ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(mf)
    ax.ticklabel_format(useMathText=True)
    ax.set_xlabel(r"$\lambda$ {\sffamily (nm)}", usetex=True)
    ax.set_ylabel(r"$B_{\lambda}$ {\sffamily (photons/cm\textsuperscript{2}/s/nm)}", usetex=True)

    # Use folder_map_xls to map the dated folder to the corresponding sample
    folder_mapping = load_folder_mapping()
    sample_label = folder_mapping[folder]
    ax.set_title(f"{sample_label} - {elapsed_time/60.:.0f} min")
    figures_output_folder = os.path.join(output_folder, folder)
    if not os.path.exists(figures_output_folder):
        os.makedirs(figures_output_folder)
    fig.savefig(os.path.join(figures_output_folder, file_tag + '.png'), dpi=600)

    plt.show()






if __name__ == '__main__':
    main()