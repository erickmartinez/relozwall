import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import os
import json
import matplotlib as mpl
from scipy.optimize import least_squares, OptimizeResult

input_folder = r'./data/brightness_data_fitspy'
echelle_excel = r'./data/echelle_db.xlsx'
output_folder = r'./data/brightness_data_fitspy_wl-calibrated'
scaling_excel = r'./data/spectra_scaling_dalpha.xlsx'

d_lines = [
    {'center_wl': 410.06, 'label': r'D$_{\delta}$'},
    {'center_wl': 434.00, 'label': r'D$_{\gamma}$'},
    {'center_wl': 486.00, 'label': r'D$_{\beta}$'},
    {'center_wl': 656.10, 'label': r'D$_{\alpha}$'}
]

use_reference = d_lines[1] # (D_alpha)

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

def jac_sum_lorentzians(b, x, y):
    m = len(x)
    n3 = len(b)
    selector = np.arange(0, n3) % 3
    h = b[selector == 0]
    mu = b[selector == 1]
    gamma = b[selector == 2]
    n = len(h)
    res = np.empty((m, n3), dtype=np.float64)
    for i in range(n):
        g = gamma[i]
        g2 = g ** 2.
        xm = x - mu[i]
        xm2 = xm ** 2.
        den = (4. * xm2 + g2) ** 2.
        res[:, 3*i] = 0.5 * g / (xm2 + 0.25*g2)
        res[:, 3*i+1] = 16. * g * h[i] * xm / den
        res[:, 3*i+2] = h[i] * (8. * xm2 - 2. * g2) / den
    return res / np.pi


def main():
    global input_folder, echelle_excel, output_folder, use_reference, scaling_excel
    # load the echelle db
    echelle_df: pd.DataFrame = pd.read_excel(echelle_excel, sheet_name=0)
    echelle_df = echelle_df[echelle_df['Label'] != 'Labsphere']
    echelle_df = echelle_df[echelle_df['Is dark'] == 0]
    echelle_df = echelle_df[echelle_df['Number of Accumulations'] >= 20]
    echelle_df = echelle_df.reset_index(drop=True)
    echelle_df = echelle_df[['Folder', 'File']]

    # Create/Update an Excel spreadsheet containing the folder, file, measured wl_center, wl_shift and reference
    # intensity
    columns = ['Folder',
            'File',
            'Reference wl (nm)',
            'Measured wl (nm)',
            'Measured intensity (photons/cm^2/s/nm)',
            'Wavelength shift (nm)']
    try:
        scaling_df: pd.DataFrame = pd.read_excel(scaling_excel, sheet_name=0)
    except FileNotFoundError as err:
        print(err)
        print(f'Will create file {scaling_excel}')
        scaling_df: pd.DataFrame = pd.DataFrame(data={
            col: [] for col in columns
        })


    # Check whether output_folder exists and create it if it doesn't
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print('******** X0 ESTIMATE (nm) ********')
    print('FROM INDEX   FROM LORENTZIAN')
    for i, row in echelle_df.iterrows():
        folder = row['Folder']
        file = row['File'].replace('.asc', '.csv')
        try:
            df = pd.read_csv(os.path.join(input_folder, folder, file), comment='#')
        except FileNotFoundError as err:
            print(err)
            continue
        # Try to locate the reference in a -/+ 0.5 nm window
        ref_wl = use_reference['center_wl']
        win_df = df[df['Wavelength (nm)'].between(ref_wl-0.6, ref_wl+0.6)].reset_index(drop=True)
        wl = win_df['Wavelength (nm)'].values
        brightness = win_df['Brightness (photons/cm^2/s/nm)'].values
        b_peak = brightness.max()
        idx_peak = np.argmin(np.abs(brightness - b_peak))
        wl_peak = wl[idx_peak]
        # fitting the peak to a Lorentzian
        guess_g = 0.2 # the FWHM
        guess_h = 0.5 * b_peak * np.pi * guess_g # The area of the peak
        guess_m = wl_peak
        b0 = np.array([guess_h, guess_m, guess_g])
        eps = float(np.finfo(np.float64).eps)
        res_lsq: OptimizeResult = least_squares(
            res_sum_lorentzians, x0=b0, args=(wl, brightness),
            loss='linear', f_scale=0.1,
            jac=jac_sum_lorentzians,
            bounds=(
                [eps, wl_peak-0.5, 1E-5],
                [1E50, wl_peak+0.5, 1E10]
            ),
            xtol=eps,
            ftol=eps,
            gtol=eps,
            x_scale='jac',
            verbose=0,
            tr_solver='exact',
            max_nfev=10000 * len(wl)
        )
        popt = res_lsq.x
        print(f'{wl_peak:<10.2f}\t{popt[1]:>15.2f}')

        # Determine the shift
        wl_shift = ref_wl - popt[1]
        df['Wavelength (nm)'] += wl_shift

        row_data = pd.DataFrame(data={
            'Folder': [folder], 'File': file, 'Reference wl (nm)': [ref_wl],
            'Measured wl (nm)': [wl_peak],
            'Measured intensity (photons/cm^2/s/nm)': [b_peak],
            'Wavelength shift (nm)': [wl_shift]
        })

        # Try finding the combination of folder and file in scaling df
        row_index = scaling_df.loc[(scaling_df['Folder'] == folder) & (scaling_df['File'] == file)]
        previous_df = scaling_df[(scaling_df['Folder'] == folder) & (scaling_df['File'] == file)]
        if len(previous_df) == 0:
            scaling_df = pd.concat([scaling_df, row_data], ignore_index=True).reset_index(drop=True)
        else:
            row_index = scaling_df.loc[(scaling_df['Folder'] == folder) & (scaling_df['File'] == file)].index[0]
            for col, val in row_data.items():
                scaling_df.loc[row_index, col] = val[0]

        out_path = os.path.join(output_folder, folder)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        df.to_csv(os.path.join(out_path, file), index=False)

    scaling_df = scaling_df.reset_index(drop=True)
    print(scaling_df)
    # scaling_df.to_excel(excel_writer=scaling_excel, index=False)

if __name__ == '__main__':
    main()