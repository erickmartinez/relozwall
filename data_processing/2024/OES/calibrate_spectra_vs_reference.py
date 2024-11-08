import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import os
import json
import matplotlib as mpl

input_folder = r'./data/brightness_data_fitspy'
echelle_excel = r'./data/echelle_db.xlsx'
output_folder = r'./data/brightness_data_fitspy_wl-calibrated'
scaling_excel = r'./data/spectra_scaling_dalpha.xlsx'

d_lines = [
    {'center_wl': 410.06, 'label': r'D$_{\delta}$'},
    {'center_wl': 433.93, 'label': r'D$_{\gamma}$'},
    {'center_wl': 486.00, 'label': r'D$_{\beta}$'},
    {'center_wl': 656.10, 'label': r'D$_{\alpha}$'}
]

use_reference = d_lines[1] # (D_alpha)


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
        win_df = df[df['Wavelength (nm)'].between(ref_wl-0.5, ref_wl+0.5)].reset_index(drop=True)
        wl = win_df['Wavelength (nm)'].values
        brightness = win_df['Brightness (photons/cm^2/s/nm)'].values
        b_peak = brightness.max()
        idx_peak = np.argmin(np.abs(brightness - b_peak))
        wl_peak = wl[idx_peak]
        # Determine the shift
        wl_shift = ref_wl - wl_peak
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
    scaling_df.to_excel(excel_writer=scaling_excel, index=False)

if __name__ == '__main__':
    main()