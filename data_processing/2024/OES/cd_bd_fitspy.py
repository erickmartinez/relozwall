import pandas as pd
import numpy as np
import re
import os

output_excel = r'./data/cd_bd_db.xlsx'
input_path = r'./data/cd_bd_fit_data/echelle_20241003'

columns = [
    "Folder", "File",
    "m04_ampli", "m04_ampli_delta", "m04_fwhm", "m04_fwhm_delta", "m04_x0", "m04_x0_delta", "m04_area",
    "m04_area_delta",
    "m10_ampli", "m10_ampli_delta", "m10_fwhm", "m10_fwhm_delta", "m10_x0", "m10_x0_delta", "m10_area",
    "m10_area_delta"
]

peak_mapping = {4:'BD', 10:'CD'}


def initialize_db():
    global columns, peak_mapping
    output_cols = columns.copy()
    # Get the peak ids of interest from the prefix of the items in `columns`
    peak_ids = []
    p = re.compile(r'm(\d+)\_.*')
    for col in columns:
        if "ampli" in col:
            # print(col)
            m = p.match(col)
            if m:
                peak_id = int(m.group(1))
                if peak_id not in peak_ids:
                    peak_ids.append(peak_id)

    for i, col in enumerate(columns):
        p = re.compile(r'm(\d+)\_(.*)?')
        m = p.match(col)
        if m:
            out_col = peak_mapping[int(m.group(1))] + '_' + m.group(2)
            # Map the column name to the agreed CD or BD peak prefix
            output_cols[i] = out_col

    cd_bd_df = pd.DataFrame(data={col: [] for col in output_cols})
    return cd_bd_df



def get_values_from_file(path_to_file):
    global columns, peak_mapping
    search_cols = columns[2::]
    search_cols = [sc for sc in search_cols if (not '_delta' in sc) and (not '_area' in sc)]
    # Get the peak ids of interest from the prefix of the items in `columns`
    peak_ids = []
    p = re.compile(r'm(\d+)\_.*')
    for col in columns:
        if "ampli" in col:
            # print(col)
            m = p.match(col)
            if m:
                peak_id = int(m.group(1))
                if peak_id not in peak_ids:
                    peak_ids.append(peak_id)
    data = {}
    with open(path_to_file, 'r') as f:
        for line in f:
            for col in search_cols:
                if col not in data:
                    p = re.compile(r'm(\d+)\_(.*)?')
                    m = p.match(col)
                    out_col = peak_mapping[int(m.group(1))] + '_' + m.group(2)
                    p = re.compile(fr'\s*{col}\:\s+(\-?\d+\.?\d*e?\+?\-?\d+)\s+\+\/\-\s+(\-?\d+\.?\d*e?\+?\-?\d+)')
                    m = p.match(line)
                    if m:
                        val = float(m.group(1))
                        err = float(m.group(2))
                        data[out_col] = val
                        data[f'{out_col}_delta'] = err
    area_params = ["ampli", "fwhm"]
    # Estimate the peak area for each peak
    for peak_id in peak_ids:
        mapped_peak = peak_mapping[peak_id]
        fwhm = data[f'{mapped_peak}_fwhm']
        fwhm_delta = data[f'{mapped_peak}_fwhm_delta']
        ampli = data[f'{mapped_peak}_ampli']
        ampli_delta = data[f'{mapped_peak}_ampli_delta']
        area, area_err = peak_area_fwhm(
            ampli=ampli, fwhm=fwhm, ampli_err=ampli_delta, fwhm_err=fwhm_delta
        )
        data[f'{mapped_peak}_area'] = area
        data[f'{mapped_peak}_area_delta'] = area_err
    return data


BY_2_SQRT_LOG2 = 0.5 / ((2. * np.log(2.)) ** 0.5)
SQRT_2PI = (2. * np.pi) ** 0.5


def peak_area_fwhm(ampli, fwhm, ampli_err=None, fwhm_err=None):
    global BY_2_SQRT_LOG2, SQRT_2PI
    s = fwhm * BY_2_SQRT_LOG2
    area = ampli * s * SQRT_2PI
    if (not ampli_err is None) and (not fwhm_err is None):
        area_err = area * np.linalg.norm([ampli_err / ampli, fwhm_err / fwhm])
        return area, area_err
    else:
        return area


def main():
    global output_excel, input_path, columns

    try:
        cd_bd_df = pd.read_excel(output_excel, sheet_name=0)
    except FileNotFoundError as ex:
        print(f'Provided path: \'{output_excel}\' does not exist.')
        print(f'Will create an empty spread sheet')
        cd_bd_df = initialize_db()

    folder = os.path.basename(input_path)
    # iterate over all the files in the folder
    data_files = [file for file in os.listdir(path=os.path.join(input_path)) if file.endswith('_stats.txt')]
    for file in data_files:
        print(f'Analyzing {folder}/{file}')
        row_data = get_values_from_file(path_to_file=os.path.join(input_path, file))
        # print(row_data)
        # Try looking up for the file and folder in cd_bd_df
        row_index = (cd_bd_df['Folder'] == folder) & (cd_bd_df['File'] == file)
        if len(cd_bd_df[row_index]) > 0:
            for key, val in row_data.items():
                cd_bd_df.loc[row_index, key] = val
        else:
            row_data['Folder'] = folder
            row_data['File'] = file
            row = pd.DataFrame(data={key: [val] for key, val in row_data.items()})
            cd_bd_df = pd.concat([cd_bd_df, row], ignore_index=True).reset_index(drop=True)
            cd_bd_df.sort_values(by=['Folder', 'File'], ascending=True, inplace=True)

    # print(cd_bd_df)
    cd_bd_df.to_excel(output_excel, index=False)

if __name__ == '__main__':
    main()
