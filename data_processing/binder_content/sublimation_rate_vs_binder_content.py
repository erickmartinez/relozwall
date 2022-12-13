import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\SPHERE_DIAMETER\film_thickness'
excel_db = 'transmission_sphere_diameter_and_binder_scan_smausz.xlsx'

def load_plt_style():
    with open('../../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

def load_data(excel_file):
    df = pd.read_excel(
        io=excel_file, sheet_name='Sheet1', skiprows=9, header=0
    )
    df = df.loc[:,~df.columns.str.match("Unnamed")]
    return df

def rse(series: pd.Series):
    return np.sqrt(series.dot(series)) / len(series)

if __name__ == '__main__':
    full_file = os.path.join(base_dir, excel_db)
    df = load_data(os.path.join(base_dir, excel_db))
    df = df[(df['Binder'] == 'Graphite') & (df['Small spheres wt %'] == 5)]
    print(df)
    df = df.groupby('Binder wt %').agg({
        'Thickness (nm)': ['mean'],
        'Thickness error (nm)': [rse],
    })
    df.sort_values(by=['Binder wt %'], inplace=True)
    print(df)
    binder_wt_pct = df.index
    thickness_nm = df['Thickness (nm)']['mean'].values
    thickness_err = df['Thickness error (nm)']['rse'].values

    load_plt_style()

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.0)

    ax.errorbar(
        binder_wt_pct, thickness_nm / 0.5 ,
        yerr=thickness_err/0.5,
        c='C0',
        marker='o', lw=1.5,
        # mec='k',
        mew=1.25, capsize=3.5, capthick=1.25, ecolor='C0', fillstyle='none'
    )

    ax.set_xlabel('Binder wt %')
    ax.set_ylabel('Deposition rate (nm/s)')
    # ax.set_xlim(0,100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.set_ylim(40, 140)
    ax.set_xlim(0, 21)


    fig.savefig(os.path.join(base_dir, 'deposition_rate_vs_binder_content.png'), dpi=600)
    plt.show()
