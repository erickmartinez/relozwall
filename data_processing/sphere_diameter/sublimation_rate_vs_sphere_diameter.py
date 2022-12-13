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
    df = df[(df['Binder'] == 'Graphite') & (df['Binder wt %'] == 10)]
    print(df)
    df = df.groupby('Small spheres wt %').agg({
        'Thickness (nm)': ['mean'],
        'Thickness error (nm)': [rse],
        'Big spheres wt %': ['mean'],
    })
    df.sort_values(by=['Small spheres wt %'], inplace=True)
    print(df)
    big_spheres_wt_pct = 100.0 - df.index
    thickness_nm = df['Thickness (nm)']['mean'].values
    thickness_err = df['Thickness error (nm)']['rse'].values

    load_plt_style()

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.0)

    ax.errorbar(
        big_spheres_wt_pct, thickness_nm / 0.5 ,
        yerr=thickness_err/0.5,
        c='C0',
        marker='o', lw=1.5,
        # mec='k',
        mew=1.25, capsize=3.5, capthick=1.25, ecolor='C0', fillstyle='none'
    )

    ax.set_xlabel('Weight % of 850 $\mathregular{\mu}$m spheres')
    ax.set_ylabel('Deposition rate (nm/s)')
    # ax.set_xlim(0,100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax.set_ylim(60,180)
    ax.set_xlim(-5, 105)
    ax.text(
        -0.0, -0.2, '185 $\mathregular{\mu}$m', transform=ax.transAxes, fontsize=12, fontweight='regular',
        va='bottom', ha='center'
    )
    ax.text(
        1.0, -0.2, '850 $\mathregular{\mu}$m', transform=ax.transAxes, fontsize=12, fontweight='regular',
        va='bottom', ha='center'
    )
    #

    fig.savefig(os.path.join(base_dir, 'deposition_rate_vs_sphere_diameter.png'), dpi=600)
    plt.show()
