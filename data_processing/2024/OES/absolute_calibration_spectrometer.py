import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
from scipy.interpolate import interp1d


def load_db():
    df = pd.read_excel('./data/echelle_db.xlsx', sheet_name='Spectrometer parameters')
    df = df[df['Folder'] == 'echelle_20240910']
    return df

def load_labsphere_calibration():
    df = pd.read_csv(
        './data/PALabsphere_2014.txt', sep=' ', comment='#',
        usecols=[0], names=['Radiance (W/cm2/ster/nm)']
    ).apply(pd.to_numeric)
    # radiance = df['Radiance (W/cm2/ster/nm)']
    n = len(df)
    wl = 350. + np.arange(n) * 10.
    df['Wavelength (nm)'] = wl
    return df[['Wavelength (nm)', 'Radiance (W/cm2/ster/nm)']]


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    plt.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}')

def main():
    db_df = load_db()
    labsphere_df = load_labsphere_calibration()
    wl_ls = labsphere_df['Wavelength (nm)'].values
    radiance_ls = labsphere_df['Radiance (W/cm2/ster/nm)'].values
    radiance_ls_interp = interp1d(x=wl_ls, y=radiance_ls)

    # Iterate for every
