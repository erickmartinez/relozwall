import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
import matplotlib as mpl
import json
import numpy as np
from data_processing.utils import latex_float
from scipy.interpolate import interp2d, RegularGridInterpolator

path_to_pec = r'./data/ADAS/PEC/B/pec93#b_llu#b0.dat'

plasma_te = 5.3
plasma_ne = 0.2475E12


def get_wavelengths(path_to_file):
    p = re.compile(r'^\s*(\d+\.?\d*)\s+A')
    wl = []
    with open(file=path_to_file, mode='r') as f:
        for line in f.readlines():
            m = p.match(line)
            if m:
                wl.append(float(m.group(1)))
    wl.sort()
    return np.array([wl])

def get_pec(path_to_file, wl) -> pd.DataFrame:
    valid_wl = get_wavelengths(path_to_file)
    if wl not in valid_wl:
        raise(KeyError(f"The wavelength: {wl:.1f} Å was not found in {path_to_file}."))
    p_str = rf'^\s*{wl:.1f}\s+A\s+(\d+)\s+(\d+)'
    p_wl = re.compile(p_str)
    i0 = 100000000000
    n_e = []
    T_e = []
    pec_i = []
    found_wl = False
    done_with_ne = False
    done_with_Te = False
    done_with_pec = False
    p_n = re.compile(r'(\d+\.?\d+[Ee][\+\-]\d+)', re.IGNORECASE)
    pec_df = pd.DataFrame(data={
        'n_e (1/cm^3)': [],
        'T_e (eV)': [],
        'PEC (photons/cm^3/s)': []
    })
    k = 0 # count the number of
    with open(path_to_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if not found_wl:
                m_wl = p_wl.match(line)
                if m_wl:
                    i0 = i
                    n = int(m_wl.group(1)) # the number of electron densities
                    m = int(m_wl.group(2)) # the number of electron temperatures
            if i > i0:
                # find the first n numbers that correspond to n_e
                if not done_with_ne:
                    m_n = p_n.findall(line)
                    for v in m_n:
                        n_e.append(float(v))
                    if len(n_e) >= n:
                        done_with_ne = True
                elif not done_with_Te:
                    m_n = p_n.findall(line)
                    for v in m_n:
                        T_e.append(float(v))
                    if len(T_e) >= m:
                        done_with_Te = True
                elif not done_with_pec:
                    m_n = p_n.findall(line)
                    for v in m_n:
                        pec_i.append(float(v))
                        k += 1
                        if k >= m*n:
                            done_with_pec = True
            if done_with_pec:
                break


    pec = np.array(pec_i).reshape(n, m)

    n_e = np.array(n_e)
    T_e = np.array(T_e)
    for i in range(m):
        for j in range(n):
            row = pd.DataFrame(data={
                'n_e (1/cm^3)': [n_e[j]],
                'T_e (eV)': [T_e[i]],
                'PEC (photons/cm^3/s)': [pec[j][i]]
            })
            pec_df = pd.concat([pec_df, row], ignore_index=True)

    return pec_df


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    rcParams.update(plot_style)
    plt.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}')

def main():
    global path_to_pec
    # wl = get_wavelengths(path_to_file=path_to_pec)
    pec_df: pd.DataFrame = get_pec(path_to_file=path_to_pec, wl=8148.2)
    # pec_df = pec_df[pec_df['T_e (eV)'] <= 100]
    distinct_ne = pec_df['n_e (1/cm^3)'].unique()
    nn = len(distinct_ne)
    cmap = mpl.colormaps.get_cmap('jet')
    norm = mpl.colors.Normalize(vmin=0, vmax=nn-1)
    colors = [cmap(norm(i)) for i in range(nn)]

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.5)
    ax.set_xlabel(r"T$_{\mathregular{e}}$ (eV)")
    ax.set_ylabel(r"S (cm$^{\mathregular{3}}$/s)")
    ax.set_title('Photon Emissivity Coefficients')
    ax.set_yscale('log')
    ax.set_xscale('log')

    bi_pec_ne = pec_df['n_e (1/cm^3)'].unique()
    bi_pec_te = pec_df['T_e (eV)'].unique()
    n_te, n_ne = len(bi_pec_te), len(bi_pec_ne)
    z_pec = np.empty((n_te, n_ne), dtype=np.float64)
    for i in range(n_te):
        for j in range(n_ne):
            dfi = pec_df[(pec_df['T_e (eV)'] == bi_pec_te[i]) & (pec_df['n_e (1/cm^3)'] == bi_pec_ne[j])].reset_index(drop=True)
            z_pec[i, j] = dfi['PEC (photons/cm^3/s)'][0]

    # interpolate the pec coefficient for b_i
    # f = interp2d(bi_pec_ne, bi_pec_te, z_pec, kind='cubic')
    f = RegularGridInterpolator((bi_pec_ne, bi_pec_te), z_pec.T)
    pec_plasma = f((plasma_ne, plasma_te))

    markers = ['o', 's', '^', 'v', 'D', 'h', '<', '>']

    T_e = None
    for i, ne in enumerate(distinct_ne):
        c = colors[i]
        ne_df = pec_df[pec_df['n_e (1/cm^3)'] == ne]
        if i == 0:
            T_e = ne_df['T_e (eV)'].values
        pec = ne_df['PEC (photons/cm^3/s)'].values
        # print(pec)
        ne_arr = f'{ne:.0E}'.split('E')
        exponent = int(ne_arr[1])
        ax.plot(T_e, pec, marker=markers[i], mfc='none', ls='-', c=c, label=fr"$n_e = 10^{{{exponent:d}}}~\mathrm{{1/cm^3}}$")

    ax.plot([plasma_te], [pec_plasma], ls='none', marker = 'x', color='k', label=fr"$S = {latex_float(pec_plasma,1)}~\mathrm{{cm^3/s}}$")
    ax.legend(loc='lower right', fontsize=10)
    fig.savefig(r'./figures/pec_814.8nm.png', dpi=600)
    plt.show()




if __name__ == '__main__':
    main()