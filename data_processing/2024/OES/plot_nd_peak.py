import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import os
import json
import numpy as np
from scipy.signal import savgol_filter

BRIGHTNESS_FOLDER = r'./data/brightness_data_fitspy_wl-calibrated'
FOLDER_MAP_XLS = r'./PISCES-A_folder_mapping.xlsx'  # Folder name to plot label database

LBL_MAPPING = {
    'echelle_20240815': 'SBR',
    'echelle_20240827': 'ABPR',
    'echelle_20241003': 'PBPR',
    'echelle_20241031': 'SBR (Low thermal contact)'
}

AXES_MAPPING = {
    'echelle_20240815': 0, 'echelle_20240827': 2, 'echelle_20241003': 3, 'echelle_20241031': 1
}

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


def main(brightness_folder, label_mapping, axis_mapping, folder_map_xls):
    folder_mapping = load_folder_mapping(folder_map_xls)
    folders = [f for f in os.listdir(brightness_folder) if not f.startswith('.')]
    n = len(folders)
    load_plot_style()
    cmap_names = ["Blues", "Oranges", "Greens", "Reds", "Purples"]
    cmaps = [mpl.colormaps.get_cmap(cmapi) for cmapi in cmap_names]
    wl_range = (333., 338.)
    tab10_colors = [f'C{i}' for i in range(n)]


    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, constrained_layout=True)
    fig.set_size_inches(3.5, 4.5)
    # fig.subplots_adjust(hspace=0, left=0.15, right=0.98, top=0.95, bottom=0.1)

    for i, folder in enumerate(folders):
        idx_axis = axis_mapping[folder]
        if folder == 'echelle_20241031':
            continue
        plot_lbl = label_mapping[folder]
        files = [f for f in os.listdir(os.path.join(brightness_folder, folder)) if f.endswith('.csv')]
        n_files = len(files)
        norm = mpl.colors.Normalize(vmin=-1, vmax=(n_files - 1))
        cmap = cmaps[i]
        colors = [cmap(norm(j)) for j in range(n_files)]
        colors = colors[::-1]
        alphas = [norm(j) for j in range(n_files)]
        alphas = alphas[::-1]
        sample_lbl = folder_mapping[folder]
        for j, file in enumerate(files):
            lbl = plot_lbl if j == 0 else None
            df = pd.read_csv(os.path.join(brightness_folder, folder, file), comment='#').apply(pd.to_numeric)
            df = df[df['Wavelength (nm)'].between(wl_range[0], wl_range[1])]
            wl = df['Wavelength (nm)'].values
            sr = df['Brightness (photons/cm^2/s/nm)'].values
            # sr = savgol_filter(
            #     sr,
            #     window_length=5,
            #     polyorder=3
            # )
            sr -= sr.min()
            axes[i].plot(wl, sr * 1E-12, color=colors[j], label=lbl, lw=1., alpha=alphas[j])
        axes[i].set_ylim(0, 6.)
        axes[i].axvline(x=335.76, ls='--', color='grey', lw=1.)
        axes[i].set_xlim(wl_range)
        axes[i].xaxis.set_major_locator(ticker.MultipleLocator(1))
        axes[i].xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
        axes[i].yaxis.set_major_locator(ticker.MultipleLocator(2))
        axes[i].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        axes[i].legend(loc='upper left', fontsize=10, frameon=True)

    # axes[-1].set_ylim(-0.1, 10.)

    axes[0].set_title('N-D Band (335.7 nm)')
    fig.supylabel('Spectral radiance (W/cm$^{\mathregular{2}}$/ster/nm) x10$^{\mathregular{12}}$')
    axes[-1].set_xlabel('$\lambda$ {\sffamily (nm)}', usetex=True)

    for i, axi in enumerate(axes.flatten()):
        panel_label = chr(ord('`') + i + 1) # starts from a
        # panel_label = chr(ord('`') + i + 3)
        axi.text(
            -0.125, 1.05, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )

    fig.savefig('./figures/fig_nd_band_plot.pdf', dpi=600)
    plt.show()


if __name__ == '__main__':
    main(brightness_folder=BRIGHTNESS_FOLDER, label_mapping=LBL_MAPPING, axis_mapping=AXES_MAPPING, folder_map_xls=FOLDER_MAP_XLS)
