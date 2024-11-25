import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
import json
import re
from scipy.signal import find_peaks
import matplotlib.ticker as ticker

path_to_pure_boron_csv = r'../data/EDX/20241122/hp_b_rod/Base(2).emsa'
path_to_pc_pebble_rod_csv = r'../data/EDX/20241122/pc_bp_rod/Base(1).emsa'


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

line_energies = {
    'B': 0.183,
    'C': 0.277,
    'N': 0.392,
    'O': 0.525
}
pattern_id = re.compile(r".*?PEAKLAB\s+\:\s+(\d+\.?\d*)\s+(\w+)\s+(\w+)\s+(\d+)")

def get_element_by_energy(energy):
    global line_energies
    elements, energies = list(line_energies), np.array(list(line_energies.values()))
    idx_energy = np.argmin(np.abs(energies-energy))
    return elements[idx_energy]


def get_identified_elements(path_to_emsa):
    global pattern_id
    identified = []
    with open(path_to_emsa, 'r') as f:
        for line in f:
            if line.startswith('#'):
                m = pattern_id.match(line)
                if m:
                    element = m.group(2)
                    energy = float(m.group(1))
                    transition = m.group(3)
                    identified.append({
                        'element': element, 'energy': energy, 'transition':transition
                    })
            else:
                break
    return identified

def main():
    global path_to_pure_boron_csv, path_to_pc_pebble_rod_csv, line_energies
    col_names = ['Energy (keV)', 'Counts']
    boron_df = pd.read_csv(path_to_pure_boron_csv, comment='#', header=None, names=col_names, usecols=[0,1]).apply(pd.to_numeric)
    pc_bp_df = pd.read_csv(path_to_pc_pebble_rod_csv, comment='#', header=None, names=col_names, usecols=[0,1]).apply(pd.to_numeric)

    boron_df = boron_df[boron_df['Energy (keV)'] <= 2.5].reset_index(drop=True)
    pc_bp_df = pc_bp_df[pc_bp_df['Energy (keV)'] <= 2.5].reset_index(drop=True)
    # boron_rod_identified = get_identified_elements(path_to_pure_boron_csv)
    # pc_boron_pebble_identified = get_identified_elements(path_to_pc_pebble_rod_csv)

    energy_b_rod = boron_df['Energy (keV)'].values
    energy_pc_b_rod = pc_bp_df['Energy (keV)'].values

    intensity_b_rod = boron_df['Counts'].values * 1E-3  # kCounts
    intensity_pc_b_pebble_rod = pc_bp_df['Counts'].values * 1E-3 # kCounts

    peaks_b_rod, _ = find_peaks(intensity_b_rod, threshold=0.05)
    peaks_pc_b_rod, _ = find_peaks(intensity_pc_b_pebble_rod, threshold=0.1)

    peaks_b_rod = energy_b_rod[peaks_b_rod]
    peaks_b_rod = np.append(peaks_b_rod, [0.524])

    peaks_pc_b_rod = energy_pc_b_rod[peaks_pc_b_rod]

    # print(peaks_b_rod)
    # print()
    # print(energy_pc_b_rod[peaks_pc_b_rod])

    load_plot_style()

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True)
    fig.set_size_inches(4.5, 4.5)

    ax1.plot(energy_b_rod, intensity_b_rod, color='C0', label='B rod')
    ax2.plot(energy_pc_b_rod, intensity_pc_b_pebble_rod, color='C1', label='poly-B pebble rod')

    for peak_energy in  peaks_b_rod:
        idx_energy = np.argmin(np.abs(energy_b_rod - peak_energy))
        intensity = intensity_b_rod[idx_energy]
        element = get_element_by_energy(peak_energy)
        ax1.plot(
            [peak_energy], [intensity], marker='|', ms=10, color='tab:red', ls='none', mew=1.5
        )

        ax1.annotate(
            text=element,
            xy=(peak_energy, intensity), xytext=(0, 15),
            xycoords='data', textcoords='offset pixels',
            ha='center', va='bottom'
        )

    for peak_energy in peaks_pc_b_rod:
        idx_energy = np.argmin(np.abs(energy_pc_b_rod - peak_energy))
        intensity = intensity_pc_b_pebble_rod[idx_energy]
        element = get_element_by_energy(peak_energy)
        ax2.plot(
            [peak_energy], [intensity], marker='|', ms=10, color='tab:red', ls='none', mew=1.5
        )

        ax2.annotate(
            text=element,
            xy=(peak_energy, intensity), xytext=(0, 15),
            xycoords='data', textcoords='offset pixels',
            ha='center', va='bottom'
        )

    for ax in (ax1, ax2):
        ax.set_ylim(0, 60)
        ax.set_xlim(0, 2.5)
        ax.legend(loc='upper right', frameon=True, fontsize=10)

    # fig.supylabel('Counts (x1000)')
    ax2.set_xlabel('Energy (keV)')

    for i, axi in enumerate([ax1, ax2]):
        axi.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        axi.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        axi.set_ylabel('Counts (x1000)')
        panel_label = chr(ord('`') + i + 1)
        axi.text(
            -0.1, 1.05, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )

    fig.savefig('fig_eds_boron.svg', dpi=600)
    plt.show()



if __name__ == '__main__':
    main()