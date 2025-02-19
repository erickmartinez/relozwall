import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from figure_sputtering_rate import load_plot_style
from pathlib import Path
import os
import re
from scipy.stats.distributions import t

BASE_DIR = r"./data/PA_probe"

def load_data(base_dir):
    base_path = Path(base_dir)
    folders = [folder  for folder in os.listdir(base_path) if not folder.startswith('.')]
    db_df = pd.DataFrame(data={
        'Folder': [], 'Shot': [], 'V_sc (V)': [], 'V_sc error (V)': [],
        'T_e (eV)': [], 'T_e error (eV)': [], 'E_i_0 (eV)': [], 'E_i_0 error (eV)': []
    })
    p = re.compile(r".*?ivdata(\d+)\_symmetrized.*")
    for folder in folders:
        path_to_results = base_path / folder / 'langprobe_results' / 'symmetrized'
        files = [file for file in os.listdir(path_to_results) if file.endswith('symmetrized.csv')]
        for file in files:
            shot_n = 0
            m = p.match(file)
            if m:
                shot_n = int(m.group(1))
            # Find the index of the row for which the value of the distance (x) = 0
            target_value = 0
            # Load the symmetrized file
            df = pd.read_csv(path_to_results / file).apply(pd.to_numeric)
            # Calculate the absolute difference between the target value and each value in the column
            abs_diff = np.abs(df['x (cm)'] - target_value)
            # Find the index of the minimum value in the absolute difference series
            closest_index = abs_diff.idxmin()
            df = df.iloc[closest_index]
            data = {
                'Folder': [folder], 'Shot': [shot_n],
                'V_sc (V)': df['V_sc (eV)'], 'V_sc error (V)': df['V_sc error (eV)'],
                'T_e (eV)': df['T_e (eV)'], 'T_e error (eV)': df['T_e error (eV)'],
                'E_i_0 (eV)': 3.* df['T_e (eV)'], 'E_i_0 error (eV)': 3.* df['T_e error (eV)'],
            }
            row = pd.DataFrame(data=data)
            db_df = pd.concat([db_df, row]).reset_index(drop=True)
    return db_df.sort_values(by=['Folder'], ascending=True).reset_index(drop=True)



def main(base_dir):
    db_df = load_data(base_dir)
    folders = db_df['Folder'].unique()
    load_plot_style()
    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True, sharex=True)
    fig.set_size_inches(6.5, 6)
    markers = ['o', 's', '^', 'D', 'v', 'p', '<', 'h', '>']
    norm = mpl.colors.Normalize(vmin=0, vmax=(len(folders)-1))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    vsc = np.array([])
    for i, folder in enumerate(folders):
        df = db_df[db_df['Folder'] == folder].reset_index(drop=True)
        markers_b, caps_b, bars_b = axes[0, 0].errorbar(
            df['Shot'], df['V_sc (V)'], yerr=df['V_sc error (V)'], capsize=2.75,
            mew=1.25, marker=markers[i], ms=8, elinewidth=1.25,
            color=colors[i],  fillstyle='none',
            ls='none',  # lw=1.25,
            label=folder,
        )
        [bar.set_alpha(0.35) for bar in bars_b]

        vsc = np.hstack([vsc, df['V_sc (V)'].values])


        markers_b, caps_b, bars_b = axes[0, 1].errorbar(
            df['Shot'], df['T_e (eV)'], yerr=df['T_e error (eV)'], capsize=2.75,
            mew=1.25, marker=markers[i], ms=8, elinewidth=1.25,
            color=colors[i],  fillstyle='none',
            ls='none',  # lw=1.25,
            label=folder,
        )
        [bar.set_alpha(0.35) for bar in bars_b]

        e_i_2 = np.abs(-4*df['T_e (eV)'].values - df['V_sc (V)'].values)
        e_i_2_err = np.linalg.norm(np.column_stack([-4*df['T_e error (eV)'].values, df['V_sc error (V)'].values]), axis=1)

        markers_b, caps_b, bars_b = axes[1, 0].errorbar(
            df['Shot'], e_i_2, yerr=e_i_2_err, capsize=2.75,
            mew=1.25, marker=markers[i], ms=8, elinewidth=1.25,
            color=colors[i], fillstyle='none',
            ls='none',  # lw=1.25,
            label=folder,
        )
        [bar.set_alpha(0.35) for bar in bars_b]

        markers_b, caps_b, bars_b = axes[1, 1].errorbar(
            df['Shot'], df['E_i_0 (eV)'], yerr=df['E_i_0 error (eV)'], capsize=2.75,
            mew=1.25, marker=markers[i], ms=8, elinewidth=1.25,
            color=colors[i],  fillstyle='none',
            ls='none',  # lw=1.25,
            label=folder,
        )
        [bar.set_alpha(0.35) for bar in bars_b]

    axes[0, 0].set_ylabel(r'$V_{\mathrm{sc}}$ {\sffamily (eV)}', usetex=True)
    axes[0, 1].set_ylabel(r'$T_{e}$ {\sffamily (eV)}', usetex=True)
    axes[1, 0].set_ylabel(r"abs($-4T_e - V_{\mathrm{sc}}$) (eV)")
    axes[1, 1].set_ylabel(r'$E_{\mathrm{}} = 3T_e$ {\sffamily (eV)}', usetex=True)


    for ax in axes.flatten():
        ax.set_xlabel(r'Shot #')
        ax.legend(loc='upper right', frameon=True, fontsize=10)

    n = len(vsc)
    v_sc_mean = np.mean(vsc)
    v_sc_std =vsc.std(ddof=1)
    t_val = t.ppf(1 - 0.05 * 0.5, n - 1)
    v_sc_se = v_sc_std * t_val / np.sqrt(n)
    axes[0, 0].axhline(y=v_sc_mean, color='tab:red', ls='--')
    axes[0, 0].axhspan(ymin=(v_sc_mean-v_sc_se), ymax=(v_sc_mean+v_sc_se), color='tab:red', alpha=0.3)
    axes[0, 0].text(
        0.95, 0.1, fr"$\langle V_{{\mathrm{{sc}}}} \rangle$ = {v_sc_mean:.0f} ± {v_sc_se:.1f} eV",
        transform=axes[0,0].transAxes,
        ha='right', va='top'
    )


    plt.show()



if __name__ == '__main__':
    main(base_dir=BASE_DIR)
