import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import json
from scipy.stats.distributions import t
import platform
import matplotlib.ticker as ticker
from data_processing.utils import get_experiment_params
import re

"""
Sample R4N137
Laser test ROW513 (40 MW/m^2)
"""

base_path = './data/tracking'
plot_files = [
    {'fn': 'P1_278_474_N1_temperature.csv', 'label': 'Pebble 1', 'marker': '^'},
    # {'fn': 'P4_300_478_N1_temperature.csv', 'label': 'Pebble 2', 'marker': '<'},
    {'fn': 'P5_267_447_N1_temperature.csv', 'label': 'Pebble 2', 'marker': '>'},
    {'fn': 'P10_288_492_N1_temperature.csv', 'label': 'Pebble 3', 'marker': 'v'},
    {'fn': 'matrix_10pt_120px_temperature.csv', 'label': 'Matrix', 'marker': 's'},
]

inl_sim = 'Documents/ucsd/Postdoc/research/manuscripts/paper2/INL/ucsd/heat_30_extra_output/laserHeating3D_out.csv'

platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'
def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def normalize_path(the_path):
    global platform_system, drive_path
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    else:
        the_path = the_path.replace('/', '\\')
    return os.path.join(drive_path, the_path)

def main():
    global base_path, plot_files, inl_sim
    inl_sim = normalize_path(inl_sim)

    load_plot_style()

    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True, height_ratios=[1, 0.5, 0.5])
    fig.set_size_inches(4.0, 5.75)
    n_files = len(plot_files)
    norm = mpl.colors.Normalize(vmin=0, vmax=(n_files - 1))
    cmap = mpl.colormaps.get_cmap('rainbow_r')
    lcolor = [cmap(norm(i)) for i in range(n_files)]
    mean_mtx_temp = 298
    max_temp = -1
    for i, item in enumerate(plot_files):
        df = pd.read_csv(os.path.join(base_path, item['fn']), comment='#').apply(pd.to_numeric)
        time_s = df['Time (s)'].values
        temperature = df['Temperature (K)'].values
        delta = df['Temperature upb (K)'].values - temperature
        max_temp = max(max_temp, temperature.max())

        if i == 0:
            temperature_sphere_max = round(temperature.max()/10.)*10.
            print(f'Max pebble temp: {temperature_sphere_max:.1f} K')
            # axes[0].plot([0, 0.2], [temperature_sphere_max, temperature_sphere_max], ls='--', color=lcolor[i])
            txt_sphere_temp_max = f'  $T_{{\\mathrm{{max}}}}$ = {temperature_sphere_max:.0f} K'
            txt_sphere_max_color = lcolor[i]
        elif i + 1 == n_files:
            msk = time_s >= 0.075
            temperature_matrix:np.ndarray = temperature[msk]
            time_mtx: np.ndarray = time_s[msk]
            time_mtx_max = time_mtx[-1]
            matrix_n = len(temperature_matrix)
            mean_mtx_temp = round(temperature_matrix.mean()/10.)*10.
            std_temp = temperature_matrix.std(ddof=1)
            t_val = t.ppf(1 - 0.05*0.05, matrix_n - 1)
            temp_delta = round(std_temp * t_val / np.sqrt(matrix_n)/10.)*10.
            print(f'Mean matrix temp: {mean_mtx_temp:.1f} ± {temp_delta:.1f} K')
            axes[0].plot([0, time_s.max()], [mean_mtx_temp, mean_mtx_temp], ls='--', color=lcolor[i])
            axes[0].plot(
                [time_mtx[0], time_mtx[-1]], [temperature_matrix[0], temperature_matrix[-1]],
                c='k', mec=lcolor[i], fillstyle='full', marker='o', ls='none',
                ms=10, zorder=5, alpha=0.75
            )
            txt_mtx_temp = f' $\\langle T \\rangle$=  {mean_mtx_temp:.0f}±{temp_delta:.0f} K'
            txt_mtx_color = lcolor[i]
        axes[0].errorbar(
            time_s, temperature, yerr=delta, #xerr=0.0025,
            marker=item['marker'],
            ms=9, mew=1.25, mfc='none', ls='none',
            capsize=2.75, elinewidth=1.25, lw=1.5,
            color=lcolor[i],
            label=item['label']
        )




    inl_df: pd.DataFrame = pd.read_csv(inl_sim).apply(pd.to_numeric)
    inl_df = inl_df.rename(columns={'time': 'time (s)'})
    inl_df = inl_df[(inl_df['time (s)'] >= 0.8) & (inl_df['time (s)'] <= 1.1)]
    time_sim = inl_df['time (s)'].values
    n_sim = len(inl_df)
    axes[1].plot(
        time_sim,
        inl_df['avg_front_temp_sphere'].values,
        marker='none', color='tab:blue', ls='-',
        label='Sphere'
    )

    temp_sim_sphere_max = round(inl_df['avg_front_temp_sphere'].max()/10)*10.
    temp_sim_sphere_mean = round(inl_df['avg_front_temp_sphere'].mean()/50.)*50.
    temp_sim_sphere_std = inl_df['avg_front_temp_sphere'].std(ddof=1)
    temp_sim_sphere_se = temp_sim_sphere_std * t.ppf(1. - 0.05*0.5, n_sim - 1) / np.sqrt(n_sim)
    temp_sim_sphere_se = round(temp_sim_sphere_se / 10.) * 10.
    txt_sim_sphere_temp_mean = f'$\\langle T \\rangle$ = {temp_sim_sphere_mean:.0f}±{temp_sim_sphere_se:.0f} K'
    txt_sim_sphere_temp_max = f'$T_{{\mathrm{{max}}}}$ = {temp_sim_sphere_max:.0f} K'

    axes[2].plot(
        time_sim,
        inl_df['avg_front_temp_matrix'].values,
        marker='none', color='tab:green', ls='-',
        label='Matrix'
    )

    temp_sim_mtx_max = round(inl_df['avg_front_temp_matrix'].max() / 10) * 10.
    temp_sim_mtx_mean = round(inl_df['avg_front_temp_matrix'].mean() / 50.) * 50.
    temp_sim_mtx_std = inl_df['avg_front_temp_matrix'].std(ddof=1)
    temp_sim_mtx_se = temp_sim_mtx_std * t.ppf(1. - 0.05 * 0.5, n_sim - 1) / np.sqrt(n_sim)
    temp_sim_mtx_se = round(temp_sim_mtx_se)
    txt_sim_mtx_temp_mean = f'$\\langle T \\rangle$ = {temp_sim_mtx_mean:.0f}±{temp_sim_mtx_se:.0f} K'
    txt_sim_mtx_temp_max = f'$T_{{\mathrm{{max}}}}$ = {temp_sim_mtx_max:.0f} K'


    axes[0].text(
        time_mtx_max, mean_mtx_temp*1.025, txt_mtx_temp, va='center', ha='left',
        color=txt_mtx_color, fontsize=10
    )

    axes[0].text(
        0.2, temperature_sphere_max, txt_sphere_temp_max, va='center', ha='left',
        color=txt_sphere_max_color, fontsize=10
    )

    axes[1].plot(
        [time_sim[0], time_sim[-1]],
        [temp_sim_sphere_mean, temp_sim_sphere_mean],
        marker='none', color='tab:blue', ls=':',
    )


    axes[1].text(
        time_sim[-7], temp_sim_sphere_mean, txt_sim_sphere_temp_mean, va='center', ha='right',
        color='tab:blue', fontsize=10, linespacing=0.7,
        bbox=dict(boxstyle="round",
                  # ec=(0.5, 0.5, 0.5),
                  ec='none',
                  fc=(1., 1., 1., 0.85),
                  pad=0.1,
                  )
    )

    axes[1].text(
        0.95, 0.95, txt_sim_sphere_temp_max, va='top', ha='right',
        color='k', fontsize=10, linespacing=0.7,
        transform=axes[1].transAxes
    )

    """
    Simulation matrix
    """
    axes[2].plot(
        [time_sim[0], time_sim[-1]],
        [temp_sim_mtx_mean, temp_sim_mtx_mean],
        marker='none', color='tab:green', ls=':',
    )

    axes[2].text(
        0.95, 0.95, txt_sim_mtx_temp_mean, va='top', ha='right',
        color='tab:green', fontsize=10,
        transform=axes[2].transAxes
    )

    print(f'Max temperature: {max_temp:.0f} K')

    axes[0].set_title('Experiment (40 MW/m$^{\mathregular{2}}$)')
    axes[1].set_title('Simulation (30 MW/m$^{\mathregular{2}}$)')

    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    axes[2].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    axes[2].xaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(250))
    axes[2].yaxis.set_major_locator(ticker.MultipleLocator(250))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(50))
    axes[2].yaxis.set_minor_locator(ticker.MultipleLocator(50))

    for ax in axes:
        ax.set_ylabel('$T$ (K)')
    axes[0].set_xlabel('$t$ (s)')
    axes[2].set_xlabel('$t$ (s)')
    axes[0].set_xlim(left=0, right=0.3)
    axes[1].set_xlim(left=0.8, right=1.1)
    axes[2].set_xlim(left=0.8, right=1.1)
    axes[0].set_ylim(bottom=2000, top=3500)
    axes[1].set_ylim(bottom=3000, top=3500)
    axes[2].set_ylim(bottom=3000, top=3500)
    axes[0].legend(loc='lower center', frameon=True, fontsize=10, ncols=2)
    axes[1].legend(loc='upper left', frameon=True, fontsize=10)
    axes[2].legend(loc='upper left', frameon=True, fontsize=10)

    for i, axi in enumerate(axes.flatten()):
        panel_label = chr(ord('`') + i + 1)
        axi.text(
            -0.175, 1.1, f'({panel_label})', transform=axi.transAxes, fontsize=12, fontweight='bold',
            va='top', ha='right'
        )

    file_tag = 'fig_ir_temperature_vs_t'
    fig.savefig(file_tag + '.png', dpi=600)
    fig.savefig(file_tag + '.pdf', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()