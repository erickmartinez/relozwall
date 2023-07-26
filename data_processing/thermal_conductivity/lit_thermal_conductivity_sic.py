import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
import os
from scipy.interpolate import interp1d
from data_processing.utils import latex_float

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\thermal_conductivity\literature'
target_temperature_k = 300.0  # K

database = [
    {
        'material': 'SiC - Slack 1964 (R52)',
        'dtf_csv': 'Slack - JAP 1964 - Thermal conductivity of pure and impure Si, SiC and diamond_R54.csv',
        'find_target': True,
        'conductivity_units': 'W/cm-K'
    },
]


def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


def main():
    load_plt_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 5.5)

    markers = ['^', 'v', '<', '>', 's', 's', 'o']
    fs = ['full', 'full', 'full', 'full', 'full', 'none', 'none']
    colors = ['k', 'k', 'tab:green', 'tab:green', 'tab:blue', 'tab:orange', 'tab:orange']
    for i, d in enumerate(database):
        m = d['material']
        csv = os.path.join(base_dir, d['dtf_csv'])
        input_columns = ['Temperature (K)', 'Thermal conductivity (W/m-K)']
        if d['conductivity_units'] == 'W/cm-K':
            input_columns = ['Temperature (K)', 'Thermal conductivity (W/cm-K)']
        df = pd.read_csv(csv, comment='#', header=None, names=input_columns)
        df = df.apply(pd.to_numeric)
        df.sort_values(by=['Temperature (K)']).reset_index(drop=True)
        temperature_k = df['Temperature (K)'].values
        thermal_conductivity = df[input_columns[1]].values
        if d['conductivity_units'] == 'W/m-K':
            thermal_conductivity *= 0.01
        f = interp1d(temperature_k, thermal_conductivity, bounds_error=False, fill_value='extrapolate')
        temperature_interp = np.linspace(0, 500, 50)
        tc_wcmk_at_target_temp = f(target_temperature_k)
        tc_wcmk_interp = f(temperature_interp)
        out_df = pd.DataFrame(data={
            'Temperature (K)': temperature_interp,
            'Thermal conductivity - interpolated (W/cm-K)': tc_wcmk_interp
        })
        out_df.to_csv(
            path_or_buf=os.path.join(base_dir, os.path.splitext(d['dtf_csv'])[0] + '_interp.csv'), index=False
        )
        lbl = f'{m}, ' + f'$K_{{ {target_temperature_k:.0f}\\;\\mathrm{{K}} }} ' \
                         f'= {latex_float(tc_wcmk_at_target_temp, significant_digits=4)}$'
        ax.plot(
            temperature_k,
            thermal_conductivity,
            marker=markers[i], fillstyle=fs[i],
            color=colors[i],
            label=lbl
        )

    ax.set_yscale('log')
    ax.set_xlabel('Temperature (K)')
    ax.set_xlim(-50, 550)
    ax.set_ylim(0.0001, 1E4)
    ax.set_ylabel('Thermal conductivity (W/cm-K)')

    ax.text(
        0.05, 0.95, f'$\\alpha_{{\mathrm{{300~K}}}} = {f(300):.2f}$ W/cm-K',
        horizontalalignment='left',
        verticalalignment='top',
        color='k',
        transform=ax.transAxes,
        fontsize=11
    )

    leg = ax.legend(
        loc='lower left', frameon=True,
        mode='expand',
        borderaxespad=0.,
        prop={'size': 8}, ncol=1,
        bbox_to_anchor=(0., 1.02, 1., .102)
    )
    plt.show()


if __name__ == '__main__':
    main()
