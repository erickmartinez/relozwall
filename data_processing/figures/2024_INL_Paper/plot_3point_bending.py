import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
import os
import tkinter as tk
from tkinter import filedialog
from data_processing.utils import get_experiment_params
import platform

platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'

initial_folder = 'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/bending_tests/2023'
gc_bending_xls = 'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/bending_tests/glassy_carbon_3pbt_db.xlsx'


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
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


def get_file_list(base_path, extension='.csv'):
    files = []
    for f in os.listdir(base_path):
        if f.startswith('3PBT') and f.endswith('.csv'):
            files.append(f)
    return files


def main():
    global initial_folder, gc_bending_xls
    initial_folder = normalize_path(initial_folder)
    gc_bending_xls = normalize_path(gc_bending_xls)
    root = tk.Tk()
    root.withdraw()
    root.update()
    rel_path = filedialog.askdirectory(initialdir=initial_folder)
    file_list = get_file_list(base_path=rel_path, extension='.csv')
    gc_df = pd.read_excel(gc_bending_xls, sheet_name=0)
    gc_cols = gc_df.columns
    gc_df[gc_cols[1::]] = gc_df[gc_cols[1::]].apply(pd.to_numeric)
    load_plot_style()
    output_path = os.path.join(rel_path, 'processed_data')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for fn in file_list:
        file_path = os.path.join(rel_path, fn)

        filetag = os.path.splitext(os.path.basename(file_path))[0]
        params = get_experiment_params(relative_path=rel_path, filename=filetag, debug=False)
        diameter = float(params['Sample diameter']['value'])
        test_id = params['Sample name']['value']
        test_id = test_id.replace('-', ',')
        support_span = float(params['Support span']['value'])

        # Load the data
        df: pd.DataFrame = pd.read_csv(file_path, comment='#').apply(pd.to_numeric)
        df.dropna(inplace=True)
        df = df.sort_values(by=['Time (s)'])
        fmax = df['Force (N)'].max()
        print(f'F_peak: {fmax:>4.2f} N')

        # find the sample ID and test ID
        sp = test_id.split(',')
        sp_arr = [s.strip() for s in sp]
        sid, tid = sp_arr[0], int(sp_arr[1])

        print(f'Looking DB for Sample ID: \'{sid}\' and Test ID: {tid}')
        params_df = gc_df[gc_df['Test ID'] == tid]
        if len(params_df) == 0:
            continue
        stored_params = params_df.iloc[0]
        # print(stored_params)
        # print(stored_params['Diameter (mm)'])
        diameter_db = stored_params['Diameter (mm)']
        diameter_err_db = stored_params['Diameter err (mm)']
        support_span_db = stored_params['Support span (mm)']
        matrix_content = stored_params['Matrix wt %']

        if diameter_db != diameter:
            print('*** Discrepancy in sample diameter ***')
            print(f'Database: {diameter_db:>4.2f} mm, File: {diameter:>4.2f} mm')
            df['Displacement (mm)'] += diameter
            df['Displacement (mm)'] -= diameter_db
        if support_span_db != support_span:
            print('*** Discrepancy in sample diameter ***')
            print(f'Database: {support_span_db:>4.2f} mm, File: {support_span:>4.2f} mm')

        d_df = df[df['Displacement (mm)'] >= 0.]

        force = d_df['Force (N)'].values
        time_s = d_df['Time (s)'].values


        # Find the breaking load
        idx_peak = (np.abs(fmax - force)).argmin()
        t_peak = time_s[idx_peak]
        d_df = d_df[d_df['Time (s)'] <= t_peak]
        df = df[df['Time (s)'] <= t_peak * 1.5]
        d_df['Displacement (um)'] = np.round(d_df['Displacement (mm)'].values*40.)*1000./40.
        d_df = d_df.sort_values(by=['Displacement (um)'])

        df_agg = d_df.groupby('Displacement (um)').agg({
            'Force (N)': ['mean', 'count', 'max'],
            'Displacement (mm)': ['mean'],
            'Displacement err (mm)': ['max']
        })

        # print(df_agg[['Displacement (mm)', 'Force (N)']])

        force = df_agg['Force (N)']['max'].values
        displacement = np.round(df_agg['Displacement (mm)']['mean']*40)/40
        # displacement = np.round(df_agg.index, decimals=2)
        displacement_err = df_agg['Displacement err (mm)']['max'].values
        force_err = np.ones_like(force) * 0.5 #/ np.sqrt(df_agg['Force (N)']['count'].values)

        if len(displacement) < 2:
            continue



        sigma_kpa = 8.0 * force * support_span / np.pi / (diameter ** 3.0) * 1E3
        sigma_kpa_err = np.zeros_like(sigma_kpa)
        for i, s in enumerate(sigma_kpa):
            sigma_kpa_err[i] = np.linalg.norm([0.5 / max(force[i], 0.01), 0.002 / support_span_db, 3. * diameter_err_db / diameter_db])
        new_d = 9.18
        new_L = 40.00
        old_L = support_span_db
        old_d = diameter_db
        new_force = (old_L / new_L) * ((new_d / old_d) ** 3.) * force
        new_force_err = np.zeros(new_force.size)
        for i, fi in enumerate(force):
            if fi == 0.:
                new_force_err[i] = force_err[i] * new_force.mean()/force.mean()
            else:
                try:
                    new_force_err[i] = new_force[i]/fi * force_err[i]
                except Exception as e:
                    print(new_force_err, force, force_err)
                    raise e

        out_df = pd.DataFrame(data={
            'Deformation (mm)': displacement,
            'Measured force (N)': force,
            'Force err (N)': force_err,
            'Force 4cm (N)': new_force,
            'Force 4cm err (N)': new_force_err,
            'f_b (KPa)': sigma_kpa,
            'f_b err (KPa)': sigma_kpa_err
        })

        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw=dict(hspace=0, height_ratios=[1, 1]))
        fig.set_size_inches(4.5, 5.)

        fig_t, axes_t = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw=dict(hspace=0, height_ratios=[1, 1]))
        fig_t.set_size_inches(4.5, 5.)


        lbl1 = f'L = {old_L*0.1:.0f} cm'
        lbl2 = f'L = {new_L*0.1:.0f} cm'

        plot_title = f'{matrix_content:.0f} % matrix, test: {tid:03d}'

        axes[0].set_title(plot_title)
        axes_t[0].set_title(plot_title)

        axes[0].errorbar(
            displacement, force, yerr=force_err, xerr=0.05,
            marker='o', mfc='none', capsize=2.5,
            ms=9, mew=1.,
            elinewidth=1.0, lw=1.75, c='C0',
            label=lbl1
        )

        axes[1].errorbar(
            displacement, new_force, yerr=new_force_err, xerr=0.05,
            marker='o', mfc='none', capsize=2.5,
            ms=9, mew=1.,
            elinewidth=1.0, lw=1.75, c='C0',
            label=lbl2
        )

        axes_t[0].errorbar(
            df['Time (s)'], df['Force (N)'], #yerr=0.5,
            marker='o', mfc='none', capsize=2.5,
            ms=9, mew=1.,
            elinewidth=1.0, lw=1.75, c='C0',
            label=lbl1
        )

        axes_t[1].errorbar(
            df['Time (s)'], df['Displacement (mm)'], #yerr=0.05,
            marker='o', mfc='none', capsize=2.5,
            ms=9, mew=1.,
            elinewidth=1.0, lw=1.75, c='C0',
            label=lbl1
        )


        for ax in axes:
            # ax.set_xlim(0., 0.5)
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.02))
            # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
            ax.set_ylabel('Load (N)')
            ax.legend(loc='upper left', frameon=True)


        for ax in axes_t:
            ax.legend(loc='upper left', frameon=True)


        axes[1].set_xlabel('Deformation (mm)')

        axes_t[0].set_ylabel('Force (N)')
        axes_t[1].set_ylabel('Deformation (mm)')
        axes_t[1].set_xlabel('Time (s)')

        fig.tight_layout()
        fig_t.tight_layout()

        # save data

        out_df.to_csv(os.path.join(output_path, filetag + '_processed.csv'), index=False)
        fig.savefig(os.path.join(output_path, filetag + '_plot.png'), dpi=300)
        fig_t.savefig(os.path.join(output_path, filetag + '_raw_plot.png'), dpi=300)


    plt.show()


if __name__ == '__main__':
    main()
