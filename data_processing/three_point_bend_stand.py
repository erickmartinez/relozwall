import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
from utils import get_experiment_params, latex_float
import pandas as pd

import utils

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\flexural_strength\displacement'
csv_file = '3PBT_R3N61-2_2022-07-28_1.csv'
sample_diameter_mm = 9.13
sample_diameter_std = 0.05
support_span_mm = 40.0
pot_pct_err = 2.2
force_gauge_pct_error = 0.2

if __name__ == '__main__':
    file_tag = os.path.splitext(csv_file)[0]
    params = get_experiment_params(
        relative_path=base_dir, filename=file_tag
    )
    sample_name = params['Sample Name']['value']
    bending_df = pd.read_csv(os.path.join(base_dir, csv_file), comment='#').apply(pd.to_numeric)
    bending_df = bending_df[bending_df['Displacement (mm)'] >= -sample_diameter_mm]
    bending_df.sort_values(by=['Displacement (mm)'], inplace=True)
    z = bending_df['Displacement (mm)'].values
    z_err = z * 0.01 * pot_pct_err
    displacement = z + sample_diameter_mm
    displacement_err = np.sqrt(z_err**2.0 + sample_diameter_std**2.0)
    force = bending_df['Force (N)'].values
    strain = 100.0 * displacement / sample_diameter_mm
    strain_err = strain * np.sqrt((displacement_err/displacement)**2.0 + (sample_diameter_std/sample_diameter_mm)**2.0)
    sigma_f = 8.0 * force * support_span_mm / np.pi / (sample_diameter_mm ** 3.0)
    sfe = (force_gauge_pct_error*0.01)**2.0 + (2.0/support_span_mm)**2.0 \
          + (sample_diameter_std/(3.0*sample_diameter_mm))**2.0
    sigma_f_err = sigma_f * np.sqrt(sfe)

    force_peak = force.max()
    idx_peak = (np.abs(force - force_peak)).argmin()
    force_peak_err = force_peak * 0.01 * force_gauge_pct_error
    displacement_peak = displacement[idx_peak]
    displacement_peak_err = displacement_err[idx_peak]
    strain_peak = strain[idx_peak]
    strain_peak_err = strain_err[idx_peak]
    sigma_f_peak = sigma_f[idx_peak]
    sigma_f_peak_err = sigma_f_err[idx_peak]

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1, nrows=2)  # , constrained_layout=True)
    fig.set_size_inches(5.0, 5.0)

    ax[0].errorbar(
        displacement, force, xerr=displacement_err, yerr=force*0.01*force_gauge_pct_error,
        capsize=2.5, mew=1.25, marker='o', ms=8, elinewidth=1.25,
        ls='-', c='C0', fillstyle='none'
    )

    ax[1].errorbar(
        strain, sigma_f, xerr=strain_err, yerr=sigma_f_err,
        capsize=2.5, mew=1.25, marker='s', ms=8, elinewidth=1.25,
        ls='-', c='C1', fillstyle='none'
    )

    ax[0].axvline(x=displacement_peak, ls='--', lw=1.0, c='k')
    ax[1].axvline(x=strain_peak, ls='--', lw=1.0, c='k')

    ax_0_txt = rf'$d_{{\mathrm{{peak}}}}={displacement_peak:.1f}\pm{displacement_peak_err:.2f}\;\mathrm{{mm}}$' + '\n' \
               + rf'$F_{{\mathrm{{peak}}}} = {force_peak:.1f}\pm{force_peak_err:.3f}\;\mathrm{{N}}$'

    ax_1_txt = rf'$\varepsilon_{{\mathrm{{peak}}}}={strain_peak:.1f}\pm{strain_peak_err:.2f}\;\%$' + '\n' \
               + rf'$\sigma_{{\mathrm{{peak}}}} = {latex_float(sigma_f_peak,2)}\pm{latex_float(sigma_f_peak_err,2)}\;\mathrm{{MPa}}$'

    ax[0].text(
        0.95, 0.95, ax_0_txt,
        transform=ax[0].transAxes,
        va='top', ha='right',
        fontsize=11,
        color='k'
    )

    ax[1].text(
        0.95, 0.95, ax_1_txt,
        transform=ax[1].transAxes,
        va='top', ha='right',
        fontsize=11,
        color='k'
    )

    ax[0].set_xlabel('Displacement (mm)')
    ax[0].set_ylabel('Force (N)')

    ax[1].set_xlabel('$\\varepsilon$ (%)')
    ax[1].set_ylabel('$\\sigma$ (MPa)')

    ax[0].set_xlim(left=0.0, right=3.0)
    ax[1].set_xlim(left=0.0, right=100.0 * 3.0 / sample_diameter_mm)

    ax[0].set_title(f'Sample: {sample_name}')

    fig.tight_layout()

    out_df = pd.DataFrame(data={
        'Displacement (mm)': displacement,
        'Displacement error (mm)': displacement_err,
        'Force (N)': force,
        'Force error (N)': 0.01 * force_gauge_pct_error * force,
        'Strain (%)': strain,
        'Strain error (%)': strain_err,
        'Stress (MPa)': sigma_f,
        'Stress error (MPa)': sigma_f_err
    })

    out_df.to_csv(os.path.join(base_dir, file_tag + '_processed.csv'), index=False)
    fig.savefig(os.path.join(base_dir, file_tag + '.png'), dpi=600)

    plt.show()