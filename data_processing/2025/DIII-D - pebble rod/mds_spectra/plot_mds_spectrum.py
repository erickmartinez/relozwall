import h5py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from pybaselines import Baseline
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.integrate import simpson


SHOT = 203780
# WL_SUM_RANGE = [4324, 4329] # < BD
WL_SUM_RANGE = [8194, 8196] # < B-I
# WL_SUM_RANGE = [4120, 4123] # < B-II
PATH_TO_DATA = Path("data")
# LINE_LABEL = 'BD'
LINE_LABEL = 'B-I'
# LINE_LABEL = 'B-II'

def main(shot, wl_sum_range, path_to_data, line_label):
    path_to_h5 = Path(path_to_data) / f'{shot}_mdspec.h5'
    try:
        from data_processing.misc_utils.plot_style import load_plot_style
        load_plot_style()
    except Exception as e:
        print(e)

    # fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig = plt.figure(layout='constrained')
    fig.set_size_inches(6.5, 5.5)
    gs = GridSpec(nrows=2, ncols=2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    """ Load the data from the HDF5 file """
    with h5py.File(path_to_h5, "r") as f:
        time_s  = np.array(f['/times'])
        wavelength = np.array(f['/wavelengths/L5'])
        intensities = np.array(f['/intensities/L5'])


    """ Create a mask for the line wavelength range """
    msk_wl_sum = (wl_sum_range[0] <= wavelength) & (wavelength <= wl_sum_range[1])

    # The sum of the brightness B(λ, t) over all times
    brightness_wl = np.sum(intensities, axis=0)
    # Estimate a baseline using Asymmetric Reweighted Penalized Least Squares (arPLS)
    baseline_fitter = Baseline()
    bkgd_1, params_1 = baseline_fitter.arpls(brightness_wl, lam=1e6)

    # Estimate the line brightness as ∫ B(λ₁ ≤ λ ≤ λ₂, t) dλ
    line_brightness = np.zeros_like(time_s)
    wl_range = wavelength[msk_wl_sum]
    for i in range(len(time_s)):
        bi = intensities[i]
        # Estimate the baseline
        baseline_fitter_i = Baseline()
        bgd_i, params_i = baseline_fitter_i.arpls(bi, lam=1e6)
        # Remove baseline and integrate
        line_brightness[i] = simpson(y=(bi[msk_wl_sum] - bgd_i[msk_wl_sum]))

    ax1.plot(wavelength, brightness_wl, label='Data')
    ax1.set_title('Survey')
    ax1.plot(wavelength, bkgd_1, ls='--', color='r', lw=1.2, label='Baseline')
    ax1.axvspan(xmin=wl_sum_range[0], xmax=wl_sum_range[1], color='gray', alpha=0.3, lw=0, label=f'{line_label} range')
    # ax1.set_xlim(4270, 4345) # <- BD range
    ax1.set_xlim(8175, 8240)  # <- B-I range
    # ax1.set_xlim(4113, 4130)  # <- B-II range
    # ax1.set_ylim(0, 2E6)  # <- BD range
    ax1.set_xlabel('$\lambda$ {\sffamily (\AA)}', usetex=True)
    ax1.set_ylabel('$\sum_{t} B_t(\lambda)$ {\sffamily (a.u.)}', usetex=True)
    ax1.legend(loc='upper left', fontsize='9', frameon=True)

    ax2.plot(wavelength[msk_wl_sum], brightness_wl[msk_wl_sum], label='Data')
    ax2.plot(wavelength[msk_wl_sum], bkgd_1[msk_wl_sum], ls='--', color='r', lw=1.25, label='Baseline')
    ax2.fill_between(wavelength[msk_wl_sum], bkgd_1[msk_wl_sum], brightness_wl[msk_wl_sum], color='C0', ls='None', alpha=0.3)
    ax2.set_xlim(WL_SUM_RANGE[0], WL_SUM_RANGE[1])
    ax2.set_xlabel('$\lambda$ {\sffamily (\AA)}', usetex=True)
    # ax2.set_ylabel('$\sum_{t} B_t(\lambda)$ {\sffamily (a.u.)}', usetex=True)
    ax2.set_title(line_label)
    ax2.legend(loc='upper right', fontsize='9', frameon=True)

    ax3.plot(time_s, line_brightness)
    ax3.set_xlabel('$t$ {\sffamily (s)}', usetex=True)
    ax3.set_ylabel(r'Line brightness (a.u.)', usetex=False)
    ax3.set_xlim(0, 4.5)
    # ax3.set_ylim(0, 7E3) # <- BD range
    line_brightness_txt = r'$\displaystyle\int_{\lambda_1}^{\lambda_2} B(\lambda, t) d\lambda$'
    ax3.text(
        0.025, 0.975, line_brightness_txt, transform=ax3.transAxes, fontsize=11,
        va='top', ha='left', usetex=True, color='k'
    )

    fig.suptitle(f'Shot #{shot}')

    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        panel_label = chr(ord('`') + i + 1)  # starts from a
        ax.text(
            -0.125, 1.15, f'({panel_label})', transform=ax.transAxes, fontsize=12, fontweight='bold',
            va='top', ha='right'
        )

    # Save the data
    msk_t_gt_0 = time_s >= 0. # Only positive time
    line_brightness_df = pd.DataFrame(data={
        'time (s)': time_s[msk_t_gt_0],
        'Line brightness (a.u.)': line_brightness[msk_t_gt_0],
    })
    path_to_data_out = path_to_data / 'line_brightness'
    path_to_data_out.mkdir(parents=True, exist_ok=True)
    path_to_csv_out = path_to_data_out / f'{shot}_line_brightness_{line_label}.csv'
    with open(path_to_csv_out, 'w') as f:
        f.write("# " + "*"*20 + "\n")
        f.write(line_brightness_df.to_csv(index=False))
        f.write(f"# Shot #{shot}\n")
        f.write(f"# {line_label} line brightness\n")
        f.write(f"# Wavelength range (Å): [{wl_sum_range[0]}, {wl_sum_range[1]}] \n")
        f.write("# " + "*" * 20 + "\n")
        line_brightness_df.to_csv(f, index=False)

    # Save the figure
    path_to_figures = Path("figures")
    path_to_figures.mkdir(parents=True, exist_ok=True)
    path_to_figure = path_to_figures / f'{shot}_line_brightness_{line_label}.png'
    fig.savefig(path_to_figure, dpi=600)

    plt.show()


if __name__ == '__main__':
    main(shot=SHOT, wl_sum_range=WL_SUM_RANGE, path_to_data=PATH_TO_DATA, line_label=LINE_LABEL)
