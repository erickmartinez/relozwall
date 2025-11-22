import h5py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from boron_i_sxb_calculator import BoronISXBCalculator
from matplotlib import ticker
from pybaselines import Baseline
import numpy as np
from scipy.integrate import simpson
from scipy import ndimage
from scipy.interpolate import make_smoothing_spline
import matplotlib.animation as animation
from typing import Union



SHOT = 203781
# WL_SUM_RANGE = [4324, 4329] # < BD
WL_SUM_RANGE = [8194, 8196] # < B-I
# WL_SUM_RANGE = [8135, 8145] # < B-I
# WL_SUM_RANGE = [4119.5, 4123] # < B-II
PATH_TO_DATA = Path("data")
# LINE_LABEL = 'BD'
LINE_LABEL = 'B-I'
# LINE_LABEL = 'B-II'
DIAMETER_MDS_SPOT = 2.3 # mds spectrometer chord L5 spot diameter (cm), aimed at DiMES
DIAMETER_DIMES_HEAD = 4.78 # diameter of DiMES head (cm)
R_DIMES = 1.485 #  DiMES major radius (m)
LP_FOLDER = r'../Langmuir Probe/data/dimes_lp'
PIXEL_WIDTH_UM = 12.8
TIME_MAX = 5 # s

PATH_TO_PEC_KNOTS_H5 = r'./sxb/boron_pec_interpolator.h5' # <- Coefficients for interpolation of ADAS PEC
PATH_TO_S_KNOT_H5 = r'./sxb/boron_ionization_interpolator.h5' # <- Coefficients for interpolation of AFAS ionization

def sxb_814(T_e, pec_knot_file=PATH_TO_PEC_KNOTS_H5, ionization_file=PATH_TO_S_KNOT_H5, ne=1E13, calculator=None):
    if calculator is None:
        calculator = BoronISXBCalculator(pec_interpolator_knots_h5=pec_knot_file,
                                         ionization_interpolator_knots_h5=ionization_file)
    return calculator.calculate_sxb(T_e=T_e, ne=ne)

def update_frame(frame, line_data, line_background, text_time, ax, intensities, msk_wl_sum, ts):
    bi = intensities[frame] # photons/s/pixel/cm^2/ster
    # Estimate the baseline
    baseline_fitter_i = Baseline()
    bgd_i, params_i = baseline_fitter_i.arpls(bi, lam=1e6)
    wl = line_data.get_xdata()
    text_time.set_text(f't = {ts[frame]*1E3:>4.0f} ms')

    # Remove all collections (this clears fill_between objects)
    for collection in ax.collections[:]:  # Use slice to avoid modifying list while iterating
        collection.remove()

    line_data.set_ydata(bi)
    line_background.set_ydata(bgd_i)

    ax.fill_between(
        wl[msk_wl_sum], bgd_i[msk_wl_sum], bi[msk_wl_sum], color='C0', ls='None',
        alpha=0.3
    )

    return line_data, line_background



def load_lp_data(shot, path_to_folder=LP_FOLDER):
    path_to_folder = Path(path_to_folder)
    with h5py.File( path_to_folder / f'{shot}_LP.h5', 'r') as h5:
        dimes_gp = h5['/LANGMUIR_DIMES']
        t_s = np.array(dimes_gp.get('time')) * 1E-3
        T_eV = np.array(dimes_gp.get('TeV'))
        n_e = np.array(dimes_gp.get('ne'))*1E13
    return t_s, T_eV, n_e

def remove_spikes_zscore(spectrum, threshold=3, window_size=5):
    """
    Remove spikes using Z-score method with local statistics.

    Parameters:
    -----------
    spectrum : array-like
        Input spectrum/signal
    threshold : float, default=3
        Z-score threshold above which points are considered spikes
    window_size : int, default=5
        Size of the local window for calculating statistics

    Returns:
    --------
    cleaned_spectrum : ndarray
        Spectrum with spikes removed
    spike_mask : ndarray
        Boolean array indicating spike locations
    """
    spectrum = np.array(spectrum)
    cleaned_spectrum = spectrum.copy()

    # Calculate local median and MAD (Median Absolute Deviation)
    local_median = ndimage.median_filter(spectrum, size=window_size)
    mad = ndimage.median_filter(np.abs(spectrum - local_median), size=window_size)

    # Calculate modified Z-score using MAD
    with np.errstate(divide='ignore', invalid='ignore'):
        modified_z_score = 0.6745 * (spectrum - local_median) / mad

    # Identify spikes
    spike_mask = np.abs(modified_z_score) > threshold

    # Replace spikes with local median
    cleaned_spectrum[spike_mask] = local_median[spike_mask]

    return cleaned_spectrum, spike_mask


def main(shot, wl_sum_range, path_to_data, line_label, diameter_mds_spot, time_max, pec_knot_file=PATH_TO_PEC_KNOTS_H5, ionization_file=PATH_TO_S_KNOT_H5):
    path_to_h5 = Path(path_to_data) / f'{shot}_mdspec.h5'
    spot_area = 0.25 * np.pi * (diameter_mds_spot ** 2)

    calculator = BoronISXBCalculator(pec_interpolator_knots_h5=pec_knot_file,
                                     ionization_interpolator_knots_h5=ionization_file)

    # Create a folder (if it does not exist) to save the generated figures
    path_to_figures = Path("figures")
    path_to_figures.mkdir(parents=True, exist_ok=True)

    t_lp, T_e, n_e = load_lp_data(shot)

    # There is an unphysical peak at t<0.1 s. Replace with T_e with value at t=0.1 for t < 0.1
    idx_0 = np.argmin(np.abs(t_lp - 0.1))
    T_e[0:idx_0] = T_e[idx_0]
    n_e[0:idx_0] = n_e[idx_0]
    # There is an unphysical peak at t>4.631 s. Replace with T_e with value at t=0.1 for t < 0.1
    idx_1 = np.argmin(np.abs(t_lp[idx_0:] - 4.631))
    if shot == 203785:
        idx_1 = np.argmin(np.abs(t_lp[idx_0:] - 3.2239))
    T_e[idx_1:] = T_e[idx_1]
    n_e[idx_1:] = n_e[idx_1]
    T_eV_despiked, _ = remove_spikes_zscore(spectrum=T_e, threshold=1, window_size=50)
    n_e_despiked, _ = remove_spikes_zscore(spectrum=n_e, threshold=1, window_size=50)

    spl_TeV = make_smoothing_spline(x=t_lp, y=T_eV_despiked, lam=None)
    spl_n_e = make_smoothing_spline(x=t_lp[n_e_despiked > 0], y=n_e_despiked[n_e_despiked>0], lam=None)
    T_eV_smooth = spl_TeV(t_lp)

    plt.plot(t_lp, n_e, color='C0', label=line_label)
    plt.plot(t_lp, spl_n_e(t_lp), color='C1', label=line_label)
    plt.show()

    try:
        from data_processing.misc_utils.plot_style import load_plot_style
        load_plot_style()
    except Exception as e:
        print(e)

    # fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    fig.set_size_inches(7.0, 6)

    ax1, ax2, ax3, ax4 = axes.flatten()

    fig_animation, ax_animation = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig_animation.set_size_inches(4., 3.)
    # gs = GridSpec(nrows=2, ncols=2, figure=fig)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[1, :])

    """ Load the data from the HDF5 file """
    with h5py.File(path_to_h5, "r") as f:
        time_s  = np.array(f['/times'])
        wavelength = np.array(f['/wavelengths/L5']) # mds wavelength vector (A)
        intensities = np.array(f['/intensities/L5']) # mds intensity vector (photons/cm2/s/ster/pixel)

    msk_time = (0 <= time_s) & (time_s <= time_max)
    time_s = time_s[msk_time]
    intensities = intensities[msk_time, :]
    """ Create a mask for the line wavelength range """
    msk_wl_sum = (wl_sum_range[0] <= wavelength) & (wavelength <= wl_sum_range[1])

    # # The sum of the brightness B(λ, t) over all times
    # brightness_wl = np.sum(intensities, axis=0) * omega_pixel # mds intensity in photons/cm2/s
    # # Estimate a baseline using Asymmetric Reweighted Penalized Least Squares (arPLS)
    # baseline_fitter = Baseline()
    # bkgd_1, params_1 = baseline_fitter.arpls(brightness_wl, lam=1e6)

    # Estimate the line brightness as ∫ B(λ₁ ≤ λ ≤ λ₂, t) dλ
    line_brightness = np.zeros_like(time_s)
    flux_bi = np.zeros_like(time_s)
    sxb_bii = np.zeros_like(time_s)
    wl_range = wavelength[msk_wl_sum]
    line_brightness_max = 0
    idx_max = 0
    # plt.plot(wavelength[msk_wl_sum], intensities[np.argmin(np.abs(time_s - 1.5)), msk_wl_sum])
    # plt.show()
    for i in range(len(time_s)):
        bi = intensities[i] # photons/s/pixel/cm^2/ster
        temp_e = spl_TeV(time_s[i])
        n_e_i = spl_n_e(time_s[i])
        branching_ratio = 1.
        sxb = sxb_814(T_e=temp_e, ne=n_e_i, calculator=calculator)
        sxb_bii[i] = sxb
        # print(f't: {time_s[i]:.3f}, T_e: {temp_e:.0f} eV, k_x: {k_x:.3E}, k_s: {k_s:.3E}, sxb: {sxb:.3E}')
        # Estimate the baseline
        baseline_fitter_i = Baseline()
        bgd_i, params_i = baseline_fitter_i.arpls(bi, lam=1e6)
        # Remove baseline and integrate
        line_brightness[i] = simpson(y=(bi[msk_wl_sum] - bgd_i[msk_wl_sum])) # photons/s/ster/cm^2
        flux_bi[i] = 4 * np.pi * line_brightness[i] * sxb * spot_area
        if line_brightness[i] > line_brightness_max:
            line_brightness_max = line_brightness[i]
            idx_max = i


    b_max = intensities[idx_max]
    baseline_fitter_max = Baseline()
    bgd_max, params_max = baseline_fitter_max.arpls(b_max, lam=1e6)



    line_data, = ax_animation.plot(wavelength, intensities[0], label='Data', color='C0')
    line_baseline, = ax_animation.plot(wavelength, bgd_max, ls='--', color='r', lw=1.2, label='Baseline')
    ax_animation.fill_between(
        wavelength[msk_wl_sum], bgd_max[msk_wl_sum], b_max[msk_wl_sum], color='C0', ls='None',
        alpha=0.3
    )
    ax_animation.set_xlim(8190, 8198)
    ax_animation.set_xlabel('$\lambda$ {\sffamily (\AA)}', usetex=True)
    ax_animation.set_ylabel(r'$I_{\lambda}$ {\sffamily (photons/s/pixel/cm\textsuperscript{2}/ster)}', usetex=True)
    ax_animation.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax_animation.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax_animation.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    ax_animation.axvspan(xmin=wl_sum_range[0], xmax=wl_sum_range[1], facecolor='gray', alpha=0.25, label='Integration range')
    ax_animation.legend(loc='upper right', fontsize='9', frameon=True)

    clock_text = ax_animation.text(
        0.05, 0.95, f't = {time_s[0]*1000:>4.0f} ms', transform=ax_animation.transAxes,
        ha='left', va='top', fontsize='9', color='black'
    )

    ani = animation.FuncAnimation(
        fig=fig_animation, func=update_frame, frames=np.arange(0, len(time_s), 1), interval=30,
        fargs=(line_data, line_baseline, clock_text, ax_animation, intensities, msk_wl_sum, time_s)
    )
    path_to_ani = path_to_figures / f'{shot}_BII_band_animation.mp4'
    ani.save(filename=path_to_ani, dpi=100, writer='ffmpeg')

    ax1.plot(wavelength, b_max, label='Data', color='C0')
    ax1.set_title('BI band')
    ax1.plot(wavelength, bgd_max, ls='--', color='r', lw=1.2, label='Baseline')
    ax1.fill_between(wavelength[msk_wl_sum], bgd_max[msk_wl_sum], b_max[msk_wl_sum], color='C0', ls='None',
                     alpha=0.3)

    ax1.set_xlim(8175, 8240)  # <- B-I range
    ax1.set_xlabel('$\lambda$ {\sffamily (\AA)}', usetex=True)
    ax1.set_ylabel(r'$B(\lambda, t_{\mathrm{m}})$ {\sffamily (photons/s/pixel)}', usetex=True)
    ax1.legend(loc='upper left', fontsize='9', frameon=True)



    ax2.plot(time_s, line_brightness*4*np.pi)
    ax2.set_xlabel('$t$ {\sffamily (s)}', usetex=True)
    ax2.set_ylabel(r'$I_{\mathrm{BI}}$ {\sffamily (photons/cm\textsuperscript{2}/s)}', usetex=True)
    ax2.set_xlim(0, 4.5)

    ax3.plot(t_lp, T_e, color='C0', alpha=0.25, label='Probe')
    p1, = ax3.plot(time_s, spl_TeV(time_s), color='C0', label='Smoothed')
    ax3.set_xlabel('$t$ {\sffamily (s)}', usetex=True)
    ax3.set_ylabel(r'$T_e$ {\sffamily (eV)}', usetex=True)
    ax3.set_xlim(0, 4.5)
    ax3.set_ylim(bottom=0, top=100)
    if shot == 203785:
        ax3.set_ylim(bottom=0, top=140)

    gamma_lbl = r'\begin{equation}\Gamma = 4\pi A_{\mathrm{spot}} \frac{S}{XB} I_{\mathrm{BI}}\end{equation}'
    ax4.plot(time_s, flux_bi, label=gamma_lbl)
    ax4.set_xlabel('$t$ {\sffamily (s)}', usetex=True)
    ax4.set_ylabel(r'$\Gamma_{\mathrm{BI}}$ {\sffamily (molecules/s)}', usetex=True)
    ax4.set_xlim(0, 4.5)

    ax4.text(
        0.025, 0.975, r'$\displaystyle \Gamma = 4\pi A_{\mathrm{spot}} \frac{S}{XB} I_{\mathrm{BII}}$',
        transform=ax4.transAxes, ha='left', va='top', fontsize='9',
        color='black', usetex=True
    )


    ax3_twin = ax3.twinx()
    p2, = ax3_twin.plot(time_s, sxb_bii, color='C1', label='SX/B')
    ax3_twin.set_ylabel('SX/B')
    ax3.yaxis.label.set_color(p1.get_color())
    ax3_twin.yaxis.label.set_color(p2.get_color())

    ax3.tick_params(axis='y', colors=p1.get_color())
    ax3_twin.tick_params(axis='y', colors=p2.get_color())
    ax3_twin.set_ylim(bottom=0, top=np.ceil(np.max(sxb_bii)/5)*5)

    ax3.legend(loc='upper right', fontsize='9', frameon=True, handles=[p1, p2])


    fig.suptitle(f'Shot #{shot}')

    for i, ax in enumerate(axes.flatten()):
        if i != 2:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        panel_label = chr(ord('`') + i + 1)  # starts from a
        ax.text(
            -0.125, 1.15, f'({panel_label})', transform=ax.transAxes, fontsize=12, fontweight='bold',
            va='top', ha='right'
        )

    # Save the data
    msk_t_gt_0 = time_s >= 0.  # Only positive time
    emission_flux_df = pd.DataFrame(data={
        'time (s)': time_s[msk_t_gt_0],
        'Line brightness (photons/cm^2/ster/s)': line_brightness[msk_t_gt_0],
        'SX/B': sxb_bii[msk_t_gt_0],
        'Flux BI (molecules/s)': flux_bi[msk_t_gt_0],
    })
    path_to_data_out = path_to_data / 'emission_flux'
    path_to_data_out.mkdir(parents=True, exist_ok=True)
    path_to_csv_out = path_to_data_out / f'{shot}_emission_flux_{line_label}.csv'
    with open(path_to_csv_out, 'w') as f:
        f.write("# " + "*" * 20 + "\n")
        f.write(f"# Shot #{shot}\n")
        f.write(f"# {line_label} line brightness\n")
        f.write(f"# Wavelength range (Å): [{wl_sum_range[0]}, {wl_sum_range[1]}] \n")
        f.write("# " + "*" * 20 + "\n")
        emission_flux_df.to_csv(f, index=False)

    # Save the figure
    path_to_figure = path_to_figures / f'{shot}_{line_label}_flux.png'
    fig.savefig(path_to_figure, dpi=600)

    plt.show()


if __name__ == '__main__':
    main(
        shot=SHOT, wl_sum_range=WL_SUM_RANGE, path_to_data=PATH_TO_DATA, line_label=LINE_LABEL,
        diameter_mds_spot=DIAMETER_MDS_SPOT, time_max=TIME_MAX,
    )
