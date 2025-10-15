import h5py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from pybaselines import Baseline
import numpy as np
from scipy.integrate import simpson
from scipy import ndimage
from scipy.interpolate import make_smoothing_spline
import matplotlib.animation as animation


SHOT = 203784
WL_SUM_RANGE = [4324, 4329] # < BD
# WL_SUM_RANGE = [8194, 8196] # < B-I
# WL_SUM_RANGE = [4120, 4123] # < B-II
PATH_TO_DATA = Path("data")
LINE_LABEL = 'BD'
# LINE_LABEL = 'B-I'
# LINE_LABEL = 'B-II'
DIAMETER_MDS_SPOT = 2.3 # mds spectrometer chord L5 spot diameter (cm), aimed at DiMES
DIAMETER_DIMES_HEAD = 4.78 # diameter of DiMES head (cm)
R_DIMES = 1.485 #  DiMES major radius (m)
LP_FOLDER = r'../Langmuir Probe/data/dimes_lp'
PIXEL_WIDTH_UM = 12.8
TIME_MAX = 5 # s

def x_rate(T_e: np.ndarray) -> np.ndarray:
    """
    Estimates the excitation rate coefficient from the
    ground state of B-H for the transition:

    .. math::\Chi^1 \Sigma^+ \to \mathrm{A}^1\Pi

    as a function of the electron temperature.

    This relationship corresponds to the modified Arrhenius function
    .. math:: k = A T_e^n\exp\left(-\frac{E_{\mathrm{act}}{T_e}\right)

    described in Kawate et al. Plasma Sources Sci. Technol. 32, 085006 (2023)
    doi: 10.1088/1361-6595/acec0c


    Parameters
    ----------
    T_e: np.ndarray
        The electron temperature in eV

    Returns
    -------
    np.ndarray:
        The excitation rate coefficient in cm^3/s

    """
    return 5.62E-8 * np.power(T_e, 0.021) * np.exp(-3.06 / T_e)

def s_rate(T_e: np.ndarray) -> np.ndarray:
    """
        Estimates the ionization rate coefficient from the
        ground state of B-H for the transition:

        .. math::\Chi^1 \Sigma^+ \to \mathrm{A}^1\Pi

        as a function of the electron temperature.

        This relationship corresponds to the modified Arrhenius function
        .. math:: k = A T_e^n\exp\left(-\frac{E_{\mathrm{act}}{T_e}\right)

        described in Kawate et al. Plasma Sources Sci. Technol. 32, 085006 (2023)
        doi: 10.1088/1361-6595/acec0c


        Parameters
        ----------
        T_e: np.ndarray
            The electron temperature in eV

        Returns
        -------
        np.ndarray:
            The ionization rate coefficient in cm^3/s

        """
    return 1.46E-8 * np.power(T_e, 0.690) * np.exp(-9.38 / T_e)

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
    return t_s, T_eV

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


def main(shot, wl_sum_range, path_to_data, line_label, diameter_mds_spot, time_max):
    path_to_h5 = Path(path_to_data) / f'{shot}_mdspec.h5'
    spot_area = 0.25 * (diameter_mds_spot ** 2)

    # Create a folder (if it does not exist) to save the generated figures
    path_to_figures = Path("figures")
    path_to_figures.mkdir(parents=True, exist_ok=True)

    t_lp, T_e = load_lp_data(shot)
    # There is an unphysical peak at t<0.1 s. Replace with T_e with value at t=0.1 for t < 0.1
    idx_0 = np.argmin(np.abs(t_lp - 0.1))
    T_e[0:idx_0] = T_e[idx_0]
    # There is an unphysical peak at t>4.631 s. Replace with T_e with value at t=0.1 for t < 0.1
    idx_1 = np.argmin(np.abs(T_e[idx_0:] - 4.631))
    T_e[idx_1:] = T_e[idx_1]
    T_eV_despiked, _ = remove_spikes_zscore(spectrum=T_e, threshold=1, window_size=50)
    spl_TeV = make_smoothing_spline(x=t_lp, y=T_eV_despiked, lam=None)
    T_eV_smooth = spl_TeV(t_lp)

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
        line_of_sight = f['/intensities/L5'].attrs['line_of_sight_m']


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
    flux_bd = np.zeros_like(time_s)
    sxb_bd = np.zeros_like(time_s)
    wl_range = wavelength[msk_wl_sum]
    line_brightness_max = 0
    idx_max = 0
    for i in range(len(time_s)):
        bi = intensities[i] # photons/s/pixel/cm^2/ster
        temp_e = spl_TeV(time_s[i])
        k_x = x_rate(temp_e)
        k_s = s_rate(temp_e)
        branching_ratio = 1.
        sxb = k_s / k_x / branching_ratio
        sxb_bd[i] = sxb
        # print(f't: {time_s[i]:.3f}, T_e: {temp_e:.0f} eV, k_x: {k_x:.3E}, k_s: {k_s:.3E}, sxb: {sxb:.3E}')
        # Estimate the baseline
        baseline_fitter_i = Baseline()
        bgd_i, params_i = baseline_fitter_i.arpls(bi, lam=1e6)
        # Remove baseline and integrate
        line_brightness[i] = simpson(y=(bi[msk_wl_sum] - bgd_i[msk_wl_sum]), x=wavelength[msk_wl_sum]) # photons/s/ster/cm^2
        flux_bd[i] = 4 * np.pi * line_brightness[i] * sxb * spot_area
        if line_brightness[i] > line_brightness_max:
            line_brightness_max = line_brightness[i]
            idx_max = i


    # line_brightness *= spot_area # photons / s / ster
    b_max = intensities[idx_max]
    baseline_fitter_max = Baseline()
    bgd_max, params_max = baseline_fitter_max.arpls(b_max, lam=1e6)



    line_data, = ax_animation.plot(wavelength, intensities[0], label='Data', color='C0')
    line_baseline, = ax_animation.plot(wavelength, bgd_max, ls='--', color='r', lw=1.2, label='Baseline')
    ax_animation.fill_between(
        wavelength[msk_wl_sum], bgd_max[msk_wl_sum], b_max[msk_wl_sum], color='C0', ls='None',
        alpha=0.3
    )
    ax_animation.set_xlim(4322, 4330)
    ax_animation.set_ylim(top=b_max[msk_wl_sum].max()*1.2, bottom=0)  # <- BD range
    ax_animation.set_xlabel('$\lambda$ {\sffamily (\AA)}', usetex=True)
    ax_animation.set_ylabel(r'$I_{\lambda}$ {\sffamily (photons/s/pixel/cm\textsuperscript{2}/ster)}', usetex=True)

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
    path_to_ani = path_to_figures / f'{shot}_BD_band_animation.mp4'
    ani.save(filename=path_to_ani, dpi=100, writer='ffmpeg')

    ax1.plot(wavelength, b_max, label='Data', color='C0')
    ax1.set_title('B-D band')
    ax1.plot(wavelength, bgd_max, ls='--', color='r', lw=1.2, label='Baseline')
    ax1.fill_between(wavelength[msk_wl_sum], bgd_max[msk_wl_sum], b_max[msk_wl_sum], color='C0', ls='None',
                     alpha=0.3)
    ax1.set_xlim(4322, 4330) # <- BD range
    # ax1.set_xlim(8175, 8240)  # <- B-I range
    # ax1.set_xlim(4113, 4130)  # <- B-II range
    ax1.set_ylim(top=b_max[msk_wl_sum].max()*1.2, bottom=0)  # <- BD range
    ax1.set_xlabel('$\lambda$ {\sffamily (\AA)}', usetex=True)
    ax1.set_ylabel(r'$B(\lambda, t_{\mathrm{m}})$ {\sffamily (photons/s/pixel)}', usetex=True)
    ax1.legend(loc='upper left', fontsize='9', frameon=True)



    ax2.plot(time_s, line_brightness)
    ax2.set_xlabel('$t$ {\sffamily (s)}', usetex=True)
    ax2.set_ylabel(r'$I_{\mathrm{BD}}$ {\sffamily (photons/cm\textsuperscript{2}/s/ster)}', usetex=True)
    ax2.set_xlim(0, 4.5)
    # ax3.set_ylim(0, 7E3) # <- BD range

    ax3.plot(t_lp, T_e, color='C0', alpha=0.25, label='Probe')
    # ax3.plot(t_lp, T_eV_despiked, color='C1', alpha=0.5, label='De-spiked')
    p1, = ax3.plot(time_s, spl_TeV(time_s), color='C0', label='Smoothed')
    ax3.set_xlabel('$t$ {\sffamily (s)}', usetex=True)
    ax3.set_ylabel(r'$T_e$ {\sffamily (eV)}', usetex=True)
    ax3.set_xlim(0, 4.5)
    ax3.set_ylim(bottom=0, top=100)

    gamma_lbl = r'\begin{equation}\Gamma = 4\pi A_{\mathrm{spot}} \frac{S}{XB} I_{\mathrm{BD}}\end{equation}'
    ax4.plot(time_s, flux_bd, label=gamma_lbl)
    ax4.set_xlabel('$t$ {\sffamily (s)}', usetex=True)
    ax4.set_ylabel(r'$\Gamma_{\mathrm{BD}}$ {\sffamily (molecules/s)}', usetex=True)
    ax4.set_xlim(0, 4.5)
    # legend_4 = ax4.legend(loc='upper left', fontsize='9', frameon=True)
    # for text in legend_4.get_texts():
    #     text.set_usetex(True)


    ax4.text(
        0.025, 0.975, r'$\displaystyle \Gamma = 4\pi A_{\mathrm{spot}} \frac{S}{XB} I_{\mathrm{BD}}$',
        transform=ax4.transAxes, ha='left', va='top', fontsize='9',
        color='black', usetex=True
    )

    ax3_twin = ax3.twinx()
    p2, = ax3_twin.plot(time_s, sxb_bd, color='C1', label='SX/B')
    ax3_twin.set_ylabel('SX/B')
    ax3.yaxis.label.set_color(p1.get_color())
    ax3_twin.yaxis.label.set_color(p2.get_color())

    ax3.tick_params(axis='y', colors=p1.get_color())
    ax3_twin.tick_params(axis='y', colors=p2.get_color())
    ax3_twin.set_ylim(bottom=0, top=np.ceil(np.max(sxb_bd)/5)*5)

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
    msk_t_gt_0 = time_s >= 0. # Only positive time
    line_brightness_df = pd.DataFrame(data={
        'time (s)': time_s[msk_t_gt_0],
        'Line brightness (photons/cm^2/ster/s)': line_brightness[msk_t_gt_0],
        'SX/B': sxb_bd[msk_t_gt_0],
        'Flux BD (molecules/s)': flux_bd[msk_t_gt_0],
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
    path_to_figure = path_to_figures / f'{shot}_{line_label}_flux.png'
    fig.savefig(path_to_figure, dpi=600)

    plt.show()


if __name__ == '__main__':
    main(
        shot=SHOT, wl_sum_range=WL_SUM_RANGE, path_to_data=PATH_TO_DATA, line_label=LINE_LABEL,
        diameter_mds_spot=DIAMETER_MDS_SPOT, time_max=TIME_MAX,
    )
