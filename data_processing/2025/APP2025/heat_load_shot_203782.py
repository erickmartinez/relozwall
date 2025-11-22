import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks
from data_processing.misc_utils.plot_style import load_plot_style
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d, make_smoothing_spline
from pathlib import Path

SHOT = 203782
PATH_TO_LP_DATA = r'./data/dimes_lp'
T_RANGE = [0.6, 4.4]

def load_lp_data(shot, path_to_folder=PATH_TO_LP_DATA):
    path_to_folder = Path(path_to_folder)
    with h5py.File( path_to_folder / f'{shot}_LP.h5', 'r') as h5:
        dimes_gp = h5['/LANGMUIR_DIMES']
        t_s = np.array(dimes_gp.get('time')) * 1E-3
        qpara = np.array(dimes_gp.get('qpara'))
    return t_s, qpara

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

def main(shot, path_to_lp_data=PATH_TO_LP_DATA, t_range=T_RANGE):
    t_s, qpara = load_lp_data(shot, path_to_lp_data)
    # Only plot within the desired range
    msk_time = (t_range[0] <= t_s) & (t_s <= t_range[1])
    t_s = t_s[msk_time]
    qpara = qpara[msk_time]
    # Despike and mooth
    qpara_despiked, _ = remove_spikes_zscore(qpara, threshold=1.5, window_size=20)
    spl_qpara = make_smoothing_spline(t_s, qpara_despiked, lam=1E-6)

    # Find the peaks of qpara
    idx_peaks, _ = find_peaks(spl_qpara(t_s), prominence=10, distance=500)
    # If there are several peaks restrict to times around 2.1 and 3.1 to get only forward and back sweep strikes
    idx_peaks = np.array([idx_peak for idx_peak in idx_peaks if (2.1 <= t_s[idx_peak]) & (t_s[idx_peak] <= 3.1)])
    t_osp_hits = t_s[idx_peaks]
    osp_range = [t_osp_hits.min(), t_osp_hits.max()]


    load_plot_style(font='Times New Roman')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 2.25)

    for t_osp_hit in osp_range:
        ax.axvline(x=t_osp_hit, color='tab:red', ls='dashed', lw=1.)

    ax.plot(t_s, spl_qpara(t_s), label=f'Smooth', color='C0')


    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$q_{\parallel}$ (MW/m\textsuperscript{2})', usetex=True)
    ax.set_xlim(t_range[0], t_range[1])
    ax.set_ylim(bottom=0, top=80)

    ax.text(
        0.025, 0.975, f'#{shot}',
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=11, fontweight='bold'
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(r'./figures/heat_load_shot_203782.{}'.format(extension), dpi=600)


    plt.show()

if __name__ == '__main__':
    main(shot=SHOT, path_to_lp_data=PATH_TO_LP_DATA)

