import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processing.misc_utils.plot_style import load_plot_style
from pathlib import Path
from pybaselines import Baseline, utils
from scipy.signal import find_peaks, peak_widths
from scipy import integrate

"""
Don't use this averaging procedure for dwell times >0.1 s !
"""


DATA_DIR = r'./data'
FIGURES_DIR = r'./figures/model'
MEAN_CURRENT_DIR = r'./data/mean_current'
INTEGRATED_CURRENT_DIR = r'./data/integrated_current'

SHOT = 203782
DiMES_R = 1.48 # m
T_RANGE = [1200, 4000]
TAU = 10

def rc_smooth(x: np.ndarray, y:np.ndarray, tau: float = 0.1) -> np.ndarray:
    """
    RC filter smoothing (simulates hardware RC low-pass filter)

    Parameters
    ----------
    x: np.ndarray
        Input signal x
    y: np.ndarray
        Input signal y
    tau: float
        Time constant

    Returns
    -------
    np.ndarray
        Filtered signal
    """

    # Average time step
    dt = np.mean(np.diff(x))

    alpha = dt / (tau + dt) # Smoothing factor

    y_smooth = np.zeros_like(y)
    y_smooth[0] = y[0]

    for i in range(1, len(x)):
        y_smooth[i] = alpha * y[i] + (1 - alpha) * y_smooth[i - 1]

    return y_smooth


def load_current_data(shot, data_dir=DATA_DIR):
    path_to_data = Path(data_dir) / f'{shot}_voltage_and_rvsout.csv'
    df = pd.read_csv(path_to_data).apply(pd.to_numeric, errors='coerce')
    return df

def find_index_intersections(x, y, y_ref, y_tolerance=1E-6, x_eps=None):
    if x_eps is None:
        x_spacing = np.mean(np.diff(x))
        x_eps = 2 * x_spacing

    idx_peaks = np.argwhere(np.abs(y - y_ref) < y_tolerance)

    if idx_peaks.size == 0:
        return []

    x_peaks = x[idx_peaks]

    # Group consecutive nearby points
    averaged_points = []
    current_group = [x_peaks[0]]

    # print(f'x_eps: {x_eps}')

    # print(f'i: 0, averaged_points {averaged_points}, current group {current_group}')

    for i in range(1, len(x_peaks)):
        if (x_peaks[i] - x_peaks[i - 1]) <= x_eps:
            # print(f'i: {i}, x_peaks[i] {x_peaks[i]}, x_peaks[i-1] {x_peaks[i-1]}, diff {x_peaks[i] - x_peaks[i - 1]}, current group {current_group}')
            current_group.append(x_peaks[i])
        else:
            # Finish current group and start new one
            averaged_points.append(np.mean(current_group))
            current_group = [x_peaks[i]]
            # print(f'i: {i}, averaged_points {averaged_points}, current group {current_group}')
    # Don't forget the last group
    averaged_points.append(np.mean(current_group))
    averaged_points = np.array(averaged_points)

    # print(averaged_points)

    idx_average =[np.argmax(np.abs(x - xi) <= x_eps) for xi in averaged_points]
    x_average = x[idx_average]

    return idx_average, x_average


def find_current_peaks(t_ms, signal, height_threshold=None, prominence_factor=0.1, width_threshold=None):
    """
    Find peaks in the baseline-corrected signal.

    Parameters:
    height_threshold: float, minimum peak height above baseline
    prominence_factor: float, prominence as fraction of signal range
    width_threshold: float, minimum peak width in time units
    """

    # Auto-determine thresholds if not provided
    if height_threshold is None:
        height_threshold = np.std(signal) * 2

    signal_range = np.max(signal) - np.min(signal)
    prominence = prominence_factor * signal_range

    # Find peaks
    peaks, properties = find_peaks(
        signal,
        height=height_threshold,
        prominence=prominence,
        distance=int(len(signal) * 0.01)  # Min 1% of data length apart
    )

    # Calculate peak widths
    widths, width_heights, left_ips, right_ips = peak_widths(
        signal, peaks, rel_height=0.92
        # signal, peaks, rel_height=0.98
    )

    # Convert width indices to time units
    time_widths = widths * (t_ms[1] - t_ms[0])  # Assuming uniform sampling

    # Filter by width if specified
    if width_threshold is not None:
        width_mask = time_widths >= width_threshold
        peaks = peaks[width_mask]
        widths = widths[width_mask]
        time_widths = time_widths[width_mask]
        left_ips = left_ips[width_mask]
        right_ips = right_ips[width_mask]

    # Store peak information
    peaks_info = {
        'indices': peaks,
        'heights': signal[peaks],
        'widths': widths,
        'time_widths': time_widths,
        'left_ips': left_ips,
        'right_ips': right_ips,
        'corrected_signal': signal
    }

    return peaks, peaks_info


def compute_charge(time, current, min_points_simpson=5):
    """
    Compute charge q = ∫I(t)dt from 0 to t using adaptive integration.

    Parameters:
    -----------
    time : array_like
        Time points (must be sorted)
    current : array_like
        Current values I(t) at each time point
    min_points_simpson : int, optional
        Minimum number of points required for Simpson's rule (default=5)

    Returns:
    --------
    charge : ndarray
        Cumulative charge at each time point
    """
    time = np.asarray(time)
    current = np.asarray(current)

    if len(time) != len(current):
        raise ValueError("Time and current arrays must have the same length")

    if len(time) < 2:
        raise ValueError("Need at least 2 data points for integration")

    # Initialize charge array
    charge = np.zeros_like(time)

    # For the first few points, use trapezoidal rule
    for i in range(1, min(min_points_simpson, len(time))):
        # Cumulative trapezoidal integration from start to current point
        charge[i] = integrate.trapezoid(current[:i + 1], time[:i + 1])

    # For remaining points with sufficient data, use Simpson's rule
    for i in range(min_points_simpson, len(time)):
        # Use Simpson's rule on the entire interval from start to current point
        # Simpson requires odd number of intervals (even number of points)
        n_points = i + 1

        if n_points % 2 == 1:  # Odd number of points (even intervals) - perfect for Simpson
            charge[i] = integrate.simpson(current[:n_points], time[:n_points])
        else:  # Even number of points - use Simpson on n-1 points + trapezoid for last interval
            charge[i] = (integrate.simpson(current[:n_points - 1], time[:n_points - 1]) +
                         integrate.trapezoid(current[n_points - 2:n_points], time[n_points - 2:n_points]))

    return charge


def compute_charge_uniform(time, current, min_points_simpson=5):
    """
    Optimized version for uniformly spaced time points.

    Parameters:
    -----------
    time : array_like
        Uniformly spaced time points
    current : array_like
        Current values I(t)
    min_points_simpson : int, optional
        Minimum points for Simpson's rule

    Returns:
    --------
    charge : ndarray
        Cumulative charge at each time point
    """
    time = np.asarray(time)
    current = np.asarray(current)

    # Check if time is uniformly spaced
    if len(time) > 2:
        dt = np.diff(time)
        if not np.allclose(dt, dt[0], rtol=1e-10):
            print("Warning: Time points not uniformly spaced. Consider using compute_charge() instead.")

    charge = np.zeros_like(time)

    # Trapezoidal rule for initial points
    for i in range(1, min(min_points_simpson, len(time))):
        charge[i] = integrate.trapezoid(current[:i + 1], dx=time[1] - time[0])

    # Simpson's rule for remaining points
    for i in range(min_points_simpson, len(time)):
        n_points = i + 1
        if n_points % 2 == 1:
            charge[i] = integrate.simpson(current[:n_points], dx=time[1] - time[0])
        else:
            charge[i] = (integrate.simpson(current[:n_points - 1], dx=time[1] - time[0]) +
                         integrate.trapezoid(current[n_points - 2:n_points], dx=time[1] - time[0]))

    return charge

def round_for_lim(value, factor):
    if value < 0:
        return np.floor(value * factor) / factor
    return np.ceil(value * factor) / factor

def main(shot, dimes_r, t_range, data_dir, fig_dir, mean_current_dir, integrated_current_dir, tau=5):
    current_df = load_current_data(shot)
    current_df = current_df[current_df['t_ms'].between(t_range[0], t_range[1])]
    t_ms = current_df['t_ms'].values
    current = current_df['current'].values
    current_rcsmooth = rc_smooth(t_ms, current, tau)
    rvsout = current_df['rvsout'].values


    idx_interections, t_intersections = find_index_intersections(x=t_ms, y=rvsout, y_ref=dimes_r, y_tolerance=1E-5)
    rvsout_inter = rvsout[idx_interections]

    # Get a baseline to the data
    baseline_fitter = Baseline(x_data=t_ms)
    bkgd_1, params_1 = baseline_fitter.snip(
        current_rcsmooth, max_half_window=600, decreasing=True, smooth_half_window=300
    )
    current_baselined = current_rcsmooth - bkgd_1

    current_peaks, current_peaks_info = find_current_peaks(
        t_ms, current_baselined, height_threshold=0.2, prominence_factor=0.25, width_threshold=None
    )


    load_plot_style()
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, constrained_layout=True)
    fig.set_size_inches((4.5, 6))

    ax1.plot(t_ms, rvsout, color='C0', label='Signal')
    ax1.plot(t_intersections, rvsout_inter, marker='o', color='k', ls='none', ms=7, mfc='none', mew=1.5)
    ax1.axhline(y=dimes_r, color='tab:red', ls='dashed', linewidth=1, label='DiMES R')


    ax1.set_ylabel('RVSOUT (A)')

    ax1.set_title(f'Shot #{shot:.0f}')
    ax1.set_xlim(t_range[0], t_range[1])
    ax1.set_ylim(1.35, 1.6)

    ax1.legend(loc='upper left')

    ax2.plot(t_ms, current_rcsmooth, color='C0', label='Smoothed')
    ax2.plot(t_ms, bkgd_1, color='tab:red', label='Background', ls='dashed', lw=1)
    ax2.set_ylabel('Current (A)')
    ax2.legend(loc='upper right', frameon=True)

    ax3.plot(t_ms, current_baselined, color='C0', label='Baselined')

    peak_mean_currents_df = pd.DataFrame(columns=[
        'Peak', 'Peak left (ms)', 'Peak time (ms)', 'Peak right (ms)', 'Peak current (A)', 'Mean current (A)', 'Charge (C)'
    ])
    peak_colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    time_integrated = np.array([])
    current_integrated = np.array([])
    t0, q0 = 0, 0
    for i, peak in enumerate(current_peaks):
        t_peak = t_ms[peak]
        current_peak = current_baselined[peak]
        peak_width = current_peaks_info['time_widths'][i]
        x_l, x_r = t_peak - 0.5*peak_width, t_peak + 0.5*peak_width
        # print(f't_peak = {t_peak}, peak_width = {peak_width}, x_l = {x_l}, x_r = {x_r}')
        ax3.plot(t_peak, current_peak, color='r', ls='none', mfc='none', mew=1.25, marker='o', ms=6)
        ax3.text(
            t_peak, current_peak+0.2, f'{i+1}', ha='center', va='bottom', color=peak_colors[i],
        )

        msk_range = (x_l <= t_ms) & (t_ms <= x_r) & (current_baselined >= 0)
        t_peak_range = t_ms[msk_range]
        c_peak_range = current_baselined[msk_range]
        mean_current = np.mean(c_peak_range)
        total_charge = integrate.simpson(y=c_peak_range, x=t_peak_range) * 1E-3

        t_integrand =  (t_peak_range - x_l) * 1E-3
        t_integrand -= t_integrand.min()
        t_integrand += t0
        q_integrated = compute_charge(time=t_integrand, current=c_peak_range) + q0
        time_integrated = np.hstack([time_integrated, t_integrand])
        current_integrated = np.hstack([current_integrated, q_integrated])
        t0 = t_integrand.max()
        q0 = q_integrated.max()


        new_df = pd.DataFrame(data={
                'Peak': [i+1],
                'Peak left (ms)': [x_l],
                'Peak time (ms)': [t_peak],
                'Peak right (ms)': [x_r],
                'Peak current (A)': [current_peak],
                'Mean current (A)': [mean_current],
                'Charge (C)': [total_charge],

            })

        if len(peak_mean_currents_df) > 0:
            peak_mean_currents_df = pd.concat([
                peak_mean_currents_df,
                new_df
            ])
        else:
            peak_mean_currents_df = new_df

        ax3.fill_between(
            t_peak_range, y1=0, y2=c_peak_range, color=peak_colors[i], ls='dashed', lw=1, alpha=0.5,
        )

        peak_txt = rf'$\langle I_{{{i+1}}} \rangle = {mean_current:.2f}$ A'
        ax3.text(
            0.025, 0.95-i*0.125, peak_txt, ha='left', va='top', color=peak_colors[i],
            transform=ax3.transAxes, usetex=True
        )


    q_df = pd.DataFrame(data={
        't (s)': time_integrated,
        'q (A)': current_integrated,
    })

    ax3.set_ylabel('Current (A)')
    ax3.legend(loc='upper right', frameon=True)
    ax3.set_ylim(top=current_baselined.max()*1.25)


    path_to_figures = Path(fig_dir)
    path_to_figures.mkdir(parents=True, exist_ok=True)
    path_to_baselined_dir = Path(data_dir) / 'baselined'
    path_to_baselined_dir.mkdir(parents=True, exist_ok=True)
    path_to_baselined_current = path_to_baselined_dir / f'{shot}_baselined_current.csv'
    path_to_mean_current_dir = Path(mean_current_dir)
    path_to_mean_current_dir.mkdir(parents=True, exist_ok=True)
    path_to_mean_current_csv = path_to_mean_current_dir / f'{shot}_mean_current.csv'
    peak_mean_currents_df.to_csv(path_to_mean_current_csv, index=False)
    current_basedlined_df = pd.DataFrame(data={'t (ms)': t_ms, 'current (A)': current_baselined})
    current_basedlined_df.to_csv(path_to_baselined_current, index=False)
    path_to_q_dir = Path(integrated_current_dir)
    path_to_q_dir.mkdir(parents=True, exist_ok=True)

    q_df.to_csv(path_to_q_dir / f'{shot}_integrated_current.csv', index=False)

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel('Time (ms)')
        for t_inter in t_intersections:
            ax.axvline(t_inter, color='k', ls='--', lw=1, alpha=0.5)

    fig.savefig(path_to_figures / f'{shot}_mean_current.png', dpi=600)

    fig_c, ax_c = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig_c.set_size_inches(4.5, 3.)

    ax_c.plot(time_integrated, current_integrated*1E3, color='C0')
    ax_c.set_xlabel('Time (s)')
    ax_c.set_ylabel('Charge (mC)')
    ax_c.set_title(f'Shot #{shot}')
    ax_c.set_xlim(round_for_lim(time_integrated.min(), factor=5), round_for_lim(time_integrated.max(), factor=5))
    ax_c.set_ylim(round_for_lim(current_integrated.min()*1E3, factor=5), round_for_lim(current_integrated.max()*1E3, factor=5))

    fig_c.savefig(path_to_figures / f'{shot}_integrated_current.png', dpi=600)



    plt.show()




if __name__ == '__main__':
    main(
        shot=SHOT, dimes_r=DiMES_R, t_range=T_RANGE, data_dir=DATA_DIR, fig_dir=FIGURES_DIR,
        mean_current_dir=MEAN_CURRENT_DIR, integrated_current_dir=INTEGRATED_CURRENT_DIR, tau=TAU
    )