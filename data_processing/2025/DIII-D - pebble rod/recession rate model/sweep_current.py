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
FIGURES_DIR = r'./figures/sweep_range'
MEAN_CURRENT_DIR = r'./data/mean_current'
INTEGRATED_CURRENT_DIR = r'./data/integrated_sweep'

SHOT = 203781
DiMES_R = 1.48 # m
T_RANGE = [1500, 3500]
PLOT_T_RANGE = [1000, 4000]
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


def round_for_lim(value, factor):
    if value < 0:
        return np.floor(value * factor) / factor
    return np.ceil(value * factor) / factor

def main(shot, dimes_r, t_range, data_dir, fig_dir, integrated_current_dir, plot_t_range, tau=5):
    current_df = load_current_data(shot)
    # current_df = current_df[current_df['t_ms'].between(t_range[0], t_range[1])]
    t_ms = current_df['t_ms'].values
    current = current_df['current'].values

    rvsout = current_df['rvsout'].values
    # Find the mean of rvsout between 1500 and 2000 ms
    msk_rvsout = (1500 <= t_ms) & (t_ms <= 2000)
    bkg_rvsout = rvsout[msk_rvsout].mean()


    idx_interections, t_intersections = find_index_intersections(x=t_ms, y=rvsout, y_ref=dimes_r, y_tolerance=1E-5)
    rvsout_inter = rvsout[idx_interections]

    # Get a baseline to the data
    # Assume a flat baseline corresponding to the data from 0 to 600 ms
    msk_baseline = t_ms <= 600
    baseline_fitter = Baseline(x_data=t_ms[msk_baseline])
    # bkgd_1, params_1 = baseline_fitter.snip(
    #     current, max_half_window=3000, decreasing=True, smooth_half_window=300
    # )
    bkgd_1, params_1 = baseline_fitter.modpoly(current[msk_baseline], poly_order=1)


    current_rcsmooth = rc_smooth(t_ms, current, tau)
    current_baselined = current_rcsmooth - bkgd_1.mean()


    # Integrate the current between the range in t_range
    msk_integration = (t_range[0] <= t_ms) & (t_ms <= t_range[1])
    t_qrange = t_ms[msk_integration]
    t_qrange -= t_qrange.min()
    current_qrange = current_baselined[msk_integration]

    integrated_current = compute_charge(time=t_qrange, current=current_qrange, min_points_simpson=5)


    load_plot_style()
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, constrained_layout=True)
    fig.set_size_inches((4.5, 6))

    ax1.plot(t_ms, rvsout, color='C0', label='Signal')
    ax1.plot(t_intersections, rvsout_inter, marker='o', color='k', ls='none', ms=7, mfc='none', mew=1.5)
    ax1.axhline(y=dimes_r, color='tab:red', ls='dashed', linewidth=1, label='DiMES R')



    ax1.set_ylabel('RVSOUT (A)')

    ax1.set_title(f'Shot #{shot:.0f}')
    ax1.set_xlim(plot_t_range[0], plot_t_range[1])
    ax1.set_ylim(1.35, 1.6)

    ax1.legend(loc='upper left')

    ax2.plot(t_ms, current, color='C0', label='Unfiltered', alpha=0.5)
    ax2.plot(t_ms, current_rcsmooth, color='C0', label='Smoothed')
    ax2.plot([t_ms.min(), t_ms.max()], [bkgd_1.mean(), bkgd_1.mean()], color='tab:red', label='Background', ls='dashed', lw=1)
    ax2.set_ylabel('Current (A)')
    ax2.legend(loc='upper right', frameon=True)

    ax3.plot(t_ms, current_baselined, color='C0', label='Baselined')

    for ax in [ax1, ax2, ax3]:
        ax.axvspan(xmin=t_range[0], xmax=t_range[1], color='grey', alpha=0.2)



    q_df = pd.DataFrame(data={
        't (s)': t_qrange*1E-3,
        'q (C)': integrated_current,
    })

    ax3.set_ylabel('Current (A)')
    ax3.legend(loc='upper right', frameon=True)
    ax3.set_ylim(top=current_baselined.max()*1.25)
    ax3.text(
        t_range[0], ax3.get_ylim()[1], 'Integration range', color='k', ha='left', va='top', fontsize=9
    )

    path_to_figures = Path(fig_dir)
    path_to_figures.mkdir(parents=True, exist_ok=True)
    path_to_baselined_dir = Path(data_dir) / 'baselined_sweep'
    path_to_baselined_dir.mkdir(parents=True, exist_ok=True)
    path_to_baselined_current = path_to_baselined_dir / f'{shot}_baselined_current.csv'
    current_basedlined_df = pd.DataFrame(data={'t (ms)': t_ms, 'current (A)': current_baselined})
    current_basedlined_df.to_csv(path_to_baselined_current, index=False)
    path_to_q_dir = Path(integrated_current_dir)
    path_to_q_dir.mkdir(parents=True, exist_ok=True)

    with open(path_to_q_dir / f'{shot}_integrated_current.csv', 'w', encoding='utf-8') as f:
        f.write('#'*50 + '\n')
        f.write(f'# SHOT: {shot}\n')
        f.write(f'# T_RANGE_CHARGE: [{t_range[0]}, {t_range[1]}]\n')
        f.write('#' * 50 + '\n')
        q_df.to_csv(f, index=False)

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel('Time (ms)')
        for t_inter in t_intersections:
            ax.axvline(t_inter, color='k', ls='--', lw=1, alpha=0.5)

    fig.savefig(path_to_figures / f'{shot}_mean_current.png', dpi=600)
    #
    fig_c, ax_c = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig_c.set_size_inches(4.5, 3.)

    ax_c.plot(t_qrange, integrated_current, color='C0')
    ax_c.set_xlabel('Time (s)')
    ax_c.set_ylabel('Charge (C)')
    ax_c.set_title(f'Shot #{shot}')
    ax_c.set_xlim(round_for_lim(t_qrange.min(), factor=5), round_for_lim(t_qrange.max(), factor=5))
    ax_c.set_ylim(round_for_lim(integrated_current.min(), factor=5), round_for_lim(integrated_current.max(), factor=5))

    fig_c.savefig(path_to_figures / f'{shot}_integrated_current.png', dpi=600)



    plt.show()




if __name__ == '__main__':
    main(
        shot=SHOT, dimes_r=DiMES_R, t_range=T_RANGE, data_dir=DATA_DIR, fig_dir=FIGURES_DIR,
        integrated_current_dir=INTEGRATED_CURRENT_DIR, plot_t_range=PLOT_T_RANGE, tau=TAU
    )