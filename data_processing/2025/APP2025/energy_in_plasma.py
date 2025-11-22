import h5py
import matplotlib.pyplot as plt
from data_processing.misc_utils.plot_style import load_plot_style
import numpy as np
from pathlib import Path
import matplotlib.ticker as ticker

PATH_TO_DATA = r'data/energy_stored_in_plasma'
SHOT = 203782

def find_index_intersections(x, y, y_ref, y_tolerance=1E-6, x_eps:float=None, r_tol=2.5E-3):
    if x_eps is None:
        x_spacing: float = np.mean(np.diff(x)).astype(float)
        x_eps = 2 * x_spacing

    # idx_peaks = np.argwhere(np.abs(y - y_ref) < y_tolerance)
    idx_peaks = np.argwhere(np.abs(np.isclose(y, y_ref, rtol=r_tol)))

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

def main(shot, path_to_data=PATH_TO_DATA):
    path_to_data = Path(path_to_data)
    path_to_h5 = path_to_data / f'{shot}_energy_in_plasma.h5'
    with h5py.File(path_to_h5, 'r') as h5:
        shot_gp = h5.get(f'{shot}')
        pinj_gp = shot_gp.get('pinj')
        pinj_time_ms = np.array(pinj_gp.get('time'))
        pinj_kw = np.array(pinj_gp.get('pinj'))
        pohm_gp = shot_gp.get('pohm')
        pohm_time_ms = np.array(pohm_gp.get('time'))
        pohm_w = np.array(pohm_gp.get('pohm'))
        prad_gp = shot_gp.get('prad')
        prad_time_ms = np.array(prad_gp.get('time'))
        prad_w = np.array(prad_gp.get('prad'))
        wmhd_gp = shot_gp.get('wmhd')
        wmhd_time_ms = np.array(wmhd_gp.get('time'))
        wmhd_j = np.array(wmhd_gp.get('wmhd'))
        rvsout_gp = shot_gp.get('rvsout')
        rvsout_time_ms = np.array(rvsout_gp.get('time'))
        rvsout_m = np.array(rvsout_gp.get(f'rvsout'))
        dimes_r = rvsout_gp.get('rvsout').attrs['dimes_r']


    # Get rvsout for times > 1 s
    msk_rvsout_time = (1000 <= rvsout_time_ms)
    rvsout_time_ms = rvsout_time_ms[msk_rvsout_time]
    rvsout_m = rvsout_m[msk_rvsout_time]

    # Get the indices where rvsout crosses dimes_r
    idx_strike_points, t_strike_points = find_index_intersections(x=rvsout_time_ms, y=rvsout_m, y_ref=dimes_r)

    load_plot_style(font='Times New Roman')
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, constrained_layout=True)
    # fig.subplots_adjust(hspace=0)
    fig.set_size_inches(4, 3.5)

    axes[0].plot(pinj_time_ms, pinj_kw*1E-3, label='Pinj')
    # axes[1].plot(pohm_time_ms, pohm_w*1E-6, label='Pohm')
    axes[1].plot(prad_time_ms, prad_w*1E-6, label='Prad')
    axes[2].plot(wmhd_time_ms, wmhd_j*1E-6, label='Wmhd')

    axes[-1].set_xlabel('Time (ms)')
    axes[-1].set_xlim(500, 4500)

    axes[0].set_ylabel('P$_{\mathregular{inj}}$ (MW)')
    # axes[1].set_ylabel('P$_{\mathregular{ohm}}$ (MW)')
    axes[1].set_ylabel('P$_{\mathregular{rad}}$ (MW)')
    axes[2].set_ylabel('Wmhd (MJ)')

    axes[0].set_ylim(bottom=0, top=5)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    # axes[1].set_ylim(bottom=0, top=4)
    # axes[1].yaxis.set_major_locator(ticker.MultipleLocator(1))
    # axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    axes[1].set_ylim(bottom=0, top=2.5)
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(1.))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    axes[2].set_ylim(bottom=0, top=0.5)
    axes[2].yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    axes[2].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    shot_number_str = f'Shot #{shot}'
    axes[-1].text(
        0.95, 0.05, shot_number_str, transform=axes[-1].transAxes,
        ha='right', va='bottom', fontsize=11,
    )

    for ax in axes:
        for t_strike_point in t_strike_points:
            ax.axvline(x=t_strike_point, color='tab:red', linestyle='--')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))

    # fig.suptitle(f'Shot #{shot}')
    fig.align_labels()

    path_to_figures = Path(r'./figures')
    path_to_figures.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf', 'svg']:
        path_to_figure = path_to_figures / f'{shot}_energy_in_plasma.{extension}'
        fig.savefig(str(path_to_figure), dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main(shot=SHOT)





