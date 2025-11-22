import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from data_processing.misc_utils.plot_style import load_plot_style
import os
import re
from scipy.signal import find_peaks
from scipy.interpolate import interp1d, make_smoothing_spline
from typing import Tuple, Dict, Union, List
import matplotlib.ticker as ticker

PATH_TO_EVAPORATION_RATES = r'./data/d3d_evaporation_rates'
PATH_TO_MASS_LOSS_RATES = r'./data/mass_loss_rate_model/model_results'
PATH_TO_MDS_EMISSION_RATS = r'./data/mds_spectra/emission_flux'
PATH_TO_SPUTTERING_DATA = r'./data/D_flux'
PATH_TO_CURRENT_DATA = r'./data/dro1'
PATH_TO_CER_FIT = r'./data/ehollmann/BII_CER_file.txt'

SHOT_TOTAL_MASS_LOSS = 203782
SHOT_BII_RATE = 203782
SHOT_EVAPORATION_RATES = 203782
SHOT_BD_RATE = 203783
SHOT_SPUTTERING_RATE = 203782
SHOT_EDGE2D_RATE = 203780
T_RANGE = [1.5, 3.5]

DIMES_R = 1.48 # m

def load_sputtered_boron(shot, path_to_folder=PATH_TO_SPUTTERING_DATA):
    path_to_folder = Path(path_to_folder)
    path_to_csv = path_to_folder / f'{shot}_d_flux.csv'
    df = pd.read_csv(path_to_csv)
    return 1E-3 * df['time_ms'].values, df['B_sputtered_jsat (B/cm^2/s)'].values

def load_evaporation_rates(shot, path_to_data=PATH_TO_EVAPORATION_RATES):
    path_to_data = Path(path_to_data)
    path_to_csv = path_to_data / f'{shot}_evaporation_rate.csv'
    df = pd.read_csv(path_to_csv, comment='#').apply(pd.to_numeric, errors='coerce')
    time = df['Time (s)'].values
    evaporation_rate = df['Evaporation rate (atoms/s)'].values #
    evaporation_rate_lb = df['Evaporation rate lb (atoms/s)'].values
    evaporation_rate_ub = df['Evaporation rate ub (atoms/s)'].values

    data = {
        'time (s)': time,
        'evaporation_rate (atoms/s)': evaporation_rate,
        'evaporation_rate_lb (atoms/s)': evaporation_rate_lb,
        'evaporation_rate_ub (atoms/s)': evaporation_rate_ub,
    }

    return data

def load_mds_emission(shot, path_to_data=PATH_TO_MDS_EMISSION_RATS) -> Dict[str, np.ndarray]:
    path_to_data = Path(path_to_data)
    files_in_dir = [fn for fn in os.listdir(str(path_to_data)) if fn.startswith(f'{shot}_') and fn.endswith('.csv')]
    path_to_file = path_to_data / files_in_dir[0]
    pattern = re.compile(fr'{shot}_emission_flux_(.*?).csv')
    match = re.search(pattern, files_in_dir[0])
    if not(match):
        raise ValueError(f'Shot {shot} not found.')

    emission_line = match.group(1).replace('-', '')
    df = pd.read_csv(path_to_file, comment='#').apply(pd.to_numeric, errors='coerce')
    time = df['time (s)'].values
    flux = df[f'Flux {emission_line} (molecules/s)'].values

    data = {
        'time (s)': time,
        'flux (molecules/s)': flux,
        'emission line': emission_line,
    }

    return data

def get_osp_hit_times(shot, path_to_folder=PATH_TO_CURRENT_DATA, dimes_r=DIMES_R):
    path_to_folder = Path(path_to_folder)
    path_to_csv = path_to_folder / f'{shot}_voltage_and_rvsout.csv'
    df = pd.read_csv(path_to_csv)
    t_ms = df['t_ms'].values
    rvsout = df['rvsout'].values
    msk_time = (1500 <= t_ms) & (t_ms <= 3000)
    t_ms = t_ms[msk_time]
    rvsout = rvsout[msk_time]
    idx_interections, t_intersections = find_index_intersections(x=t_ms, y=rvsout, y_ref=dimes_r, y_tolerance=1E-5)
    return t_intersections*1E-3, t_ms*1E-3, rvsout

def plot_b_source(ax:plt.Axes, t_hits, path_to_cer_fit=PATH_TO_CER_FIT, t_range=T_RANGE, color='C4'):
    shot = 203780
    # Load the data for OSP B source estimated from core B-V CER data #203780 and EDGE2D Pinj=1.5 MW case
    cer_df = pd.read_csv(path_to_cer_fit, sep=r'\s+', skiprows=3, names=['time [s]', 'Bdot(CER)[1e20 B/s]']).apply(
        pd.to_numeric, errors='coerce')
    t_cer = cer_df['time [s]'].values
    b_dot = cer_df['Bdot(CER)[1e20 B/s]'].values
    # print(cer_df)
    # b_dot -= b_dot.min()
    msk_time = (t_range[0] <= t_cer) & (t_cer <= t_range[1])
    t_cer = t_cer[msk_time]
    b_dot = b_dot[msk_time]
    spl_bdot = make_smoothing_spline(x=t_cer, y=b_dot, lam=0.0002)

    idx_peaks, _ = find_peaks(spl_bdot(t_cer), prominence=1, distance=20)
    idx_peaks = np.array([idx_peak for idx_peak in idx_peaks if
                               (t_range[0] <= t_cer[idx_peak]) & (t_cer[idx_peak] <= t_range[1])])

    for idx_peak in idx_peaks:
        print(f'CER peak: {t_cer[idx_peak]:.3f} s')
    for t_hit in t_hits:
        print(f'203782 OSP hit: {t_hit:.3f} s')
    dt = t_cer[idx_peaks[0]] - t_hits[0]
    t_plot = np.linspace(t_range[0], t_range[1], 500)
    # spl_bdot = make_smoothing_spline(x=t_cer-dt, y=b_dot, lam=0.00025)

    handle, = ax.plot(t_plot, spl_bdot(t_plot), color=color, label='OSP source (B-V CER + EDGE2D)')
    return handle



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

def plot_total_mass_loss(
    shot, ax: plt.Axes, path_to_mass_loss_rates=PATH_TO_MASS_LOSS_RATES,
    color='C0', t_range=T_RANGE,
):
    path_to_mass_loss_rates = Path(path_to_mass_loss_rates)
    if shot in [203780, 203781]:
        path_to_h5 = path_to_mass_loss_rates / f'203780-203781_mass_loss_model.h5'
    elif shot in np.arange(203782, 203785):
        path_to_h5 = path_to_mass_loss_rates / f'203782-203784_mass_loss_model.h5'
    else:
        raise ValueError(f'{shot} not found.')

    with h5py.File(str(path_to_h5), 'r') as h5:
        shot_gp = h5[f'/{shot}']
        time_model = np.array(shot_gp.get('time'))
        mass_loss_rate = np.array(shot_gp.get('mass_loss_rate'))
        mass_loss_rate_error = np.array(shot_gp.get('mass_loss_error'))
        qpara = np.array(shot_gp.get('qpara'))

    spl_mass_loss_rate = make_smoothing_spline(time_model*1E3, mass_loss_rate*1E-20, lam=10000)

    idx_peaks_dmdt, _ = find_peaks(spl_mass_loss_rate(time_model*1E3), prominence=10, distance=50)
    idx_peaks_dmdt = np.array([idx_peak for idx_peak in idx_peaks_dmdt if
                               (t_range[0] <= time_model[idx_peak]) & (time_model[idx_peak] <= t_range[1])])



    handle, = ax.plot(time_model, spl_mass_loss_rate(time_model*1E3), color=color, label='Pebbles + dust')
    # ax.fill_between(time_model, (mass_loss_rate - mass_loss_rate_error, mass_loss_rate + mass_loss_rate_error, qpara, color=color, alpha=0.5)
    # for idx_peak in idx_peaks_dmdt:
    #     ax.axvline(x=time_model[idx_peak], color='tab:red', linestyle='--', lw=1.)
    return handle, time_model[idx_peaks_dmdt]

def plot_mds_mass_loss(shot, label, ax: plt.Axes, color='C1', t_range=T_RANGE, scale=1, lam=1E-6):
    emission_data = load_mds_emission(shot)
    time_s = emission_data['time (s)']
    flux = emission_data['flux (molecules/s)'] * 1E-20

    msk_time = (t_range[0] <= time_s) & (time_s<=t_range[1])
    time_s = time_s[msk_time]
    flux = flux[msk_time]

    spl_flux = make_smoothing_spline(time_s, flux, lam=lam)
    handle, = ax.plot(time_s, spl_flux(time_s)*scale, label=label, c=color)
    return handle

def plot_evaporation(shot, label, ax: plt.Axes, color='C2', t_range=T_RANGE, scale=1):
    data = load_evaporation_rates(shot=shot)

    time_s = data['time (s)']
    evaporation_rate = data['evaporation_rate (atoms/s)'] * 1E-20
    evaporation_rate_lb = data['evaporation_rate_lb (atoms/s)'] * 1E-20
    evaporation_rate_ub = data['evaporation_rate_ub (atoms/s)'] * 1E-20

    spl_evaporation_rate =  make_smoothing_spline(time_s, evaporation_rate, lam=0.0001)
    spl_evaporation_rate_lb = make_smoothing_spline(time_s, evaporation_rate_lb, lam=0.001)
    spl_evaporation_rate_ub = make_smoothing_spline(time_s, evaporation_rate_ub, lam=0.001)

    handle, = ax.plot(time_s, spl_evaporation_rate(time_s)*scale, label=label, color=color)
    return handle



def main(
    shot_mass_loss=SHOT_TOTAL_MASS_LOSS,
    shot_bii_rate=SHOT_BII_RATE,
    shot_bd_rate=SHOT_BD_RATE,
    shot_evaporation=SHOT_EVAPORATION_RATES,
    shot_sputtering=SHOT_SPUTTERING_RATE,
    shot_edge2d=SHOT_EDGE2D_RATE,
    path_to_cer_fit=PATH_TO_CER_FIT,
    t_range=T_RANGE,
):
    load_plot_style(font='Times New Roman')
    # fig = plt.figure()
    # fig.set_size_inches(4., 5.)
    # subfigs = fig.subfigures(nrows=2, ncols=1, height_ratios=[1, 0.6])

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, height_ratios=[1, 1, 1])
    fig.set_size_inches(4., 5.)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.0, wspace=0.2)
    # subfigs[0].subplots_adjust(left=0.15, right=0.95, hspace=0., top=0.95, bottom=0.3)
    # subfigs[1].subplots_adjust(left=0.15, right=0.95, hspace=0., top=1., bottom=0.3)
    # ax_top = subfigs[0].subplots(nrows=2, ncols=1, sharex=True)
    # ax1, ax2 = ax_top[0], ax_top[1]
    # ax_bottom = subfigs[1].subplots(nrows=1, ncols=1, sharex=True)
    # axes = [ax1, ax2, ax_bottom]


    t_hits, t_rvsout, rvsout = get_osp_hit_times(shot=shot_mass_loss)
    # Get expected sputtered boron
    t_sputtered, b_sputtered = load_sputtered_boron(shot=shot_sputtering)
    msk_trange = (t_range[0] <= t_sputtered) & (t_sputtered <= t_range[1])
    t_sputtered = t_sputtered[msk_trange]
    b_sputtered = b_sputtered[msk_trange]
    sample_area = 0.25 * np.pi * (0.95 ** 2)
    b_sputtered_plot = b_sputtered * 1E-18 * sample_area
    spl_sputtered = make_smoothing_spline(x=t_sputtered*1E3, y=b_sputtered_plot, lam=1000000)




    line_total, t_hits  = plot_total_mass_loss(shot=shot_mass_loss, ax=axes[0], color='C0')
    line_bii = plot_mds_mass_loss(shot=shot_bii_rate, ax=axes[1], color='C1', label='Atomic B\nLocal B-II', lam=1E-6)
    line_bd = plot_mds_mass_loss(shot=shot_bd_rate, ax=axes[2], color='C5', label='Local BD', scale=1E5, lam=1E-6)

    plot_evaporation(shot=shot_evaporation, ax=axes[2], label=f'Evaporation', color='C2', scale=1E5)

    for t in t_hits:
        for ax in axes:
            ax.axvline(x=t, color='tab:red', ls='--', lw=1.,)


    plot_b_source(
        ax=axes[1], t_hits=t_hits, color='C4'
    )
    axes[0].set_ylim(0, 80)
    axes[1].set_ylim(0, 3.6)
    axes[2].set_ylim(0, 17.5)
    # axes[3].set_ylim(0, 7)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(20))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(5))
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    axes[2].yaxis.set_major_locator(ticker.MultipleLocator(5))
    axes[2].yaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    # axes[3].yaxis.set_major_locator(ticker.MultipleLocator(3))
    # axes[3].yaxis.set_minor_locator(ticker.MultipleLocator(1))
    # axes[2].tick_params(axis='y', labelcolor='C2')

    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(1.5, 3.5)
    shot_txt = f'Shot #{shot_mass_loss}'
    axes[-1].text(
        0.985, 0.025, shot_txt,
        ha='right', va='bottom', transform=axes[-1].transAxes, fontsize=10, fontweight='bold'
    )


    for ax in axes:
        ax.legend(loc='upper left')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    axes[0].set_ylabel(r'$\Gamma_{\mathrm{B}}$ ($\times 10^{20}$ atoms/s)', usetex=True, fontsize=12)
    axes[-1].set_ylabel(r'$\Gamma_{\mathrm{B}}$ ($\times 10^{15}$ atoms/s)', usetex=True)

    fig.align_labels()
    path_to_figures = Path(r'./figures')

    fig.savefig(path_to_figures / 'boron_emission_rates.svg', dpi=600, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    main()

