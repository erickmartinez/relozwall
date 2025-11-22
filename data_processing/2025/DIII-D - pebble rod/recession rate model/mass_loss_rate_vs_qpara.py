import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d, make_smoothing_spline
from scipy.integrate import simpson
from scipy.signal import find_peaks
from data_processing.misc_utils.plot_style import load_plot_style
from scipy.stats.distributions import t

PATH_TO_MASS_LOSS_FOLDER = r'./data/model_results'
SHOT = 203784
OSP_STRIKE_RANGE = [2.0, 3.1]
SMOOTH_DMDT = True

# N_A = 6.02214076e+23
N_A = 6.02214076E3

def mgrams_per_second_to_atoms_per_second(mass_mg, boron_molar_mass=10.811):
    return mass_mg*1E-3 / boron_molar_mass * N_A #x1E20 atoms/s

def atoms_per_second_to_mgrams_per_second(atoms, boron_molar_mass=10.811):
    return atoms / N_A * boron_molar_mass * 1E3

def load_mass_loss_data(shot: int, path_to_mass_loss_folder=PATH_TO_MASS_LOSS_FOLDER):
    path_to_mass_loss_folder = Path(path_to_mass_loss_folder)
    if shot in [203780, 203781]:
        path_to_h5 = path_to_mass_loss_folder / '203780-203781_mass_loss_model.h5'
    elif shot in np.arange(203782, 203785):
        path_to_h5 = path_to_mass_loss_folder / '203782-203784_mass_loss_model.h5'
    else:
        raise ValueError(f'Shot {shot} not found.')

    with h5py.File(path_to_h5, 'r') as hf:
        time_s = np.array(hf.get(f'{shot}/time'))
        mass_loss_rate = np.array(hf.get(f'{shot}/mass_loss_rate'))
        qpara = np.array(hf.get(f'{shot}/qpara'))
        mass_loss_rate_error = np.array(hf.get(f'{shot}/mass_loss_error'))

    return time_s, mass_loss_rate, mass_loss_rate_error, qpara

def main(shot, path_to_mass_loss_folder=PATH_TO_MASS_LOSS_FOLDER, osp_strike_range=OSP_STRIKE_RANGE, smooth_dmdt=SMOOTH_DMDT):
    time_s, dmdt, dmdt_delta, qpara = load_mass_loss_data(shot, path_to_mass_loss_folder)
    if smooth_dmdt:
        spl_dmdt = make_smoothing_spline(time_s, dmdt)
        spl_dmdt_delta = make_smoothing_spline(time_s, dmdt_delta)
        dmdt = spl_dmdt(time_s)

    mass_loss_g = atoms_per_second_to_mgrams_per_second(simpson(y=dmdt, x=time_s)*1E-3*1E-20)
    print(f'Total mass loss: {mass_loss_g:.3f} g')

    load_plot_style(font='Times New Roman')
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, constrained_layout=True, sharex=True)
    fig.set_size_inches(4, 4)

    osp_strike_range.sort()

    # ax1.plot(time_s, mass_loss_rate*1E-20, color='C0')
    ax1.plot(time_s, dmdt*1E-20, color='C0')
    ax1.set_ylabel(r'$dm/dt$ ($\times10^{\mathrm{20}}$ atoms/s)', usetex=True)
    ax1_g = ax1.secondary_yaxis('right', functions=(atoms_per_second_to_mgrams_per_second, mgrams_per_second_to_atoms_per_second))
    ax1_g.set_ylabel('$dm/dt$ (mg/s)', usetex=True)
    ax1.set_ylim(0, np.ceil(dmdt.max()*1.1E-20/10)*10)
    ax2.plot(time_s, qpara, color='C1')
    if shot != 203784:
        idx_peaks_qpara, _ = find_peaks(qpara, prominence=20, distance=20)
        idx_peaks_qpara = np.array([idx_peak for idx_peak in idx_peaks_qpara if
                              (osp_strike_range[0] <= time_s[idx_peak]) & (time_s[idx_peak] <= osp_strike_range[1])])
        idx_peaks_dmdt , _ = find_peaks(dmdt, prominence=200, distance=500)
        idx_peaks_dmdt = np.array([idx_peak for idx_peak in idx_peaks_dmdt if (osp_strike_range[0] <= time_s[idx_peak]) & (time_s[idx_peak] <= osp_strike_range[1])])
        ax1.plot(time_s[idx_peaks_dmdt], dmdt[idx_peaks_dmdt]*1E-20, color='tab:red', marker='x', ls='none', label='OSP hits DiMES')
        ax2.plot(time_s[idx_peaks_qpara], qpara[idx_peaks_qpara], color='tab:red', marker='x', ls='none', label='OSP hits DiMES')
    else:
        idx_peaks_qpara = np.argwhere((osp_strike_range[0] <= time_s) & (time_s<=osp_strike_range[1]))
        idx_peaks_dmdt = np.argwhere((osp_strike_range[0] <= time_s) & (time_s<=osp_strike_range[1]))
        ax1.axvspan(xmin=osp_strike_range[0], xmax=osp_strike_range[1], color='tab:red', alpha=0.1, label='OSP hits DiMES')
        ax2.axvspan(xmin=osp_strike_range[0], xmax=osp_strike_range[1], color='tab:red', alpha=0.1, label='OSP hits DiMES')


        qpara_mean = np.mean(qpara[idx_peaks_qpara])

        n_dmdt = len(dmdt[idx_peaks_dmdt])
        dmdt_mean = np.mean(dmdt[idx_peaks_dmdt])
        dmdt_std = np.std(dmdt[idx_peaks_dmdt], ddof=1)
        t_val = t.ppf(1 - 0.05 / 2, n_dmdt - 1)
        dmdt_se = dmdt_std * t_val / np.sqrt(n_dmdt)
        dmdt_delta_mean = np.linalg.norm(spl_dmdt_delta(time_s[idx_peaks_dmdt])) / n_dmdt
        dmdt_total_error = np.linalg.norm([dmdt_se, dmdt_delta_mean])

        # qpara_std = np.std(qpara[idx_peaks_qpara[0]:idx_peaks_qpara[-1] + 1], ddof=1)
        # n = idx_peaks_qpara[-1] - idx_peaks_qpara[0]
        # t_val = t.ppf(1 - 0.05 / 2, n - 1)
        # qpara_se = qpara_std * t_val / np.sqrt(n)

    ax2.set_ylabel('$q_{\mathrm{para}}$ (MW/m²)')
    ax2.set_xlabel('Time (s)')
    ax2.set_xlim(time_s[0], time_s[-1])
    ax2.set_ylim(0, 80)

    fig.suptitle(f'Shot #{shot}')
    fig.align_labels()
    output_dir = Path(r'./data/model_results/dmdt_vs_qpara')
    output_dir.mkdir(parents=True, exist_ok=True)

    path_to_h5 = output_dir / f'{shot}_dmdt_vs_qpara.h5'
    with h5py.File(str(path_to_h5), 'w') as hf:
        shot_gp = hf.create_group(f'{shot}')
        time_ds = shot_gp.create_dataset('time', data=time_s, compression='gzip')
        time_ds.attrs['units'] = 's'
        qpara_ds = shot_gp.create_dataset('qpara', data=qpara, compression='gzip')
        qpara_ds.attrs['units'] = 'MW/m2'
        dmdt_ds = shot_gp.create_dataset('dmdt', data=dmdt, compression='gzip')
        dmdt_ds.attrs['units'] = 'atoms/s'
        dmdt_ds.attrs['comment'] = 'smoothened'
        dmdt_delta_ds = shot_gp.create_dataset(f'dmdt_delta', data=spl_dmdt_delta(time_s), compression='gzip')
        dmdt_delta_ds.attrs['units'] = 'atoms/s'
        osp_hit_gp = hf.create_group(f'{shot}/osp_on_dimes')
        if shot != 203784:
            qpara_hit_ds = osp_hit_gp.create_dataset('qpara', data=qpara[idx_peaks_qpara])
            dmdt_hit_ds = osp_hit_gp.create_dataset('dmdt', data=dmdt[idx_peaks_dmdt])
            dmdt_hit_delta_ds = osp_hit_gp.create_dataset('dmdt_delta', data=spl_dmdt_delta(time_s[idx_peaks_dmdt]), compression='gzip')
        else:
            qpara_hit_ds = osp_hit_gp.create_dataset('qpara', data=np.array([qpara_mean]))
            dmdt_hit_ds = osp_hit_gp.create_dataset('dmdt', data=np.array([dmdt_mean]))
            dmdt_hit_delta_ds = osp_hit_gp.create_dataset('dmdt_delta', data=np.array([dmdt_total_error]))

        qpara_hit_ds.attrs['units'] = 'MW/m2'
        dmdt_hit_ds.attrs['units'] = 'atoms/s'

    plt.show()


if __name__ == '__main__':
    main(shot=SHOT)

