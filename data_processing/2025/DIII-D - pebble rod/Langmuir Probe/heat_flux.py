import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_smoothing_spline, interp1d, CubicSpline
from data_processing.misc_utils.plot_style import load_plot_style # < personal plot style
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage

DATA_DIR = r'./data'
FIGURES_DIR = r'./figures'
SHOT = 203785
T_RANGE = [0, 5000]
DT_LP = 1 # time step size for LP data [ms]
R_PLOT = 1.485 # major radius to look at [m]


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

def main(shot, data_dir, figures_dir, t_range, dt_lp, r_plot):
    A = (t_range[1] - t_range[0]) / dt_lp
    n_steps = int(A)  # number of time steps to use
    t_int_V = t_range[0] + np.arange(0, n_steps, 1) * dt_lp
    plot_style = load_plot_style()
    path_to_data = Path(data_dir)
    path_to_figures = Path(figures_dir)

    path_to_data.mkdir(parents=True, exist_ok=True)
    path_to_figures.mkdir(parents=True, exist_ok=True)

    # Get the h5 file for the shot
    path_to_h5 = path_to_data / f'{shot}_LP.h5'

    q_para_A = np.empty((14, n_steps))
    q_perp_A = np.empty((14, n_steps))
    T_e_A = np.empty((14, n_steps))
    ne_A = np.empty((14, n_steps))
    R_LP = np.empty(14)
    Z_LP = np.empty(14)

    for i_step in range(14):
        data = {}
        with h5py.File(path_to_h5, 'r') as h5:
            grp = h5[f'/LANGMUIR/PROBE_{i_step:03d}']
            print(f'Processing data from probe {grp.attrs["mds_index"]}...')
            for ds in grp:
                data[ds] = np.array(grp.get(ds))
                # data[ds]['attrs'] = {}
                # for key, val in ds.attrs.items():
                #     data[ds]['attrs'][key] = val

        R_LP[i_step] = data['r']  # major radius [m]
        Z_LP[i_step] = data['z']  # vertical coordinate [m]
        t_dum_V = data['time']  # time vector for this channel [ms]
        t_dum_Vr = t_dum_V + 1E-6 * np.random.random(len(t_dum_V))
        dum_V = 10 * data['heatflux']  # parallel heat flux [MW/m^2] for this channel
        alpha_V = data['angle']  # field line pitch angle [deg]
        # print(f'F/zield line pitch angle: {data["angle"]}')
        dum_V2 = dum_V * np.sin(np.radians(alpha_V))  # perp heat flux [MW/m^2] for this channel

        lam=25

        # interpolate qpara [MW/m2] onto my time grid
        spl_para = make_smoothing_spline(x=t_dum_Vr, y=dum_V, lam=lam)
        # q_para_A[i, :] = splev(x=t_int_V, tck=cs_para)
        q_para_A[i_step, :] = spl_para(t_int_V)
        # interpolate qperp [MW/m2] onto my time grid
        spl_perp = make_smoothing_spline(t_dum_Vr, dum_V2, lam=lam)
        q_perp_A[i_step, :] = spl_perp(t_int_V)

        dum_V = data['temp']  # temp eV
        spl_temp = make_smoothing_spline(x=t_dum_Vr, y=dum_V, lam=None)
        T_e_A[i_step, :] = spl_temp(t_int_V)

        dum_V = 1E-13 * data['dens']  # q_perp_A[i, :] = cs_perp(t_int_V)
        fit_V = interp1d(x=t_dum_Vr, y=dum_V, bounds_error=False)
        half_width = 2
        sigma = 2 * half_width / np.sqrt(8 * np.log(2))
        ne_A[i_step, :] = gaussian_filter1d(fit_V(t_int_V), sigma=sigma)

    qparaV = np.empty(n_steps)
    qperpV = np.empty(n_steps)
    TeV = np.empty(n_steps)
    neV = np.empty(n_steps)

    for i in range(n_steps):
        dum_V = q_para_A[:, i]
        f = interp1d(x=R_LP, y=dum_V, bounds_error=False, fill_value=0)
        qparaV[i] = f(r_plot)
        dum_V = q_perp_A[:, i]
        f = interp1d(x=R_LP, y=dum_V, bounds_error=False, fill_value=0)
        qperpV[i] = f(r_plot)
        dum_V = T_e_A[:, i]
        f = interp1d(x=R_LP, y=dum_V, bounds_error=False, fill_value=0)
        TeV[i] = f(r_plot)
        dum_V = ne_A[:, i]
        f = interp1d(x=R_LP, y=dum_V, bounds_error=False, fill_value=0)
        neV[i] = f(r_plot)


    qparaV_despiked, qparaV_spikes = remove_spikes_zscore(qparaV, threshold=2.5, window_size=40)
    qperpV_despiked, qperpV_spikes = remove_spikes_zscore(qperpV, threshold=2.5, window_size=40)
    TeV_despiked, TeV_spikes = remove_spikes_zscore(TeV, threshold=2.5, window_size=40)
    neV_despiked, neV_spikes = remove_spikes_zscore(neV, threshold=2.5, window_size=40)

    # Remove nans
    msk_nans = np.isnan(qparaV_despiked)
    spl_qpara_despiked = make_smoothing_spline(t_int_V[~msk_nans], qparaV[~msk_nans], lam=25)
    msk_nans = np.isnan(qperpV_despiked)
    spl_qperp_despiked = make_smoothing_spline(t_int_V[~msk_nans], qperpV[~msk_nans], lam=25)
    msk_nans = np.isnan(TeV_despiked)
    spl_Te_despiked = make_smoothing_spline(t_int_V[~msk_nans], TeV[~msk_nans], lam=None)
    msk_nans = np.isnan(neV_despiked)
    spl_ne_despiked = make_smoothing_spline(t_int_V[~msk_nans], neV[~msk_nans], lam=25)

    path_to_data_at_rdimes = path_to_data / 'dimes_lp'
    path_to_data_at_rdimes.mkdir(parents=True, exist_ok=True)
    path_to_data_at_rdimes_h5 = path_to_data_at_rdimes / f'{shot}_LP.h5'
    with h5py.File(path_to_data_at_rdimes_h5, 'w') as h5:
        with h5py.File(path_to_h5, 'r') as f:
            f.copy('/LANGMUIR', h5[f'/'])
        dimes_gp = h5.create_group(f'/LANGMUIR_DIMES')
        time_ds = dimes_gp.create_dataset('time', data=t_int_V, compression='gzip')
        qpara_ds = dimes_gp.create_dataset('qpara', data=qparaV, compression='gzip')
        qperp_ds = dimes_gp.create_dataset('qperp', data=qperpV, compression='gzip')
        TeV_ds = dimes_gp.create_dataset('TeV', data=TeV, compression='gzip')
        ne_ds = dimes_gp.create_dataset('ne', data=neV, compression='gzip')

        dimes_smoothed_gp = h5.create_group(f'/LANGMUIR_DIMES_SMOOTHED')
        time_smoothed_ds = dimes_smoothed_gp.create_dataset('time', data=t_int_V, compression='gzip')
        qpara_smoothed_ds = dimes_smoothed_gp.create_dataset('qpara', data=spl_qpara_despiked(t_int_V), compression='gzip')
        qperp_smoothed_ds = dimes_smoothed_gp.create_dataset('qperp', data=spl_qperp_despiked(t_int_V), compression='gzip')
        TeV_smoothed_ds = dimes_smoothed_gp.create_dataset('TeV', data=spl_Te_despiked(t_int_V), compression='gzip')
        ne_smoothed_ds = dimes_smoothed_gp.create_dataset('ne', data=spl_ne_despiked(t_int_V), compression='gzip')

        time_ds.attrs['unit'] = 'ms'
        qpara_ds.attrs['unit'] = 'MW/m2'
        qperp_ds.attrs['unit'] = 'MW/m2'
        TeV_ds.attrs['unit'] = 'eV'
        ne_ds.attrs['unit'] = 'x1E13/cm3'

    load_plot_style()
    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True, sharex=True)
    fig.set_size_inches(6, 5)

    axes[0, 0].plot(t_int_V, qparaV,  color='C0', alpha=0.35, label='Raw')
    axes[0, 1].plot(t_int_V, qperpV, color='C1', alpha=0.35, label='Raw')
    axes[1, 0].plot(t_int_V, TeV, color='C2', label='Raw', alpha=0.35)
    axes[1, 1].plot(t_int_V, neV, color='C3', label='Raw', alpha=0.35)

    axes[0, 0].plot(t_int_V, spl_qpara_despiked(t_int_V), color='C0', label='De-spiked')
    axes[0, 1].plot(t_int_V, spl_qperp_despiked(t_int_V), color='C1', label='De-spiked')
    axes[1, 0].plot(t_int_V, spl_Te_despiked(t_int_V), color='C2', label='De-spiked')
    axes[1, 1].plot(t_int_V, spl_ne_despiked(t_int_V), color='C3', label='De-spiked')

    for ax in axes.flatten():
        ax.set_xlabel('Time (s)')
        ax.set_xlim(750, 4000)
        ax.legend(loc='upper right', frameon=True, fontsize=10)

    axes[0, 0].set_ylabel(r'{\sffamily q\textsubscript{para} (MW/m\textsuperscript{2})', usetex=True)
    axes[0, 1].set_ylabel(r'{\sffamily q\textsubscript{perp} (MW/m\textsuperscript{2})', usetex=True)
    axes[1, 0].set_ylabel(r'{\sffamily T\textsubscript{e} (eV)}', usetex=True)
    axes[1, 1].set_ylabel(r'{\sffamily n\textsubscript{e} (x10\textsuperscript{13}/cm\textsuperscript{3})}',
                          usetex=True)


    axes[0, 0].set_ylim(0, 100)
    if shot == 203785:
        axes[0, 0].set_ylim(0, 500)
    axes[0, 1].set_ylim(0, 3)
    axes[1, 0].set_ylim(0, 60)
    axes[1, 1].set_ylim(0, 4)

    fig.suptitle(f'LP data for shot #{shot}')

    path_to_figures = Path(path_to_figures)
    path_to_figures.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf', 'svg']:
        path_to_figure = path_to_figures / f'{shot}_LANGMUIR.{extension}'
        fig.savefig(path_to_figure, dpi=600)

    plt.show()




if __name__ == '__main__':
    main(
        shot=SHOT,
        data_dir=DATA_DIR,
        figures_dir=FIGURES_DIR,
        t_range=T_RANGE,
        dt_lp=DT_LP,
        r_plot=R_PLOT,
    )