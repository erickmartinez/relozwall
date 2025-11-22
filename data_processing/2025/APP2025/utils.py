import h5py
from pathlib import Path
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import scipy.ndimage as ndimage
from scipy.interpolate import PPoly
from typing import Union, Tuple, Callable, List
import pandas as pd

CURRENT_FOLDER = r'./dro1'

def load_lp_data(shot, path_to_folder):
    path_to_folder = Path(path_to_folder)
    with h5py.File( path_to_folder / f'{shot}_LP.h5', 'r') as h5:
        dimes_gp = h5['/LANGMUIR_DIMES']
        t_s = np.array(dimes_gp.get('time')) * 1E-3
        T_eV = np.array(dimes_gp.get('TeV'))
        n_e = np.array(dimes_gp.get('ne')) * 1E13
        qpara = np.array(dimes_gp.get('qpara'))
        qperp = np.array(dimes_gp.get('qperp'))

    data = {
        't_s': t_s,
        'Te_eV': T_eV,
        'n_e': n_e,
        'qpara': qpara,
        'qperp': qperp,
    }
    return data

def load_model(path_to_pppl_fit) \
        -> Callable[[Union[float, np.ndarray]], Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Load the fit from

    H.W. Kugel, Y. Hirooka, J. Timberlake et al., Initial boronization of PB-X using ablation of solid boronized probes.
    PPL-2903 (1993)

    Figure 16

    to estimate the evaporation rate at each time

    Parameters
    ----------
    path_to_pppl_fit: str, pathlib.Path

    Returns
    -------
    callable:
        The evaporation model
    """
    path_to_pppl_fit = Path(path_to_pppl_fit)
    with h5py.File(str(path_to_pppl_fit), 'r') as hf:
        # Load the coefficients of the polynomial fit (in log scale) for the boron evaporation rate in
        # (atoms/cm^2/s)
        model_popt = np.array(hf['/model/popt'])

        lb_ppoly_c = np.array(hf['/model/lb_ppoly/c'])
        lb_ppoly_x = np.array(hf['/model/lb_ppoly/x'])
        ub_ppoly_c = np.array(hf['/model/ub_ppoly/c'])
        ub_ppoly_x = np.array(hf['/model/ub_ppoly/x'])

    ppoly_lb = PPoly(lb_ppoly_c, lb_ppoly_x)
    ppoly_ub = PPoly(ub_ppoly_c, ub_ppoly_x)
    def evaporation_rate_model(temperature) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The evaporation rate in atoms/cm^2/s

        Parameters
        ----------
        temperature: np.ndarray
            The temperature in Kelvin

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The evaporation rate in atoms/cm^2/s
            the lower and upper bounds of the evaporation rate
        """
        rate = np.exp(model_poly(temperature, model_popt))
        lb = np.exp(ppoly_lb(temperature))
        ub = np.exp(ppoly_ub(temperature))
        return rate, lb, ub

    return evaporation_rate_model

def model_poly(x, b) -> np.ndarray:
    """
    A polynomial model

    Parameters
    ----------
    x: np.ndarray
        The x data points the polynomial is evaluated at
    b: np.ndarray
        The coefficients of the polynomial

    Returns
    -------

    """
    n = len(b)
    r = np.zeros(len(x))
    for i in range(n):
        r += b[i] * x ** i
    return r

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

def gaussian_broadening(signal, sigma, mode='reflect', preserve_height=True):
    """
    Apply Gaussian broadening to a signal.

    Parameters:
    -----------
    signal : array-like
        Input signal to be broadened
    sigma : float
        Standard deviation of the Gaussian kernel (controls broadening width)
    mode : str, optional
        How to handle boundaries. Options: 'reflect', 'constant', 'nearest', 'mirror', 'wrap'
        Default is 'reflect'
    preserve_height : bool, optional
        If True, normalize to preserve the maximum peak height. Default is True

    Returns:
    --------
    broadened_signal : ndarray
        The broadened signal
    """
    if sigma == 0:
        return np.array(signal)

    broadened = gaussian_filter1d(signal, sigma=sigma, mode=mode)

    if preserve_height:
        # Normalize to preserve peak height
        original_max = np.max(signal)
        broadened_max = np.max(broadened)
        if broadened_max > 0:
            broadened = broadened * (original_max / broadened_max)

    return broadened

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


def load_current_data(shot, data_dir=CURRENT_FOLDER):
    path_to_data = Path(data_dir) / f'{shot}_voltage_and_rvsout.csv'
    df = pd.read_csv(path_to_data).apply(pd.to_numeric, errors='coerce')
    return df

def load_mass_loss_rate(shot, data_dir=r'./recession_rate_model/model_results'):
    path_to_data = Path(data_dir)
    if shot in [203780, 203781]:
        file = path_to_data / '203780-203781_mass_loss_model.h5'
    elif shot in np.arange(203782, 203785):
        file = path_to_data / '203782-203784_mass_loss_model.h5'
    with h5py.File(file, 'r') as f:
        time_s = np.array(f[f'{shot}/time'])
        mass_loss_rate = np.array(f[f'{shot}/mass_loss_rate'])
        mass_loss_rate_error = np.array(f[f'{shot}/mass_loss_error'])
    data = {'time': time_s,
            'mass_loss_rate': mass_loss_rate,
            'mass_loss_rate_error': mass_loss_rate_error
    }
    return data