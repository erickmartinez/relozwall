import numpy as np
import matplotlib.pyplot as plt
from utils.plot_style import load_plot_style
import MDSplus as mds
import pandas as pd
from mds_tools.getmds import getmds
from scipy.interpolate import CubicSpline


SHOT = 203780
SHOTS = np.arange(203779, 203786)
TIME_RANGE = [500, 4000]
RESISTOR = 0.31 # Ohms
TAU = 10. # ms
DIMES_R = 1.48 # m

def get_current_data(shot, tmin, tmax):
    ptname = rf'PTDATA("\dro1",{shot})'
    tname = rf"DIM_OF({ptname})"
    with mds.Connection('atlas.gat.com') as conn:
        data = np.array(conn.get(ptname))
        t_ms = np.array(conn.get(tname))

    msk_time = (tmin <= t_ms) & (t_ms <= tmax)
    data_cropped = data[msk_time]
    t_ms_cropped = t_ms[msk_time]
    return data_cropped, t_ms_cropped

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



def main(shots, tmin, tmax, tau, resistor, dimes_r):
    for shot in shots:
        tmin_s, tmax_s = tmin * 1E-3, tmax * 1E-3
        voltage, t_ms = get_current_data(shot, tmin, tmax)
        current = voltage / resistor
        current_rcsmooth = rc_smooth(t_ms, current, tau)

        rvsout, t_rvsout_s, _ = getmds(shot=shot, name='rvsout', range_min=tmin_s, range_max=tmax_s)
        cs_rvsout = CubicSpline(t_rvsout_s, rvsout)
        rvsout_at_dro1 = cs_rvsout(t_ms*1E-3)


        output_df  = pd.DataFrame(data={
            't_ms': t_ms, 'voltage': voltage, 'current': current,
            'current_rcsmooth': current_rcsmooth, 'rvsout': rvsout_at_dro1
        })

        tag = rf'{shot}_voltage_and_rvsout'

        output_df.to_csv(r'./data/' + tag + '.csv', index=False)

        load_plot_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        fig.set_size_inches(4.5, 5)

        ax1.plot(t_ms, current, color='C0', label='RAW')
        ax1.plot(t_ms, current_rcsmooth, color='tab:red', label='RC smooth')
        ax2.plot(t_ms, rvsout_at_dro1, c='C3')
        ax2.axhline(y=dimes_r, color='k', ls='--', lw='1.25')
        ax2.annotate(
            "DiMES R", (3500, dimes_r), xycoords='data', xytext=(0.8, 0.85),
            textcoords='axes fraction', ha='left', va='bottom',
            arrowprops=dict(
                arrowstyle='->', connectionstyle='angle,angleA=180,angleB=-90,rad=0',
                # patchA=None, patchB=None,
                shrinkA=2, shrinkB=2
            )
        )

        ax2.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (A)')
        ax2.set_ylabel('rvsout (m)')
        ax1.set_title(f'Shot #{shot}')
        ax1.set_xlim(tmin, tmax)
        ax2.set_ylim(bottom=1.3, top=1.6)

        fig.savefig(r'./figures/' + tag + '.png', dpi=600)

        plt.show()


if __name__ == '__main__':
    main(SHOTS, TIME_RANGE[0], TIME_RANGE[1], TAU, RESISTOR, DIMES_R)
