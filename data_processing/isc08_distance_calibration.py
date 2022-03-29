from instruments import linear_translator as lnt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares
from scipy.linalg import svd
import confidence as cf

base_path = r"G:\Shared drives\ARPA-E Project\Lab\Extrusion\Linear Translator\Calibration"
calibration_file = 'calibration_2022.csv'


def poly(x, m, b):
    return m * x + b


def poly_obj(beta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return poly(x, beta[0], beta[1]) - y


def poly_jac(beta: np.ndarray, x: np.ndarray, y: np.ndarray):
    identity = np.ones_like(x)
    return np.array([x, identity]).T


def get_pcov(res: OptimizeResult) -> np.ndarray:
    ysize = len(res.fun)
    cost = 2 * res.cost  # res.cost is half sum of squares!
    s_sq = cost / (ysize - popt.size)

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s ** 2, VT)
    pcov = pcov * s_sq

    if pcov is None:
        # indeterminate covariance
        print('Failed estimating pcov')
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(np.inf)
    return pcov


all_tol = np.finfo(np.float64).eps

if __name__ == '__main__':
    speeds = np.arange(5, 86, 5).astype(dtype=int)
    times = 0.1 * np.arange(10, 51, 5)
    columns = ["Speed Value"]
    for t in times:
        columns.append(f"d[t={t:3.2f} s] (cm)")

    columns.append('Speed (cm/s)')
    columns.append('Speed Error (cm/s)')

    calibration_df = pd.DataFrame(columns=columns)
    isc = lnt.ISC08(address='COM4')
    home = False

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots()
    fig.set_size_inches(6.0, 4.0)
    n = len(speeds)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    yerr = 0.5 * 0.254 * np.ones_like(times)

    for i, s in zip(range(n), speeds):
        distances = []
        data = {'Speed Value': s}
        for t in times:
            while not home:
                home = True if str(input('Is the post at the 6" mark? [Y/N]: ')).upper() == 'Y' else False
                if home:
                    break
                move_forward = True if str(input('Move forward? [Y/N]: ')).upper() == 'Y' else False
                speed = 60 if move_forward else -60
                isc.move_by_time(moving_time=1, speed=speed)

            isc.move_by_time(moving_time=t, speed=s)
            d = 2.54 * (float(input('Distance: (in)')) - 6.0)
            # d = t * s * 0.01
            distances.append(d)
            print(f"Moved {d:4.2f} cm")
            data[f"d[t={t:3.2f} s] (cm)"] = d
            isc.move_by_time(moving_time=t, speed=-s)
            home = False
        distances = np.array(distances, dtype=float)
        m_guess = (distances[-1] - distances[0]) / (times.max())
        res = least_squares(
            poly_obj, np.array([m_guess, 0.0]), jac=poly_jac, args=(times, distances),
            xtol=all_tol,
            ftol=all_tol,
            gtol=all_tol,
            verbose=0
        )
        popt = res.x
        pcov = get_pcov(res)
        ci = cf.confint(n, popt, pcov)
        v = popt[0]
        dv = ci[0, :].max() - v
        lbl = f"$v_{{\mathrm{{sp}}}} = {s:2d}$: {v:4.2f}±{dv:4.2f} cm/s"
        data['Speed (cm/s)'] = v
        data['Speed Error (cm/s)'] = dv
        ax.errorbar(
            times, distances, yerr=yerr, capsize=1.5, mew=1.5, marker='o', ms=8, color=colors[i], fillstyle='none',
            ls='none'
        )

        calibration_df = calibration_df.append(pd.DataFrame(data=data, index=['Speed Value']))

        ax.plot(
            times, poly(times, *popt), color=colors[i], label=lbl
        )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (cm)')
    leg = ax.legend(
        loc='upper left', frameon=True, ncol=1, fontsize=8, bbox_to_anchor=(1.05, 1),
        borderaxespad=0., prop={'size': 7}
    )
    print(calibration_df)
    fig.tight_layout()
    plt.show()
