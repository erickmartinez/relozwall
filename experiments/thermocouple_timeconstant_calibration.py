import sys, os
from scipy.linalg import svd
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
import datetime
from instruments.esp32 import DualTCLoggerTCP
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from scipy import interpolate
import data_processing.confidence as cf

data_path = r'G:\Shared drives\ARPA-E Project\Lab\Data\thermocouple time constant'
# base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\heat_flux_calibration'
tc_id = 'TCE01'

TC_LOGGER_IP = '192.168.4.3'
max_time = 60.0
tc_error_pct = 2.0
T_ss = 164
make_fit = True
csv_file = 'CAL_TCE01_2022-07-16_090756'
left_cut = 2.4
right_cut = 60.0


def model(x, b):
    return b[0] * (1.0 - np.exp(-(x - b[1]) / b[2]))


def model_obj(beta: np.ndarray, x: np.ndarray, y: np.ndarray, weights = 1.0) -> np.ndarray:
    return (model(x, beta) - y)*weights


def model_jac(b: np.ndarray, x: np.ndarray, y: np.ndarray, weights = 1.0):
    identity = np.ones_like(x)
    xb = x - b[1]
    ee = np.exp(-xb / b[2])
    b_inv = 1.0 / (b[2] ** 2.0)
    return np.array([(1.0 - ee), -b[0] / b[2] * ee, -b[0] * b_inv * ee * xb]).T


def get_pcov(res: OptimizeResult) -> np.ndarray:
    popt = res.x
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


def latex_float(f, significant_digits=2):
    significant_digits += 1
    float_str_str = f"{{val:7.{significant_digits}g}}"
    float_str = float_str_str.format(val=f).lower()

    if "e" in float_str:
        base, exponent = float_str.split("e")
        # return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
        if exponent[0] == '+':
            exponent = exponent[1::]
        return rf"{base} \times 10^{{{int(exponent)}}}"
    else:
        return float_str


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


if __name__ == '__main__':
    if not make_fit:
        print('Connecting to temperature logger...')
        tc_logger = DualTCLoggerTCP(ip_address=TC_LOGGER_IP)
        print('Connection to temperature logger successful')
        temperature_reading = tc_logger.temperature
        temperature_0 = temperature_reading[0]
        temperature_ss = temperature_reading[1] #T_ss
        print(f'T(0) = {temperature_0:5.3f} °C')
        print(f'TSS = {temperature_ss:5.3f} °C')
        input('Make sure the thermocouple on which to run the calibration is connected on the input \'TC1\'.')

        input('Place the thermocouple to be calibrated on the hotplate.')
        tc_logger.log_time = max_time
        tc_logger.start_logging()
        previous_time = 0.0
        total_time = 0.0
        start_time = time.time()
        acquisition_time = max_time

        while total_time <= acquisition_time:
            current_time = time.time()
            if (current_time - previous_time) >= 0.1:
                total_time = current_time - start_time
                progress = 100 * total_time / acquisition_time
                print(f'T = {total_time:5.1f} s / {acquisition_time:5.1f} s Progress: {progress:5.1f}%', end='\r')
                previous_time = current_time

        tc_data: pd.DataFrame = tc_logger.read_temperature_log()
        today = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        file_tag = f"CAL_{tc_id.upper()}_{today}"
        full_file_name = os.path.join(data_path, file_tag + '.csv')
        tc_data.to_csv(full_file_name, index=False)
        print(f'Data saved in:')
        print(f'{full_file_name}')

    else:
        tc_data = pd.read_csv(os.path.join(data_path, csv_file + '.csv'))
        measured_time = tc_data['Time (s)'].values
        tc1 = tc_data['TC1 (C)'].values
        tc2 = tc_data['TC2 (C)'].values
        t0 = tc1[0]
        print(f'T(0) = {t0:.2f} °C')

        idx_fit = (measured_time >= left_cut) & (measured_time <= right_cut)
        measured_time_fit = measured_time[idx_fit]
        tc1_fit = tc1[idx_fit]
        tc2_fit = tc2[idx_fit]
        dT_fit = tc1_fit - t0
        # measured_time = measured_time  # - 0.5  # measured_time.min()

        resolution_error = 0.25
        tc1_err = tc1 * tc_error_pct * 0.01
        for i in range(len(tc1_err)):
            tc1_err[i] = max(tc1_err[i], resolution_error)

        weights = 1/tc1_err/np.sqrt(2.0)
        weights /= weights.max()
        weights = weights[idx_fit]
        n = len(measured_time)
        b_guess = np.array([dT_fit.max(), left_cut*0.1, measured_time_fit.max()*0.10])
        all_tol = np.finfo(np.float64).eps
        res = least_squares(
            model_obj, b_guess, args=(measured_time_fit, dT_fit, weights),
            jac=model_jac,
            xtol=all_tol,
            ftol=all_tol,
            gtol=all_tol,
            max_nfev=10000 * n,
            # loss='soft_l1', f_scale=0.1,
            verbose=2
        )
        popt = res.x
        pcov = get_pcov(res)

        ci = cf.confint(n, popt, pcov)

        xpred = np.linspace(measured_time_fit.min(), measured_time_fit.max(), num=200)
        ypred, lpb, upb = cf.predint(xpred, measured_time_fit, tc1_fit, model, res)

        tss = popt[0]
        time_constant = popt[2]
        time_constant_ci = ci[2]

        print(time_constant, '95% CI', time_constant_ci)
        print(tss, '95% CI', ci[0])

        model_results_df = pd.DataFrame(
            data={
                'Parameter': ['Delta T (K)', 'T0 (s)', 'Time Constant (s)'],
                'Value': [tss, popt[1], time_constant],
                'Lower 95% CI': [ci[0, 0], ci[1, 0], ci[2, 0]],
                'Upper 95% CI': [ci[0, 1], ci[1, 1], ci[2, 1]]
            }
        )

        model_results_df.to_csv(os.path.join(data_path, csv_file + '_model_results.csv'), index=False)
        prediction_df = pd.DataFrame(data={
            'Time (s)': xpred,
            'Temperature (°C)': ypred,
            'Lower Prediction Band (°C)': lpb,
            'Upper Prediction Band (°C)': upb
        })
        prediction_df.to_csv(os.path.join(data_path, csv_file + '_prediction.csv'), index=False)

        model_txt = f'$\\tau$ = {time_constant:.3f} s\n95% CI: [{ci[2][0]:.4f}, {ci[2][1]:.4f}] s'
        # model_txt += f'\n$\Delta T_{{\mathrm{{ss}}}} = {latex_float(tss,significant_digits=3)} °C 95% CI: [{ci[0][0]:.4f}, {ci[0][1]:.4f}] °C$'

        with open('../data_processing/plot_style.json', 'r') as file:
            json_file = json.load(file)
            plot_style = json_file['defaultPlotStyle']
        mpl.rcParams.update(plot_style)

        fig, ax1 = plt.subplots()
        fig.set_size_inches(4.5, 3.25)

        ax1.errorbar(
            measured_time, tc1, yerr=tc1_err,
            capsize=2.75, mew=1.25, marker='o', ms=8, elinewidth=1.25,
            color='C0', fillstyle='none',
            ls='none',
            label='Data',
            zorder=1
        )

        ax1.fill_between(
            xpred, lpb+t0, upb+t0, color=lighten_color('C0', 0.2),
            label='Prediction Bands', zorder=0
        )

        ax1.plot(
            xpred, ypred+t0, color='k', label='Model', zorder=2
        )

        leg = ax1.legend(
            loc='lower right', frameon=True, ncol=1,
            # fontsize=8, bbox_to_anchor=(1.05, 1),
            # borderaxespad=0.,
            prop={'size': 10}
        )

        ax1.set_xlabel('Time (s)')
        # ax1.set_ylabel('$\Delta T$ (°C)')
        ax1.set_ylabel('Temperature (°C)')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(
            0.95,
            0.70,
            model_txt,
            fontsize=12,
            color='tab:green',
            transform=ax1.transAxes,
            va='bottom', ha='right',
            # bbox=props
        )

        ax1.ticklabel_format(useMathText=True)

        fig.tight_layout()
        fig.savefig(os.path.join(data_path, csv_file + '.png'), dpi=600)
        fig.savefig(os.path.join(data_path, csv_file + '.eps'), dpi=600)
        fig.savefig(os.path.join(data_path, csv_file + '.svg'), dpi=600)
        plt.show()
