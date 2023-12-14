import data_processing.confidence as cf
from scipy.optimize import least_squares, OptimizeResult
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import matplotlib.ticker as ticker
from data_processing.utils import latex_float_with_error
import os
import json


calibration_csv = r'./linear_pot_adc - 20231207.csv'


def model(x, b):
    return b[0] + b[1] * x

def residual(b, x, y):
    return y - model(x, b)

def jac(b, x, y):
    n, p = len(x), len(b)
    jj = np.ones(shape=(n, p), dtype=np.float64)
    xx = x.copy()
    for i in range(1, p):
        jj[:, i] = xx
        xx = xx * x
    return -jj

def poly(x, b):
    res = b[0] * np.ones_like(x)
    xx = x.copy()
    for i in range(1, len(b)):
        res += b[i] * xx
        xx *= x
    return res

def residual_poly(b, x, y):
    return y - poly(x, b)

def main():
    cal_df = pd.read_csv(calibration_csv).apply(pd.to_numeric)
    cal_df.sort_values(by=['ADC value'], inplace=True)
    adc_val = cal_df['ADC value'].values
    position = cal_df['Position (mm)'].values
    in_report = cal_df['Report'].astype(bool)

    adc_val = adc_val[in_report]
    position = position[in_report]

    file_tag = os.path.splitext(calibration_csv)[0]
    file_tag = file_tag[2::]

    eps = float(np.finfo(np.float64).eps)
    x0 = np.zeros(2)
    x0[0] = position.min()
    x0[1] = (position[-1] - position[0]) / (adc_val[-1] - adc_val[0])
    res: OptimizeResult = least_squares(
        x0=x0, fun=residual, args=(adc_val, position), jac=jac, xtol=eps, ftol=eps,
        loss='soft_l1', f_scale=0.1, verbose=2
    )

    popt = res.x

    adc_min = adc_val.min()
    adc_max = adc_val.max()
    n_adc = adc_max - adc_min
    x_pred = np.linspace(adc_min, adc_max, n_adc)
    ci = cf.confidence_interval(res)
    popt_delta = ci[:, 1] - ci[:, 0]
    y_pred, delta = cf.prediction_intervals(
        model=model, x_pred=x_pred, ls_res=res, jac=jac, new_observation=False
    )

    lpb, upb = y_pred-delta, y_pred+delta

    # get an equation for the error of the prediction
    b0 = np.array([x0[0], x0[1], 1.])
    res_d = least_squares(
        x0=b0, fun=residual_poly, args=(x_pred, delta), xtol=eps, ftol=eps, jac=jac,
        verbose=2
    )

    delta_pred = poly(x_pred, res_d.x)

    with open('../../../data_processing/plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3.)

    ax.fill_between(x_pred, lpb, upb, color='C0', alpha=0.5)
    ax.plot(adc_val, position, fillstyle='none', color='C0', marker='o', ls='none')
    ax.plot(x_pred, y_pred, color='tab:red')
    ax.plot(x_pred, y_pred+delta_pred,color='k')

    ax.set_xlabel('ADC value')
    ax.set_ylabel('Distance [mm]')
    ax.set_title(file_tag)

    info_txt = '$f(x) = b_0 + b_1 x$\n'
    info_txt += f'$b_0$: ${popt[0]:>5.3f} \pm {popt_delta[0]:>6.4f}$\n'
    info_txt += f'$b_1$: ${latex_float_with_error(popt[1], error=popt_delta[1], digits=2)}$'

    ax.text(
        0.95, 0.05, info_txt, transform=ax.transAxes, fontsize=11, color='tab:red',
        va='bottom', ha='right'
    )

    popt_e = res_d.x
    ci_d = cf.confidence_interval(res_d)
    popt_d = ci_d[:, 1] - ci_d[:, 0]
    error_txt = '$f(x) = b_0 + b_1 x + b_2 x^2$\n'
    error_txt += f'$b_0$: ${popt_e[0]:>5.3f} \pm {popt_d[0]:>6.4f}$\n'
    error_txt += f'$b_1$: ${latex_float_with_error(popt_e[1], error=popt_d[1], digits=2)}$\n'
    error_txt += f'$b_2$: ${latex_float_with_error(popt_e[2], error=popt_d[2], digits=2)}$'

    print('***** Model *****')
    for i, p in enumerate(popt):
        print(f'b[{i}]: {popt[i]:>5.3E} ± {popt_delta[i]:>5.3E}')

    print('***** Error fit *****')
    for i, p in enumerate(popt_e):
        print(f'b[{i}]: {popt_e[i]:>5.3E} ± {popt_d[i]:>5.3E}')

    ax.text(
        0.05, 0.95, error_txt, transform=ax.transAxes, fontsize=11, color='k',
        va='top', ha='left'
    )

    fig.savefig(file_tag + '.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    main()