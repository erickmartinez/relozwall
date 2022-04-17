import pandas as pd
import numpy as np
import confidence as cf
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares
from scipy.linalg import svd
import matplotlib.ticker as ticker
import time

csv_calibration = r'../instruments/string_pot_calibration.csv'

initial_position_in = 3.0 - 0.625  # initial minus position of mark in flag
flag_marker_distance_in = 0.625


def poly(x, b):
    return b[0] + b[1] * x  # + b[2] * np.power(x, 2.0)


def poly_obj(beta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return poly(x, beta) - y


def poly_jac(beta: np.ndarray, x: np.ndarray, y: np.ndarray):
    identity = np.ones_like(x)
    return np.array([identity, x]).T  # , np.power(x, 2.0)]).T


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


def cm2in(value):
    return value / 2.54


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


if __name__ == "__main__":
    calibration_df = pd.read_csv(filepath_or_buffer=csv_calibration).apply(
        pd.to_numeric)
    distance_cm = calibration_df['Distance (cm)'].values - flag_marker_distance_in * 2.54
    adc_reading = calibration_df['ADC Reading'].values.astype(np.float64)
    distance_err_cm = calibration_df['Distance Error (cm)'].values

    n = len(distance_cm)

    b0_guess = distance_cm.min()
    b1_guess = np.gradient(distance_cm[-20::], adc_reading[-20::]).mean()
    b2_guess = 0.0

    b_guess = np.array([b0_guess, b1_guess])

    print(b_guess)
    all_tol = np.finfo(np.float64).eps
    res = least_squares(
        poly_obj, b_guess, args=(adc_reading, distance_cm),
        jac=poly_jac,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        # loss='soft_l1', f_scale=0.1,
        verbose=2
    )
    popt = res.x
    pcov = get_pcov(res)
    ci = cf.confint(n, popt, pcov)

    xpred = np.linspace(adc_reading.min(), adc_reading.max())
    ypred, lpb, upb = cf.predint(xpred, adc_reading, distance_cm, poly, res)

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(4.5, 3.25)

    ax2 = ax1.twinx()

    ax1.errorbar(
        adc_reading, distance_cm, yerr=distance_err_cm,
        capsize=2.75, mew=1.25, marker='o', ms=8, elinewidth=1.25,
        color='C0', fillstyle='none',
        ls='none',
        label='Data',
        zorder=1
    )

    ax1.fill_between(
        xpred, lpb, upb, color=lighten_color('C0', 0.2),
        label='Prediction Bands', zorder=0
    )

    ax1.plot(
        xpred, ypred, color='k', label='Model', zorder=2
    )

    ymin, ymax = ax1.get_ylim()
    ax2.set_ylim(cm2in(ymin), cm2in(ymax))

    leg = ax1.legend(
        loc='lower right', frameon=True, ncol=1,
        # fontsize=8, bbox_to_anchor=(1.05, 1),
        # borderaxespad=0.,
        prop={'size': 10}
    )

    model_txt = r"$f(x) = a_0 + a_1 x$" + "\n"
    model_txt += rf"$a_0$: ${latex_float(popt[0])}$, 95% CI: [${latex_float(ci[0, 0])}, {latex_float(ci[0, 1])}$]" + "\n"
    model_txt += rf"$a_1$: ${latex_float(popt[1])}$, 95% CI: [${latex_float(ci[1, 0])}, {latex_float(ci[1, 1])}$]"  # + "\n"
    # model_txt += rf"$a_2$: ${latex_float(popt[2])}$, 95% CI: [${latex_float(ci[2, 0])}, {latex_float(ci[2, 1])}$]"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(
        0.05,
        0.95,
        model_txt,
        fontsize=9,
        # color='tab:green',
        transform=ax1.transAxes,
        va='top', ha='left',
        # bbox=props
    )

    ax1.set_xlabel('ADC Reading')
    ax1.set_ylabel('Distance (cm)')
    ax2.set_ylabel('Distance (in)')
    # ax.set_title(f"Calibration Factor: ${latex_float(1.0 / popt[1], significant_digits=4)}$")

    ax1.ticklabel_format(useMathText=True)
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1E5))
    #
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    fig.tight_layout()
    fig.savefig(f'string_pot _calibration_plot.png', dpi=600)
    plt.show()
