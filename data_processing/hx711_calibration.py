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

csv_calibration = r'../instruments/Load Cell Calibration - 20KG CELL.csv'
load_cell_range = '20kg'


def poly(x, b):
    return b[0] + b[1] * x #+ b[2] * np.power(x, 2.0)


def poly_obj(beta: np.ndarray, x: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    w = np.ones_like(x, dtype=np.float64) if weights is None else weights
    return (poly(x, beta) - y) * w


def poly_jac(beta: np.ndarray, x: np.ndarray, y: np.ndarray, weights: np.ndarray = None):
    identity = np.ones_like(x)
    return np.array([identity, x]).T


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
    calibration_df = pd.read_csv(filepath_or_buffer=csv_calibration).apply(
        pd.to_numeric)
    force_n = calibration_df['Force (N)'].values
    reading = calibration_df['Reading'].values.astype(np.float64)
    force_err = calibration_df['Force Error (N)'].values
    weights = 1.0 / force_err
    weights /= weights.max()

    n = len(force_n)

    b0_guess = 0.0
    b1_guess = np.gradient(force_n, reading).mean()
    b2_guess = 0.0

    b_guess = np.array([b0_guess, b1_guess])
    all_tol = np.finfo(np.float64).eps
    res = least_squares(
        poly_obj, b_guess, args=(reading, force_n, weights),
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

    y_model = poly(reading, popt)
    rmse = np.linalg.norm(y_model - force_n) / np.sqrt(n)

    xpred = np.linspace(reading.min(), reading.max())
    pred_mode = 'observation'  # functional
    ypred, lpb, upb = cf.predint(xpred, reading, force_n, poly, res, mode=pred_mode)
    delta = np.abs(upb - ypred)
    delta_pct = delta / ypred
    print(f'95% prediction confidence interval (max): {delta.max():.1f} (N)')
    print(f'95% prediction confidence interval (max): {100.0*delta_pct.max():.1f} (%)')

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3.25)



    ax.errorbar(
        reading, force_n, yerr=force_err,
        capsize=2.75, mew=1.25, marker='o', ms=8, elinewidth=1.25,
        color='C0', fillstyle='none',
        ls='none',
        label='Data',
        zorder=1
    )

    ax.fill_between(
        xpred, lpb, upb, color=lighten_color('C0', 0.2),
        label='Prediction Bands', zorder=0
    )

    ax.plot(
        xpred, ypred, color='C0', label='Model', zorder=2
    )

    leg = ax.legend(
        loc='lower right', frameon=True, ncol=1,
        # fontsize=8, bbox_to_anchor=(1.05, 1),
        # borderaxespad=0.,
        prop={'size': 10}
    )

    model_txt = r"$f(x) = a_0 + a_1 x$" + "\n"
    model_txt += rf"$a_0$: ${latex_float(popt[0])}$, 95% CI: [${latex_float(ci[0,0])}, {latex_float(ci[0,1])}$]" + "\n"
    model_txt += rf"$a_1$: ${latex_float(popt[1])}$, 95% CI: [${latex_float(ci[1,0])}, {latex_float(ci[1,1])}$]" + "\n"
    model_txt += rf"95% Prediction Error (avg): {delta_pct.mean()*100.0:.1f} (%)"
    # model_txt += rf"$a_2$: ${latex_float(popt[2])}$, 95% CI: [${latex_float(ci[2,0])}, {latex_float(ci[2,1])}$]"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(
        0.05,
        0.95,
        model_txt,
        fontsize=9,
        # color='tab:green',
        transform=ax.transAxes,
        va='top', ha='left',
        # bbox=props
    )


    ax.set_xlabel('HX711 ADC Reading')
    ax.set_ylabel('Force (N)')
    ax.set_title(f"Calibration Factor: $\\mathregular{{{latex_float(1.0/popt[1],significant_digits=4)}}}$")

    ax.ticklabel_format(useMathText=True)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1E5))
    #
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    fig.tight_layout()
    print(f"Calibration factor: {1 / popt[1]:7.4E}")
    fig.savefig(f'{load_cell_range}_calibration_plot.png', dpi=600)
    plt.show()



