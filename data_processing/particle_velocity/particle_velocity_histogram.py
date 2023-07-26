import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import erf, erfinv

SQRT2P = 1. / np.sqrt(2. * np.pi)
SQRT2 = 1. / np.sqrt(2.)

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC' \
            r'\LCT_R4N55_100PCT_2023-03-16_1_fitting_results'
velocity_csv = 'fitted_params.csv'

all_tol = np.finfo(np.float64).eps


def model(x, b):
    return SQRT2P / (x * b[1]) * np.exp(-(0.5 * ((np.log(x) - b[0]) / b[1]) ** 2.))


def quantile(p, mu, sigma):
    # https://en.wikipedia.org/wiki/Log-normal_distribution
    return np.exp(mu + np.sqrt(2. * sigma ** 2.) * erfinv(2. * p - 1))


def fobj(b, x, y):
    return model(x, b) - y


def jac(b, x, y):
    r = np.empty((len(x), 2), dtype=float)
    lognorm = model(x, b)
    r[:, 0] = lognorm * (np.log(x) - b[0]) / (b[1] ** 2.)
    r[:, 1] = lognorm * (((np.log(x) - b[0]) ** 2.) / (b[1] ** 3.) - 1. / b[1])
    # r[:, 2] = lognorm / b[2]
    return r


def gaussian(x, b):
    return (SQRT2P / b[1]) * np.exp(-0.5 * ((x - b[0]) / b[1]) ** 2.)


def phi(x, b):
    return 1. + erf(SQRT2 * (x - b[0]))


def skewnorm(x, b):
    return (SQRT2P / b[1]) * np.exp(-0.5 * ((x - b[1] / b[1]) ** 2.)) * (1. + erf(SQRT2 * (x - b[0])))


def jac_sk(b, x, y):
    r = np.empty((len(x), 2), dtype=float)
    ns = gaussian(x, b)
    p = phi(x, b)
    sk = ns * p
    r[:, 0] = ((x - b[0]) / b[1]) * sk - 2. * SQRT2P * ns * np.exp(-SQRT2 * (x - b[0]))
    r[:, 1] = (1 / b[1]) * ((((x - b[0]) / b[1]) ** 2.) - 1.) * sk
    return r


def fobj_sk(b, x, y):
    return skewnorm(x, b) - y


def main():
    fitting_params_df = pd.read_csv(os.path.join(base_path, velocity_csv)).apply(pd.to_numeric)
    v0 = fitting_params_df['v0 lq (cm/s)'].values

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True, frameon=True)
    fig.set_size_inches(4., 3.)

    count_v, bins_v, _ = ax.hist(v0, bins=50, density=True)
    center_v = (bins_v[:-1] + bins_v[1:]) / 2
    n = len(center_v)

    # print('center_v:', center_v)
    # print('len(center_v)', n)
    # print('count_v:', count_v)
    # print('len(bins_v)', len(count_v))
    print(f"Sum of all p's: {count_v.sum() * n * (bins_v[1] - bins_v[0])}")

    b0 = [4, 0.5]
    res = least_squares(
        fobj,
        b0,
        # loss='soft_l1', f_scale=0.1,
        jac=jac,  # '3-point',
        args=(center_v, count_v),
        bounds=([1E-15, 1E-15], [1000., 1000.]),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        diff_step=all_tol,
        max_nfev=10000 * n,
        method='trf',
        x_scale='jac',
        verbose=2
    )

    popt = res.x

    xpred = np.linspace(1., 1000., 1000)
    # ypred = model(xpred, b=[5.2, 0.8, 1.])
    ypred = model(xpred, popt)

    ax.plot(xpred, ypred, c='red',
            label=r'$f(x)=\dfrac{1}{v_0\sigma\sqrt{2\pi}}\exp\left(-\dfrac{(\ln v_0 - \mu)^2}{2\sigma^2}\right)$')

    ax.set_xlabel('$v_0$ (cm/s)')
    ax.set_ylabel('Probability density')
    ax.set_title('Pebble ejection velocity')
    ax.set_xlim(0, 1000)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(250))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))

    ax.legend(
        frameon=True, loc='upper right', fontsize=10
    )

    mu, s = popt
    # idx_mode_v = count_v == count_v.max()
    # mode_v = center_v[idx_mode_v][0] # mode obtained from the histogram
    mode_lognorm = np.exp(mu - s ** 2)
    mean_lognorm = np.exp(mu + 0.5 * s ** 2.)
    median_lognorm = np.exp(mu)
    stdev = mean_lognorm * np.sqrt(np.exp(popt[1] ** 2.) - 1.)
    q1 = quantile(p=0.025, mu=mu, sigma=s)
    q3 = quantile(p=0.975, mu=mu, sigma=s)
    # c = quantile(p=0.975, mu=mu, sigma=s)
    # sqrtn = np.sqrt(len(v0))
    """
    An attempt to get 95% confidence intervals
    """
    # r1 = mean_lognorm - s * c / sqrtn
    # r2 = mean_lognorm + s * c / sqrtn

    ax.axvspan(q1, q3, color='gold', alpha=0.1)
    ax.axvline(x=q1, color='gold', lw=1., ls='-')
    ax.axvline(x=median_lognorm, color='k', ls=':', lw=1.25)
    ax.axvline(x=q3, color='gold', lw=1., ls='-')
    ax.plot([median_lognorm], [model(median_lognorm, [mu, s])], ls='none', marker='o', ms=8, c='tab:green', mec='k')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    results_v_txt = f'Median: {median_lognorm:>3.0f} cm/s\nMode:    {mode_lognorm:>3.0f} cm/s\n'
    results_v_txt += f'$q_{{0.025}}$:    {q1:>3.0f} cm/s\n'
    results_v_txt += f'$q_{{0.975}}$:    {q3:>3.0f} cm/s'
    ax.text(
        0.94,
        0.7,
        results_v_txt,
        fontsize=11,
        # color='tab:green',
        transform=ax.transAxes,
        va='top', ha='right',
        bbox=props
    )

    ax.text(
        0.015, 0.15, '$q_{0.025}$', fontsize=11, transform=ax.transAxes, va='bottom', ha='left'
    )

    ax.text(
        0.85, 0.2, '$q_{0.975}$', fontsize=11, transform=ax.transAxes, va='bottom', ha='right'
    )

    print(f'mu:\t{popt[0]:.3f} cm/s')
    print(f'sigma:\t{popt[1]:.3f} cm/s')
    print(f'Mean:\t{mean_lognorm:.3E} cm/s')
    print(f'STD:\t{stdev:.3E} cm/s')
    # print(f'r1:\t{r1:.1f} cm/s')
    file_tag = 'v0_lognorm_stats'
    fig.savefig(os.path.join(base_path, file_tag + '.png'), dpi=600)
    fig.savefig(os.path.join(base_path, file_tag + '.eps'), dpi=600)
    fig.savefig(os.path.join(base_path, file_tag + '.svg'), dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
