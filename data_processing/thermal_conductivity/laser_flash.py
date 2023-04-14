import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import root_scalar
import json


def get_dv(ww):
    s = 0
    for i in range(1, 502):
        sign = -1.0 if i % 2 == 1 else 1.0
        n2 = i * i
        s += sign * n2 * np.exp(-n2 * ww)
    r = -2.0 * s
    return r


def get_v(ww):
    s = 0
    for i in range(1, 502):
        sign = -1 if i % 2 == 1.0 else 1.0
        n2 = i * i
        s += sign * np.exp(-n2 * ww)
    r = 1.0 + 2.0 * s
    return r


def get_wh(ww):
    return get_v(ww) - 0.5


def fprime(ww):
    return get_dv(ww)


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


if __name__ == '__main__':
    w = np.linspace(1E-2, 10, 200)
    v = get_v(w)
    dv = get_dv(w)
    all_tol = np.finfo(np.float64).eps
    # sol = root_scalar(get_wh, bracket=[1.1, 3], method='brentq', xtol=all_tol, rtol=all_tol ** 0.5,
    #                   maxiter=100 * len(w))
    sol = root_scalar(get_wh, x0=1.3, fprime=get_dv, method='newton', xtol=all_tol,
                      rtol=all_tol ** 0.5, maxiter=1000 * len(w))

    wh = sol.root
    slope = get_dv(wh)
    wx = wh - 0.5 / slope
    print(f'Slope = {slope:.3f}')

    load_plot_style()

    fig, ax1 = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)

    color = 'tab:blue'
    ax1.set_xlabel(r'$\omega = \frac{\pi^2}{L^2} \alpha t$')
    ax1.set_ylabel('$V$', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(w, v, color=color)
    ax1.plot([wh], [0.5], marker='o', mec='tab:red', mfc='none', mew=1.5)
    ax1.axvline(x=wh, color='tab:grey', ls='--', lw=1.0)
    ax1.axhline(y=0.5, color='tab:grey', ls='--', lw=1.0)
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    plt.axline((wh, 0.5), slope=slope, color="tab:grey", ls='--', lw=1.0)

    wh_txt = f'$\\omega_{{1/2}}$ = {wh:.3f}, $V$ = {get_v(wh):.3f}\n'
    wh_txt += f'$\\omega_{{x}}$ = {wx:.3f}'
    ax1.text(
        0.95, 0.05, wh_txt,
        ha='right', va='bottom',
        transform=ax1.transAxes,
        fontsize=11
    )

    color = 'tab:orange'
    ax2 = ax1.twinx()
    ax2.set_ylabel('$dV/d\\omega$', color=color)  # we already handled the x-label with ax1
    ax2.plot(w, dv, color=color, lw=1.25, ls='-.')
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_ylim(0.0, 1.0)

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 10)

    ax1.set_zorder(1)
    ax2.set_zorder(0)
    ax1.patch.set_visible(False)
    plt.show()
