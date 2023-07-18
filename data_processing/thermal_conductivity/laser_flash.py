import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import root_scalar
import pandas as pd
import json

"""
This code plots Fig 1 from W.J. Parker, et al. J. Appl. Phys. 32, 1679 (1961)
https://doi.org/10.1063/1.1728417

It also finds the roots for V(w) for different values of V and saves it in a csv file
"""

PI2 = np.pi ** 2.

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
        sign = -1.0 if i % 2 == 1.0 else 1.0
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
    sol = root_scalar(get_wh, x0=1., fprime=get_dv, method='newton', xtol=all_tol,
                      rtol=all_tol, maxiter=10000)

    wh = sol.root
    slope = get_dv(wh)
    wx = wh - 0.5 / slope
    # b = 0.5 - slope * wh
    # wx = - b / slope
    print(f'Slope = {slope:.3f}')

    """
    Make a table with the values of the intercept for different values of V
    """
    selected_v = 10. * np.arange(0, 10) 
    selected_v = np.hstack([selected_v, 33.3, 66.7])
    selected_v.sort()
    resulting_w = np.empty_like(selected_v)
    for i, vv in enumerate(selected_v):
        f = lambda y: get_v(y) - vv/100.
        soli = root_scalar(
            f, x0=0.5, fprime=get_dv, method='newton', xtol=all_tol,
            rtol=all_tol, maxiter=10000
        )
        resulting_w[i] = soli.root

    resulting_k = resulting_w / PI2

    k_df = pd.DataFrame(data={
        'V (%)': selected_v,
        'w(V)': resulting_w,
        'k(V)': resulting_k
    })

    print(k_df)
    k_df.to_csv('dimensionless_parameters.csv', index=False)

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

    wh_txt = f'$\\omega_{{1/2}}$ = {wh:.4f}, $V$ = {get_v(wh):.4f}\n'
    wh_txt += f'$\\omega_{{x}}$ = {wx:.4f}'
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
