import numpy as np
import pandas as pd
from scipy.optimize import least_squares, OptimizeResult, differential_evolution
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os
import data_processing.confidence as cf
from data_processing.utils import latex_float, lighten_color

transmission_csv = 'Transmission measurements - Deposition cone.csv'
slide_width_mm = 25.24
substrate_source_distance_in = 10.5


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


def modified_knudsen(r_, h0_, n_):
    cos_q = h0_ / np.sqrt(r_ ** 2. + h0_ ** 2.)
    return np.power(cos_q, n_ + 2.)


def fobj(b, r_, h0_, d_, w_=1.):
    return w_ * (modified_knudsen(r_, h0_, b[0]) - d_)


def fobj_de(b, r_, h0_, d_, w_):
    return 0.5 * np.linalg.norm(fobj(b, r_, h0_, d_, w_))


def jac(b, r_, h0_, d_, w_):
    cos_q = h0_ / np.sqrt(r_ ** 2. + h0_ ** 2.)
    jj = np.empty((len(r_), 1))
    jj[:, 0] = np.log(cos_q) * modified_knudsen(r_, h0_, b[0])
    return jj


def main():
    df = pd.read_csv(transmission_csv)
    columns = df.columns
    df[columns[2::]] = df[columns[2::]].apply(pd.to_numeric)
    print(df)
    slide_width_cm = slide_width_mm * 0.1
    r = np.array([(i+0) * slide_width_cm for i in range(5)])
    d = df['Film thickness (nm)'].values
    d_lb = df['Thickness lb (nm)'].values
    d_ub = df['Thickness ub (nm)'].values

    d0 = d[0]
    dn = d / d0
    dn_lb, dn_ub = d_lb / d0, d_ub / d0
    yerr = (dn - dn_lb, dn_ub - dn)
    dy = np.stack(yerr).T
    weigths = np.power(np.diff(dy, axis=1), -1)[:,0]
    weigths /= weigths.max()
    print('weights:', weigths)
    h0 = substrate_source_distance_in * 2.54
    all_tol = np.finfo(np.float64).eps
    nn = len(r)

    res_de: OptimizeResult = differential_evolution(
        func=fobj_de,
        args=(r, h0, dn, weigths),
        x0=[0.],
        bounds=[(-10, 10)],
        maxiter=nn * 1000000,
        tol=all_tol,
        atol=all_tol,
        workers=-1,
        updating='deferred',
        recombination=0.5,
        strategy='best1bin',
        mutation=(0.5, 1.5),
        init='sobol',
        polish=False,
        disp=True
    )

    res = least_squares(
        fobj,
        res_de.x,
        # loss='soft_l1', f_scale=0.1,
        jac=jac,
        args=(r, h0, dn, weigths),
        bounds=([-10., ], [10.]),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        diff_step=all_tol,
        max_nfev=10000 * nn,
        method='trf',
        x_scale='jac',
        verbose=2
    )

    print("res_de.x:", res_de.x)
    print("res_ls.x:", res.x)

    popt = res.x
    n_opt = popt[0]
    ci = cf.confidence_interval(res=res)
    dn_opt = np.max(np.abs(ci - n_opt), axis=1)[0]
    print(f'n = {n_opt:.3f}±{dn_opt}')
    xp = np.linspace(r.min(), r.max(), 500)

    def model_restricted(x, b):
        return modified_knudsen(x, h0, b[0])

    yp, lpb, upb = cf.predint(x=xp, xd=r, yd=dn, func=model_restricted, res=res, mode='observation')
    # yp = modified_knudsen(xp, h0, popt[0])

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)

    model_str = r"$\dfrac{d}{d_0} = \dfrac{h_0^2}{h^2}\cos^n(\theta)$"
    ax.errorbar(
        r, dn, yerr=yerr, xerr=0.5 * slide_width_cm, ms=7, ls='none', marker='o', color='C0', label='data', mew=1.5,
        fillstyle='none',
        capsize=2.75, elinewidth=1.25, zorder=2
    )
    ax.plot(xp, yp, ls='-', color='k', label=model_str, zorder=1)
    # ax.fill_between(xp, lpb, upb, color=lighten_color('C0', 0.35), zorder=0)

    ax.set_xlabel('$r$ [cm]')
    ax.set_ylabel('$d/d_0$')
    ax.set_title('Deposition profile')

    ax.set_xlim(0, 12)
    ax.set_ylim(0.4, 1.2)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2.))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    n_str = f"$n = {popt[0]:.1f}±{dn_opt:.1f}$"
    ax.text(
        0.95, 0.95,
        n_str,
        va='top',
        ha='right',
        transform=ax.transAxes,
        color='k',
        fontsize=11
    )

    ax.legend(
        loc='lower left', fontsize=11, frameon=True
    )

    fig.savefig('deposition_profile_cosine_law.png', dpi=600)
    fig.savefig('deposition_profile_cosine_law.svg', dpi=600)
    fig.savefig('deposition_profile_cosine_law.pdf', dpi=600)
    fig.savefig('deposition_profile_cosine_law.eps', dpi=600)
    plt.show()


if __name__ == '__main__':
    load_plot_style()
    main()
