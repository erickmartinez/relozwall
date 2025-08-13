import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from scipy.optimize import OptimizeResult, least_squares
import data_processing.confidence as cf
from data_processing.utils import lighten_color
import os
import platform
import json
import re

data_path = 'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/simulations/nu_vs_t_sim'
dt = 0.01
fit_min = 0.6

platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'


def normalize_path(the_path):
    global platform_system, drive_path
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    else:
        the_path = the_path.replace('/', '\\')
    return os.path.join(drive_path, the_path)


def load_plot_style():
    with open('../../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def model(t: np.ndarray, beta: np.ndarray):
    return beta[0] + t * beta[1]


def residual(b, t, r):
    return model(t, b) - r


def jac(b, t, r):
    n, p = len(r), len(b)
    j = np.empty((n, p), dtype=np.float64)
    j[:, 0] = np.ones(n, dtype=np.float64)
    j[:, 1] = t
    return j


def main():
    global data_path
    base_path = normalize_path(base_path)
    file_list = [fn for fn in os.listdir(base_path) if fn.endswith('.csv')]
    pattern = re.compile(r'.*?q(\d+).csv')
    structured_list = []
    for fn in file_list:
        match = pattern.match(fn)
        q = int(match.group(1))
        structured_list.append({'q': q, 'fn': fn})
    structured_list = sorted(structured_list, key=lambda d: d['q'])
    # print(structured_list)
    out_path = os.path.join(base_path, 'interpolated')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    load_plot_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 4.)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Recession (mm)')
    ax.set_xlim(0., 3.0)
    ax.set_ylim(0, 3.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    # ax.tick_params(right=True, which='both', direction='out')
    line_colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    nu_df = pd.DataFrame(columns=[
        'Q (MW/m^2)', 'nu (cm/s)', 'nu_lpb (cm/s)', 'nu_upb (cm/s)', 'nu_delta (cm/s)'
    ])
    for i, item in enumerate(structured_list):
        q = item['q']
        fn = item['fn']
        print(f'Loading {fn}')
        df: pd.DataFrame = pd.read_csv(os.path.join(base_path, fn), comment='#').apply(pd.to_numeric)
        df = df.sort_values(by=['time (s)'], ascending=True)
        df.dropna(inplace=True)
        t = df['time (s)'].values
        r = df['r (mm)'].values
        f = interp1d(x=t, y=r, assume_sorted=True, bounds_error=False, fill_value='extrapolate')
        n_points = t.max() / dt + 1
        t_interp = np.arange(0, n_points) * dt
        r_interp = f(t_interp)
        out_df = pd.DataFrame(data={
            'time (s)': t_interp,
            'r (mm)': r_interp
        })
        tag = os.path.splitext(fn)[0]
        dc = line_colors[i]
        ax.plot(t_interp, r_interp, label=f'$q$ = {q:2d} MW/m$^{{\mathregular{{2}}}}$', c=dc, lw=2.)
        msk_fit = (t_interp >= fit_min)
        t_fit = t_interp[msk_fit]
        r_fit = r_interp[msk_fit]
        print(r_fit[-1], r_fit[0])
        a0 = r_fit[0]
        a1 = (r_fit[-1] - r_fit[0]) / (t_fit[-1] - t_fit[0])
        print([a0, a1])
        res = least_squares(
            fun=residual,
            x0=[a0, a1],
            jac=jac,
            args=(t_fit, r_fit),
            loss='cauchy',
            f_scale=0.1,
            verbose=2
        )
        popt = res.x
        ci = cf.confidence_interval(res=res, confidence=0.95)
        nu = popt[1] * 0.1
        nu_ci = ci[1,:] * 0.1
        dnu = np.max(np.abs(nu_ci - nu))
        new_nu_df = pd.DataFrame(data={
            'Q (MW/m^2)': [q],
            'nu (cm/s)': [nu],
            'nu_lpb (cm/s)': [nu_ci[0]],
            'nu_upb (cm/s)': [nu_ci[1]],
            'nu_delta (cm/s)': [dnu]
        })
        nu_df = pd.concat([nu_df, new_nu_df]).reset_index(drop=True)
        # nu_txt = r' $\nu$ = ' + f'{nu:.4f} ± {dnu:.4f} cm/s\n'
        nu_txt = r' $\nu$ = ' + f'{nu:.4f} cm/s\n'
        y_pred, delta = cf.prediction_intervals(
            model=model, x_pred=t_fit, ls_res=res, jac=jac, new_observation=True,
        )
        ax.plot(t_fit, y_pred, ls='--', lw=1., c=dc)
        """
        Find the recession at 0.8 s (time at around which the steady state begins)
        """
        r_ss = f(0.8)
        ax.plot([0.8], [r_ss], c=dc, marker='o', fillstyle='full')
        pc = lighten_color(color=dc, amount=0.15)
        ax.fill_between(t_fit, y_pred - delta, y_pred + delta, color=dc, alpha=0.25)
        xt, yt = t_fit[-1], r_fit[-1]
        ha = 'right'
        va = 'bottom'
        if i + 1 == len(structured_list):
            xt, yt = t_fit.mean(), r_fit.mean()
        ax.text(
            xt, yt, nu_txt,
            va=va, ha=ha,
            color=dc, fontsize=11,
        )

        # Generate output csv file with the result from the fit in the comments
        out_csv = os.path.join(out_path, 'tag_fit_results.csv')
        with open(out_csv, 'a') as fout:
            fout.write(f'## Simulation for q = {q} MW/m^2')
            fout.write(f'## Model = y = a0 + a1 * x')
            fout.write(f'# a0: {popt[0]/10:.5f}, 95% CI [{ci[0, 0]/10:.5f}, {ci[0, 1]/10:.5f} cm')
            fout.write(f'# a1: {popt[1]/10:.5f}, 95% CI [{ci[1, 0]/10:.5f}, {ci[1, 1]/10:.5f} cm/s')
            fit_df = pd.DataFrame(data={
                'time (s)': t_fit,
                'r_fit (mm)': y_pred,
                'lpb (mm)': y_pred - delta,
                'upb (mm)': y_pred + delta
            })
            fit_df.to_csv(fout)

        out_df.to_csv(path_or_buf=os.path.join(out_path, tag + '.csv'), index=False)

    ax.legend(loc='upper left', frameon=True, fontsize=11)
    # ax.set_title('$F_{\mathrm{B}}$ = 1.5 N')
    nu_df.to_csv(os.path.join(out_path, 'fitted_nu_vs_q_interp.csv'), index=False)
    fig.savefig(os.path.join(base_path, 'fig_nu_vs_q_ft70_interp.svg'), dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
