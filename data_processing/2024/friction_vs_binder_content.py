import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import json
from matplotlib import rcParams
import os
from scipy.stats.distributions import t
from scipy.optimize import least_squares

path_to_data = './data/friction_vs_binder_content'
base_line_csv = 'FRICTION_BASELINE_XTBC003_0350C_0.51CMPS_2024-09-11_2.csv'
baseline_mean_speed = 0.545  # cm/s


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    rcParams.update(plot_style)

def poly(x, b):
  n = len(b)
  r = np.zeros(len(x))
  for i in range(n):
    r += b[i] * x**i
  return r

def res_poly(b, x, y, w=1):
  return (poly(x, b) - y)*w

def jac_poly(b, x, y, w=1):
  n = len(b)
  r = np.zeros((len(x), n))
  for i in range(n):
    r[:,i] = w*x**i
  return r


def main():
    global path_to_data, base_line_csv, baseline_mean_speed
    # load the base line
    baseline_df: pd.DataFrame = pd.read_csv(os.path.join(path_to_data, base_line_csv), comment='#').apply(pd.to_numeric)
    time_s = baseline_df['Time (s)'].values
    record_position = baseline_df['Position (cm)'].values
    force_n = baseline_df['Force (N)'].values
    force_err = np.abs(baseline_df['Force error (N)'].values)
    x0 = record_position[0]
    position = x0 + time_s * baseline_mean_speed

    load_plot_style()

    fig_b, ax_b = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig_b.set_size_inches(4., 2.5)
    # ax.errorbar(
    #     position, force_n, xerr=0.07 * 2.54, yerr=force_err,
    #     capsize=2.75, mew=1.25, marker='o', ms=8, elinewidth=1.25,
    #     color='C0', mfc='none'
    # )
    ax_b.plot(position, force_n, color='C0', marker='none')
    ax_b.fill_between(position, force_n - force_err, force_n + force_err, color='C0', alpha=0.25)

    ax_b.set_xlabel('x (cm)')
    ax_b.set_ylabel('F (N)')
    ax_b.set_xlim(40, 70.)
    ax_b.set_ylim(-5, 45)
    ax_b.xaxis.set_major_locator(ticker.MultipleLocator(5.))
    ax_b.xaxis.set_minor_locator(ticker.MultipleLocator(1.))
    ax_b.yaxis.set_major_locator(ticker.MultipleLocator(10.))
    ax_b.yaxis.set_minor_locator(ticker.MultipleLocator(2.))
    ax_b.set_title('Baseline')
    fig_b.savefig(os.path.join('./figures/friction_vs_binder_content', 'baseline.png'), dpi=600)
    fig_b.savefig(os.path.join('./figures/friction_vs_binder_content', 'baseline.pdf'), dpi=600)
    # fig_b.show()

    f_b = interp1d(position, force_n, fill_value="extrapolate", bounds_error=False)
    f_e = interp1d(position, force_err, fill_value="extrapolate", bounds_error=False)

    # Read the Excel file containing a database with the filenames of the csv results to analyze
    # and the corresponding effective average extruder speed
    db_df: pd.DataFrame = pd.read_excel(
        os.path.join('./data/friction_vs_binder_content/friction_vs_binder_db.xlsx'), sheet_name=0
    )
    db_df['mean_speed_cmps'] = db_df['mean_speed_cmps'].apply(pd.to_numeric)
    db_columns = db_df.columns
    db_df[db_columns[1::]] = db_df[db_columns[1::]].apply(pd.to_numeric)
    nrows = len(db_df)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True)
    fig.subplots_adjust(hspace=0, left=0.15, right=0.95, bottom=0.125, top=0.95)
    fig.set_size_inches(4., 5.5)

    fig_n, axes_n = plt.subplots(nrows=nrows, ncols=1, sharex=True)
    fig_n.subplots_adjust(hspace=0, left=0.15, right=0.95, bottom=0.125, top=0.95)
    fig_n.set_size_inches(4., 5.5)

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    for i, row in db_df.iterrows():
        experiment_df = pd.read_csv(os.path.join(path_to_data, row['filename'] + '.csv'), comment='#').apply(
            pd.to_numeric
        )
        mean_speed = float(row['mean_speed_cmps'])
        sample_area = float(row['sample_area'])
        sample_area_err = float(row['sample_area_err'])
        time_s = experiment_df['Time (s)'].values
        record_position = experiment_df['Position (cm)'].values
        force_n = experiment_df['Force (N)'].values
        force_err = np.abs(experiment_df['Force error (N)'].values)
        binder_content = float(row['binder_content'])
        x0 = record_position[0]
        position = x0 + time_s * mean_speed
        force_baseline = f_b(position)
        force_baseline_error = f_e(position)
        force_sample = force_n - force_baseline
        force_sample_error = np.linalg.norm(np.vstack([force_err, force_baseline_error]).T, axis=1)
        n_points = len(force_sample)
        t_val = t.ppf(1. - 0.5*0.05, n_points - 1)
        mean_force = force_sample.mean()
        force_std = force_sample.std(ddof=1)
        force_se = force_std * t_val / np.sqrt(n_points)
        axes[i].plot(position-position[0], force_sample, color=colors[i], marker='none', label=f'{binder_content:.1f} % binder')
        axes[i].fill_between(
            position-position[0], force_sample - force_sample_error, force_sample + force_sample_error, color=colors[i], alpha=0.25,
        )
        # axes[i].set_ylabel('F (N)')
        axes[i].yaxis.set_major_locator(ticker.MultipleLocator(10.))
        axes[i].yaxis.set_minor_locator(ticker.MultipleLocator(2.))
        axes[i].set_ylim(-5, 15)
        axes[i].legend(loc='upper left', fontsize=9)
        mean_force_txt = (rf'$\langle F \rangle = {mean_force:.1f}\pm{force_se:.2f}~\mathrm{{N}}$')
        axes[i].text(
            0.95, 0.95,
            mean_force_txt,
            transform=axes[i].transAxes,
            ha='right', va='top',
            fontsize=9
        )

        # Same plot but normalized by the sample area
        force_n = force_sample / sample_area
        force_n_err = force_n * np.linalg.norm(np.stack([force_sample_error/force_n, (sample_area_err / sample_area)*np.ones_like(force_n)]).T, axis=1)
        mean_force_n = force_n.mean()
        force_n_std = force_n.std(ddof=1)
        force_n_se = force_n_std * t_val / np.sqrt(n_points)

        axes_n[i].plot(position - position[0], force_n, color=colors[i], marker='none',
                     label=f'{binder_content:.1f} % binder')

        axes_n[i].fill_between(
            position - position[0], force_n - force_n_err, force_n + force_n_err,
            color=colors[i], alpha=0.25,
        )
        # axes[i].set_ylabel('F (N)')
        axes_n[i].yaxis.set_major_locator(ticker.MultipleLocator(3.))
        axes_n[i].yaxis.set_minor_locator(ticker.MultipleLocator(1.))
        axes_n[i].set_ylim(-5., 5.)
        axes_n[i].legend(loc='upper left', fontsize=9)
        mean_force_n_txt = (rf'$\langle F \rangle = {mean_force_n:.2f}\pm{force_n_se:.3f}~\mathrm{{N/cm^2}}$')
        axes_n[i].text(
            0.98, 0.95,
            mean_force_n_txt,
            transform=axes_n[i].transAxes,
            ha='right', va='top',
            fontsize=9
        )
    axes[-1].set_xlabel('x (cm)')
    fig.supylabel('F (N)')
    axes[-1].set_xlim(0, 25.)
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(5.))
    axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(1.))

    axes_n[-1].set_xlabel('x (cm)')
    fig_n.supylabel('F/S$_{\mathregular{c}}$ (N/cm$^{\mathregular{2}}$)')
    axes_n[-1].set_xlim(0, 25.)
    axes_n[-1].xaxis.set_major_locator(ticker.MultipleLocator(5.))
    axes_n[-1].xaxis.set_minor_locator(ticker.MultipleLocator(1.))

    fig.savefig(os.path.join('./figures/friction_vs_binder_content/force_vs_binder.png'), dpi=600)
    fig.savefig(os.path.join('./figures/friction_vs_binder_content/force_vs_binder.pdf'), dpi=600)

    fig_n.savefig(os.path.join('./figures/friction_vs_binder_content/force_n_vs_binder.png'), dpi=600)
    fig_n.savefig(os.path.join('./figures/friction_vs_binder_content/force_n_vs_binder.pdf'), dpi=600)

    # ax.set_title('Baseline')
    fig_ml_g, ax_mlg = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig_ml_g.set_size_inches(4., 4.)
    fig_ml_g.subplots_adjust(hspace=0, left=0.175, right=0.95, bottom=0.125, top=0.95)

    # Plot the mass los vs binder content for the extrusion test
    binder_content = db_df['binder_content'].values
    mass_loss_g = db_df['mass_loss_g'].values
    mass_loss_err_g = db_df['mass_loss_err_g'].values
    mass_loss_pct = db_df['mass_loss_pct'].values
    mass_loss_err_pct = db_df['mass_loss_err_pct'].values

    w1, w2 = np.power(mass_loss_err_g, -2), np.power(mass_loss_err_pct, -2)
    all_tol = float(np.finfo(np.float64).eps)

    lsq_poly1 = least_squares(
        res_poly, x0=[mass_loss_g[0], -0.1, 0.0, 0.], args=(binder_content, mass_loss_g, w1),
        loss='linear', f_scale=0.1,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        jac=jac_poly,
        verbose=0,
        max_nfev=10000 * len(binder_content),
        # x_scale='jac',
    )

    lsq_poly2 = least_squares(
        res_poly, x0=[mass_loss_pct[0], -0.1, 0.0, 0.], args=(binder_content, mass_loss_pct, w2),
        loss='linear', f_scale=0.1,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        jac=jac_poly,
        verbose=0,
        max_nfev=10000 * len(binder_content),
        # x_scale='jac',
    )

    xpred = np.linspace(2.5, 20, num=200)
    ypred1 = poly(xpred, lsq_poly1.x)
    ypred2 = poly(xpred, lsq_poly2.x)

    ax_mlg[0].errorbar(
        binder_content, mass_loss_g, yerr=mass_loss_err_g,
        capsize=2.75, mew=1.25, marker='^', ms=8, elinewidth=1.25,
        color='C0', fillstyle='none',
        ls='none',
        label='Mass loss',
    )

    ax_mlg[0].plot(
        xpred, ypred1, marker='none', ls='--', color='C0', lw=1., alpha=0.5
    )

    ax_mlg[1].errorbar(
        binder_content, mass_loss_pct, yerr=mass_loss_err_pct,
        capsize=2.75, mew=1.25, marker='v', ms=8, elinewidth=1.25,
        color='C1', fillstyle='none',
        ls='none',
        label='% loss',
    )

    ax_mlg[1].plot(
        xpred, ypred2, marker='none', ls='--', color='C1', lw=1., alpha=0.5
    )

    ax_mlg[0].set_ylabel('$\Delta m$ (g)')
    ax_mlg[1].set_ylabel('$\Delta m$ (%)')
    ax_mlg[-1].set_xlabel('Matrix wt %')

    ax_mlg[-1].set_xlim(0, 25)
    ax_mlg[-1].xaxis.set_major_locator(ticker.MultipleLocator(5.))
    ax_mlg[-1].xaxis.set_minor_locator(ticker.MultipleLocator(2.5))

    ax_mlg[0].set_ylim(-0.005, 0.035)

    ax_mlg[0].yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax_mlg[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.005))

    ax_mlg[1].set_ylim(-0.05, 0.35)
    ax_mlg[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax_mlg[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    fig_ml_g.savefig(os.path.join('./figures/friction_vs_binder_content/xt_dm_n_vs_binder.png'), dpi=600)
    fig_ml_g.savefig(os.path.join('./figures/friction_vs_binder_content/xt_dm_n_vs_binder.pdf'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
