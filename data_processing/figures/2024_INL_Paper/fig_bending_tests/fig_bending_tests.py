import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import data_processing.confidence as cf
from scipy.optimize import least_squares, OptimizeResult
import os
import json
from data_processing.utils import lighten_color

simulations_csv = './data/simulations_interpolated_load.csv'
support_span = 40.
diameter = 10.1

f_scaling = 30. / support_span
ft_scale = 1.
sample_diameter_cm = 0.92

latex_preamble = 'latex_preamble.tex'

experimental_files = [
    {'matrix wt %': 5, 'csv': '3PBT_R4N137 - 033_2024-01-08_1_processed.csv', 'diameter': 1.025},
    {'matrix wt %': 20, 'csv': '3PBT_R4N139 - 040_2024-01-16_2_processed.csv', 'diameter': 1.025},
    # {'matrix wt %': 20, 'csv': '3PBT_R4N139 - 038_2024-01-16_1_processed.csv'},
    # {'matrix wt %': 25, 'csv': '3PBT_R4N136 - 030_2024-01-08_1_processed.csv'},
    {'matrix wt %': 100, 'csv': 'R3N63-1_processed.csv', 'diameter': 0.95},
]

gc_mean_strength_csv = './data/gc_mean_load_vs_matrix_content.csv'
ac_mean_strength_csv = './data/ac_mean_load_vs_matrix_content.csv'


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def load_to_fb(f):
    return 8. * (f / f_scaling) * 30. / np.pi / (diameter ** 3.) * 1E3


def fb_to_load(s):
    return s * np.pi * (diameter ** 3.) / 8E3 / 30. * f_scaling


def load_to_ft(x):
    return load_to_fb(x) * ft_scale


def ft_to_load(x):
    return fb_to_load(x / ft_scale)


def model(x, b):
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    return x * b[0]


def residual(b, x, y, w=1.):
    return w * (model(x, b) - y)


def jac_model(b, x, y, w=1.):
    jac = np.ones((len(x), 1), dtype=np.float64)
    jac[:, 0] *= w * x
    return jac


def strain2deformation(x):
    global sample_diameter_cm
    sd_mm = sample_diameter_cm * 10.
    return x * sd_mm / 100.


def deformation2strain(x):
    global sample_diameter_cm
    sd_mm = sample_diameter_cm * 10.
    return x / sd_mm * 100.


def load2strength(x):
    """
    Convert the load in N to a bending strength in kPa
    Assume a span length of 4 cm and a diameter defined
    by the global variable `sample_diameter` in cm.

    Parameters
    ----------
    x: float
        The force in N.

    Returns
    -------
    float:
        The bending stress in kPa

    """
    global sample_diameter_cm
    # Direct calculation in cm gives a factor of 1E4 Pa, convert to kPa
    return 8. * x * 4. / np.pi / (sample_diameter_cm ** 3.) * 10.


def strength2load(x):
    global sample_diameter_cm
    return x * np.pi * (sample_diameter_cm ** 3.) / (8. * 4. * 10.)


def main():
    global ft_scale
    load_plot_style()

    with open(file=latex_preamble, mode='r') as file:
        preamble_json = {'text.latex.preamble': file.read().strip()}
        mpl.rcParams.update(preamble_json)

    # fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(5., 6.5)
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        # width_ratios=[1., 0.9],
        height_ratios=[1, 1.25],
    )
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[0, 0])
    axes = [ax1, ax2]

    # ax1.tick_params(right=True, which='both', direction='out')
    ax2.tick_params(right=True, which='both', direction='out')
    sec_ax_colors = ['#3b7d25ff', '#f14b2dff']

    # secax_y1 = axes[1].secondary_yaxis(
    #     'right', functions=(load_to_ft, ft_to_load)
    # )
    #
    # secax_y1.tick_params(axis='y', labelcolor=sec_ax_colors[0])

    # secax_y2 = axes[1].secondary_yaxis(
    #     1.25, functions=(load_to_ft, ft_to_load)
    # )
    # secax_y2.tick_params(axis='y', labelcolor=sec_ax_colors[1])

    # secax_y1.set_ylabel(r'$f_{\mathrm{t}}$ (kPa)', fontsize=13, color=sec_ax_colors[0])
    # secax_y2.set_ylabel(r'$f_{\mathrm{t}}$ (KPa)', fontsize=13, color=sec_ax_colors[1])

    axes[0].set_xlabel('Deformation (mm)')
    axes[0].set_ylabel('Load (N)')

    secx = axes[0].secondary_xaxis('top', functions=(deformation2strain, strain2deformation))
    secx.set_xlabel('\nBending strain (%)')

    secy = axes[0].secondary_yaxis('right', functions=(load2strength, strength2load))
    secy.set_ylabel(r'$\sigma_{\mathrm{b}}$ (kPa)')

    axes[0].set_xlim(-0.1, 0.6)
    axes[0].set_ylim(0., 2.5)
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    axes[1].set_ylabel(r'$F_{\mathrm{b}}$ (N)')
    axes[1].set_xlabel('Binder content (wt %)')
    axes[1].set_xlim(0, 30)
    axes[1].set_ylim(0.0, 2.5)

    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(5.))
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(1.))
    secx.xaxis.set_major_locator(ticker.MultipleLocator(1.))
    secx.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    # secax_y1.yaxis.set_major_locator(ticker.MultipleLocator(50))
    # secax_y1.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    # secax_y1.yaxis.set_major_locator(ticker.MultipleLocator(20))
    # secax_y1.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    simulations_df = pd.read_csv(simulations_csv).apply(pd.to_numeric)

    deformation_mm = simulations_df['Deformation (mm)'].values

    gc_strength_df = pd.read_csv(gc_mean_strength_csv).apply(pd.to_numeric)
    ac_strength_df = pd.read_csv(ac_mean_strength_csv).apply(pd.to_numeric)

    strength_20_df = gc_strength_df[gc_strength_df['Matrix content (wt %)'] == 20].iloc[0]
    print(strength_20_df)
    gc_breaking_load_4cm = strength_20_df['Matrix mean breaking load (N)'] * 30 / support_span
    print(f'Breaking load at 4 cm: {gc_breaking_load_4cm:.2f} N')
    fb_20 = 8. * gc_breaking_load_4cm * 40. / np.pi / (diameter ** 3.) * 1000.
    print(f'f_b @ 4 cm: {fb_20:.1f} kPa')
    ft_20 = 70.
    ft_scale = ft_20 / fb_20
    print(f'f_t = {ft_scale:.3f} f_b')

    colors = {60: 'tab:red', 70: 'tab:green', 80: 'k'}
    ft_values = [60, 70, 80]
    sim_lines = []
    load_max_list = []
    for k, ft in enumerate(ft_values):
        colname = f'Load at ft={ft:d} (kPa)'
        load_n = simulations_df[colname].values
        load_max = load_n.max()
        load_max_list.append(round(load_max * 100) / 100)
        idx_max = np.argmin(np.abs(load_n - load_max)) + 2
        line, = axes[0].plot(
            deformation_mm[0:idx_max], load_n[0:idx_max], c=colors[ft], lw=2.25,
            # label=rf'Simulation $f_{{\mathregular{{t}}}}$ = {ft} kPa'
            label=rf'Simulation $F_{{\mathrm{{b}}}}$ = {load_max:.1f} N'
        )
        axes[0].plot(
            deformation_mm[idx_max - 2], load_n[idx_max - 2], c=colors[ft], mew=2.,
            marker='o', ms=8, mec='k', alpha=0.5
        )

        if k == 0:
            strain = deformation2strain(deformation_mm[0:idx_max])
            sigma_b = load2strength(load_n[0:idx_max])
            weights = 1.

            all_tol = float(np.finfo(np.float64).eps)
            res: OptimizeResult = least_squares(
                x0=[5E-10],
                fun=residual,
                args=(strain, sigma_b, weights),
                jac=jac_model,
                # loss='soft_l1', f_scale=0.1,
                xtol=all_tol,  # ** 0.5,
                ftol=all_tol,  # ** 0.5,
                gtol=all_tol,  # ** 0.5,
                max_nfev=10000 * len(strain),
                x_scale='jac',
                verbose=2
            )

            popt = res.x
            ci = cf.confidence_interval(res=res)
            youngs_modulus = 100. * popt[0] * 1E-3
            youngs_delta = 100. * ci[0, 1] * 1E-3 - youngs_modulus
            print(f'Young\'s modulus: {youngs_modulus:.1f} ± {youngs_delta:.2f} MPa')
            xpred = np.linspace(strain.min(), strain.max(), 200)
            deformation_pred = strain2deformation(xpred)
            ypred, delta = cf.prediction_intervals(model=model, x_pred=xpred, ls_res=res, weights=weights, jac=jac_model)

            elastic_modulus_txt = r'$E_{\mathrm{b}} = '
            elastic_modulus_txt += fr'{youngs_modulus:.0f}~'
            elastic_modulus_txt += r'\mathrm{MPa}$'
            elastic_modulus_txt += '\n'

            axes[0].text(
                deformation_mm[idx_max-1], load_n[idx_max-1],
                elastic_modulus_txt,
                va='bottom', ha='center', color='k'
            )

            # axes[0].plot(
            #     deformation_pred, strength2load(ypred), lw=1., c='k', ls='--'
            # )

        sim_lines.append(line)

    sim_breaking_load_df = pd.DataFrame(data={
        'ft': ft_values,
        'Breaking load (N)': load_max_list
    })

    sim_breaking_load_df.to_csv(path_or_buf=os.path.join('./data', 'sim_breaking_load.csv'), index=False)
    print(sim_breaking_load_df)

    colors = ['mediumpurple', 'lightseagreen', 'saddlebrown', 'C3', 'C4', 'C5']
    markers = ['o', 's', '^', 'v', 'D', '<', '>']

    exp_lines = []

    for i, ef in enumerate(experimental_files):
        mc, csv = ef['matrix wt %'], ef['csv']
        bending_df = pd.read_csv(os.path.join('data', csv)).apply(pd.to_numeric)
        deformation = bending_df['Deformation (mm)'].values
        force_4cm = bending_df['Force 4cm (N)'].values
        force_4cm_err = bending_df['Force 4cm err (N)'].values

        # print(bending_df[['Deformation (mm)', 'Force 4cm (N)']])
        # find the number of points where force > 0
        if mc != 100:
            n = len(force_4cm[force_4cm < 0.15])
            idx_start = n - 1
            deformation = deformation[idx_start::]
            force_4cm = force_4cm[idx_start::]
            force_4cm_err = force_4cm_err[idx_start::]
            deformation -= deformation.min()
            lbl = f'{mc:>2d}% binder'
        else:
            deformation -= deformation.min()
            lbl = f'Matrix'
            f_max = force_4cm.max()
            idx_max = np.argmin(np.abs(force_4cm - f_max)) + 1
            deformation = deformation[0:idx_max]
            force_4cm = force_4cm[0:idx_max]
            force_4cm_err = force_4cm_err[0:idx_max]
        ebc = mpl.colors.to_rgba(colors[i], 0.25)

        strain = deformation2strain(deformation)
        sigma_b = load2strength(force_4cm)
        sigma_b_err = load2strength(force_4cm_err)
        weights = np.power(sigma_b_err, -2)
        # weights = 1.

        all_tol = float(np.finfo(np.float64).eps)
        res: OptimizeResult = least_squares(
            x0=[5E-10],
            fun=residual,
            args=(strain, sigma_b, weights),
            jac=jac_model,
            # loss='soft_l1', f_scale=0.1,
            xtol=all_tol,  # ** 0.5,
            ftol=all_tol,  # ** 0.5,
            gtol=all_tol,  # ** 0.5,
            max_nfev=10000 * len(deformation),
            x_scale='jac',
            verbose=2
        )

        popt = res.x
        ci = cf.confidence_interval(res=res)
        youngs_modulus = 100. * popt[0] * 1E-3
        youngs_delta = 100. * ci[0, 1] * 1E-3 - youngs_modulus
        print(f'Young\'s modulus: {youngs_modulus:.1f} ± {youngs_delta:.2f} MPa')
        xpred = np.linspace(strain.min(), strain.max(), 200)
        deformation_pred = strain2deformation(xpred)
        ypred, delta = cf.prediction_intervals(model=model, x_pred=xpred, ls_res=res, weights=weights, jac=jac_model)

        line = axes[0].errorbar(
            deformation, force_4cm, yerr=0.5,  # xerr=0.1,
            marker=markers[i], mfc='none', capsize=2.5,
            ls='none', lw=1.,
            ms=9, mew=1.25, ecolor=ebc,
            elinewidth=1.0, c=colors[i],
            label=lbl
        )

        axes[0].plot(
            deformation_pred, strength2load(ypred), lw=1., c=colors[i], ls='--'
        )

        elastic_modulus_txt = r'$E_{\mathrm{b}} = '
        elastic_modulus_txt += fr'{youngs_modulus:.0f}~'
        elastic_modulus_txt += r'\mathrm{MPa}$'
        elastic_modulus_txt += '\n'

        axes[0].text(
            deformation[-1], force_4cm[-1],
            elastic_modulus_txt,
            va='bottom', ha='center', color=colors[i]
        )


        axes[0].errorbar(
            deformation[-1], force_4cm[-1],
            marker=markers[i], mfc=colors[i], capsize=2.5,
            ls='--', lw=1.,
            ms=9, mew=1.25, ecolor=ebc,
            elinewidth=1.0, c=colors[i],
        )
        exp_lines.append(line)

    leg00 = axes[0].legend(
        handles=sim_lines,  # labels=labels[0:-2],
        loc='upper left', frameon=True, fontsize=10,
        ncols=1,
        # borderaxespad=0.
    )

    axes[0].add_artist(leg00)

    leg01 = axes[0].legend(
        handles=exp_lines,
        loc='lower right', frameon=True, fontsize=10
    )

    axes[1].errorbar(
        gc_strength_df['Matrix content (wt %)'], gc_strength_df['Matrix mean breaking load (N)'] * 30. / support_span,
        yerr=gc_strength_df['Matrix mean breaking load error (N)'] * 30. / support_span,
        marker='o', ms=9, mew=1.25, mfc='none', ls='--',
        capsize=2.75, elinewidth=1.25, lw=1.0, label='GC mean breaking load'
    )

    # 1E-3 * sigma_i / 40. / 8. * np.pi * diameter ** 3.
    ac_fb = (1E-3 * ac_strength_df['Matrix strength mean (KPa)'].values /
             40. / 8. * np.pi * np.power(ac_strength_df['Diameter mean (mm)'].values, 3.))
    axes[1].errorbar(
        ac_strength_df['Matrix content (wt %)'].values, ac_fb,
        yerr=ac_strength_df['Matrix mean breaking load error (N)'] * 30. / support_span,
        marker='s', ms=9, mew=1.25, mfc='none', ls='-.',
        capsize=2.75, elinewidth=1.25, lw=1.0, label='AC mean breaking load'
    )

    axes[1].legend(
        loc='upper left', frameon=True, fontsize=12
    )
    # fig.tight_layout()
    s_txt = r"$ f_{\mathrm{b}} = \dfrac{8 F L }{\pi D^3} = 3 f_{\mathrm{t}}$"
    # axes[1].text(
    #     0.95, 0.05, s_txt,
    #     va='bottom', ha='right',
    #     transform=axes[1].transAxes, fontsize=14, color='tab:red',
    #     # usetex=True
    # )

    for i, axi in enumerate(axes[::-1]):
        panel_label = chr(ord('`') + i + 1)
        axi.text(
            -0.15, 1.02, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )

    fig.savefig('fig_bending_tests.png', dpi=300)
    fig.savefig('fig_bending_tests.svg', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
