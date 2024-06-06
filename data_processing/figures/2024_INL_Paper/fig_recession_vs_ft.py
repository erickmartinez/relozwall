import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import platform
import os
import json
import data_processing.confidence as cf
from scipy.optimize import least_squares, OptimizeResult

ft_scale = 0.512
platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'

base_path = 'Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/simulations'

simulated_db = [
    {'lbl': 'Simulation ($F_{\mathrm{b}}$= 1.3 N)', 'file': 'Ben_Fig10_dt_sim_ft_60.csv', 'marker': 's', 'ft': 60, 'ls':'-'},
    {'lbl': 'Simulation ($F_{\mathrm{b}}$ = 1.5 N)', 'file': 'Ben_Fig10_dt_sim_ft_70.csv', 'marker': 's', 'ft': 70, 'ls':'-'},
    {'lbl': 'Simulation ($F_{\mathrm{b}}$ = 1.7 N)', 'file': 'Ben_Fig10_dt_sim_ft_80.csv', 'marker': 's', 'ft': 80, 'ls':'-'},
    {'lbl': r'Simulation ($F_{\mathrm{b}}$ = 1.5 N, 2x$\alpha$)', 'file': 'nu_vs_q_ft70_2x_conductivity.csv',
     'marker': 'x', 'ft': 70, 'ls':':'},
    {'lbl': 'Simulation ($F_{\mathrm{b}}$ = 1.5 N, no T limit)', 'file': 'Ben_Fig10_dt_sim_ft_70_no_T_limit.csv',
     'marker': 'x', 'ft': 70, 'ls': '--'},
]

simulated_breaking_load_csv = 'sim_breaking_load.csv'

experimental_csv = 'recession_vs_heat_load_30KPa.csv'

recession_db = [
    {'material': 'GC', 'file': 'gc_recession_vs_ft.csv'},
    {'material': 'AC', 'file': 'activated_carbon_recession_vs_ft.csv'}
]

fit_aggregate_df = pd.DataFrame(columns=['origin', 'heat_load', 'Fb', 'nu', 'Fb_err', 'nu_err'])


def normalize_path(the_path):
    global platform_system, drive_path
    if platform_system != 'Windows':
        the_path = the_path.replace('\\', '/')
    return os.path.join(drive_path, the_path)


def load_plot_style():
    with open('../../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def model_exp(x, b):
    return b[0] * np.exp(-x / b[1])


def residual_exp(b, x, y, w=1.):
    return (model_exp(x, b) - y) * w


def jac_exp(b, x, y, w=1.):
    m, n = len(x), len(b)
    r = np.ones((m, n))
    ee = w * np.exp(-x / b[1])
    r[:, 0] = ee
    r[:, 1] = b[0] * x * ee / (b[1] ** 2.)
    return r

def model(t: np.ndarray, beta: np.ndarray):
    return beta[0] - t * beta[1]


def residual(b, x, y, w=1):
    return (model(x, b) - y) * w


def jac(b, t, r, w=1):
    n, p = len(r), len(b)
    j = np.empty((n, p), dtype=np.float64)
    j[:, 0] = np.ones(n, dtype=np.float64) * w
    j[:, 1] = - t * w
    return j

def main():
    global base_path, simulated_db, experimental_csv, ft_scale, fit_aggregate_df
    base_path = normalize_path(base_path)
    load_plot_style()

    sim_bl_df = pd.read_csv('sim_breaking_load.csv').apply(pd.to_numeric)
    sim_ft = sim_bl_df['ft'].values.astype(int)
    sim_bl = sim_bl_df['Breaking load (N)'].values
    map_ft_bl = {sim_ft[i]: sim_bl[i] for i in range(len(sim_ft))}

    norm = mpl.colors.Normalize(vmin=25, vmax=50)
    cmap = mpl.colormaps.get_cmap('cool')

    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 7.25)
    colors_sim = ['C0', 'C3', 'C2', 'C3', 'C4']

    axes[0].set_yscale('log')
    axes[0].set_ylim(1E-5, 1)
    axes[0].set_xlabel(r'$q$ (MW/m$^{\mathregular{2}}$)')
    axes[0].set_ylabel(r'$\nu$ (cm/s)')

    axes[0].set_xlim(0, 55)
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(10))
    axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(5))

    axes[1].set_yscale('log')
    axes[1].set_xlim(0, 2.5)
    axes[1].set_ylim(1E-3, 10)
    axes[1].set_ylabel(r'$\nu$ (cm/s)')
    axes[1].set_xlabel(r'$F_{\mathrm{b}}$ (N)')
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    sim_30_50_df = pd.DataFrame(columns=['FT_KPa', 'Q', 'nu'])
    q_select = [30, 50]

    for i, r in enumerate(simulated_db):
        csv = os.path.join(base_path, r['file'])
        lbl = r['lbl']
        marker = r['marker']
        ft = r['ft']
        ls = r['ls']
        sim_df = pd.read_csv(csv, comment='#').apply(pd.to_numeric)
        sim_df['Q'] = np.round(sim_df['Heat load (MW/m2)']/5.)*5
        heat_load = sim_df['Heat load (MW/m2)'].values
        recession_rate = sim_df['Recession rate (cm/s)'].values

        match_df = sim_df[sim_df['Q'].isin(q_select)]
        n = len(match_df)
        new_df = pd.DataFrame(data={
            'FT_KPa': [ft for j in range(n)],
            'Q': match_df['Q'].values,
            'nu': match_df['Recession rate (cm/s)'].values
        })

        if i < 3:  # do not add data simulated without T limit
            sim_30_50_df = pd.concat([sim_30_50_df, new_df])

        axes[0].plot(
            heat_load, recession_rate, c=colors_sim[i], marker=marker, fillstyle='full',
            ls=ls, mew=2., label=lbl, ms=6, lw=1.25
        )

    r3n41_42_df = pd.read_csv(os.path.join(base_path, experimental_csv)).apply(pd.to_numeric)
    axes[0].errorbar(
        r3n41_42_df['Heat load (MW/m2)'],
        r3n41_42_df['Recession rate (cm/s)'],
        yerr=(r3n41_42_df['Recession rate (cm/s)']*0.5, r3n41_42_df['Recession rate (cm/s)']*1.5),
        marker='o', color='C0',
        ms=9, mew=1.25, mfc='none', ls='none',
        capsize=2.75, elinewidth=1.25, lw=1.5,
        label=r'Experiment ($F_{\mathrm{b}}$ = 1.5 N)'
    )

    markers_ft = ['o', '>']
    fill_styles = ['bottom', 'none']
    line_styles = ['none', 'none']
    plot_lines = []
    labels = []
    for i, r in enumerate(recession_db):
        csv = os.path.join(base_path, r['file'])
        ft_df = pd.read_csv(csv).apply(pd.to_numeric)
        ft_df['Q'] = np.round(ft_df['Heat load (MW/m^2)']/5.)*5.
        # ft_df['Q_5'] = np.round(ft_df['Heat load (MW/m^2)'] / 5.) * 5.
        ft_df.loc[ft_df['Q'] == 20., 'Q'] = 25.
        qs = ft_df['Q'].unique()
        print(ft_df)
        for j, qi in enumerate(qs):
            fs = fill_styles[j]
            bl_df_at_q: pd.DataFrame = ft_df[ft_df['Q'] == qi]
            # print(bl_df_at_q.columns)
            # print(bl_df_at_q[['Matrix mean breaking load (N)','Mean recession rate (cm/s)']])
            bl_df_at_q = bl_df_at_q.sort_values(by=['Matrix strength mean (KPa)', 'FT_estimate (KPa)'])
            material = r['material']
            lbl = fr'{material} ({qi:.0f} ' + r'MW/m$^{\mathregular{2}}$)'
            nu_i = bl_df_at_q['Mean recession rate (cm/s)'].values
            # ft_i = ft_df_at_q['FT_estimate (KPa)'].values
            sigma_i = bl_df_at_q['Matrix strength mean (KPa)'].values
            diameter = bl_df_at_q['Diameter mean (mm)'].values
            bl_i = 1E-3 * sigma_i / 40. / 8. * np.pi * diameter ** 3.
            ble_i =  bl_df_at_q['Matrix mean breaking load error (N)'].values * (30./40.)
            nu_err = bl_df_at_q['Standard recession rate error (cm/s)'].values

            nn = len(bl_i)
            # if i >0:
            #     nu_err /= 1.
            ub_err = nu_err
            lb_err = nu_err

            # Fix the lower error bands to make sense
            for k, eri in enumerate(lb_err):
                if eri >= nu_i[k] * 0.85:
                    lb_err[k] = nu_i[k] * 0.65

            if qi == 25:
                fit_row = pd.DataFrame(data={
                    'origin': ['Experiment' for idd in range(nn)],
                    'heat_load': [25 for idd in range(nn)],
                    'Fb': bl_i,
                    'nu': nu_i,
                    'Fb_err': ble_i,
                    'nu_err': nu_err
                })
            else:
                fit_row = pd.DataFrame(data={
                    'origin': ['Experiment' for idd in range(nn)],
                    'heat_load': [45 for idd in range(nn)],
                    'Fb': bl_i,
                    'nu': nu_i,
                    'Fb_err': ble_i,
                    'nu_err': nu_err
                })
            fit_aggregate_df = pd.concat([fit_aggregate_df, fit_row]).sort_values(by=['origin', 'heat_load', 'Fb']).reset_index(drop=True)

            ci = cmap(norm(qi))
            ebc = mpl.colors.to_rgba(ci, 0.25)
            line_i = axes[1].errorbar(
                bl_i, nu_i,
                yerr=(lb_err, ub_err),
                xerr=ble_i,#bl_df_at_q['FT_estimate error (KPa)'],
                c=ci,
                ms=9, mew=1.25, #ls='none',# mfc='none',
                ls=line_styles[i],
                fillstyle=fs, mec=ci, mfc=ci,
                capsize=2.75, elinewidth=1.25, lw=1.,
                marker=markers_ft[i],
                ecolor=ebc,
                label=lbl
            )
            # xnew = np.linspace(ft_i.min(), ft_i.max(), 300)
            # spk = len(ft_i) - 2
            # if spk % 2 == 0:
            #     spk -= 1
            # spk.
            # spl = make_interp_spline(ft_i, nu_i, k=spk)  # type: BSpline
            #
            # axes[1].plot(
            #     xnew, spl(xnew),
            #     c=cmap(norm(qi)),
            #     ls=line_styles[i],
            #     lw=1.5,
            # )
            plot_lines.append(line_i)
            labels.append(lbl)

    sim_30_50_df = sim_30_50_df.sort_values(by=['Q', 'FT_KPa'], ascending=[False, False])
    # print(sim_30_50_df)
    # load breaking load for simulations

    sim_lines = []
    sim_labels = []
    for qi in q_select:
        sim_df = sim_30_50_df[sim_30_50_df['Q'] == qi]
        ft_val = sim_df['FT_KPa'].values
        bl_sim = np.array([map_ft_bl[ft_val[i]] for i in range(len(ft_val))])
        print(sim_df)
        lbl = fr'Simulation of GC ({qi:.0f} ' + r'MW/m$^{\mathregular{2}}$)'
        ci = cmap(norm(qi))
        # line_i = axes[1].scatter(
        #     sim_df['FT_KPa'].values, sim_df['nu'].values, marker='s',
        #     ls='-', c=[ci for x in range(len(sim_df))], label=lbl
        # )
        line_i, = axes[1].plot(
            bl_sim,
            # sim_df['FT_KPa'],
            sim_df['nu'], marker='s',
            ls='-', c=ci, mfc=ci, label=lbl
        )

        sim_lines.append(line_i)
        sim_labels.append(lbl)
        nn = len(bl_sim)
        errors = np.zeros(nn, dtype=np.float64)
        if qi == 30:
            fit_row = pd.DataFrame(data={
                'origin': ['Simulation' for idd in range(nn)],
                'heat_load': [30 for idd in range(nn)],
                'Fb': bl_sim,
                'nu': sim_df['nu'],
                'Fb_err': errors,
                'nu_err': errors
            })
        else:
            fit_row = pd.DataFrame(data={
                'origin': ['Simulation' for idd in range(nn)],
                'heat_load': [50 for idd in range(nn)],
                'Fb': bl_sim,
                'nu': sim_df['nu'],
                'Fb_err': errors,
                'nu_err': errors
            })
        fit_aggregate_df = pd.concat([fit_aggregate_df, fit_row]).sort_values(by=['origin', 'heat_load', 'Fb']).reset_index(
            drop=True)

    # for pl in plot_lines:
    #     print(pl)
    # cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #              ax=ax, orientation='vertical', label=r'$f_{\mathrm{t}}$ (KPa)')
    # cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    # cbar.ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    #
    # legend_elements = [Line2D([0], [0], color=cmap(norm(30)), mfc=cmap(norm(30)), lw=1.5,
    #                           label='Simulation', marker='s'),
    #                    Line2D([0], [0], color=cmap(norm(50)), mfc=cmap(norm(30)), lw=1.5,
    #                           label='Simulation', marker='s'),
    #                    ]

    # ax.legend(handles=legend_elements, loc='lower right', frameon=True)
    axes[0].legend(loc='lower right', frameon=True, fontsize=10)

    leg10 = axes[1].legend(
        handles=plot_lines, #labels=labels[0:-2],
        loc='lower left', frameon=True, fontsize=10,
        ncols=2,
        # borderaxespad=0.
    )

    axes[1].add_artist(leg10)

    sim_lines = sim_lines[::-1]

    leg11 = axes[1].legend(
        handles=sim_lines, #labels=sim_labels[0],
        loc='upper right', frameon=True, fontsize=10,
        ncols=1,
        # borderaxespad=0.
    )

    axes[1].text(
        0.3, 0.19, 'Experiment',
        va='bottom', ha='left',
        transform=axes[1].transAxes,
        fontsize=10, fontweight='bold'
    )

    all_tol = float(np.finfo(np.float64).eps)
    fit_results_df = pd.DataFrame(columns=[
        'Origin', 'Heat load (MW/m^2)',
        'nu_0 (cm/s)', 'nu_0_lpb', 'nu_0_upb', 'delta_nu_0',
        'c (1/N)', 'c_lpb', 'c_upb', 'delta_c',
        'R^2'
    ])
    """
    Fit experimental data
    """
    fit_experiment_df = fit_aggregate_df[fit_aggregate_df['origin'] == 'Experiment']
    fit_hl = [25, 45]
    for qi in fit_hl:
        fit_experiment_hl_df = fit_experiment_df[fit_experiment_df['heat_load'] == qi]
        # fit_experiment_hl_df.eval('log_nu_err = log(nu_err)', inplace=True)
        print(fit_experiment_hl_df)
        fp = fit_experiment_hl_df['Fb'].values
        nu = fit_experiment_hl_df['nu'].values
        nu_err = fit_experiment_hl_df['nu_err'].values

        log_nu = np.log(nu)
        log_nu_err = np.log(nu_err)
        # weights = np.logspace(-1, -3, num=len(nu))
        # weights = 1. / weights

        weights = 1. / log_nu_err
        weights /= weights.max()


        # weights = np.log(weights + 1.)
        slope = np.gradient(log_nu, np.gradient(fp)).mean()

        res_i = least_squares(
            fun=residual, x0=[nu.max(), slope],
            jac=jac,
            # bounds=([1E-10, 1E-20], [np.inf, np.inf]),
            args=(fp, log_nu, weights),
            loss='linear',
            f_scale=0.1,
            xtol=all_tol,
            ftol=all_tol,
            gtol=all_tol,
            diff_step=all_tol,
            max_nfev=10000 * len(weights),
            # x_scale='jac',
            verbose=2
        )

        fit_color = 'tab:blue' if qi == 25 else 'tab:red'
        popt = res_i.x
        ci = cf.confidence_interval(res=res_i, confidence=0.95)
        nu_0, cc = np.exp(popt[0]), popt[1]
        nu_0_ci = np.exp(ci[0,:])
        cc_ci = ci[1, :]
        nu_0_delta = np.abs(nu_0_ci - nu_0).max()
        cc_delta = np.abs(cc_ci - cc).max()

        # Calculate the R-squared value
        r_squared = 1. - np.linalg.norm(res_i.fun) / np.sum((log_nu - np.mean(log_nu)) ** 2)

        print(f'Experiment nu_0: {nu_0:.3f}, 95% CI: [{nu_0_ci[0]:.4f}, {nu_0_ci[1]:.4f}], c: {cc:.3f} 95 % CI [{cc_ci[0]:.4f}, {cc_ci[1]:.4f}]')

        row_fit_params = pd.DataFrame(data={
            'Origin': ['Experiment'],
            'Heat load (MW/m^2)': [qi],
            'nu_0 (cm/s)': [nu_0],
            'nu_0_lpb': [nu_0_ci[0]],
            'nu_0_upb': [nu_0_ci[1]],
            'delta_nu_0': [nu_0_delta],
            'c (1/N)': [cc],
            'c_lpb': [cc_ci[0]],
            'c_upb': [cc_ci[1]],
            'delta_c': [cc_delta],
            'R^2': [r_squared]
        })

        fit_results_df = pd.concat([fit_results_df, row_fit_params]).reset_index(drop=True)

        x_pred = np.linspace(fp.min(), fp.max(), num=200)
        y_pred, delta = cf.prediction_intervals(
            model=model, x_pred=x_pred, ls_res=res_i, jac=jac_exp,
            new_observation=True, simultaneous=False
        )

        lpb, upb = y_pred - delta, y_pred + delta
        nu_pred = np.exp(y_pred)
        nu_lpb, nu_upb = np.exp(lpb), np.exp(upb)


        axes[1].plot(x_pred, nu_pred, c=fit_color, ls='--', lw=1.25)
        # axes[1].fill_between(x_pred, nu_lpb, nu_upb, color=fit_color, ls='--', alpha=0.15)

    """
    Fit simulations
    """
    fit_simulations_df = fit_aggregate_df[fit_aggregate_df['origin'] == 'Simulation']
    fit_hl = [30, 50]
    for qi in fit_hl:
        fit_simulation_hl_df = fit_simulations_df[fit_simulations_df['heat_load'] == qi]
        # fit_experiment_hl_df.eval('log_nu_err = log(nu_err)', inplace=True)
        print(fit_simulation_hl_df)
        fp = fit_simulation_hl_df['Fb'].values
        nu = fit_simulation_hl_df['nu'].values

        log_nu = np.log(nu)

        # weights = np.log(weights + 1.)
        slope = (log_nu[-1] - log_nu[0]) / (fp[-1] - fp[0])

        res_i = least_squares(
            fun=residual, x0=[nu.max(), slope],
            jac=jac,
            # bounds=([1E-10, 1E-20], [np.inf, np.inf]),
            args=(fp, log_nu),
            loss='linear',
            f_scale=0.1,
            xtol=all_tol,
            ftol=all_tol,
            gtol=all_tol,
            diff_step=all_tol,
            max_nfev=10000 * len(nu),
            # x_scale='jac',
            verbose=2
        )

        fit_color = 'tab:blue' if qi == 30 else 'tab:red'
        popt = res_i.x
        ci = cf.confidence_interval(res=res_i, confidence=0.95)
        nu_0, cc = np.exp(popt[0]), popt[1]
        nu_0_ci = np.exp(ci[0, :])
        cc_ci = ci[1, :]
        nu_0_delta = np.abs(nu_0_ci - nu_0).max()
        cc_delta = np.abs(cc_ci - cc).max()
        # Calculate the R-squared value
        r_squared = 1. - np.linalg.norm(res_i.fun) / np.sum((log_nu - np.mean(log_nu)) ** 2)

        print(
            f'Simulation nu_0: {nu_0:.3e}, 95% CI: [{nu_0_ci[0]:.4e}, {nu_0_ci[1]:.4e}], c: {cc:.3e} 95% CI [{cc_ci[0]:.4f}, {cc_ci[1]:.4f}]')

        row_fit_params = pd.DataFrame(data={
            'Origin': ['Simulation'],
            'Heat load (MW/m^2)': [qi],
            'nu_0 (cm/s)': [nu_0],
            'nu_0_lpb': [nu_0_ci[0]],
            'nu_0_upb': [nu_0_ci[1]],
            'delta_nu_0': [nu_0_delta],
            'c (1/N)': [cc],
            'c_lpb': [cc_ci[0]],
            'c_upb': [cc_ci[1]],
            'delta_c': [cc_delta],
            'R^2': [r_squared]
        })

        fit_results_df = pd.concat([fit_results_df, row_fit_params]).reset_index(drop=True)

        x_pred = np.linspace(fp.min(), fp.max(), num=200)
        y_pred, delta = cf.prediction_intervals(
            model=model, x_pred=x_pred, ls_res=res_i, jac=jac_exp,
            new_observation=True, simultaneous=False
        )

        lpb, upb = y_pred - delta, y_pred + delta
        nu_pred = np.exp(y_pred)
        nu_lpb, nu_upb = np.exp(lpb), np.exp(upb)

        # axes[1].plot(x_pred, nu_pred, c=fit_color, ls='-.', lw=1.25)
        # axes[1].fill_between(x_pred, nu_lpb, nu_upb, color=fit_color, ls='--', alpha=0.15)


    # Save fit results in csv
    out_csv = os.path.join(base_path, 'fit_results.csv')
    with open(out_csv, 'w') as fout:
        fout.write(f'# Model: nu = nu_0 * exp(c*Fb)')
        fit_results_df.to_csv(fout)

    for i, axi in enumerate(axes.flatten()):
        panel_label = chr(ord('`') + i + 1)
        axi.text(
            -0.15, 1.02, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )

    fig.savefig(os.path.join(base_path, 'figure_recession_heat_load_ft.png'), dpi=600)
    fig.savefig(os.path.join(base_path, 'figure_recession_heat_load_ft.pdf'), dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
