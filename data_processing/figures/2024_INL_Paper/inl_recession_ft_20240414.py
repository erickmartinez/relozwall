import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import platform
import os
import json

ft_scale = 0.512
platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'

base_path = '/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/simulations'

simulated_db = [
    {'lbl': 'Simulation (F = 1.3 N)', 'file': 'Ben_Fig10_dt_sim_ft_60.csv', 'marker': 's', 'ft': 60, 'ls':'-'},
    {'lbl': 'Simulation (F = 1.5 N)', 'file': 'Ben_Fig10_dt_sim_ft_70.csv', 'marker': 's', 'ft': 70, 'ls':'-'},
    {'lbl': 'Simulation (F = 1.7 N)', 'file': 'Ben_Fig10_dt_sim_ft_80.csv', 'marker': 's', 'ft': 80, 'ls':'-'},
    {'lbl': 'Simulation (F = 1.5 N, x2$\kappa$)', 'file': 'nu_vs_q_ft70_2x_conductivity.csv',
     'marker': 'x', 'ft': 70, 'ls':':'},
    {'lbl': 'Simulation (F = 1.5 N, no T limit)', 'file': 'Ben_Fig10_dt_sim_ft_70_no_T_limit.csv',
     'marker': 'x', 'ft': 70, 'ls': '--'},
]

simulated_breaking_load_csv = 'sim_breaking_load.csv'

experimental_csv = 'recession_vs_heat_load_30KPa.csv'

recession_db = [
    {'material': 'GC', 'file': 'gc_recession_vs_ft.csv'},
    {'material': 'AC', 'file': 'activated_carbon_recession_vs_ft.csv'}
]


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


def main():
    global base_path, simulated_db, experimental_csv, ft_scale
    base_path = normalize_path(base_path)
    load_plot_style()

    sim_bl_df = pd.read_csv('sim_breaking_load.csv').apply(pd.to_numeric)
    sim_ft = sim_bl_df['ft'].values.astype(int)
    sim_bl = sim_bl_df['Breaking load (N)'].values
    map_ft_bl = {sim_ft[i]: sim_bl[i] for i in range(len(sim_ft))}

    norm = mpl.colors.Normalize(vmin=25, vmax=50)
    cmap = mpl.colormaps.get_cmap('cool')

    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 6.75)
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
    axes[1].set_xlabel(r'F (N)')
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
        label=r'Experiment (F = 1.5 N)'
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
        qs = ft_df['Q'].unique()
        for j, qi in enumerate(qs):
            fs = fill_styles[j]
            bl_df_at_q: pd.DataFrame = ft_df[ft_df['Q'] == qi]
            print(bl_df_at_q.columns)
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
            if i >0:
                nu_err /= 1.
            ub_err = nu_err
            lb_err = nu_err

            # Fix the lower error bands to make sense
            for j, eri in enumerate(lb_err):
                if eri >= nu_i[j] * 0.85:
                    lb_err[j] = nu_i[j] * 0.65

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
