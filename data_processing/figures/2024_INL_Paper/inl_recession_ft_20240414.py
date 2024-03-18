import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import platform
import os
import json


platform_system = platform.system()
if platform_system != 'Windows':
    drive_path = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal'
else:
    drive_path = r'C:\Users\erick\OneDrive'

base_path = '/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Postdoc/research/manuscripts/paper2/figure_prep/simulations'

simulated_db = [
    {'lbl': 'Simulation ($f_{\mathrm{t}}$ = 60 KPa)', 'file': 'Ben_Fig10_dt_sim_ft_60.csv', 'marker': 's', 'ft': 60},
    {'lbl': 'Simulation ($f_{\mathrm{t}}$ = 70 KPa)', 'file': 'Ben_Fig10_dt_sim_ft_70.csv', 'marker': 's', 'ft': 70},
    {'lbl': 'Simulation ($f_{\mathrm{t}}$ = 80 KPa)', 'file': 'Ben_Fig10_dt_sim_ft_80.csv', 'marker': 's', 'ft': 80},
    {'lbl': 'Simulation ($f_{\mathrm{t}}$ = 70 KPa, not T limit)', 'file': 'Ben_Fig10_dt_sim_ft_70_no_T_limit.csv',
     'marker': 'x', 'ft': 70},
]

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
    global base_path, simulated_db, experimental_csv
    base_path = normalize_path(base_path)
    load_plot_style()

    norm = mpl.colors.Normalize(vmin=25, vmax=50)
    cmap = mpl.colormaps.get_cmap('cool')

    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(5.0, 6.5)
    colors_sim = ['C0', 'C1', 'C2', 'C3']

    axes[0].set_yscale('log')
    axes[0].set_ylim(1E-5, 1)
    axes[0].set_xlabel(r'Q (MW/m$^{\mathregular{2}}$)')
    axes[0].set_ylabel(r'$\nu$ (cm/s)')

    axes[0].set_xlim(0, 55)
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(10))
    axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(5))

    axes[1].set_yscale('log')
    axes[1].set_xlim(0, 90)
    axes[1].set_ylim(1E-3, 10)
    axes[1].set_ylabel(r'$\nu$ (cm/s)')
    axes[1].set_xlabel(r'$f_{\mathrm{t}}$ (KPa)')
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(20))
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(10))

    sim_30_50_df = pd.DataFrame(columns=['FT_KPa', 'Q', 'nu'])
    q_select = [30, 50]

    for i, r in enumerate(simulated_db):
        csv = os.path.join(base_path, r['file'])
        lbl = r['lbl']
        marker = r['marker']
        ft = r['ft']
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
            ls='-', mew=1.5, label=lbl
        )

    r3n41_42_df = pd.read_csv(os.path.join(base_path, experimental_csv)).apply(pd.to_numeric)
    axes[0].errorbar(
        r3n41_42_df['Heat load (MW/m2)'],
        r3n41_42_df['Recession rate (cm/s)'],
        # yerr=r3n41_42_df['Recession rate error (cm/s)'],
        marker='o', color='C0',
        ms=9, mew=1.25, mfc='none', ls='none',
        capsize=2.75, elinewidth=1.25, lw=1.5,
        label=r'Experiment ($f_{\mathrm{t}}$ = 30 KPa)'
    )

    markers_ft = ['o', '^']
    line_styles = ['--', '-.']
    plot_lines = []
    labels = []
    for i, r in enumerate(recession_db):
        csv = os.path.join(base_path, r['file'])
        ft_df = pd.read_csv(csv).apply(pd.to_numeric)
        ft_df['Q'] = np.round(ft_df['Heat load (MW/m^2)']/5.)*5.
        qs = ft_df['Q'].unique()
        for qi in qs:
            ft_df_at_q = ft_df[ft_df['Q'] == qi]
            ft_df_at_q = ft_df_at_q.sort_values(by=['FT_estimate (KPa)'])
            material = r['material']
            lbl = fr'{material} ({qi:.0f} ' + r'MW/m$^{\mathregular{2}}$)'
            nu_i = ft_df_at_q['Mean recession rate (cm/s)'].values
            ft_i = ft_df_at_q['FT_estimate (KPa)'].values
            nu_err = ft_df_at_q['Standard recession rate error (cm/s)'].values
            if i >0:
                nu_err /= 1.
            ub_err = nu_err
            lb_err = nu_err

            # Fix the lower error bands to make sense
            for j, eri in enumerate(lb_err):
                if eri >= nu_i[j] * 0.85:
                    lb_err[j] = nu_i[j] * 0.65

            print(ub_err.size, lb_err.size)
            ci = cmap(norm(qi))
            ebc = mpl.colors.to_rgba(ci, 0.15)
            line_i = axes[1].errorbar(
                ft_i, nu_i,
                yerr=(lb_err, ub_err),
                xerr=ft_df_at_q['FT_estimate error (KPa)'],
                c=ci,
                ms=9, mew=1.25, mfc='none', ls=line_styles[i],
                capsize=2.75, elinewidth=1.25, lw=1.25,
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

    sim_30_50_df = sim_30_50_df.sort_values(by=['Q', 'FT_KPa'], ascending=[True, False])
    print(sim_30_50_df)

    sim_lines = []
    sim_labels = []
    for qi in q_select:
        sim_df = sim_30_50_df[sim_30_50_df['Q'] == qi]
        print(sim_df)
        lbl = fr'Simulation ({qi:.0f} ' + r'MW/m$^{\mathregular{2}}$)'
        ci = cmap(norm(qi))
        # line_i = axes[1].scatter(
        #     sim_df['FT_KPa'].values, sim_df['nu'].values, marker='s',
        #     ls='-', c=[ci for x in range(len(sim_df))], label=lbl
        # )
        line_i, = axes[1].plot(
            sim_df['FT_KPa'], sim_df['nu'], marker='s',
            ls='-', c=ci, mfc=ci, label=lbl
        )
        sim_lines.append(line_i)
        sim_labels.append(lbl)

    for pl in plot_lines:
        print(pl)
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
