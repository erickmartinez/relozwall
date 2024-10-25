import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import os
import json
import numpy as np
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf
from scipy.stats.distributions import t
import matplotlib as mpl
from scipy.integrate import simpson
from data_processing.utils import latex_float, latex_float_with_error
from matplotlib.font_manager import FontProperties

k_b = 8.617333262E-5 # eV/K
condidence = 0.95
alpha = 1. - condidence

def get_t(n):
    return t.ppf(1 - alpha/2., n-1)

def r1(TK, r_m, e_a, T_m):
    x = (TK - T_m) / T_m
    a = e_a / k_b / T_m
    xx = 1. + x
    return r_m * np.exp(a*x/xx - np.power(xx, 2.)*np.exp(a*x/xx) + 1.)

def r2(TK, r_m, e_a, T_m):
    x = (TK - T_m) / T_m
    a = 0.5 * e_a / k_b / T_m
    xx = 1. + x
    inv = np.exp(-a*x/xx)+ np.power(xx, 2.)*np.exp(a*x/xx)
    return 4. * np.power(inv, -2.) * r_m

ln_4 = np.log(4.)
def r2_log(TK, r_m, e_a, T_m):
    global ln_4
    x = (TK - T_m) / T_m
    a = 0.5 * e_a / k_b / T_m
    xx = 1. + x
    inv = np.exp(-a*x/xx)+ np.power(xx, 2.)*np.exp(a*x/xx)
    return ln_4 + np.log(r_m) - 2. * np.log(inv)


def u1(TK, r_m, e_a, T_m):
    x = (TK - T_m) / T_m
    a = e_a / k_b / T_m
    xx = 1. + x
    arg = a * x / xx
    r = arg - a * x * xx * np.exp(arg)
    return r

def u2(TK, r_m, e_a, T_m):
    x = (TK - T_m) / T_m
    a = e_a / k_b / T_m
    xx = TK / T_m
    arg = a * x / xx
    r = (a+2.) * np.power(xx, 2.) * np.exp(arg) - a
    return r


def v1(TK, r_m, e_a, T_m):
    x = (TK - T_m) / T_m
    a = 0.5 * e_a / k_b / T_m
    xx = 1. + x
    arg = a * x / xx
    r = arg * (np.exp(-arg) - np.power(xx, 2.) * np.exp(arg))
    return r

def v2(TK, r_m, e_a, T_m):
    x = (TK - T_m) / T_m
    a = 0.5 * e_a / k_b / T_m
    xx = TK / T_m
    arg = a * x / xx
    r = (2.+a) * np.power(xx, 2.) * np.exp(arg) - 2. * arg * np.exp(-arg)
    return r


def model_1(x, b):
    return r1(x, b[0], b[1], b[2])

def res_1(b, x, y):
    rm = b[0]
    ea = b[1]
    tm = b[2]
    return r1(x, rm, ea, tm) - y


def jac_1(b, x, y):
    rm = b[0]
    ea = b[1]
    tm = b[2]

    rr = r1(x, rm, ea, tm)

    u_0 = (1. / rm) * np.ones(len(x), dtype=np.float64)
    u_1 = (1. / ea) * u1(x, rm, ea, tm)
    u_2 = (1. / tm) * u2(x, rm, ea, tm)
    jm = (np.stack([u_0 , u_1 , u_2 ]) * rr).T
    return jm

def jac_r2(b, x, y):
    rm = b[0]
    ea = b[1]
    tm = b[2]

    rm12 = np.sqrt(rm)

    rr = r2(x, rm, ea, tm)
    r32 = np.power(rr, 3./2.)

    v_0 = (rr / rm) * np.ones(len(x), dtype=np.float64)
    v_1 = (r32 / rm12 / ea) * v1(x, rm, ea, tm)
    v_2 = (r32 / rm12 / tm) * v2(x, rm, ea, tm)
    jm = np.array([v_0, v_1, v_2]).T
    return jm

def model_2(x, b):
    # Use 5 peak shapes to fit the boron pebble rod data
    rm_1, ea_1, tm_1 = b[0], b[1], b[2]
    rm_2, ea_2, tm_2 = b[3], b[4], b[5]
    rm_3, ea_3, tm_3 = b[6], b[7], b[8]
    rm_4, ea_4, tm_4 = b[9], b[10], b[11]
    rm_5, ea_5, tm_5 = b[12], b[13], b[14]
    rm_5, ea_6, tm_5 = b[15], b[16], b[17]
    rr1 = r1(x, rm_1, ea_1, tm_1)
    rr2 = r1(x, rm_2, ea_2, tm_2)
    rr3 = r1(x, rm_3, ea_3, tm_3)
    rr4 = r1(x, rm_4, ea_4, tm_4)
    rr5 = r1(x, rm_5, ea_5, tm_5)
    return rr1 + rr2 + rr3 + rr4 + rr5

def res_2(b, x, y):
    return model_2(x, b) - y

def jac_2(b, x, y):
    m = len(x)
    n = len(b)
    result = np.zeros((m, n), dtype=np.float64)
    result[:, 0:3] = jac_1(b[0:3], x, y)
    result[:, 3:6] = jac_1(b[3:6], x, y)
    result[:, 6:9] = jac_1(b[6:9], x, y)
    result[:, 9:12] = jac_1(b[9:12], x, y)
    result[:, 12:15] = jac_1(b[12:15], x, y)
    return result

def model_3(x, b):
    # Use 5 peak shapes to fit the boron pebble rod data
    rm_1, ea_1, tm_1 = b[0], b[1], b[2]
    rm_2, ea_2, tm_2 = b[3], b[4], b[5]
    rm_3, ea_3, tm_3 = b[6], b[7], b[8]
    rm_4, ea_4, tm_4 = b[9], b[10], b[11]
    rr1 = r1(x, rm_1, ea_1, tm_1)
    rr2 = r1(x, rm_2, ea_2, tm_2)
    rr3 = r1(x, rm_3, ea_3, tm_3)
    rr4 = r1(x, rm_4, ea_4, tm_4)
    return rr1 + rr2 + rr3 + rr4

def res_3(b, x, y):
    return model_3(x, b) - y

def jac_3(b, x, y):
    m = len(x)
    n = len(b)
    result = np.zeros((m, n), dtype=np.float64)
    result[:, 0:3] = jac_1(b[0:3], x, y)
    result[:, 3:6] = jac_1(b[3:6], x, y)
    result[:, 6:9] = jac_1(b[6:9], x, y)
    result[:, 9:12] = jac_1(b[9:12], x, y)
    return result

def model_sum1(x, b):
    nn = len(b)
    selector = np.arange(0, nn) % 3
    msk_rm = selector == 0
    msk_ea = selector == 1
    msk_tm = selector == 2
    rms = b[msk_rm]
    eas = b[msk_ea]
    tms = b[msk_tm]
    n = len(rms)
    result = np.zeros_like(x, dtype=np.float64)
    for i in range(n):
        result += r1(x, rms[i], eas[i], tms[i])
    return result

def res_sum1(b, x, y):
    return model_sum1(x, b) - y

def jac_sum1(b, x, y):
    nn = len(b)
    m = len(x)
    selector = np.arange(0, nn) % 3
    msk_rm = selector == 0
    msk_ea = selector == 1
    msk_tm = selector == 2
    rms = b[msk_rm]
    eas = b[msk_ea]
    tms = b[msk_tm]
    n = len(rms)
    result = np.zeros((m, nn), dtype=np.float64)
    for i in range(n):
        cols = 3*i + np.arange(0, 3)
        result[:, cols] = jac_1(np.array([rms[i], eas[i], tms[i]]), x, y)
    return result

def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    rcParams.update(plot_style)
    rcParams['pgf.texsystem'] = 'pdflatex'
    rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{color}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')



def draw_retention_table(ax, retained_dh, retained_d2):
    col_labels = [r'Retention m$^{\mathregular{-2}}$']
    row_labels = [r'DH ', r'D$_{\mathregular{2}}$ ', r'Total D ']
    table1_vals = [
        [fr'${latex_float(retained_dh, significant_digits=1)}$'],
        [fr'${latex_float(retained_d2, significant_digits=1)}$'],
        [fr'${latex_float(retained_dh + 2.*retained_d2, significant_digits=1)}$']
    ]
    row_colors = ['C1', 'C2', 'C0']
    table = ax.table(
        cellText=table1_vals,
        colLabels=col_labels,
        rowLabels=row_labels,
        rowColours=row_colors,
        loc='upper left',
        bbox=[0.28, 0.54, 0.48, 0.44],
        edges='closed',
        fontsize=12,
        rowLoc='left',
    )
    # table_1.PAD=-0.1

    cells = table.get_celld()

    for (row, col), cell in cells.items():
        if col == -1:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
            cell.set_text_props(color='w')
    table.set_fontsize(12)


def draw_fit_table(popt, popt_delta, ax):
    selector = np.arange(0, len(popt)) % 3
    msk_rm = selector == 0
    msk_ea = selector == 1
    msk_tm = selector == 2
    rms, rms_e = popt[msk_rm], popt_delta[msk_rm]
    eas, eas_e = popt[msk_ea], popt_delta[msk_ea]
    tms, tms_e = popt[msk_tm], popt_delta[msk_tm]

    n_peaks = len(rms)
    table = r'''\begin{tabular}{ | l | c | r |} '''
    table += r'''\hline'''
    table += r'''$P$ & \centering $E_{\mathrm{a}}$ (eV)& $T_{\mathrm{p}}$ (K)\\ \hline'''

    for i in range(len(rms)):
        order = 2 if i == 0 else 1
        table += r'''{0} & {1:.1f} ± {2:.1f} & {3:.0f} \\ \hline'''.format(
            i + 1, eas[i], eas_e[i], tms[i], tms_e[i])
    table += r'''\end{tabular}'''

    ax.text(
        0.01, 0.75, table, usetex=True, transform=ax.transAxes,
        fontsize=11, ha='left', va='top'
    )

def main():
    # The boron rod
    boron_rod_df = pd.read_csv(
        './data/20240909/Brod_mks.txt', comment='#', delimiter=r'\s+'
    ).apply(pd.to_numeric)

    boron_pebble_rod_df = pd.read_csv(
        './data/20240909/Bpebble_srs.txt', comment='#', delimiter=r'\s+'
    ).apply(pd.to_numeric)

    pc_boron_pebble_rod_df = pd.read_csv(
        './data/20241014/Bpebble_crystalline_srs.txt', comment='#', delimiter=r'\s+'
    ).apply(pd.to_numeric)

    time_r_s = boron_rod_df['Time[s]'].values
    temp_r_k = boron_rod_df['Temp[K]'].values
    d_r_total = boron_rod_df['[D/m^2/s]'].values
    dh_r = boron_rod_df['[HD/m^2/s]'].values
    d2_r = boron_rod_df['[D2/m^2/s]'].values

    time_p_s = boron_pebble_rod_df['Time[s]'].values
    temp_p_k = boron_pebble_rod_df['Temp[K]'].values
    d_p_total = boron_pebble_rod_df['[D/m^2/s]'].values
    dh_p = boron_pebble_rod_df['[HD/m^2/s]'].values
    d2_p = boron_pebble_rod_df['[D2/m^2/s]'].values

    time_pc_s = pc_boron_pebble_rod_df['Time[s]'].values
    temp_pc_k = pc_boron_pebble_rod_df['Temp[K]'].values
    d_pc_total = pc_boron_pebble_rod_df['[D/m^2/s]'].values
    dh_pc = pc_boron_pebble_rod_df['[HD/m^2/s]'].values
    d2_pc = pc_boron_pebble_rod_df['[D2/m^2/s]'].values

    all_tol = float(np.finfo(np.float64).eps)

    d_p_total[d_p_total < 0.] = 0. # d_p_total[d_p_total >= 0.].min()
    d_pc_total[d_pc_total < 0.] = 0. #d_pc_total[d_pc_total >= 0.].min()

    d2_p[d2_p < 0] = 0.# d2_p[d2_p >= 0.].min()
    d2_pc[d2_pc < 0] = 0.# d2_pc[d2_pc >= 0.].min()
    dh_p[dh_p < 0] = 0.# dh_p[dh_p >= 0.].min()
    dh_pc[dh_pc < 0] = 0.# dh_pc[dh_pc >= 0.].min()

    integrated_boron_rod = simpson(y=d_r_total, x=time_r_s)
    integrated_pebble_boron_rod = simpson(y=d_p_total, x=time_p_s)
    integrated_pc_pebble_boron_rod = simpson(y=d_pc_total, x=time_pc_s)

    integrated_boron_rod_d2 = simpson(y=d2_r, x=time_r_s)
    integrated_boron_pebble_rod_d2 = simpson(y=d2_p, x=time_p_s)
    integrated_pc_boron_rod_d2 = simpson(y=d2_pc, x=time_pc_s)

    integrated_boron_rod_dh = simpson(y=dh_r, x=time_r_s)
    integrated_pebble_boron_rod_dh = simpson(y=dh_p, x=time_p_s)
    integrated_pc_pebble_boron_rod_dh = simpson(y=dh_pc, x=time_pc_s)

    cmap = mpl.colormaps.get_cmap('rainbow')

    # Fit the data for boron rod dh data with single desorption peak of order 1
    # Tried fitting in log space but too much weight was given for data away from
    # the peak. All important data is around the peak within the same order of
    # magnitude of the peak height.
    d_max = d_r_total.max()  # get rm
    idx_max = np.argmin(np.abs(d_r_total - d_max))  # find the index of the max
    t_m = temp_r_k[idx_max]  # explicitly get tm
    # The initial guess for the least_squares
    x0 = np.array([
        d_max * 0.25, 1.0, t_m * 0.9,
        d_max * 0.9, 1.5, t_m,
    ])

    ls_res1: OptimizeResult = least_squares(
        res_sum1, x0, args=(temp_r_k, d_r_total), loss='linear', f_scale=0.1,
        bounds=(
            [
                0., 0.01, temp_r_k.min(),
                0., 1.0, temp_r_k.min()
            ],
            [
                np.inf, 3., temp_r_k.max(),
                np.inf, 6., temp_r_k.max()
            ]
        ),
        jac=jac_sum1,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        verbose=2,
        x_scale='jac',
        max_nfev=10000 * len(temp_r_k)
    )

    popt1 = ls_res1.x
    x1_pred = np.linspace(300, 1200, 2000)
    parameters_1_ci = cf.confidence_interval(res=ls_res1, level=0.95)
    print("**** Fit parameters for boron rod sample ***")
    print("**** D-H ***")
    n_popt1 = len(popt1)
    popt1_delta = np.empty(n_popt1, dtype=np.float64)
    for i, p, lci, uci in zip(range(n_popt1), popt1, parameters_1_ci[:, 0], parameters_1_ci[:, 1]):
        print(f'beta[{i}]: {p:>7.3g}, 95% CI: [{lci:>7.3g}, {uci:>7.3g}]')
        popt1_delta[i] = uci - p

    selector = np.arange(0, len(popt1)) % 3
    msk_rm = selector == 0
    msk_ea = selector == 1
    msk_tm = selector == 2
    rms, rms_e = popt1[msk_rm], popt1_delta[msk_rm]
    eas, eas_e = popt1[msk_ea], popt1_delta[msk_ea]
    tms, tms_e = popt1[msk_tm], popt1_delta[msk_tm]

    n_peaks_2 = len(rms)
    norm1 = plt.cm.colors.Normalize(vmin=0, vmax=n_peaks_2 - 1)
    colors1 = [cmap(norm1(i)) for i in range(n_peaks_2)]

    table1 = r'''\begin{tabular}{ | l | c | r |} ''' + '\n'
    table1 += r'''\hline''' + '\n'
    table1 += r'''$P$ & \centering $E_{\mathrm{a}}$ (eV)& $T_{\mathrm{p}}$ (K)\\\hline''' + '\n'

    for i in range(n_peaks_2):
        color_i = mpl.colors.to_rgba(colors1[i])
        c_txt = r"\textcolor[rgb]{%.3f, %.3f, %.3f}" % (color_i[0], color_i[1], color_i[2])
        print(c_txt)
        table1 += r'''{0}{{ {1} }} & {2:.1f} ± {3:.2f} & {4:.0f} '''.format(
            c_txt, i + 1, eas[i], eas_e[i], tms[i], tms_e[i]) + r"\\ \hline" + '\n'
    table1 += r'''\end{tabular}'''

    print(table1)
    table1 = table1.replace("\n", "")

    y1_pred, delta1 = cf.prediction_intervals(
        model=model_sum1, x_pred=x1_pred, ls_res=ls_res1, jac=jac_sum1
    )

    y1_p1 = r1(x1_pred, r_m=popt1[0], e_a=popt1[1], T_m=popt1[2])
    y1_p2 = r1(x1_pred, r_m=popt1[3], e_a=popt1[4], T_m=popt1[5])


    """"
    """
    # Fit the data for boron pebble rod d-h with multiple desorption peaks
    # The initial guess for the least_squares
    x0 = np.array([
        5E16, 0.3, 380,
        4E16, 0.3, 490,
        1E14, 0.5, 700,
        4E16, 1.0, 810,
        2E17, 2.0, 1100,
        # 2E17, 2.1, 1100,
    ])

    ls_res2: OptimizeResult = least_squares(
        res_sum1, x0, args=(temp_p_k, d_p_total), loss='linear', f_scale=0.1,
        bounds=(
            [
                0, 0.1, 200,
                0., 0.1, 420,
                0., 0.1, 650,
                0., 0.1, 800,
                0., 0.1, 900,
                # 1E15, 0.1, 1000,
            ],
            [
                1E17, np.inf, 450,
                1E17, np.inf, 600,
                1E17, np.inf, 830,
                1E17, np.inf, 950,
                # 1E18, 3., 1095,
                1E18, np.inf, 2000,
            ]
        ),
        jac=jac_sum1,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        verbose=2,
        x_scale='jac',
        max_nfev=10000 * len(temp_p_k)
    )

    popt2 = ls_res2.x
    x2_pred = np.linspace(300, 1200, 2000)
    parameters_2_ci = cf.confidence_interval(res=ls_res2, level=0.95)
    print("**** Fit parameters for boron pebble rod sample ***")
    print("**** D-H ***")
    n_popt2 = len(popt2)
    popt2_delta = np.empty(len(popt2), dtype=np.float64)

    for i, p, lci, uci in zip(range(n_popt2), popt2, parameters_2_ci[:, 0], parameters_2_ci[:, 1]):
        print(f'beta[{i}]: {p:>7.3g}, 95% CI: [{lci:>7.3g}, {uci:>7.3g}]')
        popt2_delta[i] = uci - p

    selector = np.arange(0, len(popt2)) % 3
    msk_rm = selector == 0
    msk_ea = selector == 1
    msk_tm = selector == 2
    rms, rms_e = popt2[msk_rm], popt2_delta[msk_rm]
    eas, eas_e = popt2[msk_ea], popt2_delta[msk_ea]
    tms, tms_e = popt2[msk_tm], popt2_delta[msk_tm]

    n_peaks_2 = len(rms)
    norm2 = mpl.colors.Normalize(vmin=0, vmax=n_peaks_2 - 1)
    colors2 = [cmap(norm2(i)) for i in range(n_peaks_2)]
    table2 = r'''\begin{tabular}{ | l | c | r |} '''
    table2 += r'''\hline'''
    table2 += r'''$P$ & \centering $E_{\mathrm{a}}$ (eV)& $T_{\mathrm{p}}$ (K)\\\hline'''

    for i in range(n_peaks_2):
        table2 += r'''{0} & {1:.1f} ± {2:.2f} & {3:.0f} \\ \hline'''.format(
            i + 1, eas[i], eas_e[i], tms[i], tms_e[i])
    table2 += r'''\end{tabular}'''

    y2_pred, delta2 = cf.prediction_intervals(
        model=model_sum1, x_pred=x2_pred, ls_res=ls_res2, jac=jac_sum1
    )

    y2_p1 = r1(x2_pred, r_m=popt2[0], e_a=popt2[1], T_m=popt2[2])
    y2_p2 = r1(x2_pred, r_m=popt2[3], e_a=popt2[4], T_m=popt2[5])
    y2_p3 = r1(x2_pred, r_m=popt2[6], e_a=popt2[7], T_m=popt2[8])
    y2_p4 = r1(x2_pred, r_m=popt2[9], e_a=popt2[10], T_m=popt2[11])
    y2_p5 = r1(x2_pred, r_m=popt2[12], e_a=popt2[13], T_m=popt2[14])

    load_plot_style()

    fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True)
    fig.set_size_inches(8.5, 6.)

    c0_rgb = mpl.colors.to_rgb('C0')
    c1_rgb = mpl.colors.to_rgb('C1')
    c2_rgb = mpl.colors.to_rgb('C2')
    c3_rgb = mpl.colors.to_rgb('C3')

    latex_colors = []
    for i in range(3):
        c_rgb = mpl.colors.to_rgb(f'C{i}')
        latex_colors.append(','.join(f'{cc:.3f}' for cc in c_rgb))
    """
    Plot DH and D2 separately
    """
    axes[0, 0].plot(temp_r_k, dh_r, color='C1', label='DH B rod')
    axes[0, 0].fill_between(temp_r_k, 0, dh_r, color='C1', alpha=0.2)
    axes[0, 0].plot(temp_r_k, d2_r, color='C2', label='D$_{\mathregular{2}}$ B rod')
    axes[0, 0].fill_between(temp_r_k, 0, d2_r, color='C2', alpha=0.2)
    axes[0, 0].plot(temp_r_k, d_r_total, color='C0', label='Total D B rod')

    axes[0, 0].set_title('Sintered B rod')

    draw_retention_table(
        axes[0, 0], retained_dh=integrated_boron_rod_dh, retained_d2=integrated_boron_rod_d2
    )


    axes[0, 1].plot(temp_p_k, dh_p, color='C1', label='DH B pebble rod')
    axes[0, 1].fill_between(temp_p_k, 0, dh_p, color='C1', alpha=0.2)
    axes[0, 1].plot(temp_p_k, d2_p, color='C2', label='D$_{\mathregular{2}}$ B pebble rod')
    axes[0, 1].plot(temp_p_k, d_p_total, color='C0', label='Total D B pebble rod')

    axes[0, 1].set_title('B pebble rod (sintered)')

    draw_retention_table(
        axes[0, 1], retained_dh=integrated_pebble_boron_rod_dh, retained_d2=integrated_boron_pebble_rod_d2
    )

    axes[0, 2].plot(temp_pc_k, dh_pc, color='C1', label='DH poly-B pebble rod')
    axes[0, 2].fill_between(temp_pc_k, 0, dh_pc, color='C1', alpha=0.2)
    axes[0, 2].plot(temp_pc_k, d2_pc, color='C2', label='D$_{\mathregular{2}}$ poly-B pebble rod')
    axes[0, 2].plot(temp_pc_k, d_pc_total, color='C0', label='Total D poly-B pebble rod')

    axes[0, 2].set_title('Poly B pebble rod')

    draw_retention_table(
        axes[0, 2], retained_dh=integrated_pc_pebble_boron_rod_dh, retained_d2=integrated_pc_boron_rod_d2
    )

    """
    Plot fitted peaks
    """

    axes[1,0].plot(temp_r_k, d_r_total, c='C0', ls='none', marker='o', mfc='none', label='Total D')
    axes[1,0].plot(x1_pred, y1_pred, 'k', ls='-', label=f'Fit', lw=2.)
    axes[1,0].plot(x1_pred, y1_p1, c=colors1[0], ls='-', alpha=0.8, lw=1.0)
    axes[1,0].plot(x1_pred, y1_p2, c=colors1[1], ls='-', alpha=0.8, lw=1.0)

    draw_fit_table(popt=popt1, popt_delta=popt1_delta, ax=axes[1, 0])

    axes[1,1].plot(temp_p_k, d_p_total, c='C0', ls='none', marker='o', mfc='none', label='Total D')
    axes[1,1].plot(x2_pred, y2_pred, 'k', ls='-', label=f'Fit', lw=2.)
    for i in range(n_peaks_2):
        yi = r1(x2_pred, r_m=popt2[3*i], e_a=popt2[3*i+1], T_m=popt2[3*i+2])
        axes[1,1].plot(x2_pred, yi, c=colors2[i], ls='-', alpha=0.8, lw=1.0)

    draw_fit_table(popt=popt2, popt_delta=popt2_delta, ax=axes[1, 1])

    axes[1,2].plot(temp_pc_k, d_pc_total, c='C0', ls='none', marker='o', mfc='none', label='Total D')



    for i, ax in enumerate(axes.flatten()):
        ax.set_xlim(200, 1200)
        ax.ticklabel_format(useMathText=True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))

    for ax in axes[1, :]:
        ax.legend(loc='upper left', fontsize=10)

    # heating_rate_b_rod = np.mean(np.gradient(temp_r_k)) / np.mean(np.gradient(time_r_s))
    # heating_rate_bp_rod = np.mean(np.gradient(temp_p_k)) / np.mean(np.gradient(time_p_s))
    print(f'Retained D boron rod:\t{integrated_boron_rod:.2E} 1/m^2')
    print(f'Retained D boron pebble rod:\t{integrated_pebble_boron_rod:.2E} 1/m^2')
    print(f'Retained D poly boron pebble rod:\t{integrated_pc_pebble_boron_rod:.2E} 1/m^2')

    fig.supxlabel('T (K)')
    fig.supylabel('Desorption flux (m$^{\mathregular{-2}}$ s$^{\mathregular{-1}}$)')


    fig.savefig(os.path.join('./figures', '20241014_TDS_FIT_boron_pebble_vs_boron_rod.png'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()



