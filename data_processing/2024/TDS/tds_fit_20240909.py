import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dask.array.random import normal
from matplotlib import rcParams
import os
import json
import numpy as np
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf
from scipy.stats.distributions import t
import matplotlib as mpl
from scipy.integrate import simps

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
    rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                       r'\usepackage{xcolor}')



def main():
    # The boron rod
    boron_rod_df = pd.read_csv(
        './data/20240909/Brod_mks.txt', comment='#', delimiter=r'\s+'
    ).apply(pd.to_numeric)

    boron_pebble_rod_df = pd.read_csv(
        './data/20240909/Bpebble_srs.txt', comment='#', delimiter=r'\s+'
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

    all_tol = float(np.finfo(np.float64).eps)

    d_p_total[d_p_total <= 0.] = d_p_total[d_p_total >= 0.].min()
    msk_positive_dh = dh_p >= 0.
    msk_positive_d2 = d2_p >= 0.

    dh_p[dh_p <= 0.] = dh_p[msk_positive_dh].min()
    d2_p[d2_p <= 0.] = d2_p[msk_positive_d2].min()

    integrated_boron_rod = simps(y=d_r_total, x=time_r_s)
    integrated_pebble_boron_rod = simps(y=d_p_total, x=time_p_s)

    integrated_boron_rod_d2 = simps(y=d2_r, x=time_r_s)
    integrated_pebble_boron_rod_d2 = simps(y=d2_p, x=time_p_s)

    integrated_boron_rod_dh = simps(y=dh_r, x=time_r_s)
    integrated_pebble_boron_rod_dh = simps(y=dh_p, x=time_p_s)

    # temp_p_k_p1 = temp_p_k[msk_positive_dh]
    # temp_p_k_p2 = temp_p_k[msk_positive_d2]
    # dh_p_p = dh_p[msk_positive_dh]
    # d2_p_p = d2_p[msk_positive_d2]

    cmap = mpl.colormaps.get_cmap('rainbow')

    # Fit the data for boron rod dh data with single desorption peak of order 1
    # Tried fitting in log space but too much weight was given for data away from
    # the peak. All important data is around the peak within the same order of
    # magnitude of the peak height.
    dh_r_max = dh_r.max() # get rm
    idx_dh_r_max = np.argmin(np.abs(dh_r - dh_r_max)) # find the index of the max
    t_dh_m = temp_r_k[idx_dh_r_max] # explicitly get tm
    # The initial guess for the least_squares
    x0 = np.array([
        dh_r_max*0.25, 1.0, t_dh_m*0.9,
        dh_r_max*0.9, 1.5, t_dh_m,
    ])

    ls_res1: OptimizeResult = least_squares(
        res_sum1, x0, args=(temp_r_k, dh_r), loss='linear', f_scale=0.1,
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
        max_nfev=10000 * len(temp_p_k)
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

    n_peaks = len(rms)
    norm1 = plt.cm.colors.Normalize(vmin=0, vmax=n_peaks - 1)
    colors1 = [cmap(norm1(i)) for i in range(n_peaks)]

    table1 = r'''\begin{tabular}{ | l | c | r |} ''' + '\n'
    table1 += r'''\hline''' + '\n'
    table1 += r'''$P$ & \centering $E_{\mathrm{a}}$ (eV)& $T_{\mathrm{p}}$ (K)\\\hline''' + '\n'

    for i in range(n_peaks):
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

    dh_r_txt = r'$E_{\mathrm{a}} = ' + f'{popt1[1]:.1f}\pm{popt1_delta[1]:.2f}' + r'~\mathrm{eV}$' + '\n'
    dh_r_txt += r'$T_{\mathrm{p}} = ' + f'{popt1[2]:.0f}\pm{popt1_delta[2]:.0f}' + r'~\mathrm{K}$'

    # Fit the data for boron rod d2 with single desorption peak of order 1
    d2_r_max = d2_r.max()  # get rm
    idx_d2_r_max = np.argmin(np.abs(d2_r - d2_r_max))  # find the index of the max
    t_d2_m = temp_r_k[idx_d2_r_max]  # explicitly get tm
    # The initial guess for the least_squares
    x0 = np.array([
        d2_r_max*0.25, 1., t_d2_m*0.9,
        d2_r_max*0.9, 2., t_d2_m
    ])
    ls_res2: OptimizeResult = least_squares(
        res_sum1, x0, args=(temp_r_k, d2_r), loss='linear', f_scale=0.1,
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
        max_nfev=10000 * len(temp_p_k)
    )

    popt2 = ls_res2.x
    x1_pred = np.linspace(300, 1200, 2000)
    parameters_2_ci = cf.confidence_interval(res=ls_res2, level=0.95)
    print("**** Fit parameters for boron rod sample ***")
    print("**** D2 ***")
    n_popt2 = len(popt2)

    popt2_delta = np.empty(n_popt2, dtype=np.float64)
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
    table2 = r'''\begin{tabular}{ | l | c | r |} '''
    table2 += r'''\hline'''
    table2 += r'''$P$ & \centering $E_{\mathrm{a}}$ (eV)& $T_{\mathrm{p}}$ (K)\\\hline'''

    n_peaks = len(rms)
    norm2 = plt.cm.colors.Normalize(vmin=0, vmax=n_peaks - 1)
    colors2 = [cmap(norm2(i)) for i in range(n_peaks)]
    for i in range(n_peaks):
        table2 += r'''{0} & {1:.1f} ± {2:.2f} & {3:.0f} \\ \hline'''.format(
            i + 1, eas[i], eas_e[i], tms[i], tms_e[i])
    table2 += r'''\end{tabular}'''

    y2_pred, delta2 = cf.prediction_intervals(
        model=model_sum1, x_pred=x1_pred, ls_res=ls_res2, jac=jac_sum1
    )

    y2_p1 = r1(x1_pred, r_m=popt2[0], e_a=popt2[1], T_m=popt2[2])
    y2_p2 = r1(x1_pred, r_m=popt2[3], e_a=popt2[4], T_m=popt2[5])

    d2_r_txt = r'$E_{\mathrm{a}} = ' + f'{popt2[1]:.1f}\pm{popt2_delta[1]:.2f}' + r'~\mathrm{eV}$' + '\n'
    d2_r_txt += r'$T_{\mathrm{p}} = ' + f'{popt2[2]:.0f}\pm{popt2_delta[2]:.0f}' + r'~\mathrm{K}$'

    # Fit the data for boron pebble rod d-h with multiple desorption peaks
    # The initial guess for the least_squares
    x0 = np.array([
        5E16, 0.3, 380,
        4E16, 0.3, 490,
        1E14, 0.5, 700,
        4E16, 1.0, 810,
        2E17, 2.0, 1000,
        # 2E17, 2.1, 1100,
    ])
    ls_res3: OptimizeResult = least_squares(
        res_sum1, x0, args=(temp_p_k, dh_p), loss='linear', f_scale=0.1,
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
        max_nfev=10000 * len(temp_p_k)
    )

    popt3 = ls_res3.x
    x1_pred = np.linspace(300, 1200, 2000)
    parameters_3_ci = cf.confidence_interval(res=ls_res3, level=0.95)
    print("**** Fit parameters for boron pebble rod sample ***")
    print("**** D-H ***")
    n_popt3 = len(popt3)
    popt3_delta = np.empty(len(popt3), dtype=np.float64)

    for i, p, lci, uci in zip(range(n_popt3), popt3, parameters_3_ci[:, 0], parameters_3_ci[:, 1]):
        print(f'beta[{i}]: {p:>7.3g}, 95% CI: [{lci:>7.3g}, {uci:>7.3g}]')
        popt3_delta[i] = uci - p

    selector = np.arange(0, len(popt3)) % 3
    msk_rm = selector == 0
    msk_ea = selector == 1
    msk_tm = selector == 2
    rms, rms_e = popt3[msk_rm], popt3_delta[msk_rm]
    eas, eas_e = popt3[msk_ea], popt3_delta[msk_ea]
    tms, tms_e = popt3[msk_tm], popt3_delta[msk_tm]

    n_peaks = len(rms)
    norm3 = plt.cm.colors.Normalize(vmin=0, vmax=n_peaks - 1)
    colors3 = [cmap(norm3(i)) for i in range(n_peaks)]
    table3 = r'''\begin{tabular}{ | l | c | r |} '''
    table3 += r'''\hline'''
    table3 += r'''$P$ & \centering $E_{\mathrm{a}}$ (eV)& $T_{\mathrm{p}}$ (K)\\\hline'''

    for i in range(n_peaks):
        table3 += r'''{0} & {1:.1f} ± {2:.2f} & {3:.0f} \\ \hline'''.format(
        i+1, eas[i], eas_e[i], tms[i], tms_e[i])
    table3 += r'''\end{tabular}'''

    y3_pred, delta3 = cf.prediction_intervals(
        model=model_sum1, x_pred=x1_pred, ls_res=ls_res3, jac=jac_sum1
    )

    y3_p1 = r1(x1_pred, r_m=popt3[0], e_a=popt3[1], T_m=popt3[2])
    y3_p2 = r1(x1_pred, r_m=popt3[3], e_a=popt3[4], T_m=popt3[5])
    y3_p3 = r1(x1_pred, r_m=popt3[6], e_a=popt3[7], T_m=popt3[8])
    y3_p4 = r1(x1_pred, r_m=popt3[9], e_a=popt3[10], T_m=popt3[11])
    y3_p5 = r1(x1_pred, r_m=popt3[12], e_a=popt3[13], T_m=popt3[14])
    # y3_p6= r1(x1_pred, r_m=popt3[15], e_a=popt3[16], T_m=popt3[17])

    # Fit the data for boron pebble rod d2 with multiple desorption peaks
    # The initial guess for the least_squares
    x0 = np.array([
        1E16, 0.3, 430,
        4E15, 1.1, 760,
        1E16, 1.5, 882,
        1E16, 1.5, 1050,
        # 1E16, 2.0, 1200,
    ])
    ls_res4: OptimizeResult = least_squares(
        res_sum1, x0, args=(temp_p_k, d2_p), loss='linear', f_scale=0.1,
        bounds=(
            [
                0., 0.1, 350,
                0., 0.1, 650,
                0., 0.1, 800,
                1E14, 0.1, 1000,
                # 1E15, 0.1, 1080,
            ],
            [
                1E17, 10., 600,
                1E17, 10., 850,
                1E17, 10., 950,
                # 1E18, 4., 1150,
                1E18, 10., 2000,
            ]
        ),
        jac=jac_sum1,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        verbose=2,
        max_nfev=10000 * len(temp_p_k)
    )


    popt4 = ls_res4.x
    x1_pred = np.linspace(300, 1200, 2000)
    parameters_4_ci = cf.confidence_interval(res=ls_res4, level=0.95)
    print("**** Fit parameters for boron pebble rod sample ***")
    print("**** D2 ***")
    n_popt4 = len(popt4)
    popt4_delta = np.empty(len(popt4), dtype=np.float64)

    for i, p, lci, uci in zip(range(n_popt4), popt4, parameters_4_ci[:, 0], parameters_4_ci[:, 1]):
        print(f'beta[{i}]: {p:>7.3g}, 95% CI: [{lci:>7.3g}, {uci:>7.3g}]')
        popt4_delta[i] = uci - p


    selector = np.arange(0, len(popt4)) % 3
    msk_rm = selector == 0
    msk_ea = selector == 1
    msk_tm = selector == 2
    rms, rms_e = popt4[msk_rm], popt4_delta[msk_rm]
    eas, eas_e = popt4[msk_ea], popt4_delta[msk_ea]
    tms, tms_e = popt4[msk_tm], popt4_delta[msk_tm]

    n_peaks = len(rms)
    norm4 = plt.cm.colors.Normalize(vmin=0, vmax=n_peaks - 1)
    colors4 = [cmap(norm4(i)) for i in range(n_peaks)]

    table4 = r'''\begin{tabular}{ | l | c | r |} '''
    table4 += r'''\hline'''
    table4 += r'''$P$ & \centering $E_{\mathrm{a}}$ (eV)& $T_{\mathrm{p}}$ (K)\\ \hline'''

    for i in range(len(rms)):
        order = 2 if i == 0 else 1
        table4 += r'''{0} & {1:.1f} ± {2:.1f} & {3:.0f} \\ \hline'''.format(
            i + 1, eas[i], eas_e[i], tms[i], tms_e[i])
    table4 += r'''\end{tabular}'''

    y4_pred, delta4 = cf.prediction_intervals(
        model=model_sum1, x_pred=x1_pred, ls_res=ls_res4, jac=jac_sum1
    )

    y4_p1 = r1(x1_pred, r_m=popt4[0], e_a=popt4[1], T_m=popt4[2])
    y4_p2 = r1(x1_pred, r_m=popt4[3], e_a=popt4[4], T_m=popt4[5])
    y4_p3 = r1(x1_pred, r_m=popt4[6], e_a=popt4[7], T_m=popt4[8])
    y4_p4 = r1(x1_pred, r_m=popt4[9], e_a=popt4[10], T_m=popt4[11])
    # y4_p5 = r1(x1_pred, r_m=popt4[12], e_a=popt4[13], T_m=popt4[14])



    # dh_p_txt = r'$E_{\mathrm{a}} = ' + f'{popt2[1]:.1f}\pm{popt2_delta[1]:.2f}' + r'~\mathrm{eV}$' + '\n'
    # dh_p_txt += r'$T_{\mathrm{p}} = ' + f'{popt2[2]:.0f}\pm{popt2_delta[2]:.0f}' + r'~\mathrm{K}$'

    load_plot_style()
    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    fig.set_size_inches(7.5, 6.)



    axes[0, 0].plot(temp_r_k, dh_r, c='C0', ls='none', marker='o', mfc='none', label='H-D boron rod')
    axes[0, 1].plot(temp_r_k, d2_r, c='C0', ls='none', marker='s', mfc='none', label='D$_{\mathregular{2}}$ boron rod')

    axes[0, 0].plot(x1_pred, y1_pred, 'k', ls='-', label=f'Fit', lw=2.)
    # axes[0].plot(temp_r_k, d_r_total, c='k', ls='-', label='2D$_{\mathregular{2}}$ + HD')
    # axes[0, 0].set_title('Boron rod')
    # axes[0, 1].set_title('Boron rod')

    axes[0, 0].plot(x1_pred, y1_p1, c=colors1[0], ls='-', alpha=0.8, lw=1.0)
    axes[0, 0].plot(x1_pred, y1_p2, c=colors1[1], ls='-', alpha=0.8, lw=1.0)

    axes[0, 0].text(
        0.01, 0.5, table1,
        transform=axes[0, 0].transAxes,
        fontsize=11,
        ha='left', va='center',
        usetex=True
    )

    axes[0, 1].plot(x1_pred, y2_pred, 'k', ls='-', label=f'Fit', lw=2)
    axes[0, 1].plot(x1_pred, y2_p1, c=colors2[0], ls='-', alpha=1.0, lw=1.0)
    axes[0, 1].plot(x1_pred, y2_p2, c=colors2[1], ls='-', alpha=1.0, lw=1.0)

    # axes[0].plot(temp_r_k, d_r_total, c='k', ls='-', label='2D$_{\mathregular{2}}$ + HD')
    # axes[0, 0].set_title('Boron rod')
    # axes[0, 1].set_title('Boron rod')
    axes[0, 1].text(
        0.025, 0.5, table2,
        transform=axes[0, 1].transAxes,
        color='k', fontsize=11,
        ha='left', va='center',
        usetex=True
    )

    axes[1, 0].plot(temp_p_k, dh_p, c='C0', ls='none', marker='^', mfc='none', label='H-D boron pebble rod')
    axes[1, 1].plot(temp_p_k, d2_p, c='C0', ls='none', marker='D', mfc='none', label='D$_{\mathregular{2}}$ boron pebble rod')

    axes[1, 0].plot(x1_pred, y3_pred, 'k', ls='-', lw=2, label=f'Fit')
    axes[1, 0].plot(x1_pred, y3_p1, c=colors3[0], ls='-', alpha=0.8, lw=1.0)
    axes[1, 0].plot(x1_pred, y3_p2, c=colors3[1], ls='-', alpha=0.8, lw=1.0)
    axes[1, 0].plot(x1_pred, y3_p3, c=colors3[2], ls='-', alpha=0.8, lw=1.0)
    axes[1, 0].plot(x1_pred, y3_p4, c=colors3[3], ls='-', alpha=0.8, lw=1.0)
    axes[1, 0].plot(x1_pred, y3_p5, c=colors3[4], ls='-', alpha=0.8, lw=1.0)
    # axes[1, 0].plot(x1_pred, y3_p6, c=colors3[5], ls='-', alpha=0.8, lw=1.0)

    # y3_0 = np.zeros_like(x1_pred)
    # axes[1, 0].fill_between(x1_pred, y3_0, y3_p1, color='C2', alpha=0.5)
    # axes[1, 0].fill_between(x1_pred, y3_0, y3_p2, color='C3', alpha=0.5)
    # axes[1, 0].fill_between(x1_pred, y3_0, y3_p3, color='C4', alpha=0.5)
    # axes[1, 0].fill_between(x1_pred, y3_0, y3_p4, color='C5', alpha=0.5)
    # axes[1, 0].fill_between(x1_pred, y3_0, y3_p5, color='C6', alpha=0.5)

    axes[1, 0].text(
        0.01, 0.985, table3, usetex=True, transform=axes[1, 0].transAxes,
        fontsize=11, ha='left', va='top'
    )

    axes[1, 1].plot(x1_pred, y4_pred, 'k', ls='-', lw=2, label=f'Fit')
    axes[1, 1].plot(x1_pred, y4_p1, c=colors4[0], ls='-', alpha=0.8, lw=1.0)
    axes[1, 1].plot(x1_pred, y4_p2, c=colors4[1], ls='-', alpha=0.8, lw=1.0)
    axes[1, 1].plot(x1_pred, y4_p3, c=colors4[2], ls='-', alpha=0.8, lw=1.0)
    axes[1, 1].plot(x1_pred, y4_p4, c=colors4[3], ls='-', alpha=0.8, lw=1.0)
    # axes[1, 1].plot(x1_pred, y4_p5, c=colors4[4], ls='-', alpha=0.8, lw=1.0)

    axes[1, 1].text(
        0.01, 0.985, table4, usetex=True, transform=axes[1, 1].transAxes,
        fontsize=11, ha='left', va='top'
    )



    axes[1, 0].plot(temp_p_k[~msk_positive_dh], dh_p[~msk_positive_dh], c='gray', ls=':', lw=1.0) #, label='Anomaly')
    axes[1, 1].plot(temp_p_k[~msk_positive_d2], d2_p[~msk_positive_d2], c='gray', ls=':', lw=1.0)#, label='Anomaly')
    # axes[1].plot(temp_p_k, d_p_total, c='k', ls='-', label='2D$_{\mathregular{2}}$ + HD')


    for ax in axes[0, :]:
        ax.set_ylim(0, 1.5E20)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5E19))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5E18))

    axes[1,0].set_ylim(-1.5E17, 3.5E17)
    axes[1,0].yaxis.set_major_locator(ticker.MultipleLocator(1E17))
    axes[1,0].yaxis.set_minor_locator(ticker.MultipleLocator(2.5E16))

    axes[1, 1].set_ylim(-6E16, 1.5E17)
    axes[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(5E16))
    axes[1, 1].yaxis.set_minor_locator(ticker.MultipleLocator(1E16))

    for i, ax in enumerate(axes.flatten()):
        ax.set_xlim(200, 1200)
        ax.ticklabel_format(useMathText=True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))

        loc = 'upper left' if i < 2 else 'lower left'
        ax.legend(loc=loc, fontsize=10)

    fig.supxlabel('T (K)')
    fig.supylabel('Desorption flux (m$^{\mathregular{-2}}$ s$^{\mathregular{-1}}$)')

    # heating_rate_b_rod = np.mean(np.gradient(temp_r_k)) / np.mean(np.gradient(time_r_s))
    # heating_rate_bp_rod = np.mean(np.gradient(temp_p_k)) / np.mean(np.gradient(time_p_s))
    print(f'Retained D boron rod:\t{integrated_boron_rod:.2E} 1/m^2')
    print(f'Retained D boron pebble rod:\t{integrated_pebble_boron_rod:.2E} 1/m^2')

    print(f'Retained D2 boron rod:\t{integrated_boron_rod_d2:.2E} 1/m^2')
    print(f'Retained D2 boron pebble rod:\t{integrated_pebble_boron_rod_d2:.2E} 1/m^2')

    print(f'Retained DH boron rod:\t{integrated_boron_rod_dh:.2E} 1/m^2')
    print(f'Retained DH boron pebble rod:\t{integrated_pebble_boron_rod_dh:.2E} 1/m^2')

    fig.savefig(os.path.join('./figures', '20240909_TDS_FIT_boron_pebble_vs_boron_rod.png'), dpi=600)
    fig.savefig(os.path.join('./figures', '20240909_TDS_FIT_boron_pebble_vs_boron_rod.pdf'), dpi=600)
    fig.savefig(os.path.join('./figures', '20240909_TDS_FIT_boron_pebble_vs_boron_rod.svg'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()



