import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import os
import json
import numpy as np
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf

k_b = 8.617333262E-5 # eV/K

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
    rr1 = r1(x, rm_1, ea_1, tm_1)
    rr2 = r2(x, rm_2, ea_2, tm_2)
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
    result[:, 3:6] = jac_r2(b[3:6], x, y)
    result[:, 6:9] = jac_1(b[6:9], x, y)
    result[:, 9:12] = jac_1(b[9:12], x, y)
    result[:, 12:15] = jac_1(b[12:15], x, y)
    return result



def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    rcParams.update(plot_style)
    plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'

def main():
    # The boron rod
    boron_rod_df = pd.read_csv(
        './data/20240909/Brod_mks.txt', comment='#', delimiter=r'\s+'
    ).apply(pd.to_numeric)

    boron_pebble_rod_df = pd.read_csv(
        './data/20240909/Bpebble_srs.txt', comment='#', delimiter=r'\s+'
    ).apply(pd.to_numeric)

    temp_r_k = boron_rod_df['Temp[K]'].values
    d_r_total = boron_rod_df['[D/m^2/s]'].values
    dh_r = boron_rod_df['[HD/m^2/s]'].values
    d2_r = boron_rod_df['[D2/m^2/s]'].values

    temp_p_k = boron_pebble_rod_df['Temp[K]'].values
    d_p_total = boron_pebble_rod_df['[D/m^2/s]'].values
    dh_p = boron_pebble_rod_df['[HD/m^2/s]'].values
    d2_p = boron_pebble_rod_df['[D2/m^2/s]'].values

    msk_positive_dh = dh_p >= 0.
    msk_positive_d2 = d2_p >= 0.

    temp_p_k_p1 = temp_p_k[msk_positive_dh]
    temp_p_k_p2 = temp_p_k[msk_positive_d2]
    dh_p_p = dh_p[msk_positive_dh]
    d2_p_p = d2_p[msk_positive_d2]

    # Fit the data for boron rod dh data with single desorption peak of order 1
    dh_r_max = dh_r.max() # get rm
    idx_dh_r_max = np.argmin(np.abs(dh_r - dh_r_max)) # find the index of the max
    t_dh_m = temp_r_k[idx_dh_r_max] # explicitly get tm
    # The initial guess for the least_squares
    x0 = np.array([
        dh_r_max, 1., t_dh_m
    ])
    all_tol = float(np.finfo(np.float64).eps)
    ls_res1: OptimizeResult = least_squares(
        res_1, x0, args=(temp_r_k, dh_r), loss='linear', f_scale=0.1,
        bounds=([0., -20., temp_r_k.min()], [np.inf, 20., temp_r_k.max()]),
        jac=jac_1,
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
    popt1_delta = np.empty(3, dtype=np.float64)
    for i, p, lci, uci in zip(range(n_popt1), popt1, parameters_1_ci[:, 0], parameters_1_ci[:, 1]):
        print(f'beta[{i}]: {p:>7.3g}, 95% CI: [{lci:>7.3g}, {uci:>7.3g}]')
        popt1_delta[i] = uci - p

    y1_pred, delta1 = cf.prediction_intervals(
        model=model_1, x_pred=x1_pred, ls_res=ls_res1, jac=jac_1
    )

    dh_r_txt = r'$E_{\mathrm{a}} = ' + f'{popt1[1]:.1f}\pm{popt1_delta[1]:.2f}' + r'~\mathrm{eV}$' + '\n'
    dh_r_txt += r'$T_{\mathrm{p}} = ' + f'{popt1[2]:.0f}\pm{popt1_delta[2]:.0f}' + r'~\mathrm{K}$'

    # Fit the data for boron rod d2 with single desorption peak of order 1
    d2_r_max = d2_r.max()  # get rm
    idx_d2_r_max = np.argmin(np.abs(d2_r - d2_r_max))  # find the index of the max
    t_d2_m = temp_r_k[idx_d2_r_max]  # explicitly get tm
    # The initial guess for the least_squares
    x0 = np.array([
        d2_r_max, 2., t_d2_m
    ])
    ls_res2: OptimizeResult = least_squares(
        res_1, x0, args=(temp_r_k, d2_r), loss='linear', f_scale=0.1,
        bounds=([0., -20., temp_r_k.min()], [np.inf, 20., temp_r_k.max()]),
        jac=jac_1,
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
    print("**** D-H ***")
    n_popt2 = len(popt2)
    popt2_delta = np.empty(3, dtype=np.float64)
    for i, p, lci, uci in zip(range(n_popt2), popt2, parameters_2_ci[:, 0], parameters_2_ci[:, 1]):
        print(f'beta[{i}]: {p:>7.3g}, 95% CI: [{lci:>7.3g}, {uci:>7.3g}]')
        popt2_delta[i] = uci - p

    y2_pred, delta2 = cf.prediction_intervals(
        model=model_1, x_pred=x1_pred, ls_res=ls_res2, jac=jac_1
    )

    d2_r_txt = r'$E_{\mathrm{a}} = ' + f'{popt2[1]:.1f}\pm{popt2_delta[1]:.2f}' + r'~\mathrm{eV}$' + '\n'
    d2_r_txt += r'$T_{\mathrm{p}} = ' + f'{popt2[2]:.0f}\pm{popt2_delta[2]:.0f}' + r'~\mathrm{K}$'

    # Fit the data for boron pebble rod d-h with single desorption peak of order 1
    # The initial guess for the least_squares
    x0 = np.array([
        5E16, 0.5, 370,
        1E16, 0.5, 492,
        5E13, 1., 800,
        5E16, 1., 880,
        2E17, 2., 1100,
    ])
    ls_res3: OptimizeResult = least_squares(
        res_2, x0, args=(temp_p_k_p1, dh_p_p), loss='linear', f_scale=0.1,
        bounds=(
            [
                0, 0.1, 200,
                0., 0.1, 350,
                0., 0.1, 650,
                0., 0.1, 800,
                1E15, 0.1, 1000,
            ],
            [
                1E17, np.inf, 450,
                1E17, np.inf, 600,
                1E17, np.inf, 850,
                1E17, np.inf, 950,
                1E18, np.inf, 2000,
            ]
        ),
        jac=jac_2,
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        verbose=2,
        max_nfev=10000 * len(temp_p_k_p1)
    )

    popt3 = ls_res3.x
    x1_pred = np.linspace(300, 1200, 2000)
    parameters_3_ci = cf.confidence_interval(res=ls_res3, level=0.95)
    print("**** Fit parameters for boron rod sample ***")
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

    table1_data = []
    table1 = r'''\begin{tabular}{ | l | c | c | r |} '''
    table1 += r'''\hline'''
    table1 += r'''$P$ & $n$ & \centering $E_{\mathrm{a}}$ & $T_{\mathrm{p}}$\\\hline'''

    for i in range(len(rms)):
        order = 2 if i == 1 else 1
        table1 += r'''{0} & {1} & {2:.1f} ± {3:.1f} & {4:.0f} \\ \hline'''.format(
        i+1, order, eas[i], eas_e[i], tms[i], tms_e[i])
        # table1_data.append([f"{i}", f"{order}", f"{eas[i]:.2f} ± {eas_e[i]:.2f}", f"{tms[i]:.0f} ± {tms_e[i]:.0f}"])

    # table1 += r'''\hline'''
    table1 += r'''\end{tabular}'''

    y3_pred, delta3 = cf.prediction_intervals(
        model=model_2, x_pred=x1_pred, ls_res=ls_res3, jac=jac_2
    )

    y3_p1 = r1(x1_pred, r_m=popt3[0], e_a=popt3[1], T_m=popt3[2])
    y3_p2 = r2(x1_pred, r_m=popt3[3], e_a=popt3[4], T_m=popt3[5])
    y3_p3 = r1(x1_pred, r_m=popt3[6], e_a=popt3[7], T_m=popt3[8])
    y3_p4 = r1(x1_pred, r_m=popt3[9], e_a=popt3[10], T_m=popt3[11])
    y3_p5 = r1(x1_pred, r_m=popt3[12], e_a=popt3[13], T_m=popt3[14])



    # dh_p_txt = r'$E_{\mathrm{a}} = ' + f'{popt2[1]:.1f}\pm{popt2_delta[1]:.2f}' + r'~\mathrm{eV}$' + '\n'
    # dh_p_txt += r'$T_{\mathrm{p}} = ' + f'{popt2[2]:.0f}\pm{popt2_delta[2]:.0f}' + r'~\mathrm{K}$'

    load_plot_style()
    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    fig.set_size_inches(7., 6.)

    axes[0, 0].plot(temp_r_k, dh_r, c='C0', ls='none', marker='o', mfc='none', label='H-D boron rod')
    axes[0, 1].plot(temp_r_k, d2_r, c='C0', ls='none', marker='s', mfc='none', label='D$_{\mathregular{2}}$ boron rod')

    axes[0, 0].plot(x1_pred, y1_pred, 'k', ls='-', label=f'Fit')
    # axes[0].plot(temp_r_k, d_r_total, c='k', ls='-', label='2D$_{\mathregular{2}}$ + HD')
    # axes[0, 0].set_title('Boron rod')
    # axes[0, 1].set_title('Boron rod')
    axes[0, 0].text(
        0.05, 0.1, dh_r_txt,
        transform=axes[0, 0].transAxes,
        color='k', fontsize=11,
        ha='left', va='bottom'
    )

    axes[0, 1].plot(x1_pred, y2_pred, 'k', ls='-', label=f'Fit')
    # axes[0].plot(temp_r_k, d_r_total, c='k', ls='-', label='2D$_{\mathregular{2}}$ + HD')
    # axes[0, 0].set_title('Boron rod')
    # axes[0, 1].set_title('Boron rod')
    axes[0, 1].text(
        0.05, 0.1, d2_r_txt,
        transform=axes[0, 1].transAxes,
        color='k', fontsize=11,
        ha='left', va='bottom'
    )

    axes[1, 0].plot(temp_p_k_p1, dh_p_p, c='C1', ls='none', marker='^', mfc='none', label='H-D boron pebble rod')
    axes[1, 1].plot(temp_p_k_p2, d2_p_p, c='C3', ls='-', label='D$_{\mathregular{2}}$ boron pebble rod')

    axes[1, 0].plot(x1_pred, y3_pred, 'k', ls='-', lw=2, label=f'Fit')
    axes[1, 0].plot(x1_pred, y3_p1, 'C1', ls='-')
    axes[1, 0].plot(x1_pred, y3_p2, 'C2', ls='-')
    axes[1, 0].plot(x1_pred, y3_p3, 'C3', ls='-')
    axes[1, 0].plot(x1_pred, y3_p4, 'C4', ls='-')
    axes[1, 0].plot(x1_pred, y3_p5, 'C5', ls='-')

    axes[1, 0].text(
        0.05, 0.95, table1, usetex=True, transform=axes[1, 0].transAxes,
        fontsize=9, ha='left', va='top'
    )



    axes[1, 0].plot(temp_p_k[~msk_positive_dh], dh_p[~msk_positive_dh], c='gray', ls=':') #, label='Anomaly')
    axes[1, 1].plot(temp_p_k[~msk_positive_d2], d2_p[~msk_positive_d2], c='gray', ls=':')#, label='Anomaly')
    # axes[1].plot(temp_p_k, d_p_total, c='k', ls='-', label='2D$_{\mathregular{2}}$ + HD')
    # axes[1, 0].set_title('Boron pebble rod')
    # axes[1, 1].set_title('Boron pebble rod')

    for i, ax in enumerate(axes.flatten()):
        ax.set_xlim(200, 1200)
        ax.ticklabel_format(useMathText=True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))

        loc = 'upper left' if i < 2 else 'lower left'
        ax.legend(loc=loc, fontsize=10)

    fig.supxlabel('T (K)')
    fig.supylabel('Desorption flux (m$^{\mathregular{-2}}$ s$^{\mathregular{-1}}$)')

    # fig.savefig(os.path.join('./figures', '20240909_TDS_boron_pebble_vs_boron_rod.png'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()



