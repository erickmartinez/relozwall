import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker
import logging
from scipy.optimize import least_squares, OptimizeResult
from data_processing.utils import lighten_color
import data_processing.confidence as cf

base_path = r"C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\thermal_conductivity\graphite\laser_flash\CALIBRATION_20230719"
xls_db = "flash_method_at_different_heat_loads.xlsx"


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


def poly(x, b):
    n = len(b)
    xx = np.ones_like(x)
    r = np.zeros_like(x)
    for i in range(n):
        r += xx * b[i]
        xx *= x
    return r


def fobj(b, x, y):
    return poly(x, b) - y


def jac(b, x, y):
    n = len(b)
    jj = np.zeros((len(x), n), dtype=float)
    for i in range(n):
        jj[:, i] = np.power(x, i)
    return jj


def main():
    ftc_df: pd.DataFrame = pd.read_excel(os.path.join(base_path, xls_db))
    columns = ftc_df.columns
    numeric_columns = columns[:-1]
    ftc_df[numeric_columns] = ftc_df[numeric_columns].apply(pd.to_numeric)
    ftc_df.sort_values(by=['Laser power setting (%)'], inplace=True)
    for c in columns:
        print(c)
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 5.0)
    laser_power = ftc_df['P_laser (W)'].values
    norm = mpl.colors.Normalize(vmin=laser_power.min(), vmax=laser_power.max())
    cmap = plt.cm.jet

    for i, r in ftc_df.iterrows():
        csv_origin = str(r['csv'])
        csv_file = os.path.splitext(csv_origin)[0] + '_ftd_curves.csv'
        lp = r['P_laser (W)']
        full_file = os.path.join(base_path, csv_file)
        # print(f"Plotting {full_file}")
        c_df = pd.read_csv(full_file).apply(pd.to_numeric)
        tth = c_df['t/t_h'].values
        TTm = c_df['DT/T_max'].values
        axes[0].plot(tth, TTm, c=cmap(norm(lp)))

    axes[0].set_xlabel("$t/t_{\mathrm{1/2}}$")
    axes[0].set_ylabel("$\Delta T/ \Delta T_{\mathrm{max}}$")

    p_in = ftc_df['P_in (W)'].values / 0.8 / 0.6

    b0 = [0, 1]
    all_tol = np.finfo(np.float64).eps
    n = len(p_in)
    laser_power_model = np.linspace(laser_power.min(), 3200, num=200)
    res: OptimizeResult = least_squares(
        fun=fobj, x0=b0, args=(laser_power, p_in), jac=jac, ftol=all_tol, xtol=all_tol, gtol=all_tol,
        max_nfev=10000 * n,
        # loss='soft_l1', f_scale=0.1,
        verbose=2
    )

    popt = res.x
    pcov = cf.get_pcov(res)


    ci = cf.confint(n=n, pars=popt, pcov=pcov)
    p_in_pred, lpb, upb = cf.predint(x=laser_power_model, xd=laser_power, yd=p_in, func=poly, res=res)

    axes[1].fill_between(laser_power_model, lpb, upb, color=lighten_color('C0', 0.5))
    axes[1].plot(laser_power_model, p_in_pred, ls='-', c='C0')
    axes[1].plot(laser_power, p_in, ls='none', fillstyle='none', marker='o', c='k', mew=1.5)

    axes[1].set_xlabel("$P_{\mathrm{laser}}$")
    axes[1].set_ylabel("$P_{\mathrm{in}}$")

    fit_txt = '$P_{\\mathrm{in}} = a_0 + a_1 P_{\\mathrm{laser}}$\n'
    fit_txt += f"$a_0 = {popt[0]:.1f}$ 95% CI: [{ci[0,0]:.1f}, {ci[0, 1]:.1f}]\n"
    fit_txt += f"$a_1 = {popt[1]:.3f}$ 95% CI: [{ci[1, 0]:.4f}, {ci[1, 1]:.4f}]"

    axes[1].text(
        0.95, 0.05, fit_txt, ha='right', va='bottom',
        transform=axes[1].transAxes,
        fontsize=10
    )

    plt.show()


if __name__ == '__main__':
    load_plot_style()
    main()
