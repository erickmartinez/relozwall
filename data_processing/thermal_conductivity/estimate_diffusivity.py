import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
from data_processing.utils import specific_heat_of_graphite

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\thermal_conductivity\graphite\laser_flash'
data_csv = 'LCT_GRAPHITE_TEST_100PCT_2023-04-13_9.csv'
sample_length_cm = 5.
density_g = 1.698
specific_heat_g = 0.6752  # Markelov, Volga, et al., 1973
reflectance = 40.4
time_constant = 1.681 # 0.256  # 2.1148
time_constant = 0.256  # 2.1148


PI2 = np.pi ** 2.


def load_dimensionless_params():
    df = pd.read_csv('dimensionless_parameters.csv').apply(pd.to_numeric)
    return df

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

def correct_thermocouple_response(measured_temperature, measured_time, tau):
    n = len(measured_time)
    k = int(n / 15)
    k = k + 1 if k % 2 == 0 else k
    k = max(k, 5)
    # T = savgol_filter(measured_temperature, k, 3)
    # dTdt = np.gradient(T, measured_time, edge_order=2)
    delta = measured_time[1] - measured_time[0]
    dTdt = savgol_filter(x=measured_temperature, window_length=k, polyorder=4, deriv=1, delta=delta)
    # dTdt = savgol_filter(dTdt, k - 2, 3)
    r = measured_temperature + tau * dTdt
    return savgol_filter(r, k - 4, 3)


def get_v(ww):
    s = 0
    for i in range(1, 502):
        sign = -1.0 if i % 2 == 1.0 else 1.0
        n2 = i * i
        s += sign * np.exp(-n2 * ww)
    r = 1.0 + 2.0 * s
    return r


def main():
    L2 = sample_length_cm ** 2.

    params_df = load_dimensionless_params()
    file_tag = os.path.splitext(data_csv)[0]
    data_df = pd.read_csv(os.path.join(base_dir, data_csv), comment='#').apply(pd.to_numeric)
    time_s = data_df['Measurement Time (s)'].values
    temperature = data_df['TC2 (C)'].values
    laser_peak_power = data_df['Laser output peak power (W)'].values

    specific_heat_g = specific_heat_of_graphite(temperature=temperature[0], units='C')

    msk = laser_peak_power > 0
    t_msk = time_s[msk]
    t0 = t_msk[0]
    msk_init = time_s >= t0
    time_s = time_s[msk_init] - t0
    temperature = temperature[msk_init]

    dT = temperature - temperature[0]
    # dT = correct_thermocouple_response(measured_temperature=dT, measured_time=time_s, tau=time_constant)

    v_exp = dT / dT.max()
    laser_power = laser_peak_power[msk].mean()

    dt = time_s[1] - time_s[0]
    t_model = np.linspace(time_s.min()+dt, time_s.max(), 1000)
    f = interp1d(v_exp, time_s)
    g = interp1d(time_s, v_exp, bounds_error=False, fill_value='extrapolate')

    th = f(0.5)
    ah = 0.138785 * L2 / th
    # w = PI2 * ah * time_s / L2
    w = 1.369756 * t_model / th
    v_model = get_v(w)

    t5 = 5. * th
    t10 = 10. * th
    v5 = g(t5)
    v10 = g(t10)
    vh = g(th)

    r5 = v5 / vh
    r10 = v10 / vh
    kk = np.array([
        [-0.1037162, 1.239040, -3.974433, 6.888738, -6.804883, 3.856663, -1.167799, 0.1465332],
        [0.054825246, 0.16697761, -0.28603437, 0.28356337, -0.13403286, 0.024077586, 0., 0.]
    ]).T

    rt = np.array([v10 ** i for i in range(8)])
    kc = np.dot(kk[:, 0], rt)
    a_corrected = ah * kc / 0.13885

    thermal_conductivity = a_corrected * density_g * specific_heat_g
    R_sample = 0.5 * 1.288
    area = np.pi * R_sample ** 2.
    aperture_factor = 1. - np.exp(-2. * (R_sample / (0.5*0.8164)) ** 2.)
    Q = sample_length_cm * area * dT.max() * density_g * specific_heat_g / aperture_factor / 0.4 /0.8 / 0.5

    n = len(params_df)
    alpha_list = np.empty(n, dtype=float)
    tx_list = np.empty(n, dtype=float)

    for i, r in params_df.iterrows():
        vx = 0.01 * r['V (%)']
        tx = f(vx)
        alpha_list[i] = r['k(V)'] * L2 / tx
        tx_list[i] = tx


    out_df = pd.DataFrame(data={
        'V(%)': params_df['V (%)'].values,
        'tx (s)': tx_list,
        'alpha (cm^2/s)': alpha_list
    })

    print(out_df)

    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.0)
    # ax.plot(time_s, v_exp, c='C0', marker='o', ls='none', fillstyle='none', label='Experiment')
    ax.plot(time_s/th, v_exp, c='C0', marker='o', ls='none', fillstyle='none', label='Experiment')
    ax.plot(t_model / th, v_model, c='k', fillstyle='none', label='Model')

    ax.set_xlabel('$t/t_{1/2}$')
    ax.set_ylabel('$\Delta T/ \Delta T_{\mathrm{max}}}$')
    ax.legend(
        loc='upper left', frameon=True
    )

    ax.set_xlim(0., 4.)
    ax.set_ylim(0., 1.)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    alpha_txt = f'$\\alpha_{{0.5}} = ${ah:.3f}±{0.05*ah:0.3f} cm$^{{\\mathregular{{2}}}}$/s\n'
    alpha_txt += f'$\\alpha_{{\mathrm{{corrected}}}} = ${a_corrected:.3f}±{0.05*a_corrected:.3f} cm$^{{\\mathregular{{2}}}}$/s\n\n'
    alpha_txt += f'$K$ = {thermal_conductivity:.3f}±{thermal_conductivity*0.05:.3f} W/cm-K\n'
    alpha_txt += f'$Q$ = {Q:.3f} J/cm$^{{\mathregular{{2}}}}$'

    ax.text(
        0.95, 0.05, alpha_txt,  ha='right', va='bottom',
        transform=ax.transAxes,
        fontsize=11
    )

    plt.show()


if __name__ == '__main__':
    main()




