import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
import os
from scipy.interpolate import interp1d
from data_processing.utils import latex_float
import mpmath as mp
from scipy.integrate import simps


base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\thermal_expansion'
model_coefficients_csv = 'Tsang - Carbon 2005 - thermal_expansion_model.csv'
target_temperature_k = 300.0  # K

R = 8.31446261815324
molar_mass_graphite = 12.011
mgr = R / molar_mass_graphite  # J / g-K
temperature_range_c = (-200, 2800)
mp.dps = 15;
mp.pretty = True


def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['xtick.top'] = True


def load_model_coefficients():
    df = pd.read_csv(os.path.join(base_dir, model_coefficients_csv))
    columns = df.columns
    df[columns[1:]] = df[columns[1:]].apply(pd.to_numeric)
    return df


def cte_graphite(temperature, debye_temperature, coefficients, units='K'):
    if units == 'C':
        temperature += 273.15
    debye_temperature = np.array(debye_temperature)
    coefficients = np.array(coefficients)
    Cv = get_cv(debye_temperature, temperature)
    x = np.append(Cv, temperature)
    r = np.dot(coefficients, x)
    # print('r', r)
    return r


np_poly_log = np.frompyfunc(mp.polylog, 2, 1)
np_log = np.frompyfunc(mp.log, 1, 1)
np_exp = np.frompyfunc(mp.exp, 1, 1)


def evaluate_integral(x):
    r = (12 * x ** 2.) * np_poly_log(2., np.exp(x))
    r -= (24. * x) * np_poly_log(3, np.exp(x))
    r += 24. * np_poly_log(4, np.exp(x))
    r -= np.array([(v ** 4.) / (np_exp(v) - 1.) if v != 0 else 0. for v in x])
    r -= x ** 4.
    r += np.array([(4. * v ** 3.) * np_log(1. - np_exp(v)) if v != 0 else 0. for v in x])
    return np.array([float(v.real) for v in r])


def integrand(x):
    # r = np.exp(x) * (x ** 4.0)
    # d = np.power(np.exp(x) -
    v = np.array([
        np.exp(v) * (v ** 4.0) / np.power(np.exp(v) - 1., 2.) if v != 0. else 0. for v in x
    ])
    return v


def simps_integral(x):
    r = np.empty_like(x)
    for i, xi in enumerate(x):
        t = np.linspace(0, xi, 1000)
        y = integrand(t)
        r[i] = simps(y, t)
    return r


i0 = 24. * (np.pi ** 4.) / 90.


def get_d(x):
    integral = evaluate_integral(x) - i0
    # integral = simps_integral(x)
    return (3. / x ** 3.0) * integral


def get_cv(debye_temp, temperature_k):
    D = get_d(debye_temp / temperature_k)
    return 3.0 * R * D


def main():
    model_df = load_model_coefficients()
    temperatures = np.linspace(temperature_range_c[0], temperature_range_c[1], 200)
    temperatures_k = temperatures + 273.15

    n = len(model_df)
    alpha_t = np.empty(n, dtype=np.dtype([('Reference', '<U6'), ('alpha_a (/K)', 'd'), ('alpha_c (/K)', 'd')]))

    load_plt_style()
    fig, ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 5.0)
    xfmt = ticker.ScalarFormatter(useMathText=True)
    for i, r in model_df.iterrows():
        lbl = r['Reference']
        b = np.array([[r['A'], r['B'], r['C']], [r['L'], r['M'], r['N']]])
        theta = np.array([r['Debye temperature a (K)'], r['Debye temperature c (K)']])
        a_t = cte_graphite(
            temperature=target_temperature_k, debye_temperature=theta, coefficients=b, units='K'
        )
        alpha_t[i] = (lbl, a_t[0], a_t[1])

        # alpha = cte_graphite(
        #     temperature=temperatures,
        #     debye_temperature=theta,
        #     coefficients=b,
        #     units='C'
        # )

        alpha_a = np.empty_like(temperatures)
        alpha_c = np.empty_like(temperatures)
        for j, temp in enumerate(temperatures):
            alpha_a[j], alpha_c[j] = cte_graphite(
                temperature=temp, debye_temperature=theta,
                coefficients=b, units='C'
            )

        ax[0].plot(
            temperatures,
            alpha_a,
            label=lbl
        )

        ax[1].plot(
            temperatures,
            alpha_c,
            label=lbl
        )

    for i, axi in enumerate(ax):
        axi.set_xlabel('Temperature (°C)')
        axi.legend(
            loc='best', frameon=True, fontsize=9
        )
        axi.ticklabel_format(style='sci', axis='y', scilimits=(-2,2), useMathText=True)
        axi.set_xlim(-200, 2800)
        axi.xaxis.set_major_locator(ticker.MultipleLocator(500))
        axi.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        panel_label = chr(ord('`') + i + 1)
        axi.text(
            -0.15, 1.15, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )

    ax[0].set_ylabel('$\\alpha_{\\mathrm{a}}$ (/K)')
    ax[1].set_ylabel('$\\alpha_{\\mathrm{c}}$ (/K)')
    ax[0].set_ylim(-2E-6, 2E-6)
    ax[1].set_ylim(5E-6, 5E-5)
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(1E-6))
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(5E-7))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(1E-5))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(5E-6))

    ax[0].set_title('in-plane')
    ax[1].set_title('cross-plane')

    print(alpha_t)
    fig.suptitle('CTE of graphite (Tsang 2005)')

    txt_a = f"$\\alpha_{{\\mathrm{{a}}}}$ (mean, 300 K) = $\\mathregular{{{latex_float(alpha_t['alpha_a (/K)'].mean())}}}$ (/K)"
    txt_c = f"$\\alpha_{{\\mathrm{{c}}}}$ (mean, 300 K) = $\\mathregular{{{latex_float(alpha_t['alpha_c (/K)'].mean())}}}$ (/K)"
    ax[0].text(
        0.95, 0.05, txt_a, transform=ax[0].transAxes, fontsize=9, fontweight='regular',
        va='bottom', ha='right', color='b'
    )

    ax[1].text(
        0.95, 0.05, txt_c, transform=ax[1].transAxes, fontsize=9, fontweight='regular',
        va='bottom', ha='right', color='b'
    )

    fig.savefig(os.path.join(base_dir, 'tsang2005_graphite_coefficient_of_thermal_expansion.png'), dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
