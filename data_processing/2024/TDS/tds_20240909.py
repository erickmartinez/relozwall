import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import os
import json
import numpy as np
from scipy.optimize import least_squares, OptimizeResult


over_sqrt_pi = 1. / np.sqrt(2. * np.pi)


def gaussian(x, c, sigma, mu):
    global over_sqrt_pi
    p = c * over_sqrt_pi / sigma
    arg = 0.5 * np.power((x - mu) / sigma, 2)
    return p * np.exp(-arg)

def sum_gaussians(x, b):
    global over_sqrt_pi
    m = len(x)
    nn = len(b)
    # Assume that the new b contains a list ordered like
    # (c1, sigma_1, mu1, c_2, sigma_2, mu_2, ..., c_n, sigma_n, mu_n)
    selector = np.arange(0, nn) % 3
    msk_c = selector == 0
    msk_sigma = selector == 1
    msk_mu = selector == 2
    c = b[msk_c]
    sigmas = b[msk_sigma]
    mus = b[msk_mu]
    n = len(mus)
    u = over_sqrt_pi * np.power(sigmas, -1) * c
    u = u.reshape((1, len(u)))
    v = np.zeros((n, m), dtype=np.float64)
    for i in range(len(sigmas)):
        arg = 0.5*np.power((x-mus[i])/sigmas[i], 2.)
        v[i, :] = np.exp(-arg)
    res = np.dot(u, v)
    return res.flatten()

def res_sum_gauss(b, x, y):
    return sum_gaussians(x, b) - y

def jac_sum_gauss(b, x, y):
    m, nn  = len(x), len(b)
    # Assume that the new b contains a list ordered like
    # (c1, sigma_1, mu1, c_2, sigma_2, mu_2, ..., c_n, sigma_n, mu_n)
    selector = np.arange(0, nn) % 3
    msk_c = selector == 0
    msk_sigma = selector == 1
    msk_mu = selector == 2
    c = b[msk_c]
    s = b[msk_sigma]
    mu = b[msk_mu]
    r = np.zeros((m, nn), dtype=np.float64)
    for i in range(len(s)):
        k = 3 * i
        g = gaussian(x, c[i], s[i], mu[i])
        r[:, k] = g / c[i]
        r[:, k+1] = np.power(s[i], -1) * ( np.power( (x - mu[i]) / s[i], 2) - 1.) * g
        r[:, k+2] = np.power(s[i], -2) * ( x - mu[i]) * g

    return r


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    rcParams.update(plot_style)

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

    load_plot_style()
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 5.)

    axes[0].plot(temp_r_k, dh_r, c='C0', ls='-', label='HD')
    axes[0].plot(temp_r_k, d2_r, c='C1', ls='-', label='D$_{\mathregular{2}}$')
    axes[0].plot(temp_r_k, d_r_total, c='k', ls='-', label='2D$_{\mathregular{2}}$ + HD')
    axes[0].set_title('Boron rod')

    axes[1].plot(temp_p_k, dh_p, c='C0', ls='-', label='HD')
    axes[1].plot(temp_p_k, d2_p, c='C1', ls='-', label='D$_{\mathregular{2}}$')
    axes[1].plot(temp_p_k, d_p_total, c='k', ls='-', label='2D$_{\mathregular{2}}$ + HD')
    axes[1].set_title('Boron pebble rod')

    for ax in axes:
        ax.set_xlim(200, 1200)
        ax.ticklabel_format(useMathText=True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
        ax.legend(loc='upper left')

    axes[-1].set_xlabel('T (K)')
    fig.supylabel('Desorption flux (m$^{\mathregular{-2}}$ s$^{\mathregular{-1}}$)')

    fig.savefig(os.path.join('./figures', '20240909_TDS_boron_pebble_vs_boron_rod.png'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()



