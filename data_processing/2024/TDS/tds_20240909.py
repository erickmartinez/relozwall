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
    msk_ymax = selector == 0
    msk_sigma = selector == 1
    msk_mu = selector == 2
    ymaxs = b[msk_ymax]
    sigmas = b[msk_sigma]
    mus = b[msk_mu]
    n = len(mus)
    u = over_sqrt_pi * np.power(sigmas, -1) * ymaxs
    u = u.reshape((1, len(u)))
    v = np.zeros((n, m), dtype=np.float64)
    for i in range(len(sigmas)):
        arg = 0.5*np.power((x-mus[i])/sigmas[i], 2.)
        v[i, :] = np.exp(-arg)
    # print('y.shape', y.shape, 'u.shape', u.shape, 'v.shape', v.shape)
    res = np.dot(u, v)
    return res.flatten()

def main():
    x = np.linspace(0, 10, num=200)
    beta = np.array([2., 0.5, 2., 3., 2.5, 5, 1.5, 2.5, 8.])
    y_sum = sum_gaussians(x, beta)
    y1 = gaussian(x, beta[0], beta[1], beta[2])
    y2 = gaussian(x, beta[3], beta[4], beta[5])
    y3 = gaussian(x, beta[6], beta[7], beta[8])
    print('y_sum.shape', y_sum.shape)

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3.)

    ax.plot(x, y1, c='C0', ls='-')
    ax.plot(x, y2, c='C1', ls='-')
    ax.plot(x, y3, c='C2', ls='-')
    ax.plot(x, y_sum, c='k', ls='-')

    plt.show()


if __name__ == '__main__':
    main()



