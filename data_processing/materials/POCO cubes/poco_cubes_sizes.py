import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from scipy.stats.distributions import t
from data_processing.utils import lighten_color
import json

data_str = "1.66,2.62,1.53,1.58,1.72,1.70,1.63,1.64,1.88,1.74,1.71,1.73,1.69,1.67,2.78,1.81,1.83,1.59,1.18,1.66,2.62," \
           "2.23,1.66,1.65,2.71,1.94,1.87,1.49,1.74,1.76,1.56,1.68,1.59,1.67,3.74,2.25,1.68,1.69,2.21,1.95,1.66,1.96," \
           "1.73,1.53,1.80,2.33,1.68,1.63,1.32,1.73,3.15,1.64,1.68,1.83,2.09,1.71,1.53,1.62,1.89,1.81,1.72,2.11,1.61," \
           "2.57,1.75,1.61,2.58,1.70,1.60,1.58,1.60,1.59,1.82,1.41,1.55,1.74,1.94,1.74,1.83,1.41,1.80,1.91,1.78,1.53," \
           "1.05,1.66,1.71,1.60,2.44,1.69,1.71,1.47,1.69,1.81,1.90,1.73,1.61,2.14,1.68,1.56,1.70,2.44,1.53,1.42,1.74," \
           "1.55,1.56,1.64,2.35,1.36,1.87,1.75,1.60,2.17,1.54,1.54,1.84,1.89,1.75,2.22,1.56,1.55,1.72,2.64,1.03,1.72," \
           "1.70,1.78,1.59,3.07,2.44,0.84,1.83,1.69,1.52,1.76,1.71,1.71,1.66,2.13,0.90,1.60,1.72,1.32,1.87,1.32,1.67," \
           "1.66,1.80,1.72,1.82,2.03,1.75,1.82,1.35,1.68,1.68,2.16,1.70,1.79,1.70,1.76,1.73,1.76,1.62,1.55,1.44,1.71," \
           "1.78,1.74,1.77,1.64,1.71,1.68,1.74,1.68,1.68,1.73,1.79,2.06,2.03,1.71,1.34,1.32,1.59,1.67,1.84,1.63,1.76," \
           "1.72,1.57,1.67,1.74,1.76,1.66,1.51,1.75,1.54,1.57,1.64,1.71,1.66,1.83,1.67,1.68,2.32,1.70,1.45,1.77,1.95," \
           "1.68,1.66,1.67,1.66,2.40,1.67,1.61,1.84,1.85,1.42,1.70,1.44,1.63,1.85,1.27,1.62,1.69,1.62,1.85,2.61,1.84," \
           "1.53,1.98,1.86,2.14,1.74,1.44,1.49,1.64,1.49,1.77,1.91,1.16,1.73,1.64,1.69,1.75,1.76,1.75,1.61,1.67,1.68," \
           "1.65,1.35,0.73,1.29,0.88,1.95,1.69,1.72,1.35,1.33,1.32,2.45,1.81,1.75,1.81,1.18,1.19,1.44,1.74,1.78,1.81," \
           "1.97,1.65,1.01,1.71,1.33,1.26,0.97,1.71,2.59,1.69,1.64,1.69,1.53,2.06,1.06,1.66,1.50,1.96,1.78,1.58,1.62," \
           "2.09,1.37,1.86,2.69,1.68,1.67,1.77,1.73,1.73"

seed = 1024

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
        mpl.rcParams.update(plot_style)

def main():
    data = np.array(data_str.split(","), dtype=float)
    random.seed(2024)
    random.shuffle(data)
    cube_stats = np.empty(len(data), dtype=np.dtype([('n', 'i'), ('mean', 'd'), ('std', 'd'), ('e', 'd'), ('tval', 'd')]))
    for n in range(len(cube_stats)):
        if n < 2:
            cube_stats[n] = (n+1, data[0], np.inf, np.inf, np.inf)
            continue
        x = np.array(data[0:n])
        s = np.std(x, ddof=1)
        xm = x.mean()
        tval = t.ppf(1.-0.05/2., n-1)
        cube_stats[n] = (n+1, xm, s, s*tval/xm/np.sqrt(n), tval)

        load_plot_style()

    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
    fig.set_size_inches(3.5, 5.5)

    dy = cube_stats["std"]*cube_stats["tval"]/np.sqrt(cube_stats["n"])
    axes[0].fill_between(cube_stats["n"], cube_stats["mean"]-dy, cube_stats["mean"]+dy, color=lighten_color('C0', 0.5))
    axes[0].plot(cube_stats["n"], cube_stats["mean"], lw=1.5)
    axes[1].plot(cube_stats["n"], cube_stats["std"], lw=1.5)
    axes[2].plot(cube_stats["n"], cube_stats["e"], lw=1.5)

    for i, ax in enumerate(axes):
        if i == 2:
            ax.set_xlabel("$n$")
        ax.set_xlim(left=0, right=300)
        ax.ticklabel_format(useMathText=True)

    axes[0].set_ylabel(r"$\bar{x}$ (mm)")
    axes[1].set_ylabel(r"$\sigma$ (mm)")
    axes[2].set_ylabel(r"$e$")

    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    axes[2].yaxis.set_major_locator(ticker.MultipleLocator(1.))

    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    axes[2].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    xfmt = ticker.ScalarFormatter(useOffset=True, useMathText=True, useLocale=True)
    xfmt.set_powerlimits((-2, 2))
    # axes[2].yaxis.set_major_formatter(xfmt)
    axes[2].set_yscale('log')

    axes[0].set_ylim(bottom=1.60, top=2.2)
    axes[1].set_ylim(bottom=0.3, top=0.7)
    axes[2].set_ylim(bottom=0.01, top=10.)

    s = np.std(data, ddof=1)
    xm = data.mean()
    tval = t.ppf(1. - 0.05 / 2., len(data) - 1)
    ci = tval*s/np.sqrt(len(data))

    axes[0].set_title(fr"$\bar{{x}}$ = {xm:.2f}±{ci:.2f} (mm)", fontweight='regular')
    # axes[0].set_title(fr"$\bar{{x}}$ = {xm:.1f}±{ci:.1f} (mm)", fontweight='regular')
    axes[1].set_title("Standard deviation")
    axes[2].set_title("Maximum relative error")

    fig.savefig('poco_graphite_batch_12_size.png', dpi=600)

    plt.show()

if __name__ == "__main__":
    main()