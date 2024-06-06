import numpy as np
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats.distributions import t
import json
from scipy.integrate import simpson

lot_no = 'LOT-N07F024'


def main():
    df = pd.read_excel('GC_SIZES.xlsx', sheet_name=lot_no).apply(pd.to_numeric)
    diameter_um = df['S (um)'].values
    n = len(diameter_um)
    confidence = 0.95
    alpha = 1. - confidence
    tval = t.ppf(1 - 0.5 * alpha, n - 1)
    mu = np.mean(diameter_um)
    sigma = np.std(diameter_um, ddof=1)
    delta = sigma * tval / np.sqrt(n)
    lpb, upb = mu - delta, mu + delta

    print(f'Number of measurements: {n}')
    print(f'Mean pebble size (um): {mu:.0f} ± {delta:.0f}')
    print(f'Standard deviation (um): {sigma:.0f}')
    print(f'95% confidence interval (µm): [{lpb:.0f}, {upb:.0f}]')


    x = np.linspace(500, 1500, num=1000)
    # add a 'best fit' line
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (x - mu)) ** 2))

    n_bins = 30

    with open('../../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 3.)

    # the histogram of the data
    nn, bins, patches = ax.hist(diameter_um, n_bins, density=True)
    # dx = bins[1] - bins[0]
    # bx = bins[1::]
    # idx_95 = (bx >= lpb) & (bx <= upb)
    # print(bx[idx_95])
    # ax.bar(bx[idx_95], nn[idx_95], width=dx, color='r')

    xx = np.linspace(lpb, upb, 1000)
    yy = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (xx - mu)) ** 2))
    integral_part = simpson(y=yy, x=xx)
    print(f'Percentage within 95% CI: {integral_part * 100:.3f}')

    ax.plot(x, y, '--')
    ax.set_xlabel(r'Pebble size ($\mathregular{\mu}$m)')
    ax.set_ylabel('Probability density')
    lot_txt = lot_no.replace('-',':')
    ax.set_title(f'Pebble sizes ({lot_txt})')

    results_txt = r'$\bar{D} = ' + f'{mu:.0f}' + r'\pm ' + f'{delta:.0f}$' + ' µm\n'
    results_txt += f'Sample size: {n}'
    ax.text(
        0.95, 0.95, results_txt,
        transform=ax.transAxes,
        fontsize=12,
        va='top', ha='right'
    )

    # Tweak spacing to prevent clipping of ylabel
    # fig.tight_layout()
    fig.savefig(lot_no + '.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()