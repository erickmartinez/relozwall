import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import json
from scipy.interpolate import interp1d

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\Literature\boron'
cte_aa_csv = 'Lundstrom JAC1998 - Thermal expansion of beta-rhombohedral boron_aa.csv'
cte_ac_csv = 'Lundstrom JAC1998 - Thermal expansion of beta-rhombohedral boron_ac.csv'


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


def load_data():
    df_aa = pd.read_csv(os.path.join(base_dir, cte_aa_csv), comment='#').apply(pd.to_numeric)
    df_ac = pd.read_csv(os.path.join(base_dir, cte_ac_csv), comment='#').apply(pd.to_numeric)
    temperature_aa = df_aa['Temperature (K)'].values
    temperature_ac = df_ac['Temperature (K)'].values
    cte_aa = df_aa['CTE (10^{-6}/K)'].values
    cte_ac = df_ac['CTE (10^{-6}/K)'].values

    temperature_k = np.linspace(start=300,stop=1250, num=950)
    faa = interp1d(x=temperature_aa, y=cte_aa, bounds_error=False, fill_value='extrapolate')
    fac = interp1d(x=temperature_ac, y=cte_ac, bounds_error=False, fill_value='extrapolate')

    cte_aa_interp = faa(temperature_k)
    cte_ac_interp = fac(temperature_k)
    cte_am_interp = (2. * cte_aa_interp + cte_ac_interp) / 3.

    df = pd.DataFrame(data={
        'Temperature (K)': temperature_k,
        'alpha_a (10^{-6}/K)': cte_aa_interp,
        'alpha_c (10^{-6}/K)': cte_ac_interp,
        'alpha_m (10^{-6}/K)': cte_am_interp,
    })

    df.to_csv(os.path.join(base_dir, 'Lundstrom JAC1998 - Thermal expansion of beta-rhombohedral boron.csv'), index=False)
    return df

def main():
    df = load_data()
    load_plot_style()
    temperature_k = df['Temperature (K)'].values
    cte_aa = df['alpha_a (10^{-6}/K)'].values
    cte_ac = df['alpha_c (10^{-6}/K)'].values
    cte_am = df['alpha_m (10^{-6}/K)'].values
    # cte_mean = np.mean(np.array([cte_aa, cte_ac]).T, axis=1)
    cte_am_300 = cte_am[0]

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.0)
    ax.plot(temperature_k, cte_aa, color='C0', label='$\\alpha_{a}$')
    ax.plot(temperature_k, cte_ac, color='C1', label='$\\alpha_{c}$')
    ax.plot(temperature_k, cte_am, color='C2', label='$\\alpha_{\mathrm{m}}$')

    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('$\\alpha$ (10$^{\mathregular{-6}}$/K)')

    ax.set_xlim(300,1300)
    ax.set_ylim(0, 10)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))


    ax.legend(
        loc='upper left',
        frameon=True
    )
    ax.set_title('Lundström 1998')

    print(f'CTE (average), T=300K, {cte_am_300:.2f} x 10^{-6}/K')

    fig.savefig(os.path.join(base_dir, 'Lundstrom1998_boron_b-rhombohedral_CTE_mean.png'), dpi=300)

    plt.show()

if __name__ == '__main__':
    main()