import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import json
import os

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\tumbling'
files = [
    'GLASSY_CARBON_TYPE_1_REG1_results.csv',
    'GLASSY_CARBON_TYPE_1_REG2_results.csv',
    'GLASSY_CARBON_TYPE_1_REG3_results.csv',
    'GLASSY_CARBON_TYPE_1_REG4_results.csv',
    'GLASSY_CARBON_TYPE_1_REG5_results.csv'
]
num_bins = 20


def prepare_data():
    df = pd.DataFrame(columns=['Major', 'Minor'])
    for i, f in enumerate(files):
        df = df.append(
            pd.read_csv(os.path.join(base_path, f), usecols=['Major', 'Minor']).apply(pd.to_numeric),
            ignore_index=True
        )
    df['Diameter (mm)'] = 0.5 * (df['Major'] + df['Minor'])
    return df['Diameter (mm)'].values


def estimate_interstitial(main_radius:float):
    r_tetrahedral = ((np.sqrt(6.0)/2.0) - 1.0) * main_radius
    r_octahedral = (np.sqrt(2.0) - 1.0) * main_radius
    return {
        'r_tetrahedral': r_tetrahedral,
        'r_octahedral': r_octahedral
    }


if __name__ == '__main__':
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)
    fig, ax = plt.subplots(constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)
    diameters = prepare_data()
    sigma = diameters.std()
    mu = diameters.mean()
    n, bins, patches = ax.hist(diameters, num_bins, density=True)
    new_bins = np.linspace(min(bins), max(bins), 100)
    normal_distribution = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (new_bins - mu))**2))
    ax.plot(new_bins, normal_distribution, '--', lw=1.5, color='tab:red')
    ax.set_xlabel('Diameter (mm)')
    ax.set_ylabel('Probability density')
    ax.set_title(rf'Glassy carbon $\mu={mu:.2f}$ mm (${mu/25.4:.3f}$"), $\sigma={sigma:.2f}$ mm', fontweight='regular')
    r_interstitial = estimate_interstitial(0.5*mu)
    interstitial_txt = f"$d_{{\mathrm{{tetra}}}} = {2.0*r_interstitial['r_tetrahedral']:.2f}$ mm (${2.0*r_interstitial['r_tetrahedral']/25.4:.4f}$\")\n"
    interstitial_txt += f"$d_{{\mathrm{{octa}}}} = {2.0*r_interstitial['r_octahedral']:.2f}$ mm (${2.0*r_interstitial['r_octahedral']/25.4:.4f}$\")"
    ax.text(
        0.015, 0.95, interstitial_txt,
        transform=ax.transAxes,
        fontsize=10,
        va='top', ha='left'
    )
    fig.savefig('../data/gc_spheres_diameters.png', dpi=600)
    plt.show()

