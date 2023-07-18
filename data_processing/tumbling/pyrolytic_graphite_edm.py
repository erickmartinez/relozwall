import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json


base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\tumbling\pyrolytic_graphite'
csv_list = 'xlsx_list.csv'

# def load_data():
#     df = pd.read_excel(os.path.join(base_dir, data_xls), sheet_name='averages').apply(pd.to_numeric)
#     return df

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

def main():
    files_df = pd.read_csv(os.path.join(base_dir, csv_list))
    files_df['Yield (%)'] = files_df['Yield (%)'].apply(pd.to_numeric)
    print(files_df)
    n = len(files_df)
    tumbling_averages = np.empty(n, dtype=np.dtype([
        ('Tumbling time (d)', 'd'),
        ('Mean area (mm)', 'd'), ('Area std (mm)', 'd'),
        ('Mean perimeter (mm)', 'd'), ('Perimeter std (mm)', 'd'),
        ('Mean size (mm)', 'd'), ('Size std (mm)', 'd'),
        ('Mean circularity', 'd'), ('Circularity std', 'd'),
        ('Tumbling yield (%)', 'd')
    ]))
    for i, r in files_df.iterrows():
        tumbling_time = int(r['Tumble time (d)'])
        df = pd.read_excel(os.path.join(base_dir, r['particle size xlsx']), skiprows=18, usecols=['Area', 'Length'])
        df = df.dropna()
        print(df)
        area = df['Area'].values
        perimeter = df['Length'].values
        circularity = 4. * np.pi * area / (perimeter ** 2.)
        particle_size = 2. * np.sqrt(area / np.pi)
        print(f'{tumbling_time:>3d}, {area.mean():>5.3f}')
        tumbling_averages[i] = (
            tumbling_time,
            area.mean(), np.std(area, ddof=1),
            perimeter.mean(), np.std(perimeter, ddof=1),
            particle_size.mean(), np.std(particle_size, ddof=1),
            circularity.mean(), np.std(circularity, ddof=1),
            r['Yield (%)']
        )

    tumbling_df = pd.DataFrame(data=tumbling_averages)
    print(tumbling_df)

    load_plot_style()

    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 6.5)

    axes[0].errorbar(
        tumbling_df['Tumbling time (d)'], tumbling_df['Mean size (mm)'],
        yerr=2.*tumbling_df['Size std (mm)'].values,
        ls='none',
        color='C0', marker='o', ms=8, fillstyle='none',
        capsize=2.5, mew=1.25, elinewidth=1.25,
    )

    axes[1].errorbar(
        tumbling_df['Tumbling time (d)'], tumbling_df['Mean circularity'],
        yerr=2. * tumbling_df['Circularity std'],
        ls='none',
        color='C1', marker='s', ms=8, fillstyle='none',
        capsize=2.5, mew=1.25, elinewidth=1.25,
    )

    axes[2].plot(
        tumbling_df['Tumbling time (d)'], tumbling_df['Tumbling yield (%)'],
        ls='none',
        color='C2', marker='^', ms=8, fillstyle='none',
        mew=1.25
    )

    axes[1].axhline(y=np.pi/4, ls='--', c='tab:grey', lw=1.)
    axes[1].text(
        3., np.pi/4, 'squared shape', fontsize=10, color='tab:grey', ha='left', va='bottom'
    )

    for ax in axes:
        ax.set_xlabel('Tumbiling time [days]')

    axes[0].set_ylabel('Particle size [mm]')
    axes[1].set_ylabel('Circularity')
    axes[2].set_ylabel('Tumbling yield (%)')
    axes[0].set_title('Pyrolytic graphite')

    fig.savefig(os.path.join(base_dir, 'tumbling_progress.png'), dpi=600)

    tumbling_df.to_csv(os.path.join(base_dir, 'particle_size_averages.csv'), index=False)

    plt.show()


if __name__ == '__main__':
    main()