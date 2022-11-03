import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import pandas as pd
import os
import json

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\manuscripts\paper1\inl\data_and_script_for_figures\data_and_script_for_figures\bending_test'
data_dir = 'gold'

def load_plt_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

if __name__ == '__main__':
    load_plt_style()
    s1 = pd.read_csv(os.path.join(base_path, data_dir, 'R3N61_6_processed.csv'))
    s2 = pd.read_csv(os.path.join(base_path, data_dir, 'R3N62_2_processed.csv'))
    s3 = pd.read_csv(os.path.join(base_path, data_dir, 'R3N63-1_processed.csv'))

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(4.0, 3.0)

    ax.plot(s1['disp'], s1['load'], 'k:o', lw=1.25,
             ms=8, mew=1.25, mfc='None', label='Experiment 1')
    ax.plot(s2['disp'], s2['load'], 'g:s', lw=1.25,
             ms=8, mew=1.25, mfc='None', label='Experiment 2')
    ax.plot(s3['disp'], s3['load'], 'b:^', lw=1.25,
             ms=8, mew=1.25, mfc='None', label='Experiment 3')
    beam3d = pd.read_csv(os.path.join(base_path, 'beam3d_out_oct5.csv'))

    ax.plot(-1e3*beam3d['deformation'],
             beam3d['load'], 'r-', lw=3, label='Simulation (MOOSE)')
    ax.set_xlabel('Deformation (mm)')
    ax.set_ylabel('Load (N)')
    ax.set_xlim(-0.1, 0.4)
    ax.set_ylim(0, 3)
    ax.legend(
        prop={'size':9}
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    # ax.text(
    #     -0.1, 1.15, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold',
    #     va='top', ha='right'
    # )

    fig.savefig(os.path.join(base_path, 'load_deformation.pdf'), dpi=600, format='svg')
    fig.savefig(os.path.join(base_path, 'load_deformation.svg'), dpi=600)
    fig.savefig(os.path.join(base_path, 'load_deformation.png'), dpi=600)
    fig.savefig(os.path.join(base_path, 'load_deformation.eps'), dpi=600, format='eps')
    plt.show()
