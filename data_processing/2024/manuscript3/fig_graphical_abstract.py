import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fig_tds20250303 import load_plot_style



def main():
    load_plot_style()
    dretention_df: pd.DataFrame = pd.DataFrame(data={
        'Material': ['Solid Boron', 'Boron aggregate A', 'Boron aggregate B'],
        'Retention (D/m^2)': [7.8E22, 3.2E20, 1.5E20]
    }).sort_values(by=['Retention (D/m^2)'], ascending=True)

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.5, 2.)

    y_pos = np.arange(len(dretention_df))
    ax.barh(y_pos, dretention_df['Retention (D/m^2)'])
    ax.set_xscale('log')
    ax.set_yticks(y_pos, labels=dretention_df['Material'])
    ax.set_xlim(1E20, 1E23)
    ax.set_xlabel(r'{\sffamily Deuterium retention (D/m\textsuperscript{2})}', usetex=True)
    fig.savefig(r'./figures/graphical_abstract_d_retention.svg', dpi=600)

    plt.show()

if __name__ == '__main__':
    main()