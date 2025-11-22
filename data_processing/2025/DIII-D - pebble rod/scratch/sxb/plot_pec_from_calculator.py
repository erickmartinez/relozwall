import numpy as np
from data_processing.misc_utils.plot_style import load_plot_style
import matplotlib.pyplot as plt

from boron_sxb_calculator_v1 import BoronCRModelImproved

def main():
    model = BoronCRModelImproved()
    te_range = np.logspace(np.log10(2), np.log10(100), 100)
    ne = 1E18
    pec = model.photon_emissivity_coefficient_improved(te_range, ne)

    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4, 3)

    ax.plot(te_range, pec)

    ax.set_xlabel("$T_e$")
    ax.set_ylabel("PEC")
    ax.set_yscale("log")
    ax.set_xscale("log")

    plt.show()

if __name__ == "__main__":
    main()

