import numpy as np
import matplotlib.pylab as plt
import os
from matplotlib import ticker, gridspec
import matplotlib as mpl
import json

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from heat_flux_adi import gaussian_beam

output_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\manuscripts\paper1\images'

p0 = 47
beam_diameter = 0.8164 * 1.5  # cm
sample_diameter = 0.918
sample_diameter_err = 2.0 * 0.003  # 2 standard deviations


def get_r(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def get_q(x, y):
    return gaussian_beam(r=get_r(x, y), beam_diameter=beam_diameter, beam_power=p0)


def main():
    plot_r = 1.8
    n_points = 501
    x = np.linspace(-plot_r, plot_r, num=n_points)
    y = np.linspace(-plot_r, plot_r, num=n_points)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]

    q = get_q(X, Y)
    q_min = gaussian_beam(r=get_r(x.max(), y.max()), beam_diameter=beam_diameter, beam_power=p0)
    q_max = 2.0 * p0 / (np.pi * (0.5 * beam_diameter) ** 2.0)

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots()
    fig.set_size_inches(3.25, 2.5)

    cs = ax.pcolormesh(
        x, y, q, cmap=plt.cm.jet, vmin=q_min, vmax=q_max,
        shading='gouraud', rasterized=True
    )

    ax.set_aspect('equal')
    # ax.set_xlabel('x (cm)')
    # ax.set_ylabel('y (cm)')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.tick_params(
        labelleft=False, labelbottom=False, color='k', width=1.0, left=False, right=False, bottom=False, top=False
    )
    # ax.tick_params(axis='both', which='both', direction='out')
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6.5%", pad="5%")
    cbar = fig.colorbar(cs, cax=cax)
    cbar.ax.set_ylabel('Power density (MW/m$^{\mathregular{2}}$)')
    # cbar.ax.tick_params(axis='y', right=False, labelright=False)
    cbar.ax.set_ylim(q_min, q_max)
    cbar.formatter.set_powerlimits((-3, 3))
    cbar.formatter.useMathText = True
    cbar.update_ticks()

    cax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    cax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    circle1 = plt.Circle((0, 0), 0.5 * sample_diameter, ec='w', fill=False, clip_on=False, ls=(0, (3, 1)))
    ax.add_patch(circle1)
    p_arrow = 0.5 * sample_diameter / np.sqrt(2)

    ax.annotate(
        f"Sample diameter\nØ = {sample_diameter:.3f}±{sample_diameter_err:.3f} cm",
        xy=(-p_arrow, p_arrow), xycoords="data",
        xytext=(0.05, 0.85), textcoords=ax.transAxes,
        color='w', fontsize=10,
        arrowprops=dict(
            arrowstyle="->", color="w",
            shrinkA=5, shrinkB=5,
            # patchA=None, patchB=None,
            connectionstyle='arc,angleA=-90,angleB=0,armA=20,armB=0,rad=0',

        )
    )

    # ax.set_title('Beam profile')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'beam_profile.svg'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
