import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate as integrate
import json
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\heat equation\fourier_left_flux'

N = 100  # The number of Fourier coefficients
alpha = 1E-1
length = 3.5
inital_temperature = 25.0
x_points = 100
t_max = 10.0
t_points = 100
F1 = 1E3

def u_1(x:np.ndarray, diffusion_time: np.ndarray, diffusivity: float, flux:float):
    L = x.max()
    F = flux
    return 0.5 * F * (x**2.0) / L - (F * x) + diffusivity * (flux/L) * diffusion_time

def u_t(x: np.ndarray, diffusion_time: np.ndarray, diffusivity: float, flux: float, T0: float):
    L = x.max()
    u = T0*np.ones((diffusion_time.size, x.size))
    a = np.pi / L
    for i, t in zip(range(1, diffusion_time.size), diffusion_time[1:]):
        s0 = flux * L / 3.0
        s = 0
        for n in range(1, N+1):
            an = 1.0 / (n * n)
            s += an * np.cos(n * a * x) * np.exp(-diffusivity * t * (n * a) ** 2.0)
        u[i] += s0 - 2.0*flux*L*s/np.pi/np.pi + u_1(x, t, diffusivity, flux)
        print(f'T[0,t={t:>6.3f}s] = {u[i,0]:>6.3E}, T[L,t={t:>6.3f}s] = {u[i,-1]:>6.3E}')
    return u


if __name__ == '__main__':
    x = np.linspace(0.0, length, num=x_points)
    t = np.linspace(0.0, t_max, num=t_points)
    u = u_t(x=x, diffusion_time=t, diffusivity=alpha, flux=F1, T0=inital_temperature)

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    # fig, (ax, cbar_ax) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [5, 1]})  # , constrained_layout=True)

    fig, ax = plt.subplots(ncols=1)  # , constrained_layout=True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.set_size_inches(5.0, 3.5)

    norm = mpl.colors.Normalize(vmin=0.0, vmax=t_max)
    cmap = plt.cm.jet_r

    u_ds = u[0::1]
    t_ds = t[0::1]
    for ui, ti in zip(u_ds, t_ds):
        ax.plot(x, ui, color=cmap(norm(ti)), lw=1.2)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Time (s)')

    ax.set_xlabel('x (cm)')
    ax.set_ylabel('T (°C)')

    ax.set_xlim(0, length)
    ax.set_ylim(bottom=20)

    fig.tight_layout()

    fig.savefig(os.path.join(base_dir, 'rod_with_left_flux.png'), dpi=600)
    plt.show()
