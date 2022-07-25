import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate as integrate
import json
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\heat equation\insulated_rod'

N = 100  # The number of Fourier coefficients
alpha = 1E-2
length = 3.5
t01 = 800.0
t02 = 50.0
x_points = 100
t_max = 10.0
t_points = 100


def initial_condition(x: np.ndarray, u_t0x0: float, u_t0xL: float):
    temperature = u_t0xL * np.ones_like(x)
    temperature[0:5] = u_t0x0
    return temperature


def get_an(n: int, x: np.ndarray, u_t0x0: float, u_t0xL: float, f: callable = initial_condition):
    L = x.max()
    if n == 0:
        return integrate.simps(y=f(x, u_t0x0, u_t0xL), x=x) / L
    return 2.0 * integrate.simps(y=f(x, u_t0x0, u_t0xL) * np.cos(n * np.pi * x / L), x=x) / L


def get_u(x: np.ndarray, diffusion_time: np.ndarray, diffusivity: float, u_t0x0: float, u_t0xL: float,
          f: callable = initial_condition):
    L = x.max()
    u = np.zeros((diffusion_time.size, x.size))
    t0 = initial_condition(x=x, u_t0x0=u_t0x0, u_t0xL=u_t0xL)
    u[0]=t0
    for i, ti in enumerate(diffusion_time):
        # print(f'{ti:.3f} s, T(x=0): {u[i, 0]:.3E} °C, T(x=L): {u[i, -1]:.3E} °C')
        for n in range(N+1):
            arg = ((n * np.pi / L) ** 2) * diffusivity * ti
            a_n = get_an(n, x, u_t0x0, u_t0xL, f)
            # print(f'A_{j}: {a_n}')
            if i > 0:
                u[i] =  u[i] + a_n * np.cos(n * np.pi * x / L) * np.exp(-arg)

    return u


if __name__ == '__main__':
    x = np.linspace(0.0, length, num=x_points)
    t = np.linspace(0.0, t_max, num=t_points)
    u = get_u(x=x, diffusion_time=t, diffusivity=alpha, u_t0x0=t01, u_t0xL=t02, f=initial_condition)

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
    ax.set_ylim(bottom=20, top=1000.0)

    fig.tight_layout()

    fig.savefig(os.path.join(base_dir, 'insulated_rod.png'), dpi=600)
    plt.show()
