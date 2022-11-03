import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os

from matplotlib import ticker, patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\heat equation\fourier_left_flux'

N = 100  # The number of Fourier coefficients
alpha = 1E-1
length = 3.5
inital_temperature = 25.0
x_points = 300
t_max = 20.0
t_points = 200
F1 = 1E3
pulse_length = 5.0
x_probe_1 = 0.5
x_probe_2 = length

by_p2 = np.power(np.pi, -2.0)

debug = False


def u_1(x: np.ndarray, diffusion_time: np.ndarray, diffusivity: float, flux: float):
    L = x.max()
    F = flux
    return 0.5 * F * (x ** 2.0) / L - (F * x) + diffusivity * (flux / L) * diffusion_time


def u_laser_on(x: np.ndarray, diffusion_time: np.ndarray, diffusivity: float, flux: float, T0: float):
    L = x.max()
    u = T0 * np.ones((diffusion_time.size, x.size))
    a = np.pi / L
    b = 2.0 * flux * L * by_p2
    for i, t in zip(range(1, diffusion_time.size), diffusion_time[1:]):
        s0 = flux * L / 3.0
        s = 0
        for n in range(1, N + 1):
            an = 1.0 / (n * n)
            s += an * np.cos(n * a * x) * np.exp(-diffusivity * t * (n * a) ** 2.0)
        u[i] += s0 - b * s + u_1(x, t, diffusivity, flux)
        if debug:
            print(f'[on] - ({i:4d}) T[0,t={t:>6.3f}s] = {u[i, 0]:>6.3E}, T[L,t={t:>6.3f}s] = {u[i, -1]:>6.3E}')
    return u


def get_an(n: int, x: np.ndarray, diffusivity: float, emission_time:float, flux: float, T0: float):
    L = x.max()
    if n == 0:
        return T0 + diffusivity*flux*emission_time/L
    return (2.0 * flux * L / ((n * np.pi)**2.0)) * (1.0 - np.exp(-diffusivity*emission_time*(n*np.pi/L)**2.0))


def get_ut(x: np.ndarray, diffusion_time: np.ndarray, diffusivity: float, emission_time: float, flux: float, T0: float):
    L = x.max()
    u = np.zeros((diffusion_time.size, x.size))
    msk_on = diffusion_time <= emission_time
    msk_off = diffusion_time > emission_time
    time_off = diffusion_time[msk_off]
    idx_off = len(u[msk_on])#(np.abs(diffusion_time - emission_time)).argmin() + 1
    u_on = u_laser_on(x=x, diffusion_time=diffusion_time[msk_on], diffusivity=diffusivity, flux=flux, T0=T0)
    u[msk_on, :] = u_on.copy()
    # After the laser pulse consider the solution of the heat equation for a 1D rod with insulated ends
    for i, ti in enumerate(time_off):
        for n in range(N+1):
            arg = ((n * np.pi/L) ** 2.0) * diffusivity * (ti - emission_time) #(diffusion_time[i+idx_off]-pulse_length)
            a_n = get_an(n, x, diffusivity=diffusivity, emission_time=emission_time, flux=flux, T0=T0)
            if n == 0:
                u[i+idx_off] = a_n
            else:
                u[i+idx_off] += a_n * np.cos(n * np.pi * x / L) * np.exp(-arg)
        if debug:
            print(f'[off] - ({i+idx_off:3d}) T[0,t={ti:>6.3f}s] = {u[i + idx_off, 0]:>6.3E}, T[L,t={ti:>6.3f}s] = {u[i + idx_off, -1]:>6.3E}')

    return u


if __name__ == '__main__':
    x = np.linspace(0.0, length, num=x_points)
    t = np.linspace(0.0, t_max, num=t_points)
    u = get_ut(x=x, diffusion_time=t, diffusivity=alpha, emission_time=pulse_length, flux=F1, T0=inital_temperature)
    diffusion_length = 2.0 * np.sqrt(alpha*t.max())

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
    ax.axvline(x=diffusion_length, ls='--', color='tab:grey', lw=1.25)

    ax.set_xlim(0, length)
    ax.set_ylim(bottom=20)

    fig.tight_layout()

    # Plot the temperature as a function of time for the two probes
    idx_p1 = (np.abs(x-x_probe_1)).argmin()
    idx_p2 = (np.abs(x-x_probe_2)).argmin()

    fig_t, ax_t = plt.subplots(ncols=1)  # , constrained_layout=True)
    fig_t.set_size_inches(5.0, 3.5)


    ax_t.plot(
        t, u[:, idx_p1], label=f'Probe x={x[idx_p1]:.2f} cm', zorder=2,
        # marker='o', ms=8, fillstyle='none', mew=1.0
    )

    ax_t.plot(
        t, u[:, idx_p2], label=f'Probe x={x[idx_p2]:.2f} cm', zorder=3
    )

    ax_t.set_xlabel('Time (s)')
    ax_t.set_ylabel('Temperature (°C)')
    ax_t.set_xlim(left=0.0, right=t_max)
    ax_t.set_ylim(bottom=0.0)
    ax_t.legend(
        loc='best', frameon=True
    )

    ax_t_xlim = ax_t.get_xlim()
    ax_t_ylim = ax_t.get_ylim()

    p_height = ax_t_ylim[1]
    p_width = pulse_length
    xy = (ax_t_xlim[0], ax_t_ylim[0])
    rect = patches.Rectangle(
        xy, p_width, p_height, linewidth=1, edgecolor='tab:purple', facecolor=mpl.colors.CSS4_COLORS['lavender'], alpha=0.5,
        zorder=1
    )
    ax_t.add_patch(rect)
    ax_t.text(
        0.05, 0.95, 'Laser on',
        ha='left', va='top',
        transform = ax_t.transAxes,
        fontsize=9
    )


    # ax_t.xaxis.set_minor_locator(ticker.MultipleLocator(2.0))
    # ax_t.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))

    fig_t.tight_layout()

    fig.savefig(os.path.join(base_dir, 'rod_with_left_flux.png'), dpi=600)
    plt.show()
