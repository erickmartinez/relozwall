import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os

from matplotlib import ticker, patches


base_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\heat equation\fourier_left_flux'

N = 100  # The number of Fourier coefficients
alpha = 1E-1
length = 3.5
inital_temperature = 25.0
x_points = 400
t_max = 20.0
t_points = 200
F1 = 1E3
pulse_length = 5.0
x_probe_1 = 0.5
x_probe_2 = length

by_p2 = np.power(np.pi, -2.0)

debug = False

def u_1(x: np.ndarray, diffusion_time: np.ndarray, rod_length:float, diffusivity: float, flux: float):
    L = rod_length
    F = flux
    return 0.5 * F * (x ** 2.0) / L - (F * x) + diffusivity * (flux / L) * diffusion_time


def u_laser_on(x: np.ndarray, diffusion_time: np.ndarray, rod_length:float, diffusivity: float, flux: float, T0: float):
    L = rod_length
    u = T0 * np.ones((diffusion_time.size, x.size))
    a = np.pi / L
    b = 2.0 * flux * L * by_p2
    for i, t in zip(range(1, diffusion_time.size), diffusion_time[1:]):
        s0 = flux * L / 3.0
        s = 0
        for n in range(1, N + 1):
            an = 1.0 / (n * n)
            s += an * np.cos(n * a * x) * np.exp(-diffusivity * t * (n * a) ** 2.0)
        u[i] += s0 - b * s + u_1(x, t, rod_length, diffusivity, flux)
        if debug:
            print(f'[on] - ({i:4d}) T[0,t={t:>6.3f}s] = {u[i, 0]:>6.3E}, T[L,t={t:>6.3f}s] = {u[i, -1]:>6.3E}')
    return u


def get_an(n: int, rod_length:float, diffusivity: float, emission_time:float, flux: float, T0: float):
    L = rod_length
    if n == 0:
        return T0 + diffusivity*flux*emission_time/L
    return (2.0 * flux * L / ((n * np.pi)**2.0)) * (1.0 - np.exp(-diffusivity*emission_time*(n*np.pi/L)**2.0))


def get_ut(x: np.ndarray, diffusion_time: np.ndarray, rod_length:float, diffusivity: float, emission_time: float, flux: float, T0: float):
    L = rod_length
    u = np.zeros((diffusion_time.size, x.size))
    msk_on = diffusion_time <= emission_time
    msk_off = diffusion_time > emission_time
    time_off = diffusion_time[msk_off]
    idx_off = len(u[msk_on])
    u_on = u_laser_on(x=x, diffusion_time=diffusion_time[msk_on], rod_length=L, diffusivity=diffusivity, flux=flux, T0=T0)
    u[msk_on, :] = u_on.copy()
    # After the laser pulse consider the solution of the heat equation for a 1D rod with insulated ends
    for i, ti in enumerate(time_off):
        for n in range(N+1):
            arg = ((n * np.pi/L) ** 2.0) * diffusivity * (ti - pulse_length) #(diffusion_time[i+idx_off]-pulse_length)
            a_n = get_an(n, rod_length=L, diffusivity=diffusivity, emission_time=emission_time, flux=flux, T0=T0)
            if n == 0:
                u[i+idx_off] = a_n
            else:
                u[i+idx_off] += a_n * np.cos(n * np.pi * x / L) * np.exp(-arg)
        if debug:
            print(f'[off] - ({i+idx_off:3d}) T[0,t={ti:>6.3f}s] = {u[i + idx_off, 0]:>6.3E}, T[L,t={ti:>6.3f}s] = {u[i + idx_off, -1]:>6.3E}')

    return u


if __name__ == '__main__':
    x = np.array([x_probe_1, x_probe_2])
    t = np.linspace(0.0, t_max, num=t_points)
    u = get_ut(x=x, diffusion_time=t, rod_length=length, diffusivity=alpha, emission_time=pulse_length, flux=F1, T0=inital_temperature)

    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1)  # , constrained_layout=True)
    fig.set_size_inches(5.0, 3.5)

    ax.plot(
        t, u[:, 0], label=f'Probe x={x[0]:.2f} cm', zorder=2,
        # marker='o', ms=8, fillstyle='none', mew=1.0
    )

    ax.plot(
        t, u[:, 1], label=f'Probe x={x[1]:.2f} cm', zorder=3
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlim(left=0.0, right=t_max)
    ax.set_ylim(bottom=0.0)
    ax.legend(
        loc='best', frameon=True
    )

    ax_t_xlim = ax.get_xlim()
    ax_t_ylim = ax.get_ylim()

    p_height = ax_t_ylim[1]
    p_width = pulse_length
    xy = (ax_t_xlim[0], ax_t_ylim[0])
    rect = patches.Rectangle(
        xy, p_width, p_height, linewidth=1, edgecolor='tab:purple', facecolor=mpl.colors.CSS4_COLORS['lavender'], alpha=0.5,
        zorder=1
    )
    ax.add_patch(rect)
    ax.text(
        0.05, 0.95, 'Laser on',
        ha='left', va='top',
        transform=ax.transAxes,
        fontsize=9
    )

    # ax_t.xaxis.set_minor_locator(ticker.MultipleLocator(2.0))
    # ax_t.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))

    fig.tight_layout()
    plt.show()