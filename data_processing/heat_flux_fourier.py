import numpy as np

debug = False
by_p2 = np.power(np.pi, -2.0)
N = 100


def u_1(x: np.ndarray, diffusion_time: np.ndarray, rod_length: float, diffusivity: float, flux: float):
    L = rod_length
    F = flux
    return 0.5 * F * (x ** 2.0) / L - (F * x) + diffusivity * (flux / L) * diffusion_time


def u_laser_on(x: np.ndarray, diffusion_time: np.ndarray, rod_length: float, diffusivity: float, flux: float,
               T0: float):
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


def get_an(n: int, rod_length: float, diffusivity: float, emission_time: float, flux: float, T0: float):
    L = rod_length
    if n == 0:
        return T0 + diffusivity * flux * emission_time / L
    return (2.0 * flux * L / ((n * np.pi) ** 2.0)) * (
            1.0 - np.exp(-diffusivity * emission_time * (n * np.pi / L) ** 2.0))


def get_ut(x: np.ndarray, diffusion_time: np.ndarray, rod_length: float, diffusivity: float, emission_time: float,
           flux: float, T0: float):
    L = rod_length
    u = np.zeros((diffusion_time.size, x.size))
    msk_on = diffusion_time <= emission_time
    msk_off = diffusion_time > emission_time
    time_off = diffusion_time[msk_off]
    idx_off = len(u[msk_on])
    u_on = u_laser_on(x=x, diffusion_time=diffusion_time[msk_on], rod_length=L, diffusivity=diffusivity, flux=flux,
                      T0=T0)
    u[msk_on, :] = u_on.copy()
    # After the laser pulse consider the solution of the heat equation for a 1D rod with insulated ends
    for i, ti in enumerate(time_off):
        for n in range(N + 1):
            arg = ((n * np.pi / L) ** 2.0) * diffusivity * (
                    ti - emission_time)  # (diffusion_time[i+idx_off]-pulse_length)
            a_n = get_an(n, rod_length=L, diffusivity=diffusivity, emission_time=emission_time, flux=flux, T0=T0)
            if n == 0:
                u[i + idx_off] = a_n
            else:
                u[i + idx_off] += a_n * np.cos(n * np.pi * x / L) * np.exp(-arg)
        if debug:
            print(
                f'[off] - ({i + idx_off:3d}) T[0,t={ti:>6.3f}s] = {u[i + idx_off, 0]:>6.3E}, T[L,t={ti:>6.3f}s] = {u[i + idx_off, -1]:>6.3E}')

    return u

