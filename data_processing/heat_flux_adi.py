"""
Copyright 2021 Erick R Martinez Loran

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
import matplotlib.pylab as plt
import scipy as sp
import scipy.sparse
from scipy.sparse.linalg import spsolve
import pandas as pd
import h5py
import time
from scipy.sparse.linalg import factorized
import os

base_path = '/content/drive/MyDrive/python/ADI_ROD'


def gaussian_beam(r: np.ndarray, beam_diameter: float, beam_power: float):
    """
    Estimates the gaussian profile of a laser.

    Parameters
    ----------
    r:np.ndarray
      The position in cm
    beam_diameter:float
      The diameter of the beam in cm^2
    beam_power:float
      The power of the beam in W
    Returns
    -------
    np.ndarray:
      The intensity profile of the gaussian beam
    """
    r2 = np.power(r, 2.0)
    wz_sq = (0.5 * beam_diameter) ** 2.0
    intensity = (beam_power / (0.5 * np.pi * wz_sq)) * np.exp(
        -2.0 * r2 / wz_sq
    )

    return intensity


def simulate_adi_temp(laser_power: float, r_holder: float, r_sample: float,
                      length: float, kappa_1: float, kappa_2: float,
                      k0_1: float, k0_2: float, **kwargs):
    """
    Params
    ------
    q_peak: float
      The peak value of the incomming heat flux
    r_holder: float
      The radius of the sample holder (cm)
    r_sample: float
      The radius of the sample (cm)
    length: float
      The length of the sample (cm)
    kappa_1: float
      The thermal diffusivity of the sample (cm^2/s)
    kappa_2: float
      The thermal diffusivity of the holder (cm^2/s)
    k0_1: float
      The themral conductivity of the sample (W/(cm^2 K))
    k0_2: float
      The themral conductivity of the holder (W/(cm^2 K))
    kwargs:
      M: int
        The number of cell points in the r direction
      N: int
        The number of cell points in the x direction
      dt: float
        The time increment in seconds
      t_max: float
        The maximum simulation time in seconds
      pulse_length: float
        The length of the laser pulse in seconds
      chi: float
        The absorptivity of the rod.
      beam_diameter: float
        The w(z) of the laser beam in cm
      holder_thickness: float
        The thickness of the stainless steel holder the sample is attached to (cm)
      T_a: float
        The ambient temperature (°C)
      report_every: int
        Print the progress every 'n' steps.
      debug: bool
        If true, prints debugging messages.
      save_h5: bool
        If true, saves the h5 file with the whole simulation
      x_tc_1: float
        The 'x' position of thermocouple 1 (on the surface of the rod) in cm
      x_tc_2: float
        The 'x' position of thermocouple 2 (on the surface of the rod) in cm
      emissivity: float
        The emissivity of the material
      probe_size_mm: float
        The size of the thermocouple probe in mm. Default: 3 mm

    """
    M = int(kwargs.get('r_points', 200))  # number of intervals in r
    N = int(kwargs.get('x_points', 400))  # number of intervals in x
    dt = float(kwargs.get('dt', 1.0E-3))  # seconds
    t_max = float(kwargs.get('t_max', 2.0)) # seconds
    pulse_length = float(kwargs.get('pulse_length', 1.0)) # seconds
    chi = float(kwargs.get('chi', 1.0)) # The absorption coefficient
    emissivity = float(kwargs.get('emissivity', 1.0))
    beam_diameter = float(kwargs.get('beam_diameter', 0.8165)) # cm
    holder_thickness = float(kwargs.get('holder_thickness_cm', 2.54)) # cm
    T_a = float(kwargs.get('T_a', 20.0)) # °C
    report_every = int(kwargs.get('report_every', 10)) # iterations
    debug = bool(kwargs.get('debug', False))
    save_h5 = bool(kwargs.get('save_h5', False))
    x_tc_1 = float(kwargs.get('x_tc_1', 1.0)) # Position of probe 1 in cm along the x axis and the surface of the rod
    x_tc_2 = float(kwargs.get('x_tc_2', 2.0)) # Position of probe 2 in cm along the x axis and the surface of the rod
    probe_size = float(kwargs.get('probe_size_mm', 2.0)) # The size of the probe in mm
    R = r_holder
    R_sample = r_sample
    L = length  # the length of the cylinder in cm
    # Stefan-Boltzmann constant
    sb = emissivity * 5.670374419E-12  # W cm^{-2} K^{-4}
    if L < holder_thickness:
        print(f"Warning: setting the sample length to the thickness of the holder ({holder_thickness:.3f} cm)")
        length = holder_thickness
    exposed_x = length - holder_thickness

    # The output simulation time
    time_s = np.arange(0.0, t_max + dt, dt, dtype=np.float64)
    # The output temperature at point 1
    temperature_p1 = T_a * np.ones_like(time_s, dtype=np.float64)
    # The output temperature at point 2
    temperature_p2 = T_a * np.ones_like(time_s, dtype=np.float64)

    dr = R / M
    dx = L / N

    # The x vector array
    x = dx * np.arange(0, N + 1)
    # The r vector array
    r = dr * np.arange(0, M + 1)

    # Determine the the number of points to average the temperature at the position of the probe
    probe_size_idx = int(probe_size * 0.1 / dx)
    probe_idx_delta = int(0.5 * probe_size_idx)

    # The gaussian beam input as a function of the laser power and beam diameter
    q = gaussian_beam(r=r, beam_power=laser_power, beam_diameter=beam_diameter)
    # Shadow the beam outside the sample diameter
    q[r > R_sample] = 0.0

    # The output filename
    hf_filename = f'./ADI_k1_{kappa_1:.2E}_chi_{chi:.2f}_P{laser_power:.2E}'

    # Constants for the matrices and vectors
    alpha = 0.5 * dt
    beta_1 = kappa_1 * alpha / (dr ** 2.0)
    gamma_1 = 0.5 * kappa_1 * alpha / dr
    mu_1 = kappa_1 * alpha / (dx ** 2.0)

    # a_cal_1 = dx * chi / kappa_1
    s_cal_1 = 2.0 * dx / k0_1
    z_cal_1 = 2.0 * dr * sb * (beta_1 + gamma_1 / R_sample) / k0_1

    beta_2 = kappa_2 * alpha / (dr ** 2.0)
    gamma_2 = 0.5 * kappa_2 * alpha / dr
    mu_2 = kappa_2 * alpha / (dx ** 2.0)

    s_cal_2 = 2.0 * dx / k0_2
    z_cal_2 = 2.0 * dr * sb * (beta_2 + gamma_2 / R) / k0_2
    a_cal_2 = dx * chi / kappa_2

    k_cal_1 = (kappa_1 - kappa_2) / kappa_1
    k_cal_2 = kappa_2 / kappa_1
    k_cal_3 = (kappa_2 - kappa_1) / kappa_2
    k_cal_4 = kappa_1 / kappa_2

    msk_holder = r > R_sample
    idx_r = (np.abs(r - R_sample)).argmin()
    idx_h = (np.abs(exposed_x - x)).argmin()

    idx_x1 = (np.abs(x_tc_1 - x)).argmin()
    idx_x2 = (np.abs(x_tc_2 - x)).argmin()

    if debug:
        print(f'q_max: {laser_power:.3E} W/cm^2')
        print(f'beam diameter: {beam_diameter:.3f} cm')
        print(f'Size of x: {x.size}, N: {N}')
        print(f'Size of r: {r.size}, M: {M}')
        print(f'r(idx={idx_r}) = {r[idx_r]} cm')
        print(f'R = {r[-1]} cm')
        print(f'Exposed length: {exposed_x:.2f} cm')
        print("***** Thermal conductivities *****")
        print(f"K01: {k0_1:5.3E}, K02: {k0_2:5.3E}")
        print("***** Thermal diffusivities *****")
        print(f"kappa_1: {kappa_1:5.3E}, kappa_2: {kappa_2:5.3E}")

        print(f'dr: {dr:.3E}, dx: {dx:.3E}')
        print(f'Emissivity: {emissivity:6.3f}')
        print(f'kappa_1: {kappa_1:.3E}, kappa_2: {kappa_2:.3E}')
        print(f'beta_1: {beta_1:.3E}, beta_2: {beta_2:.3E}')
        print(f'gamma_1: {gamma_1:.3E}, gamma_2: {gamma_2:.3E}')
        print(f'mu_1: {mu_1:.3E}, mu_2: {mu_2:.3E}')
        print(f'k_cal_1: {k_cal_1:.3E}, k_cal_2: {k_cal_2:.3E}')
        print(f'k_cal_3: {k_cal_3:.3E}, k_cal_4: {k_cal_4:.3E}')

    # The matrix A_1
    d_0 = np.zeros(M)
    d_1 = np.array([1.0 + 2.0 * beta_1 if i <= idx_r else 1.0 + 2.0 * beta_2 for i in range(M + 1)])
    d_2 = np.zeros(M)

    for i in range(1, M):
        if i <= idx_r:
            bb = beta_1
            gg = gamma_1
        else:
            bb = beta_2
            gg = gamma_2
        d_0[i - 1] = -bb + gg / r[i]
        d_2[i] = -bb - gg / r[i]

    d_0[-1] = -2.0 * beta_2
    d_2[0] = -2.0 * beta_1
    d_0[idx_r - 1] = -beta_1 * (1.0 + k_cal_4) + (gamma_1 / r[idx_r]) * (1.0 - k_cal_4)
    d_1[idx_r] = 1.0 + beta_1 * (2.0 - k_cal_3) - k_cal_3 * (gamma_1 / r[idx_r])
    d_2[idx_r] = 0.0
    d_0[idx_r] = -beta_2 * (1.0 + k_cal_4) + (gamma_2 / r[idx_r + 1]) * (1.0 - k_cal_4)
    d_1[idx_r + 1] = 1.0 + beta_2 * (2.0 - k_cal_3) - k_cal_3 * (gamma_2 / r[idx_r + 1])
    d_2[idx_r + 1] = 0.0

    diagonals = [d_0, d_1, d_2]
    A1 = sp.sparse.diags(diagonals, [-1, 0, 1], format='csc')

    # The matrix A_2
    d_0 = np.zeros(M)
    d_1 = np.array([1.0 + 2.0 * beta_1 if i <= idx_r else 1.0 for i in range(M + 1)])
    d_2 = np.zeros(M)

    for i in range(1, M):
        if i <= idx_r:
            bb = beta_1
            gg = gamma_1
        else:
            bb = 0.0
            gg = 0.0
        d_0[i - 1] = -bb + gg / r[i]
        d_2[i] = -bb - gg / r[i]

    d_0[idx_r - 1] = -2.0 * beta_1
    d_2[idx_r] = 0.0
    d_2[0] = -2.0 * beta_1

    diagonals = [d_0, d_1, d_2]
    A2 = sp.sparse.diags(diagonals, [-1, 0, 1], format='csc')

    # Matrix B1
    d_0 = -mu_1 * np.ones(N, dtype=np.float64)
    d_1 = (1.0 + 2.0 * mu_1) * np.ones(N + 1, dtype=np.float64)
    d_2 = -mu_1 * np.ones(N, dtype=np.float64)
    d_0[-1] = -2.0 * mu_1
    d_2[0] = -2.0 * mu_1
    diagonals = [d_0, d_1, d_2]
    B1 = sp.sparse.diags(diagonals, [-1, 0, 1], format='csc')

    # Matrix B2
    d_0 = -mu_2 * np.ones(N, dtype=np.float64)
    d_1 = (1.0 + 2.0 * mu_2) * np.ones(N + 1, dtype=np.float64)
    d_2 = -mu_2 * np.ones(N, dtype=np.float64)

    for i in range(idx_h):
        d_0[i] = 0.0
        d_1[i] = 1.0
        d_2[i] = 0.0

    d_0[-1] = -2.0 * mu_2
    d_2[idx_h] = -2.0 * mu_2

    diagonals = [d_0, d_1, d_2]
    B2 = sp.sparse.diags(diagonals, [-1, 0, 1], format='csc')

    solve_A1 = factorized(A1)
    solve_A2 = factorized(A2)
    solve_B1 = factorized(B1)
    solve_B2 = factorized(B2)

    # Create the structure of the output dataset (if saving it)
    if save_h5:
        # Save data to h5 file
        with h5py.File(hf_filename + '.h5', 'w') as hf:
            hf.create_dataset('data/x', data=x)
            hf.create_dataset('data/r', data=r)
            hf['/data'].attrs['kappa_1'] = kappa_1
            hf['/data'].attrs['kappa_2'] = kappa_2
            hf['/data'].attrs['K0_1'] = k0_1
            hf['/data'].attrs['K0_2'] = k0_2
            hf['/data'].attrs['M'] = M
            hf['/data'].attrs['N'] = N
            hf['/data'].attrs['R_sample'] = R_sample
            hf['/data'].attrs['R'] = R
            hf['/data'].attrs['dr'] = dr
            hf['/data'].attrs['dx'] = dx
            hf['/data'].attrs['dt'] = dt
            hf['/data'].attrs['idx_r'] = idx_r
            hf.create_dataset('data/q', data=q)
            hf.create_dataset('data/time', data=np.arange(0, t_max + dt, dt))

    U_k1 = T_a * np.ones((M + 1, N + 1), dtype=np.float64)
    U_kh = T_a * np.ones((M + 1, N + 1), dtype=np.float64)

    mu_j = np.array([mu_1 if i <= idx_r else mu_2 for i in range(M + 1)])
    w_j = np.array([1.0 / k0_1 if i <= idx_r else 1.0 / k0_2 for i in range(M + 1)])
    w_j_0 = np.array([1.0 / k0_1 if i <= idx_r else 0.0 for i in range(M + 1)])

    t_now = 0
    count = 0

    eye_m = np.ones(M + 1, dtype=np.float64)
    d = np.zeros(M + 1, dtype=np.float64)
    b = np.zeros(N + 1, dtype=np.float64)

    u_min = np.inf
    u_max = -np.inf

    sb4 = sb ** 0.25  # =  0.03928268256554753 <- avoid multiplying for 1E-12
    z_cal_1_4 = z_cal_1 ** 0.25
    z_cal_2_4 = z_cal_2 ** 0.25
    T_a_k = T_a + 273.15
    T_a_4 = T_a_k ** 4.0

    def radiative_term(temperature: np.ndarray) -> np.ndarray:
        return sb*(np.power(c2k(temperature), 4.0) - T_a_4)

    def c2k(temperature: np.ndarray) -> np.ndarray:
        return temperature + 273.15



    start_time = time.time()
    step_time = time.time()
    while t_now <= t_max:
        if count % (report_every - 1) == 0 and debug:
            print(
                f'Time: {t_now:4.3f} s, T(r={r[0]:3.3f}, x={x[0]:3.3f}) = {U_k1[0, 0]:5.1f}, T(r={r[idx_r]:3.3f}, x={x[idx_x1]:3.3f}) = {U_k1[idx_r, idx_x1-probe_idx_delta:idx_x1+probe_idx_delta].mean():5.1f}, T(r={r[-1]:3.3f}, x={x[idx_h]:3.3f}) = {U_k1[-1, idx_h]:5.1f}, Wall Time: {(time.time() - start_time) / 60.0:8.2f} min, Step Time: {(time.time() - step_time):6.3E} s')

        temperature_p1[count] = U_k1[idx_r, idx_x1-probe_idx_delta:idx_x1+probe_idx_delta].mean()
        temperature_p2[count] = U_k1[idx_r, idx_x2]

        u_min = min(u_min, U_k1.flatten().flatten().min())
        u_max = max(u_max, U_k1.flatten().flatten().max())

        if save_h5:
            with h5py.File(hf_filename + '.h5', 'a') as hf:
                ds_name = f'/data/T_{count:d}'
                hf.create_dataset(ds_name, data=U_k1)
                hf[ds_name].attrs['time (s)'] = t_now

        # Update in r
        for j in range(0, N + 1):
            if j < idx_h:
                d[idx_r] = radiative_term(U_k1[M, j])
            else:
                d[-1] = radiative_term(U_k1[M, j])

            if j == 0:
                if t_now <= pulse_length:
                    w1 = (eye_m - 2.0 * mu_j) * U_k1[:, 0] + 2.0 * mu_j * (
                            U_k1[:, 1] + dx * w_j * chi * q - dx * w_j_0 * radiative_term(U_k1[:, j])) + d
                else:
                    w1 = (eye_m - 2.0 * mu_j) * U_k1[:, 0] + 2.0 * mu_j * (
                            U_k1[:, 1] - dx * w_j_0 * radiative_term(U_k1[:, j])) + d
            elif j == N:
                w1 = 2.0 * mu_j * (U_k1[:, j - 1] + dx * w_j * radiative_term(U_k1[:, j])) \
                     + (eye_m - 2.0 * mu_j) * U_k1[:, j] + d
            else:
                w1 = mu_j * U_k1[:, j - 1] + (eye_m - 2.0 * mu_j) * U_k1[:, j] + mu_j * U_k1[:, j + 1] + d
                if j < idx_h:
                    w1[idx_r + 1:] = U_k1[idx_r + 1:, j]
                elif j == idx_h:
                    if t_now <= pulse_length:
                        w_e = (eye_m - 2.0 * mu_j) * U_k1[:, idx_h] + 2.0 * mu_j * (
                                U_k1[:, idx_h + 1] + dx * w_j * chi * q - dx * w_j * radiative_term(U_k1[:, idx_h])) + d
                    else:
                        w_e = (eye_m - 2.0 * mu_j) * U_k1[:, idx_h] + 2.0 * mu_j * (
                                U_k1[:, idx_h + 1] - dx * w_j * radiative_term(U_k1[:, idx_h])) + d
                    w1[idx_r + 1:] = w_e[idx_r + 1:]

            if (np.isnan(w1).any() or np.isinf(w1).any()) and debug:
                print(f"w1 has nan")
                idx_nan = np.logical_or(np.isnan(w1), np.isinf(w1))
                for k in range(M + 1):
                    if idx_nan[k]:
                        print(f'r = {r[k]:.3f} cm, w1[{k}] = {w1[k]}, U_kh = {U_k1[k, j]}')
                raise ValueError("NaN found in w2")

            ss = solve_A2 if j < idx_h else solve_A1

            U_kh[:, j] = ss(w1)

        # update in x
        for i in range(0, M + 1):
            g_i = 2.0 * dx * w_j[i] * (chi * q[i] - radiative_term(U_kh[
                i, 0])) if t_now <= pulse_length else -2.0 * dx * w_j[i] * radiative_term(U_kh[i, 0])
            b_n = s_cal_1 * radiative_term(U_kh[i, -1]) if i <= idx_r else s_cal_2 * radiative_term(U_kh[i, -1])
            b[0] = g_i
            b[-1] = b_n

            if i == 0:
                w2 = (1.0 - 2.0 * beta_1) * U_kh[0, :] + 2.0 * beta_1 * U_kh[1, :] + mu_1 * b
            elif i == idx_r:
                w2 = (beta_1 * (1.0 + k_cal_4) - (gamma_1 / R_sample) * (1.0 - k_cal_4)) * U_kh[idx_r - 1, :] + (
                        1.0 - beta_1 * (2.0 - k_cal_3) + k_cal_3 * (gamma_1 / r[idx_r])) * U_kh[idx_r, :] + mu_1 * b
                w2[:idx_h] = 2.0 * beta_1 * U_kh[i - 1, :idx_h] + (1.0 - 2.0 * beta_1) * U_kh[i, :idx_h] \
                                 + (beta_1 + gamma_1 / R_sample) * (2.0 * dr / k0_1) * radiative_term(U_kh[i, :idx_h]) \
                                 + mu_1 * b[:idx_h]
            elif i == (idx_r + 1):
                w2 = (beta_2 * (1.0 + k_cal_4) - (gamma_2 / r[idx_r + 1]) * (1.0 - k_cal_4)) * U_kh[idx_r, :] \
                     + (1.0 - beta_2 * (2.0 - k_cal_3) + k_cal_3 * (gamma_2 / r[idx_r + 1])) * U_kh[idx_r + 1, :] + mu_2 * b
                w2[:idx_h] = U_kh[i, :idx_h]
            elif i == M:
                w2 = 2.0 * beta_2 * U_kh[i - 1, :] + (1.0 - 2.0 * beta_2) * U_kh[i, :] + (beta_2 + gamma_2 / R) * (
                        2.0 * dr / k0_2) * radiative_term(U_kh[M, :]) + mu_2 * b
                w2[:idx_h] = U_kh[i, :idx_h]
            elif i != 0 and i < idx_r:
                w2 = (beta_1 - gamma_1 / r[i]) * U_kh[i - 1, :] + (1.0 - 2.0 * beta_1) * U_kh[i, :] + (
                        beta_1 + gamma_1 / r[i]) * U_kh[i + 1, :] + mu_1 * b
            elif i > (idx_r + 1) and i != M:
                w2 = (beta_2 - gamma_2 / r[i]) * U_kh[i - 1, :] + (1.0 - 2.0 * beta_2) * U_kh[i, :] + (
                        beta_2 + gamma_2 / r[i]) * U_kh[i + 1, :] + mu_2 * b
                w2[:idx_h] = U_kh[i, :idx_h]

            if (np.isnan(w2).any() or np.isinf(w2).any()) and debug:
                print(f"w2 has a nan")
                idx_nan = np.logical_or(np.isnan(w2), np.isinf(w2))
                for k in range(M + 1):
                    if idx_nan[k]:
                        print(f'r = {r[i]:.3f}, x = {x[k]:.3f} cm,  w2 = {w2[k]}, U_kh = {U_kh[i, k]}')
                print(w2)
                raise ValueError("NaN found in w2")

            ss = solve_B1 if i <= idx_r else solve_B2

            U_k1[i, :] = ss(w2)

        U_k1[idx_r + 1:, 0:idx_h] = T_a
        t_now += dt
        count += 1

        if debug:
            step_time = time.time()

    # Save the minimum and maximum temperatures (if save_h5 is True)
    if save_h5:
        with h5py.File(hf_filename + '.h5', 'a') as hf:
            hf['data'].attrs['T_min'] = u_min
            hf['data'].attrs['T_max'] = u_max
        return hf_filename
    else:
        return time_s, temperature_p1, temperature_p2
