import numpy as np
from scipy.special import erfinv, erfc


# Eric Hollmann's model
def simulate_1d_temperature(
        q0: float, length: float, k0: float, rho: float, **kwargs
):
    """
    Simulates the temperature profile as a function of time for two points in z:
    one at :math:`z=0`, and at an specified point.

    Parameters
    ----------
    q0: (float) The input heat flux.
    length: (float) The length of the rod in cm.
    k0: (float) The thermal conductivity of the rod in W/K/m.
    rho: (float) The density of the rod in kg/m^3.
    **kwargs:
      z_points: (int) The number of intervals in the z axis. Default: 500
      t_steps: (int) The number of time intervals. Default: 1000
      t_max: (float) The duration of the simulation in s. Default 1.0 s
      pulse_length: (float) The duration of the pulse in s. Default: 0.5 s
      alpha: (float) The absorbance of the front surface :math:`\alpha \in [0,1]`. Default: 0.8
      T0: (float) The initial temperature in K. Default: 300 K
      z_tc_1: (float) The position of the z probe in cm. Default: 1.0 cm
      cp: (float) The heat capacity of the rod in J/kg/K. Default 710 J/kg/K
      debug: (bool) If true, show debugging messages.

    Returns
    -------
    np.ndarray: The simulation time in seconds.
    np.ndarray: A :math:`M \times N` array with the temperatures of the probe :math:`m \in {0,\ldots,M}` for time `t_n, n \in {0,\ldots,N}`
    """
    N = int(kwargs.get('z_points', 500))  # number of intervals in x
    M = int(kwargs.get('t_steps', 1000))  # number of time steps
    t_max = float(kwargs.get('t_max', 1.0))
    pulse_length = float(kwargs.get('pulse_length', 0.5))
    alpha = float(kwargs.get('alpha', 0.8))
    T0 = float(kwargs.get('T0', 300.0))
    debug = bool(kwargs.get('debug', False))
    z_tc_1 = float(kwargs.get('z_tc_1', 1.0))
    cp = float(kwargs.get('cp', 710.0))
    probe_size = float(kwargs.get('probe_size_mm', 2.0))
    Zprobe = [0, z_tc_1]

    sb = 5.670374419E-8  # W m^{-2} K^{-4}
    q0mks = q0 * 1e6  # front surface heat flux [W/m2]
    chi = k0 / (cp * rho)  # thermal diffusivity [m^2/s]
    dt = t_max / M  # step size in t [s]
    tV = dt * np.arange(0.0, M + 1, 1, dtype=np.float64)  # time [s]
    dT0 = 2.0 * alpha * q0mks * np.sqrt(dt / (np.pi * rho * cp * k0))  # temperature excursion step size [K]
    dz = length * 1e-2 / M  # step size in z [m]
    zV = dz * np.arange(0.0, N + 1, dtype=np.float64)
    TA = np.zeros((M + 1, N + 1), dtype=np.float64)
    TAdum = TA.copy()
    probe_size_idx = int(probe_size * 1E-3 / dz)
    probe_idx_delta = int(0.5 * probe_size_idx)

    if debug:
        print(f"L: {length} cm")
        print(f"T0: {T0} K")
        print(f"rho: {rho} kg/m^3")
        print(f"t_max: {t_max} s")
        print(f"pulse_length: {pulse_length} s")
        print(f"dt: {dt} s")
        print(f"dT0: {dT0} s")
        print(f"dz: {dz} m")
        print(f"chi: {chi} m^2/s")
        print(f"q0mks: {q0mks:05.3E}")
        print(f"cp: {cp} J/kg/K")

    for i in range(1, len(tV)):
        tval = tV[i]
        zv2 = np.power(zV, 2.0) / (4.0 * chi * tval)
        TV = dT0 * np.sqrt(dt / tval) * np.exp(-zv2)
        TAdum[i, :] = TV

    idx_probes = np.empty(len(Zprobe), dtype=int)
    for i, v in enumerate(Zprobe):
        idx_probes[i] = (np.abs(zV - v * 1E-2)).argmin()

    if debug:
        print("Probing temperature at positions:")
        for ip in idx_probes:
            print(f'z = {zV[ip] * 100.0:3.2f} cm')

    if debug:
        check_probe_idx = [zV[i] for i in idx_probes]
        print(check_probe_idx)

    iit0 = (np.abs(tV - pulse_length)).argmin()
    for iit in range(1, M + 1):
        iit1 = iit - iit0 + 1
        iit1 = max(iit1, 1)
        TV = np.sum(TAdum[iit1:iit, :], axis=0)
        TA[iit, :] = TV

    result = np.empty((idx_probes.size, M + 1), dtype=np.float64)
    for i, v in enumerate(Zprobe):
        if i > 0:
            idx_1 = idx_probes[i] - probe_idx_delta
            idx_2 = idx_probes[i] + probe_idx_delta
            result[i, :] = TA[:, idx_1:idx_2].mean(axis=1) + T0 - 273.15
        else:
            result[i, :] = TA[:, idx_probes[i]] + T0 - 273.15

    return tV, result


