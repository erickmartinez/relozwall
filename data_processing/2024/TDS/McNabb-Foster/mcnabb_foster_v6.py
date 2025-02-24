import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

"""
+----------+-------------+------------+-------------+-----------+-----+
| Material | D0 (cm^2/s) | D0 (m^2/s) | Ed (kJ/mol) |  Ed (eV)  | Ref |
+----------+-------------+------------+-------------+-----------+-----+
| B4C      |      2.3E-4 |     2.3E-8 |     96 ± 10 | 1.0 ± 0.1 | [1] |
| B2O3     |      4.9E-5 |     4.9E-9 |    124 ± 15 | 2.3 ± 0.2 | [1] |
+----------+-------------+------------+-------------+-----------+-----+


[1] K. Schnarr and H. Münzel, "Release of tritium from boron carbide." (1990) J. Chem. Soc., Faraday Trans. 86, 651-656
doi: 10.1039/FT9908600651
"""

class ThermalDesorptionSimulator:
    def __init__(
            self,
            T0=300,  # Initial temperature (K)
            Tf=1200,  # Final temperature (K)
            beta=0.3,  # Heating rate (K/s)
            C0=5e25,  # Peak concentration (1/m³)
            L=1e-6,  # Sample thickness (m)
            D0=1e-6,  # Diffusion pre-exponential (m²/s)
            Ed=1.5,  # Diffusion activation energy (eV)
            k0=1e-15,  # Trapping coefficient pre-exponential (m³/s)
            Et=1.2,  # Trapping energy (eV)
            p0=1e13,  # Detrapping frequency factor (1/s)
            Eb=1.0,  # Binding energy (eV) for detrapping
            N=1e27,  # Trap density (1/m³)
            nx=1001  # Number of spatial points
    ):
        # Physical parameters (same as original)
        self.L = L  # Sample thickness (m)
        self.D0 = D0  # Diffusion pre-exponential (m²/s)
        self.Ed = Ed  # Diffusion activation energy (eV)
        self.k0 = k0  # Trapping coefficient pre-exponential (m³/s)
        self.Et = Et  # Trapping energy (eV)
        self.p0 = p0  # Detrapping frequency factor (1/s)
        self.Eb = Eb  # Binding energy (eV) for detrapping
        self.N = N  # Trap density (1/m³)

        # Initial conditions and numerical parameters (same as original)
        self.T0 =T0  # Initial temperature (K)
        self.Tf = Tf  # Final temperature (K)
        self.beta = beta  # Heating rate (K/s)
        self.nx = nx  # Number of spatial points
        self.dX = 1.0 / (self.nx - 1)  # Grid spacing in normalized coordinates
        self.X = np.linspace(0, 1, self.nx)  # Normalized spatial grid
        self.x = self.X * self.L  # Physical spatial grid

        # Initial concentration profile (Gaussian)
        self.C0 = C0  # Peak concentration (1/m³)
        self.sigma = 0.025  # Width of Gaussian in normalized coordinates
        self.u0 = np.exp(-(self.X - 0.05) ** 2 / (2 * self.sigma ** 2))

        # Threshold for considering concentration effectively zero
        self.zero_threshold = 1e-10

    def D(self, T):
        """Temperature-dependent diffusion coefficient"""
        return self.D0 * np.exp(-self.Ed / (8.617e-5 * T))

    def k(self, T):
        """Temperature-dependent trapping coefficient"""
        return self.k0 * np.exp(-self.Et / (8.617e-5 * T))


    def p(self, T):
        """Temperature-dependent detrapping coefficient"""
        return self.p0 * np.exp(-self.Eb / (8.617e-5 * T))

    def calculate_parameters(self, t):
        """Calculate normalized parameters at given time"""
        T = self.T0 + self.beta * t
        D_ref = self.D(self.Tf)  # Reference diffusion coefficient
        D_t = self.D(T)  # Current diffusion coefficient

        # Modified normalized parameters using D_ref
        lambda_param = self.N * self.k(T) * self.L ** 2 / D_ref
        nu_param = self.k(T) * self.C0 * self.L ** 2 / D_ref
        mu_param = self.p(T) * self.L ** 2 / D_ref
        D_ratio = D_t / D_ref  # Normalized diffusion coefficient

        return T, D_ratio, lambda_param, nu_param, mu_param

    def pde_system(self, t, y):
        """Define the PDE system with temperature-dependent diffusion"""
        nx = self.nx
        u = y[:nx]
        w = y[nx:]

        # Calculate parameters including D_ratio
        T, D_ratio, lambda_param, nu_param, mu_param = self.calculate_parameters(t)

        # Initialize derivative arrays
        dudt = np.zeros(nx)
        dwdt = np.zeros(nx)

        # Modified diffusion term with temperature-dependent D
        # Implement ∂(D(T)∂u/∂x)/∂x using central differences
        for i in range(1, nx - 1):
            # Calculate flux at i+1/2 and i-1/2
            flux_right = D_ratio * (u[i + 1] - u[i]) / self.dX
            flux_left = D_ratio * (u[i] - u[i - 1]) / self.dX

            # Calculate divergence of flux
            dudt[i] = (flux_right - flux_left) / self.dX

        # Zero flux boundary conditions
        # dudt[0] = 0
        # dudt[-1] = 0

        # Trapping terms (same as original)
        dwdt = lambda_param * u - nu_param * u * w - mu_param * w
        dudt -= dwdt

        # Zero flux boundary conditions
        # dudt[0] = 0
        dudt[-1] = 0
        dwdt[0] = 0
        dwdt[-1] = 0

        return np.concatenate([dudt, dwdt])

    # Rest of the methods remain the same
    def get_initial_conditions(self):
        """Calculate initial equilibrium conditions"""
        T = self.T0
        k_val = self.k(T)
        p_val = self.p(T)
        n0 = k_val * self.u0 * self.C0 / (k_val * self.u0 * self.C0 + p_val)
        w0 = n0 * self.N / self.C0
        return np.concatenate([self.u0, w0])

    def concentrations_depleted(self, t, y):
        """Event function that returns 0 when total integrated concentration is effectively zero"""
        # Sum up all mobile and trapped concentrations
        total_inventory = np.sum(np.abs(y))
        return total_inventory - self.zero_threshold

    def simulate(self, t_max):
        """Run the simulation"""
        y0 = self.get_initial_conditions()
        tau_max = self.D(self.Tf) * t_max / (self.L ** 2)
        dtau = (self.dX **2) / 2
        ntaus_stability = int(tau_max // dtau + 1)
        print(f"dtau_stability: {dtau}, ntaus_stability: {ntaus_stability}")
        ntaus = np.clip(ntaus_stability, a_min=101, a_max=1001)
        dtau = tau_max / (ntaus - 1)
        print(f"dtau: {dtau}, ntaus: {ntaus}")
        taus = np.arange(ntaus) * dtau


        solution = solve_ivp(
            self.pde_system,
            (0, tau_max+dtau),
            y0,
            # method='LSODA',
            method='Radau',
            # t_eval=np.linspace(0, tau_max,1000),
            t_eval=taus,
            rtol=1e-6,
            atol=1e-8,
            events=self.concentrations_depleted  # Add event detection
        )

        # Calculate temperatures and physical quantities
        temps = self.T0 + self.beta * solution.t * (self.L ** 2) / self.D(self.Tf)
        C_mobile = solution.y[:self.nx, :] * self.C0
        C_trapped = solution.y[self.nx:, :] * self.C0

        # Calculate flux using temperature-dependent D
        fluxes = np.zeros(len(temps))
        for i, T in enumerate(temps):
            D_t = self.D(T)
            fluxes[i] = -D_t * self.C0 * (solution.y[0, i] - solution.y[2, i]) / (2 * self.dX * self.L)

        return {
            'time': solution.t,
            'temperature': temps,
            'C_mobile': C_mobile,
            'C_trapped': C_trapped,
            'flux': fluxes,
            'terminated_early': len(solution.t_events[0]) > 0  # Check if simulation stopped due to event
        }


# Run simulation
simulator = ThermalDesorptionSimulator(
    T0=300,  # Initial temperature (K)
    Tf=1200,  # Final temperature (K)
    beta=0.3,  # Heating rate (K/s)
    C0=5e25,  # Peak concentration (1/m³)
    L=1e-6,  # Sample thickness (m)
    D0=1e-8,  # Diffusion pre-exponential (m²/s)
    Ed=1.07,  # Diffusion activation energy (eV)
    k0=1e-15,  # Trapping coefficient pre-exponential (m³/s)
    Et=1.2,  # Trapping energy (eV)
    p0=1e13,  # Detrapping frequency factor (1/s)
    Eb=1.2,  # Binding energy (eV) for detrapping
    N=1e27,  # Trap density (1/m³)
    nx=1001  # Number of spatial points
)
t_max = (simulator.Tf - simulator.T0) / simulator.beta
solution = simulator.simulate(t_max)

# Calculate desorption flux at surface
temps = simulator.T0 + simulator.beta * solution['time']

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(solution['temperature'], solution['flux'])
plt.xlabel('Temperature (K)')
plt.ylabel(r'{\sffamily Desorption Flux (1/m\textsuperscript{2}/s)}', usetex=True)
plt.title('Thermal Desorption Spectrum')
plt.grid(True)
plt.show()