import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class TDSModel:
    # Physical constants
    kB = 8.617333262e-5  # Boltzmann constant (eV/K)

    def __init__(self,
                 L=2e-6,  # Sample thickness (m)
                 nx=50,  # Number of spatial points
                 D0=8.5e-8,  # Pre-exponential factor (m²/s)
                 Ed=0.85,  # Diffusion activation energy (eV)
                 Etrap=0.1,  # Trapping energy (eV)
                 c0=1e23,  # Initial concentration (atoms/m³)
                 ntrap=5e21,  # Trap density (traps/m³)
                 p0=1e13,  # Base trapping rate (s⁻¹)
                 beta=0.5,  # Heating rate (K/s)
                 T0=300,  # Initial temperature (K)
                 Tmax=1200):  # Maximum temperature (K)

        # Store physical parameters
        self.L = L
        self.nx = nx
        self.D0 = D0
        self.Ed = Ed
        self.Etrap = Etrap
        self.c0 = c0
        self.ntrap = ntrap
        self.p0 = p0
        self.beta = beta
        self.T0 = T0
        self.Tmax = Tmax
        self.kB = 8.617333262e-5  # Boltzmann constant (eV/K)

        # Set up normalized spatial grid
        self.dX = 1.0 / (nx - 1)
        self.X = np.linspace(0, 1, nx)

        # Reference diffusion coefficient at T0
        self.D_ref = self.D(T0)

        # Set up normalized time grid
        self.setup_time_grid()

    def D(self, T):
        """Temperature-dependent diffusion coefficient"""
        return self.D0 * np.exp(-self.Ed / (self.kB * T))

    def setup_time_grid(self):
        """Create normalized time grid"""
        # Physical end time
        t_max = (self.Tmax - self.T0) / self.beta

        # Normalized time parameters
        tau_max = self.D_ref * t_max / (self.L ** 2)

        # Choose number of time points based on stability criterion
        # dt_stability = 0.25 * self.dx**2 / D_max
        # n_points = min(int(tmax / dt_stability), 10000)  # Cap at 10000 points
        D_ratio_max = self.D(self.Tmax) / self.D_ref
        dtau = 0.25 * self.dX ** 2 / D_ratio_max  # Stability criterion
        n_points = min(int(tau_max / dtau) + 1, 10000)  # Cap number of points

        self.tau = np.linspace(0, tau_max, n_points)
        self.t = self.tau * self.L ** 2 / self.D_ref
        self.T = self.T0 + self.beta * self.t

        print(f"Number of time steps: {len(self.tau)}")
        print(f"Normalized time step: {self.tau[1] - self.tau[0]:.2e}")

    def normalized_rates(self, T):
        """Calculate normalized trapping/detrapping rates"""
        D_T = self.D(T)
        # ν = p*C0*L²/D
        nu = (self.p0 * np.exp(-0.1 / (self.kB * T)) *
              self.c0 * self.L ** 2 / D_T)
        # μ = q*L²/D
        mu = (self.p0 * np.exp(-self.Etrap / (self.kB * T)) *
              self.L ** 2 / D_T)
        return nu, mu

    def dudt(self, U, tau):
        """System of ODEs in normalized form
        U = [u, w] where u = Cm/C0, w = Ct/C0"""

        # Get temperature at current time
        t = tau * self.L ** 2 / self.D_ref
        T = self.T0 + self.beta * t
        if T > self.Tmax:
            return np.zeros_like(U)

        # Split U into mobile and trapped concentrations
        u = U[:self.nx]  # normalized mobile concentration
        w = U[self.nx:]  # normalized trapped concentration

        # Initialize derivatives
        dudt = np.zeros(self.nx)
        dwdt = np.zeros(self.nx)

        # Get normalized rates and diffusion ratio
        nu, mu = self.normalized_rates(T)
        D_ratio = self.D(T) / self.D_ref

        # Diffusion term (second order central difference)
        for i in range(1, self.nx - 1):
            dudt[i] = D_ratio * (u[i + 1] - 2 * u[i] + u[i - 1]) / self.dX ** 2

        # Boundary conditions (u = 0 at surfaces)
        # dudt[0] = 0
        dudt[-1] = 0

        # Trapping-detrapping terms
        theta_max = self.ntrap / self.c0  # Maximum trap occupancy
        for i in range(self.nx):
            theta = w[i] / theta_max
            theta = np.clip(theta, 0.0, 1.0)
            dwdt[i] = nu * (1 - theta) * u[i] - mu * theta
            dudt[i] -= dwdt[i]

        # Apply boundary conditions
        # Neumann for trapped concentration
        dwdt[0] = 0  # dw/dx = 0 at x=0
        dwdt[-1] = 0  # dw/dx = 0 at x=L
        #
        # # Dirichlet for mobile concentration
        # u[0] = u[-1] = 0
        # dudt[0] = dudt[-1] = 0
        dudt[-1] = 0

        return np.concatenate([dudt, dwdt])

    def simulate(self):
        """Run the TDS simulation in normalized form"""
        # Initial conditions (Gaussian profile)
        x_peak = 0.25  # Peak at L/4
        width = 0.15  # Width of 0.15*L
        U0 = np.zeros(2 * self.nx)
        U0[:self.nx] = np.exp(-(self.X - x_peak) ** 2 / (2 * width ** 2))
        U0[self.nx:] = 0.0

        # Solve the system
        self.solution = odeint(self.dudt, U0, self.tau)
                               # rtol=1e-4, atol=1e-4,
                               # mxstep=1000)

        # Calculate desorption flux
        self.flux = np.array([
            self.D(T) * self.c0 / self.L *
            (-3*self.solution[i, 0] + 4*self.solution[i, 1] - self.solution[i, 2]) / (2 * self.dX)
            for i, T in enumerate(self.T)
        ])

    def plot_results(self, save_plots=False):
        """Plot results in physical units"""
        # TDS spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(self.T, self.flux, 'b-', linewidth=2)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Desorption flux (m⁻²s⁻¹)')
        plt.title('TDS Spectrum of Deuterium from Boron')
        plt.grid(True)
        if save_plots:
            plt.savefig('tds_spectrum.png')
        plt.show()

        # Concentration profiles
        plt.figure(figsize=(12, 8))
        times = [0, len(self.t) // 4, len(self.t) // 2, 3 * len(self.t) // 4, -1]

        plt.subplot(2, 1, 1)
        for i in times:
            T = self.T[i]
            plt.plot(self.X * self.L * 1e6,
                     self.solution[i, :self.nx] * self.c0,
                     label=f'T = {T:.0f} K')
        plt.xlabel('Position (μm)')
        plt.ylabel('Mobile D concentration (m⁻³)')
        plt.title('Mobile Deuterium Profiles')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        for i in times:
            T = self.T[i]
            plt.plot(self.X * self.L * 1e6,
                     self.solution[i, self.nx:] * self.c0,
                     label=f'T = {T:.0f} K')
        plt.xlabel('Position (μm)')
        plt.ylabel('Trapped D concentration (m⁻³)')
        plt.title('Trapped Deuterium Profiles')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        if save_plots:
            plt.savefig('concentration_profiles.png')
        plt.show()


# Example usage
model = TDSModel()
model.simulate()
model.plot_results(save_plots=True)