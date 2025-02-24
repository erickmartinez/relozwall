import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class ThermalDesorptionSimulator:
    def __init__(self):
        # Physical parameters
        self.L = 1e-6  # Sample thickness (m)
        self.D0 = 1e-6  # Diffusion pre-exponential (m²/s)
        self.Ed = 1.4 # Diffusion activation energy (eV)
        self.k0 = 1e-15  # Trapping coefficient pre-exponential (m³/s)
        self.Et = 1.1  # Trapping energy (eV)
        self.p0 = 1e13  # Detrapping frequency factor (1/s)
        self.Eb = 1.1  # Binding energy (eV) for detrapping
        self.N = 1e28  # Trap density (1/m³)

        # Initial conditions
        self.T0 = 300  # Initial temperature (K)
        self.Tf = 1200  # Final temperature (K)
        self.beta = 0.5  # Heating rate (K/s)

        # Numerical parameters
        self.nx = 101  # Number of spatial points
        self.dX = 1.0 / (self.nx - 1)  # Grid spacing in normalized coordinates
        self.X = np.linspace(0, 1, self.nx)  # Normalized spatial grid
        self.x = self.X * self.L  # Physical spatial grid

        # Initial concentration profile (Gaussian)
        self.C0 = 1e24  # Peak concentration (1/m³)
        self.sigma = 0.1  # Width of Gaussian in normalized coordinates
        self.u0 = np.exp(-(self.X - 0.1) ** 2 / (2 * self.sigma ** 2))

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
        D = self.D(T)

        # Normalized parameters
        lambda_param = self.N * self.k(T) * self.L ** 2 / D
        nu_param = self.k(T) * self.C0 * self.L ** 2 / D
        mu_param = self.p(T) * self.L ** 2 / D

        return T, lambda_param, nu_param, mu_param

    def pde_system(self, t, y):
        """Define the PDE system for scipy solver"""
        nx = self.nx
        u = y[:nx]
        w = y[nx:]

        # Calculate parameters
        T, lambda_param, nu_param, mu_param = self.calculate_parameters(t)

        # Initialize derivative arrays
        dudt = np.zeros(nx)
        dwdt = np.zeros(nx)

        # Central differences for diffusion term (using normalized coordinate X)
        for i in range(1, nx - 1):
            dudt[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / self.dX ** 2

        # Boundary conditions (zero flux at boundaries)
        dudt[0] = 0
        dudt[-1] = 0

        # Trapping terms according to equation 8
        dwdt = lambda_param * u - nu_param * u * w - mu_param * w
        dudt -= dwdt

        return np.concatenate([dudt, dwdt])

    def get_initial_conditions(self):
        """Calculate initial equilibrium conditions"""
        T = self.T0
        k_val = self.k(T)
        p_val = self.p(T)

        # Initial trapped fraction from eq. 10
        n0 = k_val * self.u0 * self.C0 / (k_val * self.u0 * self.C0 + p_val)
        w0 = n0 * self.N / self.C0

        return np.concatenate([self.u0, w0])

    def calculate_flux(self, temps, solution):
        """Calculate desorption flux at the surface"""
        # Calculate flux using finite difference at the surface
        # Note: -1 index is the surface, -2 is one step inside
        u = solution.y[:self.nx]
        return -self.D(temps) * self.C0 * (u[0] - u[2]) / (2*self.dX * self.L)

    def get_concentrations(self, solution):
        """Convert normalized solutions to physical concentrations"""
        # Extract normalized solutions
        u = solution.y[:self.nx, :]  # mobile species
        w = solution.y[self.nx:, :]  # trapped species

        # Convert to physical concentrations
        C_mobile = u * self.C0  # mobile concentration
        C_trapped = w * self.C0  # trapped concentration

        return C_mobile, C_trapped

    def simulate(self, t_max):
        """Run the simulation"""
        # t_span = (0, t_max)
        y0 = self.get_initial_conditions()

        tau_max = self.D(self.Tf) * t_max / (self.L ** 2)
        t_span = (0, tau_max)
        dtau = self.dX / 2
        ntaus = int(tau_max // dtau + 1)
        print(f"tau_max: {tau_max}, dtau: {dtau:.3E}, ntaus: {ntaus}")
        taus = np.arange(ntaus) * dtau

        solution = solve_ivp(
            self.pde_system,
            t_span,
            y0,
            # method='BDF',
            method='LSODA',
            # t_eval=np.linspace(0, tau_max, 1000),
            t_eval=taus,
            rtol=1e-6,
            atol=1e-8,
        )

        # Calculate temperatures
        temps = self.T0 + self.beta * solution.t * (self.L ** 2) / self.D(self.Tf)

        # Calculate physical quantities
        C_mobile, C_trapped = self.get_concentrations(solution)
        fluxes = self.calculate_flux(temps, solution)

        return {
            'time': solution.t,
            'temperature': temps,
            'C_mobile': C_mobile,
            'C_trapped': C_trapped,
            'flux': fluxes
        }


# Run simulation
simulator = ThermalDesorptionSimulator()
t_max = (simulator.Tf - simulator.T0) / simulator.beta
solution = simulator.simulate(t_max)

# Calculate desorption flux at surface
temps = simulator.T0 + simulator.beta * solution['time']

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(solution['temperature'], solution['flux'])
plt.xlabel('Temperature (K)')
plt.ylabel('Desorption Flux (a.u.)')
plt.title('Thermal Desorption Spectrum')
plt.grid(True)
plt.show()