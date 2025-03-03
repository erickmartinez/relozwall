import numpy as np
from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt


class StableTDSSimulator:
    """
    A numerically stable thermal desorption simulator implementing the Wilson-Baskes model.
    Designed to reproduce Figure 1 from Oya et al. 2021 paper with robust convergence.
    """
    kB = 8.617333262145179e-05  # Boltzmann constant in eV/K

    def __init__(
            self,
            T0=300,  # Initial temperature (K)
            Tf=1300,  # Final temperature (K)
            beta=0.5,  # Heating rate (K/s)
            D_W_ratio=2e-3,  # D/W ratio for trap sites
            trap_depth=2e-6,  # Maximum trap depth (m)
            Et=1.8,  # Detrapping energy (eV)
            L=1e-3,  # Sample thickness (m)
            nx=201,  # Number of spatial grid points
            v0=1e13,  # Attempt frequency (1/s)
            lambda_jump=3e-10,  # Jump distance (m)
            W_density=6.3e28,  # Tungsten atomic density (atoms/m³)
            verbose=True  # Print diagnostic information
    ):
        # Physical parameters
        self.T0 = T0
        self.Tf = Tf
        self.beta = beta
        self.L = L
        self.Et = Et
        self.D_W_ratio = D_W_ratio
        self.trap_depth = trap_depth
        self.v0 = v0
        self.lambda_jump = lambda_jump
        self.W_density = W_density
        self.verbose = verbose

        # Create a non-uniform mesh with refinement near the surface
        self.nx = nx
        self.setup_mesh()

        # Calculate trap concentration from D/W ratio
        self.trap_conc = self.D_W_ratio * self.W_density

        # Initialize concentration profiles
        self.setup_initial_conditions()

        # For stability assessment
        self.zero_threshold = 1e-10

        if self.verbose:
            # Print simulation parameters for diagnostics
            self.print_parameters()

    def setup_mesh(self):
        """
        Create a non-uniform spatial mesh with refinement near x=0 where gradients
        are expected to be steepest.
        """
        # Create stretching function for non-uniform mesh
        # More points near x=0, fewer points at large x
        s = np.linspace(0, 1, self.nx) ** 1.5  # Stretch parameter controls non-uniformity

        self.x = self.L * s

        # Calculate finite difference spacing parameters
        self.dx = np.zeros(self.nx)
        for i in range(self.nx - 1):
            self.dx[i] = self.x[i + 1] - self.x[i]
        self.dx[-1] = self.dx[-2]  # Extrapolate last spacing

        if self.verbose:
            print(f"Spatial grid: {self.nx} points")
            print(f"Min dx: {np.min(self.dx):.2e} m, Max dx: {np.max(self.dx):.2e} m")

    def D(self, T):
        """
        Frauenfelder diffusivity for deuterium in tungsten (modified from H).
        D = 2.9e-7 * exp(-0.39/kT) m²/s
        """
        return 2.9e-7 * np.exp(-0.39 / (self.kB * T))

    def D_ratio(self, T):
        """Calculate the ratio of diffusivity at temperature T to maximum diffusivity"""
        D_max = self.D(self.Tf)
        return self.D(T) / D_max

    def K_r(self, T):
        """
        Anderl recombination coefficient.
        Kr = 3.2e-15 * exp(-1.16/kT) m³/s
        """
        return 3.2e-15 * np.exp(-1.16 / (self.kB * T))

    def setup_initial_conditions(self):
        """Set up the initial concentration profiles"""
        # Initialize arrays for mobile and trapped concentrations
        self.C_mobile = np.zeros(self.nx)
        self.C_trapped = np.zeros(self.nx)

        # Find the index corresponding to the trap depth
        trap_idx = np.searchsorted(self.x, self.trap_depth)
        trap_idx = min(trap_idx, self.nx - 1)

        # Set uniform trap concentration up to the trap depth
        self.C_trapped[:trap_idx + 1] = self.trap_conc

        # Total initial trapping sites (uniform throughout material)
        self.trap_sites = np.ones(self.nx) * self.trap_conc

        # Calculate total trapped deuterium (atoms/m²)
        total_trapped = simpson(y=self.C_trapped, x=self.x)

        if self.verbose:
            print(f"Trap concentration: {self.trap_conc:.2e} traps/m³")
            print(f"Total trapped deuterium: {total_trapped:.2e} atoms/m²")
            print(f"Trap depth: {self.trap_depth * 1e6:.1f} μm")
            print(f"Trap depth index: {trap_idx}/{self.nx}")

    def print_parameters(self):
        """Print simulation parameters for diagnostics"""
        print("\n=== Simulation Parameters ===")
        print(f"Temperature range: {self.T0} K - {self.Tf} K")
        print(f"Heating rate: {self.beta} K/s")
        print(f"Detrapping energy: {self.Et} eV")
        print(f"D/W ratio: {self.D_W_ratio:.2e}")

        # Calculate important physical scales
        t_sim = (self.Tf - self.T0) / self.beta
        D_max = self.D(self.Tf)
        D_min = self.D(self.T0)
        diffusion_length = np.sqrt(2 * D_max * t_sim)

        print("\n=== Physical Scales ===")
        print(f"Simulation time: {t_sim:.1f} s")
        print(f"Diffusivity range: {D_min:.2e} - {D_max:.2e} m²/s")
        print(f"Maximum diffusion length: {diffusion_length * 1e6:.1f} μm")

        # Calculate stability parameters
        min_dx = np.min(self.dx)
        dt_diff_stability = 0.5 * min_dx ** 2 / D_max

        print("\n=== Numerical Stability ===")
        print(f"Minimum grid spacing: {min_dx * 1e6:.2f} μm")
        print(f"Diffusion stability limit (dt): {dt_diff_stability:.2e} s")
        print(f"Number of grid points: {self.nx}")
        print("=" * 30 + "\n")

    def get_temperature(self, t):
        """Calculate the temperature at time t using linear heating with rate beta."""
        return self.T0 + self.beta * t

    def pde_system(self, t, y):
        """
        Define the coupled PDE system for diffusion, trapping, and detrapping.
        Implements the Wilson-Baskes model with numerically stable discretization.
        """
        # Extract concentrations from state vector
        nx = self.nx
        C_m = y[:nx]  # Mobile concentration
        C_t = y[nx:]  # Trapped concentration

        # Current temperature
        T = self.get_temperature(t)

        # Current diffusivity
        D_t = self.D(T)

        # Initialize derivatives
        dCm_dt = np.zeros(nx)
        dCt_dt = np.zeros(nx)

        # === Interior Points: Non-uniform mesh diffusion ===
        for i in range(1, nx - 1):
            # Forward and backward spacing
            dx_forward = self.x[i + 1] - self.x[i]
            dx_backward = self.x[i] - self.x[i - 1]
            dx_center = 0.5 * (dx_forward + dx_backward)

            # Calculate fluxes using non-uniform mesh
            flux_right = D_t * (C_m[i + 1] - C_m[i]) / dx_forward
            flux_left = D_t * (C_m[i] - C_m[i - 1]) / dx_backward

            # Update based on flux balance
            dCm_dt[i] = (flux_left - flux_right) / dx_center

        # === Boundary Conditions ===
        # Left boundary (x=0): Recombination boundary condition
        dCm_dt[0] = 2.0 * self.K_r(T) * C_m[0] ** 2 / D_t

        # Right boundary (x=L): No-flux condition
        # Use backward difference for better stability
        dx_last = self.x[-1] - self.x[-2]
        dCm_dt[-1] = D_t * (C_m[-2] - C_m[-1]) / dx_last ** 2

        # === Trapping/Detrapping Terms ===
        # Available trapping sites
        open_traps = np.maximum(0, self.trap_sites - C_t)  # Prevent negative values

        # Trapping rate: mobile → trapped
        trapping_term = D_t * C_m * open_traps / (self.lambda_jump ** 2 * self.W_density)

        # Detrapping rate: trapped → mobile
        detrapping_term = self.v0 * C_t * np.exp(-self.Et / (self.kB * T))

        # Net change in trapped concentration
        dCt_dt = trapping_term - detrapping_term

        # Adjust mobile concentration rate accordingly
        dCm_dt -= dCt_dt

        return np.concatenate([dCm_dt, dCt_dt])

    def concentrations_depleted(self, t, y):
        """Event function that returns 0 when total concentration is effectively zero"""
        total_inventory = np.sum(np.abs(y))
        return total_inventory - self.zero_threshold

    def simulate_tds(self):
        """
        Run TDS simulation with robust numerical methods.
        Uses adaptive time stepping and tight error control.
        """
        if self.verbose:
            print(f"Running TDS simulation with {self.Et:.2f} eV trap...")

        # Set initial conditions
        y0 = np.concatenate([self.C_mobile, self.C_trapped])

        # Calculate simulation time
        t_max = (self.Tf - self.T0) / self.beta

        # Create evaluation points with emphasis on desorption peak regions
        # Use non-uniform sampling to better resolve peaks
        s = np.linspace(0, 1, 1000) ** 0.8  # More points in lower temperature region
        t_eval = s * t_max

        # Run the solver with tight error control
        solution = solve_ivp(
            self.pde_system,
            (0, t_max),
            y0,
            method='LSODA',  # LSODA automatically handles stiff/non-stiff transitions
            t_eval=t_eval,
            rtol=1e-8,  # Tight relative tolerance
            atol=1e-10,  # Tight absolute tolerance
            max_step=t_max / 100,  # Limit maximum step size
            events=self.concentrations_depleted
        )

        if self.verbose:
            print(f"Solver status: {solution.message}")
            print(f"Number of function evaluations: {solution.nfev}")
            print(f"Number of time steps: {len(solution.t)}")

        # Extract results
        time = solution.t
        temp = self.get_temperature(time)
        C_m = solution.y[:self.nx, :]
        C_t = solution.y[self.nx:, :]

        # Calculate desorption flux (2*Kr*C²)
        flux = np.zeros_like(time)
        for i, t in enumerate(time):
            T = self.get_temperature(t)
            flux[i] = 2.0 * self.K_r(T) * C_m[0, i] ** 2

        # Calculate total retention at each time point
        retention = np.zeros_like(time)
        for i in range(len(time)):
            retention[i] = simpson(C_t[:, i], self.x)

        return {
            'time': time,
            'temperature': temp,
            'flux': flux,
            'C_mobile': C_m,
            'C_trapped': C_t,
            'retention': retention,
            'success': solution.success
        }

    def plot_tds_spectrum(self, results, title=None):
        """Plot the TDS spectrum (temperature vs. flux)"""
        plt.figure(figsize=(10, 6))
        plt.plot(results['temperature'], results['flux'], 'b-', linewidth=2)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Desorption Flux (atoms/m²/s)')

        if title:
            plt.title(title)
        else:
            plt.title(f'TDS Spectrum - Et={self.Et} eV, D/W={self.D_W_ratio:.1e}, Depth={self.trap_depth * 1e6:.1f} μm')

        plt.grid(True)
        plt.tight_layout()
        return plt.gcf()

    def plot_concentration_profiles(self, results, num_profiles=5):
        """Plot concentration profiles at different times during heating"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Select evenly spaced time indices
        n_times = len(results['time'])
        indices = np.linspace(0, n_times - 1, num_profiles, dtype=int)

        # Plot mobile concentration profiles
        for idx in indices:
            T = results['temperature'][idx]
            t = results['time'][idx]
            ax1.plot(self.x * 1e6, results['C_mobile'][:, idx],
                     label=f'T={T:.0f}K, t={t:.0f}s')

        ax1.set_xlabel('Depth (μm)')
        ax1.set_ylabel('Mobile Concentration (atoms/m³)')
        ax1.set_yscale('log')
        ax1.set_title('Mobile D Concentration Profiles')
        ax1.legend()
        ax1.grid(True)

        # Plot trapped concentration profiles
        for idx in indices:
            T = results['temperature'][idx]
            ax2.plot(self.x * 1e6, results['C_trapped'][:, idx],
                     label=f'T={T:.0f}K')

        ax2.set_xlabel('Depth (μm)')
        ax2.set_ylabel('Trapped Concentration (atoms/m³)')
        ax2.set_title('Trapped D Concentration Profiles')
        ax2.set_yscale('log')
        ax2.grid(True)

        plt.tight_layout()
        return fig


# Function to reproduce Figure 1 from Oya paper with robust convergence
def reproduce_figure1_stable(extra_parameters=None):
    """
    Reproduce Figure 1 from Oya's paper with cases 2, 3, and 4 using
    the numerically stable simulator.

    Args:
        extra_parameters: Optional dict of additional parameters to pass to simulators
    """
    plt.figure(figsize=(12, 8))

    # Default parameters for all cases
    base_params = {
        'nx': 301,
        'verbose': True
    }

    # Add any extra parameters provided
    if extra_parameters:
        base_params.update(extra_parameters)

    # Case 2: Et = 1.8 eV, D/W = 2e-3, depth = 2 μm
    sim_case2 = StableTDSSimulator(
        Et=1.8,
        D_W_ratio=2e-3,
        trap_depth=2e-6,
        **base_params
    )
    results_case2 = sim_case2.simulate_tds()

    if results_case2['success']:
        plt.plot(results_case2['temperature'], results_case2['flux'], 'r-',
                 label='Case 2: Et=1.8 eV, D/W=2e-3, depth=2 μm')
    else:
        print("Case 2 simulation did not converge")

    # Case 3: Et = 1.8 eV, D/W = 2e-3, depth = 25 μm
    sim_case3 = StableTDSSimulator(
        Et=1.8,
        D_W_ratio=2e-3,
        trap_depth=25e-6,
        **base_params
    )
    results_case3 = sim_case3.simulate_tds()

    if results_case3['success']:
        plt.plot(results_case3['temperature'], results_case3['flux'], 'g-',
                 label='Case 3: Et=1.8 eV, D/W=2e-3, depth=25 μm')
    else:
        print("Case 3 simulation did not converge")

    # Case 4: Et = 1.8 eV, D/W = 2e-4, depth = 25 μm
    sim_case4 = StableTDSSimulator(
        Et=1.8,
        D_W_ratio=2e-4,
        trap_depth=25e-6,
        **base_params
    )
    results_case4 = sim_case4.simulate_tds()

    if results_case4['success']:
        plt.plot(results_case4['temperature'], results_case4['flux'], 'b-',
                 label='Case 4: Et=1.8 eV, D/W=2e-4, depth=25 μm')
    else:
        print("Case 4 simulation did not converge")

    plt.xlabel('Temperature (K)')
    plt.ylabel('Desorption Flux (atoms/m²/s)')
    plt.title('Reproduction of Figure 1 from Oya et al. 2021')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return plt.gcf()


# Function to run a single TDS simulation with detailed diagnostics
def run_single_simulation(Et=1.8, D_W_ratio=2e-3, trap_depth=2e-6, **kwargs):
    """
    Run a single TDS simulation with detailed output for debugging.

    Args:
        Et: Detrapping energy in eV
        D_W_ratio: D/W ratio
        trap_depth: Trap depth in meters
        **kwargs: Additional parameters to pass to the simulator

    Returns:
        simulator: The simulator object
        results: The simulation results
    """
    simulator = StableTDSSimulator(
        Et=Et,
        D_W_ratio=D_W_ratio,
        trap_depth=trap_depth,
        verbose=True,
        **kwargs
    )

    results = simulator.simulate_tds()

    if results['success']:
        print("\nSimulation successful!")
        # Plot the TDS spectrum
        simulator.plot_tds_spectrum(results)
        plt.figure()

        # Plot concentration profiles
        simulator.plot_concentration_profiles(results)

        # Plot the retention
        plt.figure(figsize=(8, 5))
        plt.plot(results['temperature'], results['retention'])
        plt.xlabel('Temperature (K)')
        plt.ylabel('Total D Retention (atoms/m²)')
        plt.title('D Retention vs. Temperature')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print("\nSimulation failed to converge.")

    return simulator, results


if __name__ == "__main__":
    # Run a test simulation with a single trap
    run_single_simulation(
        Et=1.8,  # Detrapping energy (eV)
        D_W_ratio=2e-3,  # D/W ratio
        trap_depth=2e-6,  # Trap depth (m)
        nx=301,  # Number of grid points
        L=1e-3  # Sample thickness (m)
    )

    # Reproduce Figure 1 with robust convergence
    # reproduce_figure1_stable()