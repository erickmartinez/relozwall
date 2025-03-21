"""
Author: Erick Martinez Loran
erickrmartinez@gmail.com
"""
import numpy as np
from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt
import scipy.stats.distributions as distributions
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, OptimizeResult
import data_processing.confidence as cf
from data_processing.heat_equation_1d_rod_with_laser_pulse import debug


def model_poly(x, b) -> np.ndarray:
    n = len(b)
    r = np.zeros(len(x))
    for i in range(n):
        r += b[i] * x ** i
    return r


def res_poly(b, x, y, w=1.):
    return (model_poly(x, b) - y) * w


def jac_poly(b, x, y, w=1):
    n = len(b)
    r = np.zeros((len(x), n))
    for i in range(n):
        r[:, i] = w * x ** i
    return r

def fit_polynomial(x, y, weights=1, poly_order:int=10, loss:str= 'soft_l1', f_scale:float=1.0, tol:float=None) -> OptimizeResult:
    """
    Fits the curve to a polynomial function
    Parameters
    ----------
    x: np.ndarray
        The x values
    y: np.ndarray
        The y values
    weights: float
        The weights for the residuals
    poly_order: int
        The degree of the polynomial to be used
    loss: str
        The type of loss to be used
    f_scale: float
        The scaling factor for the outliers
    tol: float
        The tolerance for the convergence

    Returns:
    -------
    OptimizeResult:
        The least squares optimized result
    """
    if tol is None:
        tol = float(np.finfo(np.float64).eps)

    x0 = [(0.01) ** (i-1) for i in range(poly_order+1)]
    try:
        ls_res: OptimizeResult = least_squares(
            res_poly,
            x0=x0,
            args=(x, y, weights),
            loss=loss, f_scale=f_scale,
            jac=jac_poly,
            xtol=tol,
            ftol=tol,
            gtol=tol,
            verbose=0,
            x_scale='jac',
            method='trf',
            max_nfev=10000 * poly_order
        )
    except ValueError as e:
        for xi, yi, r in zip(x, y, res_poly(b=x0, x=x, y=y)):
            print(xi, yi, r)
        raise e

    return ls_res

class ThermalDesorptionSimulator:
    kB = 8.617333262145179e-05  # Boltzmann constant in eV/K

    def __init__(
            self,
            T0=300,  # Initial temperature (K)
            Tf=1200,  # Final temperature (K)
            beta=0.3,  # Heating rate (K/s)
            trap_filling=6.5E22,  # Total retained deuterium in traps (1/m²)
            L=6.77E-6,  # Sample thickness (m)
            D0=2.3E-8,  # Diffusion pre-exponential (m²/s)
            Ed=1.0,  # Diffusion activation energy (eV)
            Et=2.2,  # Trapping energy (eV)
            lam=1E-9,  # The jump distance
            v0=1e13,  # attempt frequency for detrapping (1/s)
            density_host=2.31,  # Trap density (g/cm³)
            atomic_mass_host=10.811,  # g/mol
            gamma_ampl=2.0598166,
            gamma_shape=6.1870771,
            gamma_scale=2.43103442,
            gamma_loc=-5.04063028,
            kr=3.2e-15,  # Recombination coefficient (m⁴/s)
            Er=1.16,  # The recombination energy (eV)
            nx=201,  # Number of spatial points
            adapt_mesh=True,  # Enable adaptive meshing
            debug=False
    ):
        # Physical parameters
        self.L = L
        self.D0 = D0
        self.Ed = Ed
        self.kr = kr
        self.Er = Er
        self.Et = Et
        self.lam = lam
        self.v0 = v0
        self.N = self.get_atomic_density(density_host, atomic_mass_host)
        self.trap_filling = trap_filling # Total retained deuterium in traps (1/m²)

        # Initial conditions and numerical parameters
        self.T0 = T0
        self.Tf = Tf
        self.beta = beta

        # Enhanced spatial discretization
        self.adapt_mesh = adapt_mesh
        self.nx = nx

        self.debug= debug

        # Set up the mesh (potentially non-uniform)
        self.setup_mesh()

        # Initial concentration profile
        self.gamma_ampl = gamma_ampl
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.gamma_loc = gamma_loc
        C0, w0 = self.estimate_CT0()
        self.C0 = min(C0, 0.5*self.N)
        self.w0 = w0
        self.theta_0 = np.ones_like(self.x)

        # Threshold for considering concentration effectively zero
        self.zero_threshold = 1e-10


        if self.debug:
            # Calculate the expected diffusion length for adaptive time stepping
            self.diffusion_length = np.sqrt(self.D(self.Tf) * (self.Tf - self.T0) / self.beta)
            print(f"Expected diffusion length at max temp: {self.diffusion_length:.2E} m")
            print(f"Sample thickness: {self.L:.2E} m")
            print(f"Atomic density: {self.N:.2E} atoms/m³")
            print(f"C0: {self.C0:.2E} atoms/m³")

        integrated_implanted_profile = simpson(y=w0, x=self.x) * C0
        if self.debug:
            print(f"Desorption from traps: {integrated_implanted_profile:.2E} 1/m²")

        # The interpolating functions for desorption rate at temperature are not initially defined
        self.desorption_at_temperature = None

    def set_experimental_tds(self, temperature, desorption_flux):
        """
        Generates interpolating functions for the desorption rate as a function of temperature

        Parameters
        ----------
        temperature: np.ndarray
            The desorption temperature (K)
        desorption_flux: np.ndarray
            The desorption rate (atoms/m^2/s)
        """
        self.desorption_at_temperature = interp1d(x=temperature, y=desorption_flux, bounds_error=False, fill_value=0.0)


    def setup_mesh(self):
        """Set up a three-part spatial mesh with refinement near boundaries and uniform in the center"""
        if self.adapt_mesh:
            # Parameters to control mesh properties
            alpha = 3.0  # Controls mesh density near boundaries
            boundary_frac = 0.4  # Fraction of mesh points allocated to each boundary region

            # Total number of points
            n_total = self.nx

            # Number of points in each region
            n_left = int(n_total * boundary_frac)
            n_right = int(n_total * boundary_frac)
            n_center = n_total - n_left - n_right

            # Left boundary region (concentrated near x=0)
            s_left = np.linspace(0, 1, n_left)
            X_left = (s_left ** alpha) * (boundary_frac)

            # Center region (uniform mesh)
            X_center = np.linspace(boundary_frac, 1 - boundary_frac, n_center)

            # Right boundary region (concentrated near x=L)
            s_right = np.linspace(0, 1, n_right)
            X_right = 1 - (1 - s_right) ** alpha * boundary_frac

            # Combine all regions
            self.X = np.unique(np.concatenate([X_left, X_center, X_right]))
            self.nx = len(self.X)

            if self.debug:
                print(f"Mesh points: Left boundary={n_left}, Center={n_center}, Right boundary={n_right}")
                print(f"Total mesh points after combining: {self.nx}")
        else:
            # Uniform mesh
            self.X = np.linspace(0, 1, self.nx)

        # Physical coordinate and mesh spacing
        self.x = self.X * self.L

        # Calculate local mesh spacing for finite difference
        self.dX = np.zeros(self.nx)
        self.dX[1:-1] = (self.X[2:] - self.X[:-2]) / 2  # Central differences
        self.dX[0] = self.X[1] - self.X[0]  # Forward difference
        self.dX[-1] = self.X[-1] - self.X[-2]  # Backward difference

        # Compute grid spacing for second derivatives (non-uniform mesh)
        self.dx_forward = np.zeros(self.nx - 1)
        self.dx_backward = np.zeros(self.nx - 1)
        self.dx_forward = self.X[1:] - self.X[:-1]
        self.dx_backward = np.roll(self.dx_forward, 1)
        self.dx_backward[0] = self.dx_forward[0]

        if self.debug:
            print(f"Using {'adaptive' if self.adapt_mesh else 'uniform'} mesh with {self.nx} points")
            print(f"Min dX: {np.min(self.dx_forward):.2E}, Max dX: {np.max(self.dx_forward):.2E}")

            # Additional debug info for adaptive mesh
            if self.adapt_mesh:
                left_region = self.X[self.X <= X_left.max()]
                center_region = self.X[(self.X > X_left.max()) & (self.X < X_right.min())]
                right_region = self.X[self.X >= X_right.min()]

                print(f"Left region (X≤{boundary_frac:.1f}): {len(left_region)} points")
                print(f"Center region ({boundary_frac:.1f}<X<{1 - boundary_frac:.1f}): {len(center_region)} points")
                print(f"Right region (X≥{1 - boundary_frac:.1f}): {len(right_region)} points")

    @staticmethod
    def get_atomic_density(density, atomic_mass):
        """Calculate atomic density from mass density"""
        N_A = 6.02214076e+23  # Avogadro's number
        return density / atomic_mass * N_A * 1E6  # Convert to atoms/m³

    def estimate_CT0(self):
        """Estimate initial trapped concentration profile"""
        # Use gamma distribution for implantation profile or uniform distribution
        if hasattr(self, 'gamma_ampl') and self.gamma_ampl > 0 and 1==0:
            distribution = self.gamma_ampl * distributions.gamma.pdf(
                self.x * 1E10, self.gamma_shape, loc=self.gamma_loc, scale=self.gamma_scale
            )
            s = simpson(y=distribution, x=self.x)
            w0 = distribution
            # Normalize to match total retention
            nD = self.trap_filling / s
        else:
            # Uniform distribution
            w0 = np.ones_like(self.x)
            # For uniform distribution, scale by length
            nD = self.trap_filling / self.L

        return nD, w0

    def D(self, T):
        """Temperature-dependent diffusion coefficient"""
        return self.D0 * np.exp(-self.Ed / (self.kB * T))

    def K_r(self, T):
        """Temperature-dependent recombination coefficient"""
        return self.kr * np.exp(-self.Er / (self.kB * T))

    def calculate_parameters(self, t):
        """Calculate normalized parameters at given time"""
        T = self.T0 + self.beta * t
        D_ref = self.D(self.Tf)  # Reference diffusion coefficient
        D_t = self.D(T)  # Current diffusion coefficient

        # Modified normalized parameters
        mu_param = self.C0 * self.L ** 2 / self.N / (self.lam ** 2)
        nu_param = self.v0 * self.L ** 2 / D_ref

        D_ratio = D_t / D_ref  # Normalized diffusion coefficient

        return T, D_ratio, mu_param, nu_param

    def pde_system(self, tau, y):
        """Define the PDE system with temperature-dependent diffusion"""
        nx = self.nx
        u = y[:nx]  # Mobile concentration
        w = y[nx:]  # Trapped concentration

        # Calculate parameters including D_ratio
        t = tau * (self.L ** 2) / self.D(self.Tf)
        T, D_ratio, mu_param, nu_param = self.calculate_parameters(t)

        # Initialize derivative arrays
        dudt = np.zeros(nx)
        dwdt = np.zeros(nx)

        # Standard central finite difference for interior points
        # Since D_ratio is not a function of x, standard finite difference is appropriate
        for i in range(1, nx - 1):
            # For non-uniform mesh, we need to adjust the central difference formula
            # Following the Taylor series expansion for non-uniform grid
            dx_left = self.X[i] - self.X[i - 1]
            dx_right = self.X[i + 1] - self.X[i]

            # Second derivative formula for non-uniform mesh
            dudt[i] = D_ratio * 2.0 * (
                    (dx_left * u[i + 1] - (dx_left + dx_right) * u[i] + dx_right * u[i - 1]) /
                    (dx_left * dx_right * (dx_left + dx_right))
            )



        # Trapping/detrapping terms
        dwdt = mu_param * D_ratio * u * (self.theta_0 - w) - nu_param * w * np.exp(-self.Et / (self.kB * T))
        dudt -= dwdt

        # Left boundary (x=0): Recombination boundary condition
        # Following the derivation: J = -D ∂C/∂x = -2*Kr*C^2 at x=0
        # The flux balance at the boundary should be:
        # D * (∂C/∂x) = 2 * Kr * C^2

        dx_first = self.X[1] - self.X[0]
        # First calculate a ghost point value that would satisfy the flux boundary condition
        # If we use central difference: (u[1] - u[-1])/(2*dx) = 2*Kr*u[0]^2/D
        # Then u[-1] = u[1] - 4*dx*Kr*u[0]^2/D

        K_r_T = self.K_r(T)
        D_T = self.D(T)
        if self.desorption_at_temperature is None:
            ghost_val = u[1] - 4 * self.L * self.C0 * dx_first * K_r_T * u[0] ** 2 / D_T
        else:
            ghost_val = u[1] - 2 * self.L * dx_first * self.desorption_at_temperature(T) / D_T / self.C0

        # Now use this ghost point in the central difference formula for second derivative
        dudt[0] = D_ratio * (u[1] - 2 * u[0] + ghost_val) / (dx_first ** 2)

        # Right boundary (x=L): Symmetry boundary condition (∂u/∂x = 0)
        # Using a forward difference approximation
        dx_last = self.X[-1] - self.X[-2]
        dudt[-1] = D_ratio * (2.0 * u[-2] - 2.0 * u[-1]) / (dx_last ** 2)



        return np.concatenate([dudt, dwdt])

    def concentrations_depleted(self, t, y):
        """Event function that returns 0 when total integrated concentration is effectively zero"""
        # Sum up all mobile and trapped concentrations
        total_inventory = np.sum(np.abs(y))
        return total_inventory - self.zero_threshold

    def get_initial_conditions(self):
        """Calculate initial equilibrium conditions"""
        # Initial mobile concentration is zero
        u0 = np.zeros_like(self.x)
        return np.concatenate([u0, self.w0])

    def simulate(self, t_max=None):
        """Run the simulation"""
        if t_max is None:
            t_max = (self.Tf - self.T0) / self.beta

        y0 = self.get_initial_conditions()

        # Calculate appropriate time scaling
        tau_max = self.D(self.Tf) * t_max / (self.L ** 2)
        if self.debug:
            print(f"tau_max: {tau_max:.3E}")

        # Adaptive time stepping based on the courant condition
        # The smallest possible time step for stability
        min_dx = np.min(self.dx_forward)
        dtau_min = (min_dx ** 2) / 2.0

        # Choose a reasonable number of time steps - between stability limit and reasonable upper bound
        ntaus_min = max(int(tau_max / dtau_min), 100)
        ntaus_max = 1001  # Upper limit to prevent excessive computation
        ntaus = min(ntaus_min, ntaus_max)

        dtau = tau_max / (ntaus - 1)
        if self.debug:
            print(f"Using {ntaus} time steps with dtau: {dtau:.6E} (stability limit: {dtau_min:.6E})")

        # Create evaluation points with higher density at lower temperatures
        # where the desorption peaks typically occur
        s = np.linspace(0, 1, ntaus)
        s_eval = s ** 0.6  # Concentrate points at lower temperatures
        taus = s_eval * tau_max

        # Configure solver with appropriate tolerances
        solution = solve_ivp(
            self.pde_system,
            (0., tau_max),
            y0,
            method='Radau',  # Adaptive method that handles stiff systems well
            t_eval=taus,
            rtol=1e-9,  # Relative tolerance
            atol=1e-11,  # Absolute tolerance
            # max_step=tau_max / 50,  # Maximum step size
            # events=self.concentrations_depleted,  # Add event detection
        )

        # Calculate temperatures and physical quantities
        temps = self.T0 + self.beta * solution.t * (self.L ** 2) / self.D(self.Tf)
        C_M = solution.y[:self.nx, :] * self.C0
        C_T = solution.y[self.nx:, :] * self.C0


        kr, Er, kr_delta, Er_delta, fit_result = None, None, None, None, None
        if self.desorption_at_temperature :
            kr, Er, kr_delta, Er_delta, fit_result = self.fit_Kr(temps, solution.y[:self.nx, :])

        # Calculate flux using recombination rate K
        flux = np.zeros(len(temps))
        for i, T in enumerate(temps):
            flux[i] = 2. * self.K_r(T) * C_M[0, i] ** 2.

        result = {
            'time': solution.t * (self.L ** 2) / self.D(self.Tf),  # Convert to physical time
            'temperature': temps,
            'C_M': C_M,
            'C_T': C_T,
            'flux': flux,
            'terminated_early': len(solution.t_events[0]) > 0 if solution.t_events else False
        }

        if self.desorption_at_temperature:
            result['KR fit result'] = {
                'kr': kr, 'kr_delta': kr_delta,
                'Er': Er, 'Er_delta': Er_delta,
                'least_squares_result': fit_result
            }

        return result

    def fit_Kr(self, temperature, u):
        x = -1. / temperature / self.kB
        # Get the desorption flux from the interpolated value
        J_TDS = self.desorption_at_temperature(temperature)
        u0 = u[0,:]

        msk_pos = (J_TDS > 0) & (u0 > 0)
        J_TDS = J_TDS[msk_pos]
        u0 = u0[msk_pos]
        y = np.log(J_TDS) - 2 * np.log(u0)
        x = x[msk_pos]


        fit_result = fit_polynomial(x=x, y=y, poly_order=1, loss='linear', f_scale=0.1)
        popt = fit_result.x
        if self.debug:
            print('popt:', popt)
        try:
            ci = cf.confidence_interval(fit_result)
        except IndexError as e:
            print("len(x):", len(x), "len(y):", len(y))
            raise e
        delta = np.abs(ci[:, 1] - popt)
        # log_kr = popt[0] - np.log(2)
        kr = np.exp(popt[0]) / 2 / (self.C0 ** 2)
        kr_delta = kr * delta[0]

        Er = popt[1]
        Er_delta = delta[1]

        if self.debug:
            print(f"kr: {kr:.3E} ± {kr_delta:.3E} m^4/2, Er: {Er:.3E} ± {Er_delta:.3E} eV")

        self.kr = kr
        self.Er = Er
        return kr, Er, kr_delta, Er_delta, fit_result


    def plot_results(self, solution, save_path=None):
        """Plot the simulation results"""
        # Create figure with 2 subplots: TDS spectrum and concentration profiles
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot TDS spectrum
        ax1.plot(solution['temperature'], solution['flux'], 'b-', linewidth=2, marker='o', ms=3, mfc='None')
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Desorption Flux (1/m²/s)')
        ax1.set_title('Thermal Desorption Spectrum')
        ax1.grid(True)

        # Plot concentration profiles at selected points
        n_profiles = 5
        indices = np.linspace(0, len(solution['time']) - 1, n_profiles, dtype=int)

        for i in indices:
            T = solution['temperature'][i]
            ax2.plot(self.x * 1e6, solution['C_M'][:, i], label=f'Mobile, T={T:.0f}K', marker='o', ms=3, mfc='None')
            ax2.plot(self.x * 1e6, solution['C_T'][:, i], '--', label=f'Trapped, T={T:.0f}K', marker='o', ms=3, mfc='None')

        ax2.set_xlabel('Position (μm)')
        ax2.set_ylabel('Concentration (1/m³)')
        ax2.set_title('Concentration Profiles')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)

        return fig, ax1, ax2


# Example usage:
def run_simulation(save_plot=False):
    # simulator = ThermalDesorptionSimulator(
    #     T0=300,  # Initial temperature (K)
    #     Tf=1300,  # Final temperature (K)
    #     # beta=0.5,  # Heating rate (K/s)
    #     # trap_filling=3.16E21,  # Total retained deuterium (1/m²)
    #     # L=25e-6,  # Sample thickness (m)
    #     # D0=2.9e-7,  # Diffusion pre-exponential (m²/s)
    #     # Ed=0.39,  # Diffusion activation energy (eV)
    #     # kr=3.2e-15,  # Recombination coefficient (m³/s)
    #     # Er=1.16,
    #     # Et=1.8,  # Trapping energy (eV)
    #     # lam=0.3E-9,  # Jump distance (m)
    #     # v0=1e13,  # Attempt frequency (1/s)
    #     # density_host=19.3,  # Host density (g/cm³)
    #     # atomic_mass_host=183.84,  # Atomic mass (g/mol)
    #     nx=51,  # Number of spatial points
    #     adapt_mesh=True,  # Use adaptive mesh
    #     debug=True
    # )

    TDS_FILE_SBR = r'./data/TDS/20250213/Brod_mks_v3.txt'
    import pandas as pd
    sintered_boron_rod_df = pd.read_csv(TDS_FILE_SBR, comment='#', delimiter=r'\s+').apply(pd.to_numeric)
    sintered_boron_rod_df = sintered_boron_rod_df[sintered_boron_rod_df['[D2/m^2/s]'] > 0]
    temperature_sbr = sintered_boron_rod_df['Temp[K]'].values
    desorption_flux = sintered_boron_rod_df['[D2/m^2/s]'].values
    # desorption_flux_min_pos = desorption_flux[desorption_flux>0].min()
    # desorption_flux[desorption_flux<=0] = float(np.finfo(float).eps)

    simulator = ThermalDesorptionSimulator(
        T0=temperature_sbr.min(),  # Initial temperature (K)
        Tf=temperature_sbr.max(),  # Final temperature (K)
        # beta=0.5,  # Heating rate (K/s)
        # trap_filling=3.24E+20,  # Total retained deuterium (1/m²)
        trap_filling=5.5E+22,  # Total retained deuterium (1/m²)
        L=4.3E-06,  # Sample thickness (m)
        D0=2.3e-8,  # Diffusion pre-exponential (m²/s)
        Ed=1.,  # Diffusion activation energy (eV)
        # kr=3.2e-15,  # Recombination coefficient (m³/s)
        # Er=1.16,
        # Et=2.623371661,  # Trapping energy (eV)
        Et=2.16,
        # lam=0.3E-9,  # Jump distance (m)
        # v0=1e13,  # Attempt frequency (1/s)
        # density_host=19.3,  # Host density (g/cm³)
        # atomic_mass_host=183.84,  # Atomic mass (g/mol)
        nx=51,  # Number of spatial points
        adapt_mesh=True,  # Use adaptive mesh
        debug=True
    )


    # simulator.set_experimental_tds(temperature=temperature_sbr, desorption_flux=desorption_flux)


    # Run the simulation
    t_max = (simulator.Tf - simulator.T0) / simulator.beta
    print(f"Physical simulation time: {t_max:.3f} seconds")
    solution = simulator.simulate(t_max)

    # Plot and optionally save the results
    fig, ax1, ax2 = simulator.plot_results(solution, save_path='tds_simulation.png' if save_plot else None)
    ax1.plot(temperature_sbr, desorption_flux, color='tab:green', ls='-', marker='none', label='Experiment')
    plt.show()

    return simulator, solution


if __name__ == "__main__":
    simulator, solution = run_simulation(save_plot=True)