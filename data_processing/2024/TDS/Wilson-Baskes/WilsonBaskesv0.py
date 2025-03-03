import numpy as np
from holoviews.plotting.bokeh.styles import font_size
from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt
import scipy.stats.distributions as distributions

"""
+----------+-------------+------------+-------------+-----------+-----+
| Material | D0 (cm^2/s) | D0 (m^2/s) | Ed (kJ/mol) |  Ed (eV)  | Ref |
+----------+-------------+------------+-------------+-----------+-----+
| B4C      |      2.3E-4 |     2.3E-8 |     96 ± 10 | 1.0 ± 0.1 | [1] |
| B2O3     |      4.9E-5 |     4.9E-9 |    124 ± 15 | 2.3 ± 0.2 | [1] |
+----------+-------------+------------+-------------+-----------+-----+




Taking lambda parameter from beta-rhombohedral boron lattice parameter = 10 Å [2]

Assume that total retention is the integral of the trapped species over the distance.

[1] K. Schnarr and H. Münzel, "Release of tritium from boron carbide." (1990) J. Chem. Soc., Faraday Trans. 86, 651-656
doi: 10.1039/FT9908600651

[2] R.E. Hughes, C.H.L. Kennard, D.B. Sullenger, H.A. Weakliem, D.E. Sands, J.L. Hoard, "The Structure of β-Rhombohedral Boron"
(1963) Journal of the American Chemical Society, 85, 361
doi: https://doi.org/10.1021/ja00886a036
"""


class ThermalDesorptionSimulator:
    kB = 8.617333262145179e-05
    def __init__(
            self,
            T0=300,  # Initial temperature (K)
            Tf=1200,  # Final temperature (K)
            beta=0.3,  # Heating rate (K/s)
            total_retention=1E23,  # Total retained deuterium from desorption experiments (1/m²)
            L=25e-6,  # Sample thickness (m)
            D0=2.3E-8,  # Diffusion pre-exponential (m²/s)
            Ed=1.0,  # Diffusion activation energy (eV)
            Et=1.2,  # Trapping energy (eV)
            lam=1E-9,  # The jump distance, taken
            v0=1e13,  # attempt frequency for detrapping (1/s)
            density_host=2.31,  # Trap density (g/cm³)
            atomic_mass_host=10.811,  # g/mol
            gamma_ampl=2.0598166,
            gamma_shape=6.1870771,
            gamma_scale=2.43103442,
            gamma_loc=-5.04063028,
            kr=3.2e-15,  # Recombination coefficient (m⁴/s)
            Er=1.61,  # The recombination energy (eV)
            nx=2001  # Number of spatial points
    ):
        # Physical parameters (same as original)
        self.L = L  # Sample thickness (m)
        self.D0 = D0  # Diffusion pre-exponential (m²/s)
        self.Ed = Ed  # Diffusion activation energy (eV)
        self.kr = kr  # Recombination coefficient (m⁴/s)
        self.Er = Er  # The recombination energy (eV)
        self.Et = Et  # Trapping energy (eV)
        self.lam = lam  # Trap density (1/m³)
        self.v0 = v0  # attempt frequency for detrapping (1/s)
        self.N = self.get_atomic_density(density_host, atomic_mass_host)
        self.total_retention = total_retention


        # Initial conditions and numerical parameters (same as original)
        self.T0 =T0  # Initial temperature (K)
        self.Tf = Tf  # Final temperature (K)
        self.beta = beta  # Heating rate (K/s)
        self.nx = nx  # Number of spatial points
        self.dX = 1.0 / (self.nx - 1)  # Grid spacing in normalized coordinates
        self.X = np.linspace(0, 1, self.nx)  # Normalized spatial grid
        self.x = self.X * self.L  # Physical spatial grid

        # Initial concentration profile (Gaussian)
        self.gamma_ampl = gamma_ampl
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.gamma_loc = gamma_loc
        C0, w0 = self.estimate_CT0()
        self.C0 = C0
        self.w0 = w0
        self.theta_0 = np.ones_like(self.x)

        # Threshold for considering concentration effectively zero
        self.zero_threshold = 1e-10

        print(f"Atomic density: {self.N:.2E} atoms/m^3")
        print(f"C0: {self.C0:.2E} atoms/m^3")
        integrated_implanted_profile = simpson(y=w0, x=self.x) * C0
        print(f"Desorption from traps: {integrated_implanted_profile:.2E} 1/m^2")

    @staticmethod
    def get_atomic_density(density, atomic_mass):
        N_A = 6.02214076e+23
        return density / atomic_mass * N_A * 1E6

    def estimate_CT0(self):
        distribution = self.gamma_ampl * distributions.gamma.pdf(
            self.x*1E10, self.gamma_shape, loc=self.gamma_loc, scale=self.gamma_scale
        )

        s = simpson(y=distribution, x=self.x)
        # nD = self.total_retention / s
        # w0 = distribution #/ nD
        nD = self.total_retention / self.L
        w0 = np.ones_like(self.x)
        return nD, w0


    def D(self, T):
        """Temperature-dependent diffusion coefficient"""
        return self.D0 * np.exp(-self.Ed / (self.kB * T))

    def K_r(self, T):
        """Temperature-dependent recombination coefficient"""
        return self.kr * np.exp(-self.Er/(self.kB * T))


    def calculate_parameters(self, t):
        """Calculate normalized parameters at given time"""
        T = self.T0 + self.beta * t
        D_ref = self.D(self.Tf)  # Reference diffusion coefficient
        D_t = self.D(T)  # Current diffusion coefficient

        # Modified normalized parameters using D_ref
        mu_param = self.C0 * self.L ** 2 / self.N / (self.lam ** 2)
        nu_param = self.v0 * self.L ** 2 / D_ref

        D_ratio = D_t / D_ref  # Normalized diffusion coefficient

        return T, D_ratio, mu_param, nu_param

    def pde_system(self, tau, y):
        """Define the PDE system with temperature-dependent diffusion"""
        nx = self.nx
        u = y[:nx]
        w = y[nx:]

        # Calculate parameters including D_ratio
        t = tau *  (self.L ** 2) / self.D(self.Tf)
        T, D_ratio, mu_param, nu_param = self.calculate_parameters(t)

        # Initialize derivative arrays
        dudt = np.zeros(nx)
        dwdt = np.zeros(nx)

        # Modified diffusion term with temperature-dependent D
        for i in range(1, nx - 1):
            dudt[i] = D_ratio * (u[i + 1] - 2 * u[i] + u[i - 1]) / (self.dX ** 2)

        dudt[0] = D_ratio * (u[1] -2*u[0]) / (self.dX ** 2) # Assume u[0-dx] = 0
        dudt[-1] = D_ratio * (u[-2] - 2 * u[-1] ) / (self.dX ** 2)  # Assume that u[L+dx] = 0

        # At x=0 the flux is J = -kr*e^(-Er/kT)*CM[0]^2
        # J = -D ∂C(x)/∂x|x=0 + x ∂C_T/∂t|x=0 = -2*kr*e^(-Er/kT)*C(x)^2|x=0
        # -D ∂C(x)/∂x|x=0 = -2 kr e^(-Er/kT) C(0)^2
        #    ∂C(x)/∂x|x=0 = 2 kr e^(-Er/kT) C(0)^2 / D(T)
        # (C0 / L) ∂u/∂X |X=0 = 2 C0^2 kr e^(-Er/kT) u(0)^2 / D(T)
        #          ∂u/∂X |X=0 = 2 (L/C0) C0 kr e^(-Er/kT) * u(0)^2 / D(T)
        #          ∂u/∂X |X=0 = 2 L C0 kr e^(-Er/kT) * u(0)^2 / D(T)
        # ∂u/∂t = D(T)/D_max ∂u/∂x - ∂w/∂t = D_ratio 2 L kr e^(-Er/kT) * u(0)^2 - ∂w/∂t
        # dudt[0] = 2. * self.C0 * self.L * self.K_r(T) * u[0]**2 / self.D(T) - ∂w/∂t

        # Trapping terms (same as original)
        dwdt = mu_param * D_ratio *  u * (self.theta_0 - w) - nu_param * w * np.exp(-self.Et/(self.kB * T))
        dudt -= dwdt


        return np.concatenate([dudt, dwdt])


    def concentrations_depleted(self, t, y):
        """Event function that returns 0 when total integrated concentration is effectively zero"""
        # Sum up all mobile and trapped concentrations
        total_inventory = np.sum(np.abs(y))
        return total_inventory - self.zero_threshold

    def get_initial_conditions(self):
        """Calculate initial equilibrium conditions"""
        # D0 = self.D(self.T0)
        # Df = self.D(self.Tf)
        # # Modified normalized parameters using D_ref
        # mu_param = self.C0 * self.L ** 2 / self.N / (self.lam ** 2)
        # nu_param = self.v0 * self.L ** 2 / Df

        # u0 = (Df/D0) * (nu_param / mu_param) * np.exp(-self.Et / (self.kB * self.T0)) / (self.theta_0 - self.w0)
        u0 = np.zeros_like(self.w0)
        return np.concatenate([u0, self.w0])


    def simulate(self, t_max):
        """Run the simulation"""
        y0 = self.get_initial_conditions()
        tau_max = self.D(self.Tf) * t_max / (self.L ** 2)
        print(f"tau_max: {tau_max:.3E}")
        dtau = (self.dX **2) / 2
        ntaus_stability = int(tau_max // dtau + 1)
        print(f"dtau_stability: {dtau}, ntaus_stability: {ntaus_stability}")
        ntaus = np.clip(ntaus_stability, a_min=101, a_max=1001)
        dtau = tau_max / (ntaus - 1)
        print(f"dtau: {dtau}, ntaus: {ntaus}")
        taus = np.arange(ntaus) * dtau
        solution = solve_ivp(
            self.pde_system,
            (0, tau_max),
            y0,
            method='LSODA',
            t_eval=taus,
            # rtol=1e-6,
            # atol=1e-8,
            events=self.concentrations_depleted  # Add event detection
        )

        # Calculate temperatures and physical quantities
        temps = self.T0 + self.beta * solution.t * (self.L ** 2) / self.D(self.Tf)
        C_M = solution.y[:self.nx, :] * self.C0
        C_T = solution.y[self.nx:, :] * self.C0

        # Calculate flux using recombination rate K
        flux = np.zeros(len(temps))
        for i, T in enumerate(temps):
            flux[i] = 2. * self.K_r(T) * C_M[0, i] ** 2.
            # D_t = self.D(T)
            # flux[i] = -D_t * self.C0 * (solution.y[0, i] - solution.y[2, i]) / (2 * self.dX * self.L)

        return {
            'time': solution.t,
            'temperature': temps,
            'C_M': C_M,
            'C_T': C_T,
            'flux': flux,
            # 'terminated_early': len(solution.t_events[0]) > 0  # Check if simulation stopped due to event
        }


# Run simulation
simulator = ThermalDesorptionSimulator(
    T0=300,  # Initial temperature (K)
    Tf=1200,  # Final temperature (K)
    beta=0.3,  # Heating rate (K/s)
    total_retention=3.1E22,  # Total retained deuterium from desorption experiments (1/m²)
    L=2e-6,  # Sample thickness (m)
    D0=2.3E-8,  # Diffusion pre-exponential (m²/s)
    Ed=1.,  # Diffusion activation energy (eV)
    kr=3.2e-15,  # Recombination coefficient (m³/s)
    Et=2.5,  # Trapping energy (eV)
    lam=1E-9, # The jump distance, taken from lattice parameter
    v0=1e13,  # attempt frequency for detrapping (1/s)
    density_host=2.31,  # Host density (g/cm³)
    atomic_mass_host=10.811, # g/mol
)
# simulator = ThermalDesorptionSimulator(
#     T0=300,  # Initial temperature (K)
#     Tf=1300,  # Final temperature (K)
#     beta=0.5,  # Heating rate (K/s)
#     total_retention=2.5E20,  # Total retained deuterium from desorption experiments (1/m²)
#     L=25e-6,  # Sample thickness (m)
#     D0=4.1E-7,  # Diffusion pre-exponential (m²/s)
#     Ed=0.043,  # Diffusion activation energy (eV)
#     kr=3.2e-15,  # Recombination coefficient (m³/s)
#     Et=1.8,  # Trapping energy (eV)
#     lam=0.3E-9, # The jump distance, taken from lattice parameter
#     v0=1e13,  # attempt frequency for detrapping (1/s)
#     density_host=19.3,  # Host density (g/cm³)
#     atomic_mass_host=183.84, # g/mol
# )
t_max = (simulator.Tf - simulator.T0) / simulator.beta
print(f"t_max: {t_max:.3E}")
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



