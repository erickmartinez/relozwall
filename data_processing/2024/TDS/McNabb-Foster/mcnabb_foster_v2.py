import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class TDSModel:
    def __init__(self,
                 L=2e-6,  # Sample thickness (m) - typical implantation depth for D in B
                 nx=200,  # Number of spatial points - reduced for better performance
                 D0=8.5e-7,  # Pre-exponential factor (m²/s) for D in B
                 Ed=0.85,  # Diffusion activation energy (eV) for D in B
                 Etrap=0.7,  # Trapping energy (eV) - typical for D in B
                 c0=1e23,  # Initial concentration (atoms/m³)
                 ntrap=5e22,  # Trap density (traps/m³)
                 p0=1e13,  # Base trapping rate (s⁻¹)
                 beta=0.5,  # Heating rate (K/s)
                 T0=300,  # Initial temperature (K)
                 Tmax=1000,  # Maximum temperature (K)
                 implant_depth=0.5e-6,  # Implantation depth (m)
                 implant_width=0.3e-6):  # Width of implantation profile (m)

        self.L = L
        self.nx = nx
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)

        self.D0 = D0
        self.Ed = Ed
        self.Etrap = Etrap
        self.c0 = c0
        self.ntrap = ntrap
        self.p0 = p0
        self.beta = beta
        self.T0 = T0
        self.Tmax = Tmax
        self.implant_depth = implant_depth
        self.implant_width = implant_width

        self.kB = 8.617333262e-5  # Boltzmann constant (eV/K)

        # Calculate appropriate time step based on diffusion coefficient
        D_max = self.D(Tmax)
        self.dt = min(0.25 * self.dx ** 2 / D_max, 0.1)  # Stability criterion

    def D(self, T):
        """Temperature-dependent diffusion coefficient"""
        return self.D0 * np.exp(-self.Ed / (self.kB * T))

    def p(self, T):
        """Temperature-dependent trapping rate"""
        return self.p0 * np.exp(-0.1 / (self.kB * T))  # Small activation energy for trapping

    def q(self, T):
        """Detrapping rate"""
        return self.p0 * np.exp(-self.Etrap / (self.kB * T))

    def initial_profile(self):
        """Generate initial concentration profile based on implantation parameters"""
        return self.c0 * np.exp(-(self.x - self.implant_depth) ** 2 /
                                (2 * self.implant_width ** 2))

    def dCdt(self, C, t):
        """System of ODEs for diffusion and trapping"""
        T = self.T0 + self.beta * t
        if T > self.Tmax:
            return np.zeros_like(C)

        # Split C into mobile and trapped concentrations
        Cm = C[:self.nx]  # Mobile deuterium
        Ct = C[self.nx:]  # Trapped deuterium

        dCmdt = np.zeros(self.nx)
        dCtdt = np.zeros(self.nx)

        # Diffusion term (second order central difference)
        D = self.D(T)
        for i in range(1, self.nx - 1):
            dCmdt[i] = D * (Cm[i + 1] - 2 * Cm[i] + Cm[i - 1]) / (self.dx ** 2)

        # Boundary conditions (surface concentration = 0)
        dCmdt[0] = 0
        dCmdt[-1] = 0

        # Trapping-detrapping terms
        p = self.p(T)
        q = self.q(T)
        for i in range(self.nx):
            theta = Ct[i] / self.ntrap
            if theta < 0.0: theta = 0.0  # Physical constraints
            if theta > 1.0: theta = 1.0
            dCtdt[i] = p * (1 - theta) * Cm[i] - q * theta * self.ntrap
            dCmdt[i] -= dCtdt[i]

        return np.concatenate([dCmdt, dCtdt])

    def simulate(self):
        """Run the TDS simulation"""
        # Time grid based on heating rate and stability criterion
        tmax = (self.Tmax - self.T0) / self.beta
        nsteps = int(tmax / self.dt) + 1
        self.t = np.linspace(0, tmax, nsteps)

        # Initial conditions with Gaussian implantation profile
        C0 = np.zeros(2 * self.nx)
        C0[:self.nx] = self.initial_profile()  # Initial mobile concentration
        C0[self.nx:] = 0.0  # Initial trapped concentration

        # Solve the system
        self.solution = odeint(self.dCdt, C0, self.t,
                               rtol=1e-5, atol=1e-5)  # Adjusted tolerances

        # Calculate desorption flux at x=0
        self.T = self.T0 + self.beta * self.t
        self.flux = np.array([
            -self.D(T) * (self.solution[i, 1] - self.solution[i, 0]) / self.dx
            for i, T in enumerate(self.T)
        ])

    def plot_results(self, save_plots=False):
        """Plot the TDS spectrum and concentration profiles"""
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
            plt.plot(self.x * 1e6, self.solution[i, :self.nx],
                     label=f'T = {T:.0f} K')
        plt.xlabel('Position (μm)')
        plt.ylabel('Mobile D concentration (m⁻³)')
        plt.title('Mobile Deuterium Profiles')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        for i in times:
            T = self.T[i]
            plt.plot(self.x * 1e6, self.solution[i, self.nx:],
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


# Example usage with realistic parameters for D in B
model = TDSModel(
    L=2e-6,  # 2 μm sample thickness
    nx=200,  # Reasonable spatial resolution
    D0=8.5e-7,  # Pre-exponential factor for D in B
    Ed=0.85,  # Activation energy for diffusion
    Etrap=0.7,  # Trapping energy
    c0=1e23,  # Initial concentration
    ntrap=5e22,  # Trap density
    p0=1e13,  # Base trapping rate
    beta=0.5,  # Heating rate (K/s)
    implant_depth=0.5e-6,  # 0.5 μm implantation depth
    implant_width=0.3e-6  # 0.3 μm width of implantation profile
)

model.simulate()
model.plot_results(save_plots=True)