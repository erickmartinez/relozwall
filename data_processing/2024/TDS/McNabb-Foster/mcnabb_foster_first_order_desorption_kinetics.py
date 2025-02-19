import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class TDSModel:
    def __init__(self,
                 L=1e-6,  # Sample thickness (m)
                 nx=100,  # Number of spatial points
                 D0=1e-6,  # Pre-exponential factor (m²/s)
                 Ed=0.5,  # Diffusion activation energy (eV)
                 Etrap=0.1,  # Trapping energy (eV)
                 c0=1e24,  # Initial concentration (atoms/m³)
                 ntrap=1e24,  # Trap density (traps/m³)
                 beta=0.3,  # Heating rate (K/s)
                 T0=300,  # Initial temperature (K)
                 Tmax=1100,  # Maximum temperature (K)
                 k_d0=1e13,  # Pre-exponential factor for desorption (s⁻¹)
                 Ed_des=0.8):  # Desorption activation energy (eV)

        self.L = L
        self.nx = nx
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)

        self.D0 = D0
        self.Ed = Ed
        self.Etrap = Etrap
        self.c0 = c0
        self.ntrap = ntrap
        self.beta = beta
        self.T0 = T0
        self.Tmax = Tmax
        self.k_d0 = k_d0
        self.Ed_des = Ed_des

        self.kB = 8.617333262e-5  # Boltzmann constant (eV/K)

    def D(self, T):
        """Temperature-dependent diffusion coefficient"""
        return self.D0 * np.exp(-self.Ed / (self.kB * T))

    def k_d(self, T):
        """Temperature-dependent first-order desorption rate"""
        return self.k_d0 * np.exp(-self.Ed_des / (self.kB * T))

    def p(self, T):
        """Trapping rate"""
        return 1e13  # Typical attempt frequency (s⁻¹)

    def q(self, T):
        """Detrapping rate"""
        return self.p(T) * np.exp(-self.Etrap / (self.kB * T))

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

        # First-order desorption boundary condition at surface
        k_des = self.k_d(T)
        dCmdt[0] = -k_des * Cm[0]

        # No-flux boundary condition at back surface
        dCmdt[-1] = 0

        # Trapping-detrapping terms
        p = self.p(T)
        q = self.q(T)
        for i in range(self.nx):
            theta = Ct[i] / self.ntrap
            if theta > 1.0: theta = 1.0  # Ensure occupancy ≤ 1
            dCtdt[i] = p * (1 - theta) * Cm[i] - q * theta
            dCmdt[i] -= dCtdt[i]

        return np.concatenate([dCmdt, dCtdt])

    def simulate(self):
        """Run the TDS simulation"""
        # Time grid
        tmax = (self.Tmax - self.T0) / self.beta
        self.t = np.linspace(0, tmax, 1000)

        # Initial conditions
        C0 = np.zeros(2 * self.nx)
        C0[:self.nx] = self.c0  # Initial mobile concentration
        C0[self.nx:] = 0.0  # Initial trapped concentration

        # Solve the system
        self.solution = odeint(self.dCdt, C0, self.t)

        # Calculate desorption flux at x=0 using central difference
        self.T = self.T0 + self.beta * self.t

        # Total desorption flux combines diffusive and first-order desorption
        self.diff_flux = np.array([
            -self.D(T) * (self.solution[i, 2] - self.solution[i, 0]) / (2 * self.dx)
            for i, T in enumerate(self.T)
        ])

        self.des_flux = np.array([
            self.k_d(T) * self.solution[i, 0]
            for i, T in enumerate(self.T)
        ])

        self.flux = self.diff_flux + self.des_flux

    def plot_results(self):
        """Plot the TDS spectrum with separate diffusive and desorption contributions"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.T, self.flux, 'b-', label='Total flux')
        plt.plot(self.T, self.diff_flux, 'g--', label='Diffusive flux')
        plt.plot(self.T, self.des_flux, 'r--', label='Desorption flux')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Desorption flux (a.u.)')
        plt.title('TDS Spectrum')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot concentration profiles at different times
        plt.figure(figsize=(10, 6))
        times = [0, len(self.t) // 4, len(self.t) // 2, 3 * len(self.t) // 4, -1]
        for i in times:
            T = self.T[i]
            plt.plot(self.x * 1e6, self.solution[i, :self.nx],
                     label=f'T = {T:.0f} K')
        plt.xlabel('Position (μm)')
        plt.ylabel('Mobile D concentration (m⁻³)')
        plt.title('Concentration Profiles')
        plt.grid(True)
        plt.legend()
        plt.show()


# Example usage
model = TDSModel()
model.simulate()
model.plot_results()