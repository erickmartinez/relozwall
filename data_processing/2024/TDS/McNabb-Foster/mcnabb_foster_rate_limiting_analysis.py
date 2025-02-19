import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class TDSModel:
    def __init__(self,
                 L=1e-6,  # Sample thickness (m)
                 nx=100,  # Number of spatial points
                 D0=1e-6,  # Pre-exponential factor (m²/s)
                 Ed=0.3,  # Diffusion activation energy (eV)
                 Etrap=0.1,  # Trapping energy (eV)
                 c0=1e24,  # Initial concentration (atoms/m³)
                 ntrap=1e21,  # Trap density (traps/m³)
                 beta=0.3,  # Heating rate (K/s)
                 T0=300,  # Initial temperature (K)
                 Tmax=1100,  # Maximum temperature (K)
                 k_d0=0,#1e13,  # Pre-exponential factor for desorption (s⁻¹)
                 Ed_des=0.8,  # Desorption activation energy (eV)
                 k_r0=1E-28):#1e-28):  # Surface recombination coefficient (m⁴/s)

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
        self.k_r0 = k_r0

        self.kB = 8.617333262e-5  # Boltzmann constant (eV/K)

    def D(self, T):
        """Temperature-dependent diffusion coefficient"""
        return self.D0 * np.exp(-self.Ed / (self.kB * T))

    def k_d(self, T):
        """Temperature-dependent first-order desorption rate"""
        return self.k_d0 * np.exp(-self.Ed_des / (self.kB * T))

    def k_r(self, T):
        """Temperature-dependent surface recombination coefficient"""
        # Assuming weak temperature dependence for recombination
        return self.k_r0 * np.exp(-0.1 / (self.kB * T))

    def p(self, T):
        """
        Temperature-dependent trapping rate
        Typically assumed to be diffusion-limited
        """
        return self.D(T) * self.ntrap

    def q(self, T):
        """
        Temperature-dependent detrapping rate
        Follows Arrhenius behavior with trap binding energy
        """
        nu = 1e13  # Attempt frequency (s⁻¹)
        return nu * np.exp(-self.Etrap / (self.kB * T))

    def characteristic_times(self, T, C_surface):
        """Calculate characteristic times for different processes"""
        # Diffusion time
        t_diff = self.L ** 2 / self.D(T)

        # First-order desorption time
        t_des = 1 / self.k_d(T)

        # Surface recombination time
        t_recomb = 1 / (self.k_r(T) * C_surface)

        return t_diff, t_des, t_recomb

    def rate_limiting_process(self, T, C_surface):
        """Determine the rate-limiting process based on characteristic times"""
        t_diff, t_des, t_recomb = self.characteristic_times(T, C_surface)
        times = {'Diffusion': t_diff, 'Desorption': t_des, 'Recombination': t_recomb}

        # Rate-limiting step has the longest characteristic time
        rate_limiting = max(times.items(), key=lambda x: x[1])[0]

        # Calculate relative contributions
        total_time = sum(times.values())
        contributions = {k: v / total_time for k, v in times.items()}

        return rate_limiting, contributions

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

        # Combined surface kinetics boundary condition
        k_des = self.k_d(T)
        k_rec = self.k_r(T)
        dCmdt[0] = -k_des * Cm[0] - k_rec * Cm[0] ** 2

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

        # Calculate various fluxes and rate-limiting steps
        self.T = self.T0 + self.beta * self.t

        # Store rate-limiting process analysis
        self.rate_limiting = []
        self.process_contributions = []

        # Calculate fluxes and analyze rate-limiting steps
        self.diff_flux = np.zeros_like(self.t)
        self.des_flux = np.zeros_like(self.t)
        self.recomb_flux = np.zeros_like(self.t)

        for i, T in enumerate(self.T):
            C_surface = self.solution[i, 0]

            # Calculate fluxes
            self.diff_flux[i] = -self.D(T) * (self.solution[i, 2] - self.solution[i, 0]) / (2 * self.dx)
            self.des_flux[i] = self.k_d(T) * C_surface
            self.recomb_flux[i] = self.k_r(T) * C_surface ** 2

            # Analyze rate-limiting step
            rls, contributions = self.rate_limiting_process(T, C_surface)
            self.rate_limiting.append(rls)
            self.process_contributions.append(contributions)

        self.flux = self.diff_flux + self.des_flux + self.recomb_flux

    def plot_results(self):
        """Plot the TDS spectrum with detailed analysis"""
        # Plot 1: Desorption fluxes
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.T, self.flux, 'b-', label='Total flux')
        plt.plot(self.T, self.diff_flux, 'g--', label='Diffusive flux')
        plt.plot(self.T, self.des_flux, 'r--', label='Desorption flux')
        plt.plot(self.T, self.recomb_flux, 'm--', label='Recombination flux')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Desorption flux (a.u.)')
        plt.title('TDS Spectrum')
        plt.grid(True)
        plt.legend()

        # Plot 2: Process contributions
        plt.subplot(2, 1, 2)
        contributions = np.array(self.process_contributions)
        plt.plot(self.T, [c['Diffusion'] for c in contributions], 'g-', label='Diffusion')
        plt.plot(self.T, [c['Desorption'] for c in contributions], 'r-', label='Desorption')
        plt.plot(self.T, [c['Recombination'] for c in contributions], 'm-', label='Recombination')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Relative contribution')
        plt.title('Rate-Limiting Process Analysis')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot 3: Concentration profiles
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