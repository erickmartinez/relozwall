import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class TDSModel:
    # Physical constants
    kB = 8.617333262e-5  # Boltzmann constant (eV/K)

    def __init__(self,
                 D0=1e-2,  # Diffusion pre-exponential (m^2/s)
                 Ed=0.2,  # Diffusion activation energy (eV)
                 k0=1e-15,  # Trapping coefficient (m^3/s)
                 p0=1e13,  # Detrapping pre-exponential (1/s)
                 Eb=0.8,  # Binding energy (eV)
                 N=1e20,  # Trap density (1/m^3)
                 L=1e-6,  # Sample thickness (m)
                 nx=100,  # Number of spatial points
                 beta=0.3,  # Heating rate (K/s)
                 C0=1e24  # Reference concentration (1/m^3)
                 ):
        # Physical parameters
        self.D0 = D0
        self.Ed = Ed
        self.k0 = k0
        self.p0 = p0
        self.Eb = Eb
        self.N = N
        self.L = L
        self.beta = beta
        self.C0 = C0

        # Temperature range
        self.T0 = 300  # Initial temperature (K)
        self.Tf = 1200  # Final temperature (K)

        # Numerical parameters
        self.nx = nx
        self.x = np.linspace(0, self.L, self.nx)
        self.dx = self.x[1] - self.x[0]

        # Normalized spatial coordinate
        self.X = self.x / self.L
        self.dX = self.dx / self.L

        # Calculate stable time step from Neumann criterion
        self.dtau = 0.45 * self.dX ** 2  * self.D(self.T0)/self.D(self.T0) # Safety factor of 0.45 for stability

    def D(self, T):
        """Temperature-dependent diffusion coefficient"""
        return self.D0 * np.exp(-self.Ed / (self.kB * T))

    def p(self, T):
        """Temperature-dependent detrapping rate"""
        return self.p0 * np.exp(-self.Eb / (self.kB * T))

    def get_time_grid(self, n_points=1000):
        """Generate time grid based on Neumann stability criterion"""
        # Calculate total dimensional time
        total_time = (self.Tf - self.T0) / self.beta

        # Convert to dimensionless time
        total_dtau = total_time * self.D0 / self.L ** 2

        # Calculate number of points needed for stability
        n_points_stable = int(np.ceil(total_dtau / self.dtau)) + 1
        print(f"n_points_stable: {n_points_stable}, dtau: {self.dtau}")

        # Create uniform grid with stable time step
        taus = np.linspace(0, total_dtau, n_points)

        temps = self.T0 + self.beta * taus * self.L ** 2 / self.D0

        return taus, temps

    def temporal_derivative(self, y, tau, T):
        """Right-hand side of the McNabb-Foster equations using dimensionless time tau"""
        # Cast inputs to float64 for stability
        # T = float(T)
        u = y[:self.nx].astype(np.float64)
        w = y[self.nx:].astype(np.float64)

        D = self.D(T)
        p_T = self.p(T)

        # Dimensionless parameters
        ll = self.N * self.k0 * self.L ** 2 / D  # λ = NkL²/D
        nu = self.k0 * self.C0 * self.L ** 2 / D  # ν = kC₀L²/D
        mu = p_T * self.L ** 2 / D  # μ = pL²/D

        # Initialize derivatives
        du_dtau = np.zeros_like(u)
        dw_dtau = np.zeros_like(w)

        # Interior points - central differences for diffusion
        for i in range(1, self.nx - 1):
            # Calculate trapping terms separately for clarity
            trap_term = ll * u[i]
            detrap_term = nu * u[i] * w[i]
            release_term = mu * w[i]

            # Equation [8]: ∂w/∂τ = λu - νuw - μw
            print("trap_term", trap_term.shape)
            print("detrap_term", detrap_term.shape)
            print("release_term", release_term.shape)
            dw_dtau[i] = trap_term - detrap_term - release_term

            # Equation [7]: ∂u/∂τ + ∂w/∂τ = ∂²u/∂X²
            du_dtau[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / self.dX ** 2 - dw_dtau[i]

        # Boundary conditions
        du_dtau[0] = 0  # Zero concentration at surface
        du_dtau[-1] = (u[-2] - u[-1]) / self.dX ** 2  # Zero flux at back
        dw_dtau[0] = dw_dtau[1]
        dw_dtau[-1] = dw_dtau[-2]

        return np.concatenate([du_dtau, dw_dtau])

    def run_simulation(self):
        """Run TDS simulation"""
        # Initial conditions - Gaussian implantation profile
        Rp = 0.2 * self.L
        dRp = 0.1 * self.L
        # Normalize initial concentration by C0
        u0 = self.C0 * np.exp(-(self.x - Rp) ** 2 / (2 * dRp ** 2))
        u0 = u0 / self.C0  # Normalize to get dimensionless u
        w0 = np.zeros_like(u0)
        y0 = np.concatenate([u0, w0])

        # Generate time grid using Neumann criterion
        taus, temps = self.get_time_grid()

        # Solve the system
        solution = odeint(self.temporal_derivative, y0, taus,
                          args=(temps,), rtol=1e-6, atol=1e-8)

        # Calculate desorption flux (positive when pointing outwards at x=0)
        D_t = self.D(temps)
        flux = D_t * (solution[:, 1] - solution[:, 0]) / self.dx * self.C0

        return temps, flux


# Example usage:
if __name__ == "__main__":
    # Create model
    model = TDSModel(beta=0.3, C0=1e24)

    # Run simulation
    T, flux = model.run_simulation()

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(T, flux / np.max(flux), 'b-', linewidth=2)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Normalized Desorption Flux')
    plt.title('TDS Spectrum - McNabb-Foster Model')
    plt.xlim(300, 1200)
    plt.grid(True)
    plt.show()