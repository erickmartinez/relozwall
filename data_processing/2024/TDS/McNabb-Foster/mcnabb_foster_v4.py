import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class TDSModel:
    # Physical constants
    kB = 8.617333262e-5  # Boltzmann constant (eV/K)

    def __init__(self,
                 D0=1e-6,  # Diffusion pre-exponential (m^2/s)
                 Ed=0.85,  # Diffusion activation energy (eV)
                 k0=1e-15,  # Trapping coefficient (m^3/s)
                 p0=1e13,  # Detrapping pre-exponential (1/s)
                 Eb=2.1,  # Binding energy (eV)
                 N=1e23,  # Trap density (1/m^3)
                 L=1e-6,  # Sample thickness (m)
                 nx=100,  # Number of spatial points
                 beta=0.3,  # Heating rate (K/s)
                 C0=1e24  # Reference concentration (1/m^3)
                 ):
        # Store parameters
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

        # Spatial grid
        self.nx = nx
        self.x = np.linspace(0, self.L, self.nx)
        self.dx = self.x[1] - self.x[0]

        # Normalized coordinates
        self.X = self.x / self.L
        self.dX = self.dx / self.L

    def D(self, T):
        """Temperature-dependent diffusion coefficient"""
        return self.D0 * np.exp(-self.Ed / (self.kB * T))

    def p(self, T):
        """Temperature-dependent detrapping rate"""
        return self.p0 * np.exp(-self.Eb / (self.kB * T))

    def get_time_grid(self):
        """Generate time grid"""
        total_time = (self.Tf - self.T0) / self.beta
        # Convert to dimensionless time
        total_tau = total_time * self.D0 / self.L ** 2

        # Create time points
        n_points = 1000
        taus = np.linspace(0, total_tau, n_points)
        temps = self.T0 + self.beta * taus * self.L ** 2 / self.D0

        return taus, temps

    def system_equations(self, tau, y, T):
        """System of McNabb-Foster equations in vector form"""
        # Split state vector into u and w components
        u = y[:self.nx]
        w = y[self.nx:]

        # Get coefficients at current temperature
        D_T = self.D(T)
        p_T = self.p(T)

        # Dimensionless parameters
        ll = self.N * self.k0 * self.L ** 2 / D_T  # λ = NkL²/D
        nu = self.k0 * self.C0 * self.L ** 2 / D_T  # ν = kC₀L²/D
        mu = p_T * self.L ** 2 / D_T  # μ = pL²/D

        # Create Laplacian operator matrix for diffusion term
        diag = np.ones(self.nx - 2) / self.dX ** 2
        D2 = np.diag(-2 * diag) + np.diag(diag[:-1], k=1) + np.diag(diag[:-1], k=-1)

        # Initialize derivatives
        du_dt = np.zeros(self.nx)
        dw_dt = np.zeros(self.nx)

        # Interior points
        du_dt[1:-1] = np.dot(D2, u[1:-1])
        dw_dt[1:-1] = ll * u[1:-1] - nu * u[1:-1] * w[1:-1] - mu * w[1:-1]

        # Boundary conditions
        du_dt[0] = 0  # Zero concentration at surface
        du_dt[-1] = (u[-2] - u[-1]) / self.dX ** 2  # Zero flux at back
        dw_dt[0] = dw_dt[1]
        dw_dt[-1] = dw_dt[-2]

        # Couple equations
        du_dt -= dw_dt

        return np.concatenate([du_dt, dw_dt])

    def run_simulation(self):
        """Run TDS simulation"""
        # Initial conditions - Gaussian implantation profile
        Rp = 0.2 * self.L
        dRp = 0.1 * self.L
        u0 = self.C0 * np.exp(-(self.x - Rp) ** 2 / (2 * dRp ** 2))
        u0 = u0 / self.C0  # Normalize
        w0 = np.zeros_like(u0)
        y0 = np.concatenate([u0, w0])

        # Get time grid
        taus, temps = self.get_time_grid()

        # Solve system using solve_ivp
        def wrapped_system(t, y):
            T = np.interp(t, taus, temps)
            return self.system_equations(t, y, T)

        solution = solve_ivp(
            wrapped_system,
            t_span=(taus[0], taus[-1]),
            y0=y0,
            t_eval=taus,
            method='BDF',
            rtol=1e-6,
            atol=1e-8
        )

        # Calculate desorption flux
        D_t = self.D(temps)
        flux = D_t * (solution.y[1, :] - solution.y[0, :]) / self.dx * self.C0

        return temps, flux


# Example usage:
if __name__ == "__main__":
    model = TDSModel(
        D0=1e-3,  # Diffusion pre-exponential (m^2/s)
        Ed=0.5,  # Diffusion activation energy (eV)
        k0=1e-15,  # Trapping coefficient (m^3/s)
        p0=1e13,  # Detrapping pre-exponential (1/s)
        Eb=1.5,  # Binding energy (eV)
        N=1e23,  # Trap density (1/m^3)
        L=1e-6,  # Sample thickness (m)
        nx=100,  # Number of spatial points
        beta=0.3,  # Heating rate (K/s)
        C0=1e24  # Reference concentration (1/m^3)
    )
    T, flux = model.run_simulation()

    plt.figure(figsize=(10, 6))
    plt.plot(T, flux, 'b-', linewidth=2)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Normalized Desorption Flux')
    plt.title('TDS Spectrum - McNabb-Foster Model')
    plt.xlim(300, 1200)
    plt.grid(True)
    plt.show()