import numpy as np
from sklearn.neighbors import KernelDensity
from particle_distribution_from_circular_surface import generate_particle_distribution
import matplotlib.pyplot as plt
from n_e_profile_fit import load_plot_style
from matplotlib.patches import Circle
import h5py


class GridEmissivityCalculator:
    def __init__(self, photoemission_coeff=5.1e-11):  # cm³/s
        """
        Initialize calculator with photoemission coefficient
        """
        self.photoemission_coeff = photoemission_coeff

    @staticmethod
    def extract_ne_coefficients(filename):
        """
        Extract electron density polynomial coefficients from file
        """
        coefficients = {}
        reading_ne = False
        ne_order = None

        with open(filename, 'r') as file:
            for line in file:
                if '# *********** Polynomial fit to n_e ***********' in line:
                    reading_ne = True
                    continue

                if reading_ne:
                    if '# Order:' in line:
                        ne_order = int(line.split(':')[1].strip())
                        continue

                    if 'a_' in line:
                        parts = line.split(':')
                        if len(parts) == 2:
                            index = int(parts[0].split('a_')[1])
                            value = float(parts[1].split('-/+')[0].strip())
                            coefficients[index] = value
                            if index == ne_order:
                                break

        return coefficients

    def evaluate_ne_polynomial(self, r):
        """
        Evaluate electron density polynomial at given radial positions

        Parameters:
        -----------
        r : float or ndarray
            Radial position(s) at which to evaluate ne

        Returns:
        --------
        float or ndarray
            Electron density at the given position(s)
        """
        return sum(coeff * r ** power for power, coeff in self.ne_coefficients.items())

    def load_electron_density(self, filename):
        """
        Load electron density from polynomial coefficients file
        """
        self.ne_coefficients = self.extract_ne_coefficients(filename)
        self.ne_interpolator = self.evaluate_ne_polynomial
        self.max_radius = 4.0  # Set a reasonable maximum radius

        # Generate sample points for visualization
        r = np.linspace(0, self.max_radius, 100)
        ne = self.evaluate_ne_polynomial(r)

        return r, ne

    def create_grid(self, bounds, grid_spacing):
        """
        Create 3D grid for emissivity calculation
        """
        x = np.arange(bounds[0][0], bounds[0][1] + grid_spacing, grid_spacing)
        y = np.arange(bounds[1][0], bounds[1][1] + grid_spacing, grid_spacing)
        z = np.arange(bounds[2][0], bounds[2][1] + grid_spacing, grid_spacing)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return X, Y, Z

    def calculate_particle_density(self, positions, grid_points, cell_volume):
        """
        Calculate particle density using sklearn KDE
        Returns density in particles/cm³
        """
        try:
            import cupy as cp
            from cuml.neighbors import KernelDensity as cuKDE
            USE_GPU = True
        except ImportError:
            from sklearn.neighbors import KernelDensity
            USE_GPU = False

        points = np.column_stack((
            grid_points[0].ravel(),
            grid_points[1].ravel(),
            grid_points[2].ravel()
        ))

        if USE_GPU:
            # GPU-accelerated KDE
            positions_gpu = cp.asarray(positions)
            points_gpu = cp.asarray(points)

            kde = cuKDE(bandwidth='scott')
            kde.fit(positions_gpu)
            log_density = kde.score_samples(points_gpu)
            density = cp.exp(log_density).get()
        else:
            # CPU vectorized KDE
            # Calculate optimal bandwidth using Scott's rule
            n_samples = len(positions)
            bandwidth = np.power(n_samples, -1 / 7)  # Scott's rule for 3D
            # Fit KDE
            kde = KernelDensity(bandwidth=bandwidth, kernel='epanechnikov', algorithm='ball_tree')
            kde.fit(positions)
            # kde = KernelDensity(bandwidth='scott')
            # kde.fit(positions)
            log_density = kde.score_samples(points)
            density = np.exp(log_density)



        # Get log density and convert to density
        # log_density = kde.score_samples(points)
        # density = np.exp(log_density)

        # Normalize to preserve total particle count
        total_particles = len(positions)
        volume_integral = np.sum(density) * cell_volume
        if volume_integral > 0:
            density *= total_particles / volume_integral

        return density.reshape(grid_points[0].shape)

    def calculate_grid_emissivity(self, positions=None, bounds=None, grid_spacing=None,
                                  density_file=None, save_density=None):
        """
        Calculate emissivity on 3D grid with cylindrically symmetric electron density
        Can load pre-calculated density from file or save new density calculation.

        Parameters
        ----------
        positions : ndarray, optional
            Particle positions
        bounds : tuple, optional
            Grid bounds ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        grid_spacing : float, optional
            Grid spacing
        density_file : str, optional
            File to load pre-calculated density from
        save_density : str, optional
            File to save calculated density to
        """
        if density_file is not None:
            # Load pre-calculated density and grid
            n_particles, (X, Y, Z) = self.load_density_grid(density_file)
        else:
            # Create grid and calculate density
            X, Y, Z = self.create_grid(bounds, grid_spacing)
            cell_volume = grid_spacing ** 3
            n_particles = self.calculate_particle_density(positions, (X, Y, Z), cell_volume)

            # Save density if requested
            if save_density is not None:
                self.save_density_grid(save_density, n_particles, (X, Y, Z))

        # Calculate radial distance in xy-plane for electron density
        R_xy = np.sqrt(X ** 2 + Y ** 2)  # Distance from z-axis
        n_e = self.ne_interpolator(R_xy)

        # Calculate emissivity
        emissivity = n_particles * n_e * self.photoemission_coeff

        return X, Y, Z, emissivity

    def integrate_cylinder(self, X, Y, Z, emissivity, diameter, axis_point1, axis_point2):
        """
        Integrate emissivity over cylindrical volume with proper cylindrical geometry

        Parameters:
        -----------
        X, Y, Z : ndarray
            Grid coordinates
        emissivity : ndarray
            Emissivity values on grid
        diameter : float
            Cylinder diameter in cm
        axis_point1, axis_point2 : tuple
            Points defining cylinder axis (x,y,z)

        Returns:
        --------
        float
            Integrated emissivity divided by cylinder cap area
        """
        p1 = np.array(axis_point1)
        p2 = np.array(axis_point2)

        axis = p2 - p1
        axis_length = np.linalg.norm(axis)
        axis_unit = axis / axis_length
        radius = diameter / 2

        points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        # Calculate perpendicular distance and projection for each point
        v = points - p1
        proj = np.dot(v, axis_unit)
        perp = v - np.outer(proj, axis_unit)
        dist = np.linalg.norm(perp, axis=1)

        # Mask points inside cylinder
        mask = (dist <= radius) & (proj >= 0) & (proj <= axis_length)
        mask = mask.reshape(X.shape)

        # Calculate grid spacing
        grid_spacing = X[1, 0, 0] - X[0, 0, 0]

        # Calculate effective volume element for each point
        # For points near the cylinder surface, we need to account for partial volumes
        dist_reshaped = dist.reshape(X.shape)

        # Calculate the fraction of each grid cell that lies within the cylinder
        # For simplicity, we use a linear approximation for cells intersecting the surface
        volume_fraction = np.ones_like(dist_reshaped)
        surface_cells = (dist_reshaped > (radius - grid_spacing)) & (dist_reshaped <= radius)
        volume_fraction[surface_cells] = (radius - dist_reshaped[surface_cells]) / grid_spacing

        # Calculate effective volume element for each point
        volume_element = grid_spacing ** 3 * volume_fraction

        # Calculate the true cylindrical volume for normalization
        true_cylinder_volume = axis_length * np.pi * radius ** 2

        # Sum up the grid cell volumes to get the actual integration volume
        integration_volume = np.sum(volume_element[mask])

        # Apply volume correction factor
        volume_correction = true_cylinder_volume / integration_volume

        # Calculate total emissivity with volume correction
        total_emissivity = np.sum(emissivity[mask] * volume_element[mask]) * volume_correction

        # Divide by cylinder cap area
        cap_area = np.pi * radius ** 2

        return total_emissivity / cap_area

    @staticmethod
    def save_density_grid(filename, density, grid_points):
        """
        Save particle density and grid points to HDF5 file
        """

        with h5py.File(filename, 'w') as f:
            # Save density and grid points
            f.create_dataset('density', data=density)
            f.create_dataset('grid_x', data=grid_points[0])
            f.create_dataset('grid_y', data=grid_points[1])
            f.create_dataset('grid_z', data=grid_points[2])

    @staticmethod
    def load_density_grid(filename):
        """
        Load particle density and grid points from HDF5 file

        Returns
        -------
        tuple
            (density, (grid_x, grid_y, grid_z))
        """
        import h5py

        with h5py.File(filename, 'r') as f:
            density = f['density'][:]
            grid_x = f['grid_x'][:]
            grid_y = f['grid_y'][:]
            grid_z = f['grid_z'][:]

        return density, (grid_x, grid_y, grid_z)


def plot_electron_density(r, ne):
    """
    Plot the radial electron density profile
    """
    plt.figure(figsize=(6, 4))
    plt.plot(r, ne, 'b-')
    plt.xlabel('Radial Distance from Z-axis (cm)')
    plt.ylabel('Electron Density (cm⁻³)')
    plt.title('Cylindrically Symmetric Electron Density Profile')
    plt.grid(True)
    plt.show()


def main(angle_distribution_file, electron_density_file, particle_density_file,
         grid_spacing, bounds, cylinder_diameter, cylinder_axis,
         num_particles=10000000, surface_radius=0.5, max_distance=10.0,
         save_file=False):
    """
    Main function to calculate and visualize grid-based emissivity
    """

    # Initialize calculator
    calculator = GridEmissivityCalculator()

    # Load electron density
    r, ne = calculator.load_electron_density(electron_density_file)
    plot_electron_density(r, ne)

    if save_file:
        # Generate particle distribution
        positions, _ = generate_particle_distribution(
            num_particles, surface_radius, max_distance, angle_distribution_file
        )
        X, Y, Z, emissivity = calculator.calculate_grid_emissivity(
            positions, bounds, grid_spacing, save_density=particle_density_file
        )
    else:
        # Calculate emissivity on grid
        X, Y, Z, emissivity = calculator.calculate_grid_emissivity(
            None, bounds, grid_spacing, density_file=particle_density_file
        )

    # Calculate integrated intensity
    intensity = calculator.integrate_cylinder(
        X, Y, Z, emissivity,
        cylinder_diameter,
        cylinder_axis[0], cylinder_axis[1]
    )

    # Visualize results
    fig = plt.figure(figsize=(15, 5))

    # XZ slice at Y=0
    ax1 = fig.add_subplot(131)
    mid_y = emissivity.shape[1] // 2
    im1 = ax1.pcolormesh(X[:, mid_y, :], Z[:, mid_y, :],
                         emissivity[:, mid_y, :], shading='auto')
    ax1.set_title('Emissivity XZ Slice (Y=0)')
    ax1.set_xlabel('X (cm)')
    ax1.set_ylabel('Z (cm)')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='Emissivity (photons/cm³/s)')

    circle = Circle((0., 0.25), 0.5, fill=False, color='red')
    ax1.add_patch(circle)

    # XY slice at Z=0
    ax2 = fig.add_subplot(132)
    mid_z = emissivity.shape[2] // 2
    # Assume z=zmin at idx 1
    idx_z0 = int(0.252 // grid_spacing)
    im2 = ax2.pcolormesh(X[:, :, idx_z0], Y[:, :, idx_z0],
                         emissivity[:, :, idx_z0], shading='auto')
    ax2.set_title('Emissivity XY Slice (Z=0)')
    ax2.set_xlabel('X (cm)')
    ax2.set_ylabel('Y (cm)')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='Emissivity (photons/cm³/s)')

    # Radial profile at Z=0
    ax3 = fig.add_subplot(133)
    r_grid = np.sqrt(X[:, :, mid_z] ** 2 + Y[:, :, mid_z] ** 2)
    r_unique = np.unique(r_grid)
    emissivity_radial = [np.mean(emissivity[:, :, mid_z][r_grid == r]) for r in r_unique]
    ax3.plot(r_unique, emissivity_radial, 'k-')
    ax3.set_title('Radial Emissivity Profile (Z=0)')
    ax3.set_xlabel('Radial Distance (cm)')
    ax3.set_ylabel('Emissivity (photons/cm³/s)')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    return intensity


if __name__ == "__main__":
    # Example usage
    ANGLE_DIST_FILE = r'trimsp_simulations/d_on_b_40keV_polar_angle_dist.csv'
    ELECTRON_DENSITY_FILE = r"./data/PA_probe/20240815/langprobe_results/symmetrized/lang_results_gamma_ivdata0004_symmetrized_fit.csv"
    THERMAL_VELOCITY = 1e4  # cm/s
    PARTICLE_DENS_FILE = r"./data/emissivity_simulations/20241224_particle_density.hd5"

    # Grid parameters
    GRID_SPACING = 0.02  # cm
    BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (-0.25, 1.5))  # cm

    # Cylinder parameters
    CYLINDER_DIAMETER = 1.0  # cm
    CYLINDER_AXIS = [(0, 1.0, 0.25), (0, -1.0, 0.25)]  # Points defining cylinder axis

    load_plot_style()

    intensity = main(
        ANGLE_DIST_FILE,
        ELECTRON_DENSITY_FILE,
        PARTICLE_DENS_FILE,
        GRID_SPACING,
        BOUNDS,
        CYLINDER_DIAMETER,
        CYLINDER_AXIS
    )

    print(f"Integrated intensity per unit area: {intensity:.2e} photons/cm²/s")