import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure
from skimage.transform import probabilistic_hough_line
from typing import List, Tuple, Optional
import warnings
import tifffile
from pathlib import Path

STACK_FILE =  r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Research/Data/2025/laser_tests/thermal_images/LCT_R5N16-0912_100PCT_2025-09-15_1/LCT_R5N16-0912_100PCT_2025-09-15_1_temperature_stack_removed_reflections.tif'
REFERENCE_ROD_DIAMETER = 1.27 # cm
MEASURED_ELLIPSE_RADII = [85.37, 155.85] # minor and major radius in pixels The major radius should be vertical
BEAM_CENTER = [1006, 416] # The beam center in pixels (measured in imageJ), used to draw a circular mask around it
# BEAM_CENTER = [1000, 436] # The beam center in pixels (measured in imageJ), used to draw a circular mask around it
BEAM_DIAMETER = 448 # The beam diameter in pixels.
REFLECTIONS_FRAME_REF = 59 # The number of the frame containing strong spurious reflections but no particles
FLARE_CENTER = [625, 608] # The center of the 468 px diameter circle around which we want to avoid the flare
PLOT_FRAME = 63

class ParticleTrajectoryAnalyzer:
    """
    Analyzes particle trajectories from temperature field images.

    Detects linear streaks in a circular region of interest and estimates
    particle velocities and sizes.
    """

    def __init__(self,
                 center_x: int = 1006,
                 center_y: int = 416,
                 analysis_diameter_pixels: int = 448,
                 exclusion_diameter_cm: float = 1.0,
                 pixel_scale_cm: float = 1.27 / 156,  # cm per pixel
                 min_streak_length: int = 10,
                 max_streak_length: int = 200):
        """
        Initialize the particle trajectory analyzer.

        Parameters:
        -----------
        center_x : int
            X-coordinate of analysis disc center (pixels)
        center_y : int
            Y-coordinate of analysis disc center (pixels)
        analysis_diameter_pixels : int
            Diameter of analysis region (pixels)
        exclusion_diameter_cm : float
            Diameter of central exclusion zone (cm)
        pixel_scale_cm : float
            Scale factor: cm per pixel
        min_streak_length : int
            Minimum detectable streak length (pixels)
        max_streak_length : int
            Maximum detectable streak length (pixels)
        """
        self.center_x = center_x
        self.center_y = center_y
        self.analysis_radius = analysis_diameter_pixels / 2
        self.exclusion_radius = exclusion_diameter_cm / pixel_scale_cm / 2  # Convert to pixels
        self.pixel_scale_cm = pixel_scale_cm
        self.min_streak_length = min_streak_length
        self.max_streak_length = max_streak_length

        print(f"Analysis region: radius = {self.analysis_radius:.1f} pixels "
              f"({self.analysis_radius * pixel_scale_cm:.2f} cm)")
        print(f"Exclusion zone: radius = {self.exclusion_radius:.1f} pixels "
              f"({exclusion_diameter_cm:.2f} cm)")

    def create_analysis_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a boolean mask for the annular analysis region.

        Parameters:
        -----------
        shape : tuple
            Shape of the image (height, width)

        Returns:
        --------
        mask : np.ndarray
            Boolean mask (True for analysis region)
        """
        y, x = np.ogrid[:shape[0], :shape[1]]

        # Distance from center
        dist_from_center = np.sqrt((x - self.center_x) ** 2 + (y - self.center_y) ** 2)

        # Annular mask: inside analysis circle but outside exclusion circle
        mask = (dist_from_center <= self.analysis_radius) & \
               (dist_from_center >= self.exclusion_radius)

        return mask

    def preprocess_temperature_field(self,
                                     temp_array: np.ndarray,
                                     threshold_percentile: float = 85) -> np.ndarray:
        """
        Preprocess temperature array to enhance particle streaks.

        Parameters:
        -----------
        temp_array : np.ndarray
            2D array of temperature values
        threshold_percentile : float
            Percentile for temperature thresholding (higher = stricter)

        Returns:
        --------
        binary_streaks : np.ndarray
            Binary image with enhanced streaks
        """
        # Apply analysis mask
        mask = self.create_analysis_mask(temp_array.shape)
        masked_temp = np.where(mask, temp_array, np.nan)

        # Calculate threshold based on percentile of valid data
        valid_temps = masked_temp[~np.isnan(masked_temp)]
        threshold = np.percentile(valid_temps, threshold_percentile)

        # Create binary image of hot streaks
        binary = (masked_temp > threshold) & mask

        # Morphological operations to enhance linear features
        # Remove small isolated pixels
        binary = morphology.remove_small_objects(binary, min_size=3)

        # Slight dilation to connect nearby pixels in streaks
        binary = morphology.binary_dilation(binary, morphology.disk(1))

        return binary.astype(np.uint8)

    def detect_streaks(self,
                       binary_image: np.ndarray,
                       threshold: int = 10,
                       line_gap: int = 5) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect linear streaks using probabilistic Hough transform.

        Parameters:
        -----------
        binary_image : np.ndarray
            Binary image with potential streaks
        threshold : int
            Minimum number of points for line detection
        line_gap : int
            Maximum gap between points on a line

        Returns:
        --------
        lines : list
            List of line segments as ((x0, y0), (x1, y1))
        """
        # Apply probabilistic Hough line transform
        lines = probabilistic_hough_line(
            binary_image,
            threshold=threshold,
            line_length=self.min_streak_length,
            line_gap=line_gap
        )

        # Filter by length
        filtered_lines = []
        for line in lines:
            (x0, y0), (x1, y1) = line
            length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

            if self.min_streak_length <= length <= self.max_streak_length:
                filtered_lines.append(line)

        return filtered_lines

    def calculate_velocities(self,
                             lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                             exposure_time: float) -> np.ndarray:
        """
        Calculate particle velocities from detected streaks.

        Parameters:
        -----------
        lines : list
            List of detected line segments
        exposure_time : float
            Camera exposure time (seconds)

        Returns:
        --------
        velocities : np.ndarray
            Array of velocities (cm/s) for each detected particle
        """
        velocities = []

        for line in lines:
            (x0, y0), (x1, y1) = line

            # Calculate streak length in pixels
            length_pixels = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

            # Convert to physical length
            length_cm = length_pixels * self.pixel_scale_cm

            # Calculate velocity
            velocity_cm_s = length_cm / exposure_time
            velocities.append(velocity_cm_s)

        return np.array(velocities)

    def analyze(self,
                temp_array: np.ndarray,
                exposure_time: float,
                threshold_percentile: float = 85) -> dict:
        """
        Complete analysis pipeline.

        Parameters:
        -----------
        temp_array : np.ndarray
            2D temperature array (1440 x 1080)
        exposure_time : float
            Camera exposure time (seconds)
        threshold_percentile : float
            Percentile for temperature thresholding

        Returns:
        --------
        results : dict
            Dictionary containing analysis results
        """
        print(f"\nAnalyzing temperature field (shape: {temp_array.shape})")
        print(f"Exposure time: {exposure_time} s")

        # Preprocess
        binary_image = self.preprocess_temperature_field(temp_array, threshold_percentile)

        # Detect streaks
        lines = self.detect_streaks(binary_image, threshold=50, line_gap=1)
        print(f"Detected {len(lines)} particle trajectories")

        if len(lines) == 0:
            print("Warning: No streaks detected. Try adjusting threshold_percentile.")
            return {
                'num_particles': 0,
                'lines': [],
                'velocities': np.array([]),
                'binary_image': binary_image,
                'mask': self.create_analysis_mask(temp_array.shape)
            }

        # Calculate velocities
        velocities = self.calculate_velocities(lines, exposure_time)

        # Statistics
        print(f"\nVelocity Statistics:")
        print(f"  Mean: {np.mean(velocities):.2f} cm/s")
        print(f"  Std:  {np.std(velocities):.2f} cm/s")
        print(f"  Min:  {np.min(velocities):.2f} cm/s")
        print(f"  Max:  {np.max(velocities):.2f} cm/s")

        return {
            'num_particles': len(lines),
            'lines': lines,
            'velocities': velocities,
            'streak_lengths_cm': velocities * exposure_time,
            'binary_image': binary_image,
            'mask': self.create_analysis_mask(temp_array.shape)
        }

    def visualize_results(self,
                          temp_array: np.ndarray,
                          results: dict,
                          save_path: Optional[str] = None):
        """
        Visualize detected particle trajectories.

        Parameters:
        -----------
        temp_array : np.ndarray
            Original temperature array
        results : dict
            Results from analyze() method
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original temperature field with mask
        ax = axes[0]
        masked_temp = np.where(results['mask'], temp_array, np.nan)
        im = ax.imshow(masked_temp, cmap='jet', origin='lower')

        # Draw analysis region boundaries
        circle_outer = plt.Circle((self.center_x, self.center_y),
                                  self.analysis_radius,
                                  color='cyan', fill=False, linewidth=2, label='Analysis region')
        circle_inner = plt.Circle((self.center_x, self.center_y),
                                  self.exclusion_radius,
                                  color='blue', fill=False, linewidth=2, label='Exclusion zone')
        ax.add_patch(circle_outer)
        ax.add_patch(circle_inner)
        ax.set_title('Temperature Field\n(Analysis Region)')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.legend(loc='upper right')
        plt.colorbar(im, ax=ax, label='Temperature')

        # Binary image with detected streaks
        ax = axes[1]
        ax.imshow(results['binary_image'], cmap='gray', origin='lower')
        for line in results['lines']:
            (x0, y0), (x1, y1) = line
            ax.plot([x0, x1], [y0, y1], 'r-', linewidth=2, alpha=0.7)
        ax.set_title(f'Detected Streaks\n({results["num_particles"]} particles)')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

        # Velocity histogram
        ax = axes[2]
        if len(results['velocities']) > 0:
            ax.hist(results['velocities'], bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(results['velocities']), color='red',
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(results["velocities"]):.1f} cm/s')
            ax.set_xlabel('Velocity (cm/s)')
            ax.set_ylabel('Count')
            ax.set_title('Particle Velocity Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No particles detected',
                    ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        plt.show()


def estimate_particle_size(velocity_cm_s: float,
                           fluid_density: float = 1.225e-3,  # g/cm³ (air at STP)
                           particle_density: float = 2.5,  # g/cm³ (typical solid)
                           fluid_viscosity: float = 1.81e-4,  # g/(cm·s) (air at 20°C)
                           method: str = 'stokes') -> dict:
    """
    Estimate particle size from velocity measurements.

    Several methods are available depending on the flow conditions.

    Parameters:
    -----------
    velocity_cm_s : float
        Measured particle velocity (cm/s)
    fluid_density : float
        Fluid density (g/cm³), default is air at STP
    particle_density : float
        Particle density (g/cm³)
    fluid_viscosity : float
        Dynamic viscosity (g/(cm·s)), default is air at 20°C
    method : str
        Estimation method: 'stokes', 'drag', or 'empirical'

    Returns:
    --------
    results : dict
        Dictionary with particle size estimates and relevant parameters
    """
    g = 981  # gravitational acceleration (cm/s²)

    results = {'velocity_cm_s': velocity_cm_s}

    if method == 'stokes':
        """
        Stokes' Law (valid for low Reynolds number, Re < 1):
        For settling particles: v = (2/9) * (ρ_p - ρ_f) * g * r² / μ
        For drag-limited motion: v = F_drag / (6πμr)

        Assumes laminar flow around spherical particles.
        """
        # Terminal velocity assumption (gravity-driven)
        # r = sqrt(9 * μ * v / (2 * (ρ_p - ρ_f) * g))

        diameter_cm = np.sqrt(18 * fluid_viscosity * velocity_cm_s /
                              ((particle_density - fluid_density) * g))
        diameter_microns = diameter_cm * 1e4

        # Calculate Reynolds number for validation
        Re = fluid_density * velocity_cm_s * diameter_cm / fluid_viscosity

        results.update({
            'method': 'Stokes Law (settling)',
            'diameter_cm': diameter_cm,
            'diameter_microns': diameter_microns,
            'reynolds_number': Re,
            'valid': Re < 1,
            'notes': 'Valid for Re < 1 (laminar flow). If Re > 1, try drag method.'
        })

    elif method == 'drag':
        """
        For higher Reynolds numbers (1 < Re < 1000), use empirical drag correlations.
        Iterative solution required.
        """
        # Initial guess using Stokes
        d_guess = np.sqrt(18 * fluid_viscosity * velocity_cm_s /
                          ((particle_density - fluid_density) * g))

        # Iterative refinement using drag coefficient correlation
        for _ in range(10):
            Re = fluid_density * velocity_cm_s * d_guess / fluid_viscosity

            # Empirical drag coefficient (Schiller-Naumann)
            if Re < 1:
                Cd = 24 / Re
            elif Re < 1000:
                Cd = 24 / Re * (1 + 0.15 * Re ** 0.687)
            else:
                Cd = 0.44  # Constant for high Re

            # Update diameter estimate
            # F_drag = Cd * 0.5 * ρ_f * v² * A = (ρ_p - ρ_f) * V * g
            # A = π * d² / 4,  V = π * d³ / 6
            d_new = (4 * (particle_density - fluid_density) * g * d_guess ** 3 /
                     (3 * Cd * fluid_density * velocity_cm_s ** 2)) ** (1 / 2)

            if abs(d_new - d_guess) / d_guess < 0.01:
                break
            d_guess = d_new

        diameter_cm = d_guess
        diameter_microns = diameter_cm * 1e4
        Re_final = fluid_density * velocity_cm_s * diameter_cm / fluid_viscosity

        results.update({
            'method': 'Drag coefficient (iterative)',
            'diameter_cm': diameter_cm,
            'diameter_microns': diameter_microns,
            'reynolds_number': Re_final,
            'drag_coefficient': Cd,
            'valid': 1 < Re_final < 1000,
            'notes': 'Valid for 1 < Re < 1000 (transitional regime).'
        })

    elif method == 'empirical':
        """
        Empirical correlation based on observed streak characteristics.
        This requires calibration with known particles.
        """
        # Placeholder - would need experimental calibration
        # Example: diameter ∝ velocity^α for specific conditions
        alpha = 0.5  # Calibration parameter
        k = 0.01  # Calibration constant

        diameter_cm = k * velocity_cm_s ** alpha
        diameter_microns = diameter_cm * 1e4

        results.update({
            'method': 'Empirical correlation',
            'diameter_cm': diameter_cm,
            'diameter_microns': diameter_microns,
            'notes': 'Requires experimental calibration. Values are illustrative only.'
        })

    return results


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("PARTICLE TRAJECTORY ANALYZER - DEMONSTRATION")
    print("=" * 70)

    # Create synthetic temperature field for demonstration
    # np.random.seed(42)
    # temp_field = np.random.normal(300, 10, (1440, 1080))  # Background temperature
    #
    # # Add some synthetic streaks (hot particle trails)
    # center_x, center_y = 1006, 416
    # for i in range(15):
    #     # Random position in annular region
    #     angle = np.random.uniform(0, 2 * np.pi)
    #     radius = np.random.uniform(100, 224)  # Between exclusion and analysis radius
    #     x_start = int(center_x + radius * np.cos(angle))
    #     y_start = int(center_y + radius * np.sin(angle))
    #
    #     # Random streak direction and length
    #     streak_angle = np.random.uniform(0, 2 * np.pi)
    #     streak_length = np.random.randint(15, 150)
    #
    #     # Draw streak
    #     for t in range(streak_length):
    #         x = int(x_start + t * np.cos(streak_angle))
    #         y = int(y_start + t * np.sin(streak_angle))
    #         if 0 <= x < 1080 and 0 <= y < 1440:
    #             # Hot streak with Gaussian profile
    #             temp_field[y, x] += 30 * np.exp(-0.5 * (t / 20) ** 2)

    # Load the data from the tiff file
    patht_to_tiff = Path(STACK_FILE)
    with tifffile.TiffFile(str(patht_to_tiff)) as tif:
        temp_field = np.array(tif.pages[PLOT_FRAME].asarray())

    # Initialize analyzer
    analyzer = ParticleTrajectoryAnalyzer(
        center_x=BEAM_CENTER[0],
        center_y=BEAM_CENTER[1],
        analysis_diameter_pixels=848,
        exclusion_diameter_cm=2.5,
        pixel_scale_cm=1.27 / 156,
        min_streak_length=20,
        max_streak_length=5000
    )

    # Perform analysis
    exposure_time = 100E-6  # seconds (10 ms exposure)
    results = analyzer.analyze(temp_field, exposure_time, threshold_percentile=85)

    # Visualize
    analyzer.visualize_results(temp_field, results, save_path='./particle_trajectories.png')

    # Particle size estimation example
    if len(results['velocities']) > 0:
        print("\n" + "=" * 70)
        print("PARTICLE SIZE ESTIMATION")
        print("=" * 70)

        mean_velocity = np.mean(results['velocities'])

        print(f"\nUsing mean velocity: {mean_velocity:.2f} cm/s")
        print("\n--- Method 1: Stokes Law (Low Re) ---")
        size_stokes = estimate_particle_size(mean_velocity, method='stokes')
        print(f"Estimated diameter: {size_stokes['diameter_microns']:.1f} μm")
        print(f"Reynolds number: {size_stokes['reynolds_number']:.3f}")
        print(f"Valid: {size_stokes['valid']}")

        print("\n--- Method 2: Drag Coefficient (Higher Re) ---")
        size_drag = estimate_particle_size(mean_velocity, method='drag')
        print(f"Estimated diameter: {size_drag['diameter_microns']:.1f} μm")
        print(f"Reynolds number: {size_drag['reynolds_number']:.3f}")
        print(f"Drag coefficient: {size_drag['drag_coefficient']:.3f}")

        print("\n" + "=" * 70)
        print("PARTICLE SIZE ESTIMATION - METHODOLOGY")
        print("=" * 70)
        print("""
From trajectory data, particle size can be estimated using:

1. **Stokes' Law** (for small, slow particles, Re < 1):
   - Assumes laminar flow around spherical particles
   - d = √(18μv / ((ρₚ - ρf)g))
   - Where: μ = viscosity, v = velocity, ρₚ = particle density, 
            ρf = fluid density, g = gravity
   - Best for particles < ~50 μm in gases

2. **Drag Coefficient Method** (for 1 < Re < 1000):
   - Uses empirical drag correlations (e.g., Schiller-Naumann)
   - Iteratively solves for diameter given velocity
   - More accurate for larger/faster particles

3. **Thermal Signature Analysis**:
   - Streak width may correlate with particle size
   - Thermal wake characteristics
   - Requires careful calibration

4. **Important Considerations**:
   - Assumes spherical particles
   - Requires knowing: fluid properties, particle density
   - Flow field affects trajectories (convection, turbulence)
   - For thermal imaging: emissivity, particle temperature

**Recommended Approach**:
1. Measure velocities from streaks (as coded above)
2. Estimate Re number
3. Choose appropriate model (Stokes vs. drag)
4. Validate with known calibration particles
5. Consider thermal wake width as secondary indicator
        """)