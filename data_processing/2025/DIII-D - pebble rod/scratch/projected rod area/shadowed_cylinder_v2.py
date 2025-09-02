import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


class CylinderShadowAnalyzer:
    """
    Analyzes shadow and illumination of a cylinder under collimated light with raytracing.

    The cylinder axis is along the z-direction, and the light source is
    characterized by a normal vector at angle theta from the z-axis.
    """

    def __init__(self, diameter, height, theta):
        """
        Initialize the cylinder shadow analyzer.

        Parameters:
        -----------
        diameter : float
            Cylinder diameter
        height : float
            Cylinder height
        theta : float
            Angle between light source normal and cylinder axis (radians)
        """
        self.diameter = diameter
        self.radius = diameter / 2
        self.height = height
        self.theta = theta

        # Light direction vector (normalized)
        # Light comes from direction opposite to the normal
        self.light_dir = np.array([-np.sin(theta), 0, -np.cos(theta)])

    def calculate_shadowed_area(self):
        """Calculate the total shadowed area of the cylinder."""
        curved_shadow = self._curved_surface_shadow()
        top_face_shadow = self._circular_face_shadow(face='top')
        bottom_face_shadow = self._circular_face_shadow(face='bottom')

        return curved_shadow + top_face_shadow + bottom_face_shadow

    def _curved_surface_shadow(self):
        """Calculate shadow area on the curved surface of the cylinder."""
        light_xy = np.sqrt(self.light_dir[0] ** 2 + self.light_dir[1] ** 2)

        if light_xy == 0:  # Light parallel to cylinder axis
            return 0

        shadow_angle_range = np.pi
        effective_shadow_width = shadow_angle_range * self.radius
        height_factor = abs(self.light_dir[2])

        return effective_shadow_width * self.height * light_xy

    def _circular_face_shadow(self, face='top'):
        """Calculate shadow area on circular face (top or bottom)."""
        if face == 'top':
            face_normal = np.array([0, 0, 1])
        else:
            face_normal = np.array([0, 0, -1])

        dot_product = np.dot(-self.light_dir, face_normal)

        if dot_product <= 0:
            return np.pi * self.radius ** 2
        else:
            return 0

    def calculate_total_surface_area(self):
        """Calculate total surface area of the cylinder."""
        curved_area = 2 * np.pi * self.radius * self.height
        circular_faces_area = 2 * np.pi * self.radius ** 2
        return curved_area + circular_faces_area

    def calculate_illuminated_fraction(self):
        """Calculate the fraction of cylinder surface that is illuminated."""
        total_area = self.calculate_total_surface_area()
        shadowed_area = self.calculate_shadowed_area()
        illuminated_area = total_area - shadowed_area

        return max(0, min(1, illuminated_area / total_area))

    def raytrace_cylinder_surface(self, n_theta=100, n_z=50):
        """
        Raytrace the cylinder surface to determine illumination.

        Parameters:
        -----------
        n_theta : int
            Number of angular divisions around cylinder
        n_z : int
            Number of height divisions

        Returns:
        --------
        X, Y, Z : arrays
            Coordinate arrays for surface points
        illumination : array
            Illumination values (0 = shadow, 1 = fully lit)
        """
        # Create surface grid
        theta_range = np.linspace(0, 2 * np.pi, n_theta)
        z_range = np.linspace(0, self.height, n_z)
        THETA, Z = np.meshgrid(theta_range, z_range)

        X = self.radius * np.cos(THETA)
        Y = self.radius * np.sin(THETA)

        # Calculate surface normals
        normals = np.zeros((n_z, n_theta, 3))
        normals[:, :, 0] = np.cos(THETA)  # Normal x-component
        normals[:, :, 1] = np.sin(THETA)  # Normal y-component
        normals[:, :, 2] = 0  # Normal z-component (curved surface)

        # Calculate illumination using dot product
        illumination = np.zeros((n_z, n_theta))

        for i in range(n_z):
            for j in range(n_theta):
                # Dot product between surface normal and light direction
                dot_product = np.dot(normals[i, j], -self.light_dir)
                # Only illuminate if surface faces the light (dot product > 0)
                illumination[i, j] = max(0, dot_product)

        return X, Y, Z, illumination

    def raytrace_circular_faces(self, n_r=30, n_theta=60):
        """
        Raytrace the circular faces (top and bottom) of the cylinder.

        Returns:
        --------
        dict : Contains X, Y, Z coordinates and illumination for both faces
        """
        # Create radial grid for circular faces
        r_range = np.linspace(0, self.radius, n_r)
        theta_range = np.linspace(0, 2 * np.pi, n_theta)
        R, THETA = np.meshgrid(r_range, theta_range)

        X_face = R * np.cos(THETA)
        Y_face = R * np.sin(THETA)

        faces = {}

        # Top face (z = height)
        Z_top = np.full_like(X_face, self.height)
        top_normal = np.array([0, 0, 1])
        top_illumination = max(0, np.dot(top_normal, -self.light_dir))
        faces['top'] = {
            'X': X_face, 'Y': Y_face, 'Z': Z_top,
            'illumination': np.full_like(X_face, top_illumination)
        }

        # Bottom face (z = 0)
        Z_bottom = np.zeros_like(X_face)
        bottom_normal = np.array([0, 0, -1])
        bottom_illumination = max(0, np.dot(bottom_normal, -self.light_dir))
        faces['bottom'] = {
            'X': X_face, 'Y': Y_face, 'Z': Z_bottom,
            'illumination': np.full_like(X_face, bottom_illumination)
        }

        return faces

    def cast_shadow_rays(self, n_rays=500):
        """
        Cast shadow rays to visualize light paths and shadows.

        Returns:
        --------
        dict : Ray information for visualization
        """
        # Generate random points on a plane perpendicular to light direction
        # This represents the light source plane

        # Create basis vectors for the light source plane
        if abs(self.light_dir[2]) < 0.9:
            u = np.cross(self.light_dir, [0, 0, 1])
        else:
            u = np.cross(self.light_dir, [1, 0, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(self.light_dir, u)
        v = v / np.linalg.norm(v)

        # Random points on the source plane
        source_size = max(self.radius * 3, self.height)
        u_coords = np.random.uniform(-source_size, source_size, n_rays)
        v_coords = np.random.uniform(-source_size, source_size, n_rays)

        # Light source plane center (positioned away from cylinder)
        source_center = np.array([0, 0, self.height / 2]) - self.light_dir * source_size * 2

        # Ray starting points
        ray_starts = source_center[:, np.newaxis] + u[:, np.newaxis] * u_coords + v[:, np.newaxis] * v_coords

        # Ray directions (all parallel for collimated light)
        ray_directions = np.tile(self.light_dir[:, np.newaxis], (1, n_rays))

        return {
            'starts': ray_starts.T,
            'directions': ray_directions.T,
            'source_center': source_center
        }


def visualize_cylinder_with_shadows(analyzer, show_rays=True, n_rays=100):
    """Visualize the cylinder with raytraced shadows and optional light rays."""
    fig = plt.figure(figsize=(15, 12))

    # Create custom colormap for shadows
    colors_shadow = ['#000080', '#4169E1', '#87CEEB', '#FFFF00']  # Dark blue to yellow
    n_bins = 100
    cmap_shadow = LinearSegmentedColormap.from_list('shadow', colors_shadow, N=n_bins)

    # Main 3D plot
    ax1 = fig.add_subplot(221, projection='3d')

    # Raytrace cylinder surface
    X, Y, Z, illumination = analyzer.raytrace_cylinder_surface()

    # Plot cylinder surface with shadows
    surf = ax1.plot_surface(X, Y, Z, facecolors=cmap_shadow(illumination),
                            alpha=0.8, linewidth=0, antialiased=True)

    # Raytrace and plot circular faces
    faces = analyzer.raytrace_circular_faces()

    # Top face
    ax1.plot_surface(faces['top']['X'], faces['top']['Y'], faces['top']['Z'],
                     facecolors=cmap_shadow(faces['top']['illumination']),
                     alpha=0.8, linewidth=0)

    # Bottom face
    ax1.plot_surface(faces['bottom']['X'], faces['bottom']['Y'], faces['bottom']['Z'],
                     facecolors=cmap_shadow(faces['bottom']['illumination']),
                     alpha=0.8, linewidth=0)

    # Add light rays if requested
    if show_rays:
        ray_info = analyzer.cast_shadow_rays(n_rays)

        # Draw subset of rays
        n_display_rays = min(20, n_rays)
        indices = np.random.choice(n_rays, n_display_rays, replace=False)

        for i in indices:
            start = ray_info['starts'][i]
            direction = ray_info['directions'][i]
            end = start + direction * (analyzer.radius * 4)

            ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                     'yellow', alpha=0.3, linewidth=0.5)

    # Draw light direction arrow
    arrow_start = np.array([analyzer.radius * 2.5, 0, analyzer.height / 2])
    arrow_direction = analyzer.light_dir * analyzer.radius

    ax1.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
               arrow_direction[0], arrow_direction[1], arrow_direction[2],
               color='red', arrow_length_ratio=0.1, linewidth=3)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Raytraced Cylinder Shadows\nθ = {np.degrees(analyzer.theta):.1f}°')

    # Set equal aspect ratio
    max_range = max(analyzer.radius * 3, analyzer.height)
    ax1.set_xlim([-max_range / 2, max_range / 2])
    ax1.set_ylim([-max_range / 2, max_range / 2])
    ax1.set_zlim([0, analyzer.height])

    # Side view (2D projection)
    ax2 = fig.add_subplot(222)

    # Project cylinder onto xz-plane for side view
    theta_side = np.linspace(-np.pi / 2, np.pi / 2, 50)
    x_side = analyzer.radius * np.cos(theta_side)
    z_bottom = np.zeros_like(x_side)
    z_top = np.full_like(x_side, analyzer.height)

    # Create illumination map for side view
    illumination_side = np.zeros_like(theta_side)
    for i, theta in enumerate(theta_side):
        normal = np.array([np.cos(theta), 0, 0])
        illumination_side[i] = max(0, np.dot(normal, -analyzer.light_dir))

    # Plot cylinder outline
    ax2.fill_between(x_side, z_bottom, z_top,
                     color=cmap_shadow(np.mean(illumination_side)), alpha=0.7)
    ax2.plot(x_side, z_bottom, 'k-', linewidth=2)  # Bottom edge
    ax2.plot(x_side, z_top, 'k-', linewidth=2)  # Top edge
    ax2.plot([x_side[0], x_side[0]], [0, analyzer.height], 'k-', linewidth=2)  # Left edge
    ax2.plot([x_side[-1], x_side[-1]], [0, analyzer.height], 'k-', linewidth=2)  # Right edge

    # Draw light direction
    ax2.arrow(analyzer.radius * 2, analyzer.height / 2,
              analyzer.light_dir[0] * analyzer.radius, analyzer.light_dir[2] * analyzer.radius,
              head_width=0.1, head_length=0.1, fc='red', ec='red')

    ax2.set_xlim([-analyzer.radius * 2, analyzer.radius * 3])
    ax2.set_ylim([-0.5, analyzer.height + 0.5])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('Side View with Shadow')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Shadow intensity map
    ax3 = fig.add_subplot(223)

    # Create 2D unwrapped view of cylinder surface
    theta_unwrap = np.linspace(0, 2 * np.pi, 100)
    z_unwrap = np.linspace(0, analyzer.height, 50)
    THETA_unwrap, Z_unwrap = np.meshgrid(theta_unwrap, z_unwrap)

    # Calculate illumination for unwrapped surface
    illumination_unwrap = np.zeros_like(THETA_unwrap)
    for i in range(len(z_unwrap)):
        for j in range(len(theta_unwrap)):
            normal = np.array([np.cos(theta_unwrap[j]), np.sin(theta_unwrap[j]), 0])
            illumination_unwrap[i, j] = max(0, np.dot(normal, -analyzer.light_dir))

    im = ax3.imshow(illumination_unwrap, extent=[0, 2 * np.pi, 0, analyzer.height],
                    aspect='auto', origin='lower', cmap=cmap_shadow)
    ax3.set_xlabel('Angular Position (radians)')
    ax3.set_ylabel('Height')
    ax3.set_title('Unwrapped Cylinder Surface\nIllumination Map')

    # Add angular markers
    ax3.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax3.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    plt.colorbar(im, ax=ax3, label='Illumination Intensity')

    # Analysis results
    ax4 = fig.add_subplot(224)
    ax4.axis('off')

    # Display numerical results
    total_area = analyzer.calculate_total_surface_area()
    shadowed_area = analyzer.calculate_shadowed_area()
    illuminated_fraction = analyzer.calculate_illuminated_fraction()

    results_text = f"""
    CYLINDER SHADOW ANALYSIS

    Parameters:
    • Diameter: {analyzer.diameter:.2f}
    • Height: {analyzer.height:.2f}
    • Light angle θ: {np.degrees(analyzer.theta):.1f}°

    Results:
    • Total surface area: {total_area:.3f}
    • Shadowed area: {shadowed_area:.3f}
    • Illuminated area: {total_area - shadowed_area:.3f}
    • Illuminated fraction: {illuminated_fraction:.1%}

    Light Direction Vector:
    • x: {analyzer.light_dir[0]:.3f}
    • y: {analyzer.light_dir[1]:.3f}
    • z: {analyzer.light_dir[2]:.3f}
    """

    ax4.text(0.05, 0.95, results_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.show()

    return fig


def test_illumination_vs_height():
    """Test and plot illuminated fraction vs cylinder height."""
    diameter = 2.0
    theta = np.pi / 4  # 45 degrees
    heights = np.linspace(0.5, 10, 50)

    illuminated_fractions = []

    for height in heights:
        analyzer = CylinderShadowAnalyzer(diameter, height, theta)
        illuminated_fractions.append(analyzer.calculate_illuminated_fraction())

    plt.figure(figsize=(10, 6))
    plt.plot(heights, illuminated_fractions, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Cylinder Height')
    plt.ylabel('Illuminated Fraction')
    plt.title(f'Illuminated Fraction vs Height (θ = {np.degrees(theta):.1f}°, d = {diameter})')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.show()

    return heights, illuminated_fractions


def test_illumination_vs_theta():
    """Test and plot illuminated fraction vs angle theta."""
    diameter = 2.0
    height = 5.0
    thetas = np.linspace(0, np.pi / 2, 50)  # 0 to 90 degrees

    illuminated_fractions = []

    for theta in thetas:
        analyzer = CylinderShadowAnalyzer(diameter, height, theta)
        illuminated_fractions.append(analyzer.calculate_illuminated_fraction())

    plt.figure(figsize=(10, 6))
    plt.plot(np.degrees(thetas), illuminated_fractions, 'r-', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Angle θ (degrees)')
    plt.ylabel('Illuminated Fraction')
    plt.title(f'Illuminated Fraction vs Light Angle (h = {height}, d = {diameter})')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.show()

    return thetas, illuminated_fractions


def compare_multiple_angles():
    """Compare raytraced visualizations for multiple light angles."""
    diameter = 2.0
    height = 4.0
    angles = [0, np.pi / 6, np.pi / 4, np.pi / 3]  # 0°, 30°, 45°, 60°

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    colors_shadow = ['#000080', '#4169E1', '#87CEEB', '#FFFF00']
    cmap_shadow = LinearSegmentedColormap.from_list('shadow', colors_shadow, N=100)

    for i, theta in enumerate(angles):
        ax = axes[i]
        analyzer = CylinderShadowAnalyzer(diameter, height, theta)

        # Raytrace cylinder surface
        X, Y, Z, illumination = analyzer.raytrace_cylinder_surface()

        # Plot cylinder surface with shadows
        ax.plot_surface(X, Y, Z, facecolors=cmap_shadow(illumination),
                        alpha=0.8, linewidth=0, antialiased=True)

        # Raytrace and plot circular faces
        faces = analyzer.raytrace_circular_faces()

        ax.plot_surface(faces['top']['X'], faces['top']['Y'], faces['top']['Z'],
                        facecolors=cmap_shadow(faces['top']['illumination']),
                        alpha=0.8, linewidth=0)

        ax.plot_surface(faces['bottom']['X'], faces['bottom']['Y'], faces['bottom']['Z'],
                        facecolors=cmap_shadow(faces['bottom']['illumination']),
                        alpha=0.8, linewidth=0)

        # Draw light direction arrow
        arrow_start = np.array([analyzer.radius * 2, 0, analyzer.height / 2])
        arrow_direction = analyzer.light_dir * analyzer.radius

        ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                  arrow_direction[0], arrow_direction[1], arrow_direction[2],
                  color='red', arrow_length_ratio=0.1, linewidth=3)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'θ = {np.degrees(theta):.0f}° (Illuminated: {analyzer.calculate_illuminated_fraction():.1%})')

        # Set equal aspect ratio
        max_range = max(analyzer.radius * 2.5, analyzer.height)
        ax.set_xlim([-max_range / 2, max_range / 2])
        ax.set_ylim([-max_range / 2, max_range / 2])
        ax.set_zlim([0, analyzer.height])

    plt.tight_layout()
    plt.show()


def run_complete_analysis():
    """Run complete analysis with raytracing visualization."""
    # Example parameters
    diameter = 2.0
    height = 4.0
    theta = np.pi / 4  # 45 degrees

    # Create analyzer
    analyzer = CylinderShadowAnalyzer(diameter, height, theta)

    # Print results
    print(f"Cylinder Raytracing Analysis:")
    print(f"Diameter: {diameter}")
    print(f"Height: {height}")
    print(f"Light angle θ: {np.degrees(theta):.1f}°")
    print(f"Total surface area: {analyzer.calculate_total_surface_area():.3f}")
    print(f"Shadowed area: {analyzer.calculate_shadowed_area():.3f}")
    print(f"Illuminated fraction: {analyzer.calculate_illuminated_fraction():.3f}")
    print()

    # Create detailed raytraced visualization
    print("Generating raytraced visualization...")
    visualize_cylinder_with_shadows(analyzer, show_rays=True, n_rays=100)

    # Compare multiple angles
    print("Comparing multiple light angles...")
    compare_multiple_angles()

    # Run parameter studies
    print("Running height analysis...")
    heights, height_illumination = test_illumination_vs_height()

    print("Running angle analysis...")
    thetas, theta_illumination = test_illumination_vs_theta()

    return analyzer


if __name__ == "__main__":
    # Run the complete analysis with raytracing
    analyzer = run_complete_analysis()

    # Additional test cases
    print("\nTesting edge cases with raytracing:")

    # Light parallel to cylinder axis (θ = 0)
    test1 = CylinderShadowAnalyzer(2.0, 5.0, 0.0)
    print(f"θ = 0°: Illuminated fraction = {test1.calculate_illuminated_fraction():.3f}")

    # Light perpendicular to cylinder axis (θ = 90°)
    test2 = CylinderShadowAnalyzer(2.0, 5.0, np.pi / 2)
    print(f"θ = 90°: Illuminated fraction = {test2.calculate_illuminated_fraction():.3f}")