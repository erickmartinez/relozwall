import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches


class CylinderShadowAnalyzer:
    """
    Analyzes shadow and illumination of a cylinder under collimated light.

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
        # Assuming light comes from a plane with normal at angle theta from z-axis
        # Light direction is opposite to the normal
        self.light_dir = np.array([-np.sin(theta), 0, -np.cos(theta)])

    def calculate_shadowed_area(self):
        """
        Calculate the total shadowed area of the cylinder.

        Returns:
        --------
        float : Total shadowed area
        """
        # For a cylinder, we need to consider:
        # 1. Shadow on the curved surface
        # 2. Shadow on the top/bottom circular faces

        curved_shadow = self._curved_surface_shadow()
        top_face_shadow = self._circular_face_shadow(face='top')
        bottom_face_shadow = self._circular_face_shadow(face='bottom')

        return curved_shadow + top_face_shadow + bottom_face_shadow

    def _curved_surface_shadow(self):
        """Calculate shadow area on the curved surface of the cylinder."""
        # The curved surface shadow depends on the angle theta
        # For collimated light, the shadow forms an elliptical pattern

        # Project light direction onto xy-plane
        light_xy = np.sqrt(self.light_dir[0] ** 2 + self.light_dir[1] ** 2)

        if light_xy == 0:  # Light parallel to cylinder axis
            return 0

        # The shadowed region on curved surface forms a sinusoidal pattern
        # Shadow area = radius * height * integral of shadowing function

        # Angular range where surface is shadowed
        # This is approximately half the cylinder when viewed from the side
        shadow_angle_range = np.pi  # Half the circumference

        # Effective shadow width considering the light angle
        effective_shadow_width = shadow_angle_range * self.radius

        # Height projection factor
        height_factor = abs(self.light_dir[2])  # cos component

        return effective_shadow_width * self.height * light_xy

    def _circular_face_shadow(self, face='top'):
        """Calculate shadow area on circular face (top or bottom)."""
        # Determine if the face is facing towards or away from light
        if face == 'top':
            face_normal = np.array([0, 0, 1])
        else:  # bottom
            face_normal = np.array([0, 0, -1])

        # Dot product to determine if face is illuminated
        dot_product = np.dot(-self.light_dir, face_normal)

        if dot_product <= 0:  # Face is not illuminated (in shadow)
            return np.pi * self.radius ** 2
        else:  # Face is illuminated
            return 0

    def calculate_total_surface_area(self):
        """Calculate total surface area of the cylinder."""
        curved_area = 2 * np.pi * self.radius * self.height
        circular_faces_area = 2 * np.pi * self.radius ** 2
        return curved_area + circular_faces_area

    def calculate_illuminated_fraction(self):
        """
        Calculate the fraction of cylinder surface that is illuminated.

        Returns:
        --------
        float : Illuminated fraction (0 to 1)
        """
        total_area = self.calculate_total_surface_area()
        shadowed_area = self.calculate_shadowed_area()
        illuminated_area = total_area - shadowed_area

        return max(0, min(1, illuminated_area / total_area))


def visualize_cylinder_and_light(analyzer, ax=None):
    """Visualize the cylinder and light direction in 3D."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Generate cylinder surface points
    theta_range = np.linspace(0, 2 * np.pi, 50)
    z_range = np.linspace(0, analyzer.height, 20)
    THETA, Z = np.meshgrid(theta_range, z_range)

    X = analyzer.radius * np.cos(THETA)
    Y = analyzer.radius * np.sin(THETA)

    # Plot cylinder surface
    ax.plot_surface(X, Y, Z, alpha=0.3, color='lightblue', label='Cylinder')

    # Plot top and bottom circles
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    x_circle = analyzer.radius * np.cos(theta_circle)
    y_circle = analyzer.radius * np.sin(theta_circle)

    # Top circle
    ax.plot(x_circle, y_circle, analyzer.height, 'b-', linewidth=2)
    # Bottom circle
    ax.plot(x_circle, y_circle, 0, 'b-', linewidth=2)

    # Draw light direction arrow
    arrow_start = np.array([analyzer.radius * 2, 0, analyzer.height / 2])
    arrow_direction = analyzer.light_dir * analyzer.radius

    ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
              arrow_direction[0], arrow_direction[1], arrow_direction[2],
              color='yellow', arrow_length_ratio=0.1, linewidth=3,
              label=f'Light Direction (θ={np.degrees(analyzer.theta):.1f}°)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Cylinder under Collimated Light\nθ = {np.degrees(analyzer.theta):.1f}°')

    # Set equal aspect ratio
    max_range = max(analyzer.radius * 2, analyzer.height)
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range])

    return ax


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


def run_complete_analysis():
    """Run complete analysis with 3D visualization and plots."""
    # Example parameters
    diameter = 2.0
    height = 4.0
    theta = np.pi / 6  # 30 degrees

    # Create analyzer
    analyzer = CylinderShadowAnalyzer(diameter, height, theta)

    # Print results
    print(f"Cylinder Analysis:")
    print(f"Diameter: {diameter}")
    print(f"Height: {height}")
    print(f"Light angle θ: {np.degrees(theta):.1f}°")
    print(f"Total surface area: {analyzer.calculate_total_surface_area():.3f}")
    print(f"Shadowed area: {analyzer.calculate_shadowed_area():.3f}")
    print(f"Illuminated fraction: {analyzer.calculate_illuminated_fraction():.3f}")
    print()

    # Create 3D visualization
    fig = plt.figure(figsize=(15, 10))

    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    visualize_cylinder_and_light(analyzer, ax1)

    # Test different configurations
    ax2 = fig.add_subplot(222, projection='3d')
    analyzer2 = CylinderShadowAnalyzer(diameter, height, np.pi / 3)  # 60 degrees
    visualize_cylinder_and_light(analyzer2, ax2)

    plt.tight_layout()
    plt.show()

    # Run parameter studies
    print("Running height analysis...")
    heights, height_illumination = test_illumination_vs_height()

    print("Running angle analysis...")
    thetas, theta_illumination = test_illumination_vs_theta()

    return analyzer


if __name__ == "__main__":
    # Run the complete analysis
    analyzer = run_complete_analysis()

    # Additional test cases
    print("\nTesting edge cases:")

    # Light parallel to cylinder axis (θ = 0)
    test1 = CylinderShadowAnalyzer(2.0, 5.0, 0.0)
    print(f"θ = 0°: Illuminated fraction = {test1.calculate_illuminated_fraction():.3f}")

    # Light perpendicular to cylinder axis (θ = 90°)
    test2 = CylinderShadowAnalyzer(2.0, 5.0, np.pi / 2)
    print(f"θ = 90°: Illuminated fraction = {test2.calculate_illuminated_fraction():.3f}")

    # Light at 45°
    test3 = CylinderShadowAnalyzer(2.0, 5.0, np.pi / 4)
    print(f"θ = 45°: Illuminated fraction = {test3.calculate_illuminated_fraction():.3f}")