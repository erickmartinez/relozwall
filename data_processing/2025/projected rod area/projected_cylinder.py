import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_interaction_vs_angle(diameter, height):
    """
    Analyze how ion beam interaction area changes with beam angle.
    """
    angles = np.linspace(0, 90, 91)
    total_areas = []
    top_face_areas = []
    curved_surface_areas = []

    for angle in angles:
        total, top_face, curved = calculate_ion_beam_interaction_area(diameter, height, angle)
        total_areas.append(total)
        top_face_areas.append(top_face)
        curved_surface_areas.append(curved)

    plt.figure(figsize=(12, 8))

    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(angles, total_areas, 'b-', linewidth=3, label='Total Ion Interaction Area')
    plt.plot(angles, top_face_areas, 'r--', linewidth=2, label='Top Face Contribution')
    plt.plot(angles, curved_surface_areas, 'g--', linewidth=2, label='Curved Surface Contribution')

    plt.xlabel('Ion Beam Angle θ (degrees)')
    plt.ylabel('Interaction Area')
    plt.title('Ion Beam Interaction Area vs Beam Angle')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Physical interpretation
    plt.subplot(2, 1, 2)
    plt.plot(angles, np.cos(np.radians(angles)), 'r:', linewidth=2, label='cos(θ) - Top Face Factor')
    plt.plot(angles, np.sin(np.radians(angles)), 'g:', linewidth=2, label='sin(θ) - Curved Surface Factor')

    plt.xlabel('Ion Beam Angle θ (degrees)')
    plt.ylabel('Angular Factor')
    plt.title('Angular Factors: cos(θ) for Top Face, sin(θ) for Curved Surface')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def calculate_ion_beam_interaction_area(diameter, height, theta_degrees):
    """
    Calculate the effective area of ion beam interaction with a vertically aligned cylinder.
    This represents the physical cross-sectional area that intercepts the ion beam.

    Parameters:
    diameter (float): Diameter of the cylinder
    height (float): Height of the cylinder
    theta_degrees (float): Angle in degrees between cylinder axis and ion beam direction

    Returns:
    tuple: (total_interaction_area, top_face_area, curved_surface_area)
    """
    # Convert angle to radians
    theta = np.radians(theta_degrees)

    # Radius of cylinder
    radius = diameter / 2

    # Calculate ion beam interaction areas

    # 1. Top circular face - flux-weighted effective area
    # When beam hits at angle theta, effective area = original_area * cos(theta)
    # This is also equal to the geometric projection of the circle
    top_face_area = np.pi * radius ** 2 * abs(np.cos(theta))

    # 2. Curved cylindrical surface - only the half facing the beam
    # The curved surface has a "frontal cross-section" visible to the beam
    # This is NOT the full circumference, but only the portion facing the beam
    # Effective area = (π * radius) * height * sin(theta)
    # where π * radius is the "frontal width" (half circumference)
    curved_surface_area = np.pi * radius * height * abs(np.sin(theta))

    # 3. Bottom face is completely shadowed - no contribution
    bottom_face_area = 0

    # Total ion beam interaction area
    total_interaction_area = top_face_area + curved_surface_area

    return total_interaction_area, top_face_area, curved_surface_area


def calculate_cylinder_projection_area(diameter, height, theta_degrees):
    """
    Calculate the projected area of a vertically aligned cylinder onto a tilted plane.
    This is the geometric projection (shadow) and matches the ion beam interaction area.

    Parameters:
    diameter (float): Diameter of the cylinder
    height (float): Height of the cylinder
    theta_degrees (float): Angle in degrees between cylinder axis and plane normal

    Returns:
    tuple: (total_projected_area, circular_projection_area, rectangular_projection_area)
    """
    # For comparison - this should match the ion beam calculation
    return calculate_ion_beam_interaction_area(diameter, height, theta_degrees)


def calculate_current_vs_height(diameter, initial_height, theta_degrees, height_recession):
    """
    Calculate how the ion beam interaction area (and thus current) changes
    as the cylinder height decreases due to surface recession.

    Parameters:
    diameter (float): Diameter of the cylinder
    initial_height (float): Initial height of the cylinder
    theta_degrees (float): Angle between cylinder axis and ion beam direction
    height_recession (float): Amount the top surface has receded (Δh)

    Returns:
    tuple: (new_total_area, area_change, current_change_ratio)
    """
    # Original interaction area
    original_area, _, _ = calculate_ion_beam_interaction_area(diameter, initial_height, theta_degrees)

    # New height after recession
    new_height = initial_height - height_recession

    # New interaction area
    new_area, _, _ = calculate_ion_beam_interaction_area(diameter, new_height, theta_degrees)

    # Changes
    area_change = new_area - original_area
    current_change_ratio = new_area / original_area if original_area > 0 else 0

    return new_area, area_change, current_change_ratio


def plot_cylinder_projection_on_plane(diameter, height, theta_degrees, n_points=100, plane_offset=None):
    """
    Plot the actual projection of the cylinder onto the tilted plane.
    This shows how the cylinder would look when viewed from the plane's normal direction.

    Parameters:
    plane_offset (float): Distance to shift the plane above the cylinder.
                         If None, defaults to height * 0.3
    """
    radius = diameter / 2
    theta_rad = np.radians(theta_degrees)

    # Default plane offset to avoid intersection
    if plane_offset is None:
        plane_offset = height * 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left plot: 3D cylinder with projection plane
    ax1 = fig.add_subplot(121, projection='3d')

    # Create cylinder surface
    z = np.linspace(0, height, 50)
    theta_cyl = np.linspace(0, 2 * np.pi, 50)
    Z, THETA = np.meshgrid(z, theta_cyl)
    X = radius * np.cos(THETA)
    Y = radius * np.sin(THETA)

    # Plot cylinder
    ax1.plot_surface(X, Y, Z, alpha=0.3, color='lightblue', label='Cylinder')

    # Create and plot the projection plane (shifted upward)
    plane_size = max(diameter, height) * 1.5
    xx, yy = np.meshgrid(np.linspace(-plane_size / 2, plane_size / 2, 20),
                         np.linspace(-plane_size / 2, plane_size / 2, 20))

    # Plane equation: normal vector is (sin(theta), 0, cos(theta))
    # Plane passes through a point above the cylinder
    plane_center_z = height + plane_offset
    zz = plane_center_z - np.tan(theta_rad) * xx

    ax1.plot_surface(xx, yy, zz, alpha=0.2, color='yellow')

    # Add projection lines from cylinder edges to plane
    # Draw a few projection lines to show the projection direction
    projection_lines_x = []
    projection_lines_y = []
    projection_lines_z = []

    # Select some points on the cylinder to draw projection lines
    sample_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    sample_heights = [0, height]  # Top and bottom circles

    for z_cyl in sample_heights:
        for t in sample_angles:
            x_cyl = radius * np.cos(t)
            y_cyl = radius * np.sin(t)

            # Find where the projection line intersects the plane
            # Projection direction is along the plane normal: (sin(theta), 0, cos(theta))
            # Line: (x_cyl, y_cyl, z_cyl) + s * (sin(theta), 0, cos(theta))
            # Plane: z = plane_center_z - tan(theta) * x

            # Solve for intersection parameter s
            if abs(np.cos(theta_rad)) > 1e-10:  # Avoid division by zero
                s = (plane_center_z - z_cyl - np.tan(theta_rad) * x_cyl) / np.cos(theta_rad)

                x_proj = x_cyl + s * np.sin(theta_rad)
                y_proj = y_cyl
                z_proj = z_cyl + s * np.cos(theta_rad)

                # Draw projection line
                ax1.plot([x_cyl, x_proj], [y_cyl, y_proj], [z_cyl, z_proj],
                         'gray', alpha=0.5, linewidth=1)

                projection_lines_x.append([x_cyl, x_proj])
                projection_lines_y.append([y_cyl, y_proj])
                projection_lines_z.append([z_cyl, z_proj])

    # Add normal vector
    normal_length = height * 0.4
    normal_x = normal_length * np.sin(theta_rad)
    normal_z = normal_length * np.cos(theta_rad)
    ax1.quiver(0, 0, plane_center_z, normal_x, 0, normal_z,
               color='red', arrow_length_ratio=0.1, linewidth=3, label='Plane Normal')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Cylinder and Projection Plane (θ={theta_degrees}°)\nPlane offset: {plane_offset:.1f}')

    # Set axis limits to show both cylinder and plane nicely
    max_extent = max(radius, height + plane_offset) * 1.2
    ax1.set_xlim([-max_extent, max_extent])
    ax1.set_ylim([-max_extent, max_extent])
    ax1.set_zlim([0, height + plane_offset + max_extent * 0.3])

    # Right plot: Actual projection on the plane
    ax2 = fig.add_subplot(122)

    # For the projection calculation, we need to project points onto the plane
    # and then express them in the plane's coordinate system

    # Generate points on cylinder surface
    theta_points = np.linspace(0, 2 * np.pi, n_points)
    z_points = np.linspace(0, height, n_points // 2)

    projected_points = []

    # Project cylinder edge circles (top and bottom)
    for z_cyl in [0, height]:
        for t in theta_points:
            x_cyl = radius * np.cos(t)
            y_cyl = radius * np.sin(t)

            # Project point onto plane and convert to plane coordinates
            # The projection is along the plane normal direction
            # In plane coordinates: u is along y-axis, v is in the plane's x-z direction
            u = y_cyl  # y-coordinate stays the same
            v = (x_cyl - (z_cyl - plane_center_z) * np.tan(theta_rad)) * np.cos(theta_rad)

            projected_points.append([u, v])

    # Project cylinder side edges
    for t in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:  # Four generator lines
        for z_cyl in z_points:
            x_cyl = radius * np.cos(t)
            y_cyl = radius * np.sin(t)

            # Project point onto plane
            u = y_cyl
            v = (x_cyl - (z_cyl - plane_center_z) * np.tan(theta_rad)) * np.cos(theta_rad)

            projected_points.append([u, v])

    projected_points = np.array(projected_points)

    # Plot the projection
    ax2.scatter(projected_points[:, 0], projected_points[:, 1],
                s=1, alpha=0.6, color='blue', label='Projected cylinder surface')

    # Draw the projection outline more clearly
    # Top and bottom ellipses
    t_ellipse = np.linspace(0, 2 * np.pi, 100)

    # Top circle projection
    top_u = radius * np.sin(t_ellipse)
    top_v = (radius * np.cos(t_ellipse) - (height - plane_center_z) * np.tan(theta_rad)) * np.cos(theta_rad)
    ax2.plot(top_u, top_v, 'r-', linewidth=2, label='Top circle projection')

    # Bottom circle projection
    bottom_u = radius * np.sin(t_ellipse)
    bottom_v = (radius * np.cos(t_ellipse) - (0 - plane_center_z) * np.tan(theta_rad)) * np.cos(theta_rad)
    ax2.plot(bottom_u, bottom_v, 'g-', linewidth=2, label='Bottom circle projection')

    # Side edges (tangent lines)
    for angle in [0, np.pi]:  # Front and back tangent lines
        x_edge = radius * np.cos(angle)

        # Top point
        u1 = radius * np.sin(angle)
        v1 = (x_edge - (height - plane_center_z) * np.tan(theta_rad)) * np.cos(theta_rad)

        # Bottom point
        u2 = radius * np.sin(angle)
        v2 = (x_edge - (0 - plane_center_z) * np.tan(theta_rad)) * np.cos(theta_rad)

        ax2.plot([u1, u2], [v1, v2], 'orange', linewidth=2, alpha=0.8)

    # Calculate and display ion beam interaction area
    total_area, top_face_area, curved_area = calculate_ion_beam_interaction_area(diameter, height, theta_degrees)

    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('U (plane coordinate)')
    ax2.set_ylabel('V (plane coordinate)')
    ax2.set_title(
        f'Ion Beam Interaction Cross-Section\nTotal Area = {total_area:.3f}\n(Top: {top_face_area:.3f}, Curved: {curved_area:.3f})')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return total_area


def visualize_cylinder_projection(diameter, height, theta_degrees):
    """
    Visualize the cylinder and its projection plane (original function maintained for compatibility).
    """
    return plot_cylinder_projection_on_plane(diameter, height, theta_degrees)


def analyze_current_vs_recession(diameter, height, theta_degrees):
    """
    Analyze how current changes with height recession for ion beam interaction.
    """
    recession_values = np.linspace(0, height * 0.8, 50)
    total_areas = []
    area_changes = []
    current_ratios = []

    for recession in recession_values:
        new_area, area_change, current_ratio = calculate_current_vs_height(
            diameter, height, theta_degrees, recession)
        total_areas.append(new_area)
        area_changes.append(area_change)
        current_ratios.append(current_ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Total interaction area vs height recession
    ax1.plot(recession_values, total_areas, 'b-', linewidth=2, label='Total Ion Interaction Area')
    ax1.set_xlabel('Height Recession Δh')
    ax1.set_ylabel('Ion Interaction Area')
    ax1.set_title(f'Ion Beam Interaction Area vs Height Recession\n(θ = {theta_degrees}°)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Current ratio (normalized) vs height recession
    ax2.plot(recession_values, current_ratios, 'r-', linewidth=2, label='Current Ratio (I/I₀)')
    ax2.set_xlabel('Height Recession Δh')
    ax2.set_ylabel('Current Ratio (I/I₀)')
    ax2.set_title(f'Normalized Current vs Height Recession\n(θ = {theta_degrees}°)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1.1])

    # Add linear fit for small recessions
    small_recession_mask = recession_values <= height * 0.2
    if np.sum(small_recession_mask) > 5:
        # Convert arrays to numpy arrays and use boolean indexing properly
        recession_small = np.array(recession_values)[small_recession_mask]
        ratios_small = np.array(current_ratios)[small_recession_mask]

        p = np.polyfit(recession_small, ratios_small, 1)
        ax2.plot(recession_small, np.polyval(p, recession_small),
                 'r--', alpha=0.7, label=f'Linear fit: slope = {p[0]:.4f}')
        ax2.legend()

    plt.tight_layout()
    plt.show()

    return recession_values, total_areas, current_ratios


# Example usage
if __name__ == "__main__":
    # Cylinder parameters
    diameter = 4.0  # units
    height = 6.0  # units
    theta = 30  # degrees

    # Calculate ion beam interaction area
    total_area, top_face_area, curved_area = calculate_ion_beam_interaction_area(diameter, height, theta)

    print(f"Cylinder Parameters:")
    print(f"  Diameter: {diameter}")
    print(f"  Height: {height}")
    print(f"  Ion beam angle θ: {theta}°")
    print(f"\nIon Beam Interaction Areas:")
    print(f"  Top face area: {top_face_area:.3f}")
    print(f"  Curved surface area: {curved_area:.3f}")
    print(f"  Total interaction area: {total_area:.3f}")

    print(f"\nPhysical Interpretation:")
    print(f"  • Top face: π×r²×cos({theta}°) = {top_face_area:.3f}")
    print(f"  • Curved surface: π×r×h×sin({theta}°) = {curved_area:.3f}")
    print(f"  • Current ∝ Total interaction area = {total_area:.3f}")

    # Demonstrate height recession effect
    recession = height * 0.1  # 10% recession
    new_area, area_change, current_ratio = calculate_current_vs_height(diameter, height, theta, recession)
    print(f"\nHeight Recession Analysis:")
    print(f"  After {recession:.2f} height recession:")
    print(f"  • New interaction area: {new_area:.3f}")
    print(f"  • Area change: {area_change:.3f}")
    print(f"  • Current ratio (I/I₀): {current_ratio:.3f}")

    # Visualizations
    projected_area = plot_cylinder_projection_on_plane(diameter, height, theta)
    analyze_interaction_vs_angle(diameter, height)
    analyze_current_vs_recession(diameter, height, theta)