import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_sphere_surface(radius, nx, ny, spacing_factor=1.0):
    """
    Generate a surface made of spheres

    Parameters:
    radius : float
        Radius of each sphere
    nx, ny : int
        Number of points in x and y direction
    spacing_factor : float
        Controls spacing between spheres (1.0 = touching spheres)

    Returns:
    X, Y : 2D arrays of coordinates
    Z : 2D array of heights
    """
    # Create grid
    x = np.linspace(0, (nx - 1) * 2 * radius * spacing_factor, nx * 200)
    y = np.linspace(0, (ny - 1) * 2 * radius * spacing_factor, ny * 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Place spheres
    for i in range(nx):
        for j in range(ny):
            center_x = i * 2 * radius * spacing_factor
            center_y = j * 2 * radius * spacing_factor

            # Calculate distance from each point to sphere center
            dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

            # Calculate sphere height at each point (upper surface)
            sphere_z = np.where(dist <= radius,
                                np.sqrt(radius ** 2 - dist ** 2),
                                0)

            # Take maximum height at each point
            Z = np.maximum(Z, sphere_z)

    return X, Y, Z


def calculate_rms_roughness(Z):
    """
    Calculate RMS roughness of a surface
    """
    # Calculate mean height
    mean_height = np.mean(Z)

    # Calculate RMS roughness
    rms = np.sqrt(np.mean((Z - mean_height) ** 2))

    return rms


# Example usage
if __name__ == "__main__":
    # Parameters
    radius = 1.0  # sphere radius
    nx, ny = 4, 4  # number of spheres
    spacing_factors = np.linspace(1.0, 1.5, 5)  # different spacings to try

    # Calculate roughness for different spacings
    rms_values = []
    for spacing in spacing_factors:
        X, Y, Z = generate_sphere_surface(radius, nx, ny, spacing)
        rms = calculate_rms_roughness(Z)
        rms_values.append(rms)
        print(f"Spacing factor: {spacing:.2f}, RMS roughness: {rms:.3f} * radius")

    # Plot surface for one case
    X, Y, Z = generate_sphere_surface(radius, nx, ny, spacing_factors[0])

    fig = plt.figure(figsize=(10, 5))

    # Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_title('Surface Profile')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Roughness vs spacing plot
    ax2 = fig.add_subplot(122)
    ax2.plot(spacing_factors, rms_values, 'o-')
    ax2.set_xlabel('Spacing Factor')
    ax2.set_ylabel('RMS Roughness (relative to radius)')
    ax2.set_title('RMS Roughness vs Sphere Spacing')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()