import numpy as np
import matplotlib.pyplot as plt
from matplotlib.contour import QuadContourSet
from skimage import measure
from typing import Union, Optional
import warnings


def contourc_matplotlib(Z: np.ndarray,
                        x: Optional[np.ndarray] = None,
                        y: Optional[np.ndarray] = None,
                        levels: Optional[Union[int, np.ndarray]] = None) -> np.ndarray:
    """
    Python implementation of MATLAB's contourc using matplotlib.
    Creates contours in a separate figure that doesn't interfere with plotting.
    """
    m, n = Z.shape

    # Set default coordinates
    if x is None:
        x = np.arange(n)
    if y is None:
        y = np.arange(m)

    # Create meshgrid for matplotlib
    X, Y = np.meshgrid(x, y)

    # Create a completely separate figure for contour computation
    fig_temp = plt.figure(figsize=(1, 1))
    ax_temp = fig_temp.add_subplot(111)

    try:
        # Handle levels parameter
        if levels is None:
            cs = ax_temp.contour(X, Y, Z)
        elif isinstance(levels, int):
            cs = ax_temp.contour(X, Y, Z, levels)
        else:
            cs = ax_temp.contour(X, Y, Z, levels)

        # Convert matplotlib contours to MATLAB format
        result_row1 = []
        result_row2 = []

        # Get contour paths from allsegs
        for level_idx, level_value in enumerate(cs.levels):
            if level_idx < len(cs.allsegs):
                segments = cs.allsegs[level_idx]

                for segment in segments:
                    if len(segment) > 0:
                        vertices = np.array(segment)

                        result_row1.append(level_value)
                        result_row1.extend(vertices[:, 0].tolist())

                        result_row2.append(len(vertices))
                        result_row2.extend(vertices[:, 1].tolist())

        if len(result_row1) == 0:
            return np.array([]).reshape(2, 0)

        return np.array([result_row1, result_row2])

    finally:
        # Always close the temporary figure
        plt.close(fig_temp)


def contourc_skimage(Z: np.ndarray,
                     x: Optional[np.ndarray] = None,
                     y: Optional[np.ndarray] = None,
                     levels: Optional[Union[int, np.ndarray]] = None) -> np.ndarray:
    """
    Python implementation of MATLAB's contourc using scikit-image.
    """
    m, n = Z.shape

    # Set default coordinates
    if x is None:
        x = np.arange(n)
    if y is None:
        y = np.arange(m)

    # Handle levels parameter
    if levels is None:
        z_min, z_max = Z.min(), Z.max()
        level_values = np.linspace(z_min, z_max, 10)[1:-1]  # Exclude min/max
    elif isinstance(levels, int):
        z_min, z_max = Z.min(), Z.max()
        level_values = np.linspace(z_min, z_max, levels + 2)[1:-1]  # Exclude min/max
    else:
        level_values = np.asarray(levels)

    result_row1 = []
    result_row2 = []

    for level in level_values:
        # Skip levels outside data range
        if level <= Z.min() or level >= Z.max():
            continue

        try:
            contours = measure.find_contours(Z, level)

            for contour in contours:
                if len(contour) > 0:
                    # Convert indices to actual coordinates
                    y_indices = contour[:, 0]
                    x_indices = contour[:, 1]

                    # Interpolate to get actual x, y coordinates
                    x_coords = np.interp(x_indices, np.arange(n), x)
                    y_coords = np.interp(y_indices, np.arange(m), y)

                    result_row1.append(level)
                    result_row1.extend(x_coords.tolist())

                    result_row2.append(len(contour))
                    result_row2.extend(y_coords.tolist())

        except (ValueError, RuntimeError):
            continue

    if len(result_row1) == 0:
        return np.array([]).reshape(2, 0)

    return np.array([result_row1, result_row2])


def plot_contour_matrix(M: np.ndarray, ax, color: str, label: str):
    """
    Plot contours from contour matrix format.
    """
    if M.size == 0:
        ax.text(0.5, 0.5, f'No contours found\n({label})',
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        return 0

    i = 0
    contour_count = 0

    while i < M.shape[1]:
        try:
            level_val = M[0, i]
            num_vertices = int(M[1, i])

            if num_vertices <= 0 or i + num_vertices >= M.shape[1]:
                i += 1
                continue

            x_coords = M[0, i + 1:i + 1 + num_vertices]
            y_coords = M[1, i + 1:i + 1 + num_vertices]

            if contour_count == 0:
                ax.plot(x_coords, y_coords, color=color, linewidth=2,
                        alpha=0.9, label=f'{label}')
            else:
                ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.9)

            contour_count += 1
            i += 1 + num_vertices

        except (IndexError, ValueError):
            break

    return contour_count


def simple_test():
    """Simple test that should definitely work"""
    print("=== SIMPLE CONTOUR TEST ===")

    # Create simple test data
    # x = np.linspace(-2, 2, 50)
    # y = np.linspace(-2, 2, 50)
    x = np.arange(-5, 5.5, 0.5)
    y = np.arange(-5, 5.5, 0.5)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2

    print(f"Data shape: {Z.shape}")
    print(f"Data range: {Z.min():.2f} to {Z.max():.2f}")

    # Test levels
    # test_levels = [1, 4, 9]  # Simple integer levels
    test_levels = [5, 10, 15, 20]
    print(f"Testing levels: {test_levels}")

    # Generate contour matrices
    print("\nGenerating matplotlib contours...")
    M1 = contourc_matplotlib(Z, x, y, test_levels)
    print(f"Matplotlib matrix shape: {M1.shape}")

    print("\nGenerating skimage contours...")
    M2 = contourc_skimage(Z, x, y, test_levels)
    print(f"Skimage matrix shape: {M2.shape}")

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # Top left: Original surface with standard contours
    im1 = axes[0, 0].contourf(X, Y, Z, 20, alpha=0.6, cmap='viridis')
    cs1 = axes[0, 0].contour(X, Y, Z, test_levels, colors='black', linewidths=2)
    axes[0, 0].clabel(cs1, inline=True, fontsize=8)
    axes[0, 0].set_title('Original: Standard Matplotlib Contours')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')

    # Add an axes for the colorbar
    fig.colorbar(im1, ax=axes.ravel().tolist())

    # Top right: From matplotlib matrix
    axes[0, 1].contourf(X, Y, Z, 20, alpha=0.4, cmap='viridis')
    count1 = plot_contour_matrix(M1, axes[0, 1], 'red', 'Matplotlib Matrix')
    axes[0, 1].set_title(f'From Matplotlib Matrix ({count1} contours)')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    if count1 > 0:
        axes[0, 1].legend()

    # Bottom left: From skimage matrix
    axes[1, 0].contourf(X, Y, Z, 20, alpha=0.4, cmap='viridis')
    count2 = plot_contour_matrix(M2, axes[1, 0], 'blue', 'Skimage Matrix')
    axes[1, 0].set_title(f'From Skimage Matrix ({count2} contours)')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    if count2 > 0:
        axes[1, 0].legend()

    # Bottom right: Comparison overlay
    axes[1, 1].contourf(X, Y, Z, 20, alpha=0.3, cmap='viridis')
    axes[1, 1].contour(X, Y, Z, test_levels, colors='black', linewidths=1, linestyles='--', alpha=0.7)
    count1_overlay = plot_contour_matrix(M1, axes[1, 1], 'red', 'Matplotlib')
    count2_overlay = plot_contour_matrix(M2, axes[1, 1], 'blue', 'Skimage')
    axes[1, 1].set_title('Comparison Overlay')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    if count1_overlay > 0 or count2_overlay > 0:
        axes[1, 1].legend()

    for ax in axes.flatten():
        ax.set_aspect('equal')

    # plt.tight_layout()
    plt.show()

    # Print matrix details
    if M1.size > 0:
        print(f"\nMatplotlib matrix first 10 elements:")
        print(f"Row 1: {M1[0, :min(10, M1.shape[1])]}")
        print(f"Row 2: {M1[1, :min(10, M1.shape[1])]}")

    if M2.size > 0:
        print(f"\nSkimage matrix first 10 elements:")
        print(f"Row 1: {M2[0, :min(10, M2.shape[1])]}")
        print(f"Row 2: {M2[1, :min(10, M2.shape[1])]}")

    return M1, M2


def matlab_paraboloid_test():
    """Test with exact MATLAB example"""
    print("\n=== MATLAB PARABOLOID EXAMPLE ===")

    # Exact MATLAB example
    x = np.arange(-5, 5.5, 0.5)
    y = np.arange(-5, 5.5, 0.5)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2

    print(f"MATLAB example - Z shape: {Z.shape}, range: {Z.min():.1f} to {Z.max():.1f}")

    # Test specific levels from MATLAB documentation
    levels = [5, 10, 15, 20]
    M = contourc_matplotlib(Z, x, y, levels)

    print(f"Contour matrix shape: {M.shape}")
    if M.size > 0:
        print("First contour info:")
        print(f"Level: {M[0, 0]}, Vertices: {int(M[1, 0])}")
        if M.shape[1] > 2:
            print(f"First vertex: ({M[0, 1]:.4f}, {M[1, 1]:.4f})")

    # Simple plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original
    ax1.contourf(X, Y, Z, 20, alpha=0.6)
    cs = ax1.contour(X, Y, Z, levels, colors='black', linewidths=2)
    # ax1.clabel(cs, inline=True)
    ax1.set_title('MATLAB Example: Original')
    ax1.set_aspect('equal')

    # From matrix
    ax2.contourf(X, Y, Z, 20, alpha=0.3)
    count = plot_contour_matrix(M, ax2, 'red', 'From Matrix')
    ax2.set_title(f'From Contour Matrix ({count} contours)')
    ax2.set_aspect('equal')
    if count > 0:
        ax2.legend()

    plt.tight_layout()
    plt.show()

    return M


if __name__ == "__main__":
    # Run simple test first
    M1, M2 = simple_test()

    # Then MATLAB example
    M_matlab = matlab_paraboloid_test()

    print(f"\n=== SUMMARY ===")
    print(f"Simple test - Matplotlib: {M1.shape}, Skimage: {M2.shape}")
    print(f"MATLAB test - Matrix: {M_matlab.shape}")
    print("Done!")