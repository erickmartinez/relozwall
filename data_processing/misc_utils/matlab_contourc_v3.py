# Simple test script to run the contourc functions and see plots
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import contour as mpl_contour
from skimage import measure
from typing import Union, Optional, Tuple
import warnings


# Copy the functions from the main code here for easy execution
def contourc_matplotlib(Z: np.ndarray,
                        x: Optional[np.ndarray] = None,
                        y: Optional[np.ndarray] = None,
                        levels: Optional[Union[int, np.ndarray]] = None) -> np.ndarray:
    """Python implementation of MATLAB's contourc using matplotlib."""
    m, n = Z.shape

    # Set default coordinates
    if x is None:
        x = np.arange(n)
    if y is None:
        y = np.arange(m)

    # Create meshgrid for matplotlib
    X, Y = np.meshgrid(x, y)

    # Handle levels parameter
    if levels is None:
        cs = plt.contour(X, Y, Z)
        plt.close()
    elif isinstance(levels, int):
        cs = plt.contour(X, Y, Z, levels)
        plt.close()
    else:
        cs = plt.contour(X, Y, Z, levels)
        plt.close()

    # Convert matplotlib contours to MATLAB format
    result_row1 = []
    result_row2 = []

    # Get contour paths
    for level_idx, level_value in enumerate(cs.levels):
        if level_idx < len(cs.allsegs):
            segments = cs.allsegs[level_idx]

            for segment in segments:
                if len(segment) > 0:
                    vertices = np.array(segment)

                    result_row1.append(level_value)
                    result_row1.extend(vertices[:, 0])

                    result_row2.append(len(vertices))
                    result_row2.extend(vertices[:, 1])

    if len(result_row1) == 0:
        return np.array([]).reshape(2, 0)

    return np.array([result_row1, result_row2])


def plot_contour_matrix(M: np.ndarray, ax, color: str, label: str):
    """Plot contours from contour matrix format."""
    if M.size == 0:
        ax.text(0.5, 0.5, f'No contours found\n({label})',
                transform=ax.transAxes, ha='center', va='center')
        return

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
                        alpha=0.8, label=f'{label} (level={level_val:.1f})')
            else:
                ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)

            contour_count += 1
            i += 1 + num_vertices

        except (IndexError, ValueError) as e:
            print(f"Error plotting contour at index {i}: {e}")
            break

    print(f"{label}: Found {contour_count} contour segments")


# Create test data (MATLAB paraboloid example)
print("Creating test data...")
x = np.arange(-5, 5.5, 0.5)
y = np.arange(-5, 5.5, 0.5)
X, Y = np.meshgrid(x, y)
Z = X ** 2 + Y ** 2

print(f"Data shape: {Z.shape}")
print(f"Data range: {Z.min():.1f} to {Z.max():.1f}")

# Test contour generation
print("\nGenerating contours...")
test_levels = [5, 10, 15, 20]
M1 = contourc_matplotlib(Z, x, y, test_levels)

print(f"Contour matrix shape: {M1.shape}")

if M1.size > 0:
    print("First few values of contour matrix:")
    print("Row 1 (levels & x-coords):", M1[0, :10])
    print("Row 2 (n_vertices & y-coords):", M1[1, :10])

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: Original surface
print("\nCreating plots...")
im = axes[0].contourf(X, Y, Z, 20, alpha=0.7, cmap='viridis')
axes[0].contour(X, Y, Z, test_levels, colors='black', linewidths=1)
axes[0].set_title('Original Surface with Contours')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
plt.colorbar(im, ax=axes[0])

# Right plot: Reconstructed contours
axes[1].contourf(X, Y, Z, 20, alpha=0.3, cmap='viridis')
plot_contour_matrix(M1, axes[1], 'red', 'Reconstructed')
axes[1].set_title('Reconstructed Contours from Matrix')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].legend()

plt.tight_layout()
plt.show()

print("Plots should now be displayed!")
print("\nTo use this in your own code, copy the functions and run:")
print("M = contourc_matplotlib(your_Z_matrix, your_x_vector, your_y_vector, your_levels)")