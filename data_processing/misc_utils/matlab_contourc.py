import numpy as np
import matplotlib.pyplot as plt
from matplotlib import contour as mpl_contour
from skimage import measure
from typing import Union, Optional, Tuple
import warnings


def contourc_matplotlib(Z: np.ndarray,
                        x: Optional[np.ndarray] = None,
                        y: Optional[np.ndarray] = None,
                        levels: Optional[Union[int, np.ndarray]] = None) -> np.ndarray:
    """
    Python implementation of MATLAB's contourc using matplotlib.

    Parameters:
    -----------
    Z : np.ndarray
        2D array of height values
    x : np.ndarray, optional
        1D array of x-coordinates (default: column indices)
    y : np.ndarray, optional
        1D array of y-coordinates (default: row indices)
    levels : int or np.ndarray, optional
        Number of levels (int) or specific level values (array)

    Returns:
    --------
    np.ndarray
        Contour matrix in MATLAB format
    """
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
        # Let matplotlib choose automatic levels
        cs = plt.contour(X, Y, Z)
        plt.close()  # Close the figure to avoid display
    elif isinstance(levels, int):
        # Specific number of levels
        cs = plt.contour(X, Y, Z, levels)
        plt.close()
    else:
        # Specific level values
        cs = plt.contour(X, Y, Z, levels)
        plt.close()

    # Convert matplotlib contours to MATLAB format
    contour_matrix = []

    # Get contour paths - cs.allsegs contains the line segments for each level
    for level_idx, level_value in enumerate(cs.levels):
        # cs.allsegs[level_idx] contains all segments for this level
        if level_idx < len(cs.allsegs):
            segments = cs.allsegs[level_idx]

            for segment in segments:
                if len(segment) > 0:
                    vertices = np.array(segment)
                    # First column: [level_value, num_vertices]
                    header = [level_value, len(vertices)]
                    contour_matrix.extend(header)

                    # Add x-coordinates
                    contour_matrix.extend(vertices[:, 0])
                    # Add y-coordinates
                    contour_matrix.extend(vertices[:, 1])

    # Reshape to 2-row format like MATLAB
    if len(contour_matrix) == 0:
        return np.array([]).reshape(2, 0)

    # Convert to numpy array and reshape
    contour_array = np.array(contour_matrix)
    # Reshape to alternate between x and y rows
    result = []
    i = 0
    while i < len(contour_array):
        level_val = contour_array[i]
        num_vertices = int(contour_array[i + 1])

        if i + 2 + 2 * num_vertices > len(contour_array):
            break

        # Extract x and y coordinates
        x_coords = contour_array[i + 2:i + 2 + num_vertices]
        y_coords = contour_array[i + 2 + num_vertices:i + 2 + 2 * num_vertices]

        # Build result in MATLAB format: [level, x1, x2, ..., xn]
        #                                [n_vertices, y1, y2, ..., yn]
        row1 = [level_val] + list(x_coords)
        row2 = [num_vertices] + list(y_coords)

        if len(result) == 0:
            result = [row1, row2]
        else:
            result[0].extend(row1)
            result[1].extend(row2)

        i += 2 + 2 * num_vertices

    return np.array(result) if result else np.array([]).reshape(2, 0)


def contourc_skimage(Z: np.ndarray,
                     x: Optional[np.ndarray] = None,
                     y: Optional[np.ndarray] = None,
                     levels: Optional[Union[int, np.ndarray]] = None) -> np.ndarray:
    """
    Python implementation of MATLAB's contourc using scikit-image.

    Parameters: Same as contourc_matplotlib
    """
    m, n = Z.shape

    # Set default coordinates
    if x is None:
        x = np.arange(n)
    if y is None:
        y = np.arange(m)

    # Handle levels parameter
    if levels is None:
        # Choose reasonable automatic levels
        z_min, z_max = Z.min(), Z.max()
        level_values = np.linspace(z_min, z_max, 10)
    elif isinstance(levels, int):
        # Specific number of levels
        z_min, z_max = Z.min(), Z.max()
        level_values = np.linspace(z_min, z_max, levels)
    else:
        # Specific level values
        level_values = np.asarray(levels)

    contour_matrix = []

    for level in level_values:
        # Find contours at this level using skimage
        try:
            contours = measure.find_contours(Z, level)
        except ValueError:
            # Skip if level is outside data range
            continue

        for contour in contours:
            if len(contour) > 0:
                # Convert indices to actual coordinates
                # contour is in (row, col) format, we need (x, y)
                y_indices = contour[:, 0]
                x_indices = contour[:, 1]

                # Interpolate to get actual x, y coordinates
                x_coords = np.interp(x_indices, np.arange(n), x)
                y_coords = np.interp(y_indices, np.arange(m), y)

                # Build result in MATLAB format
                row1 = [level] + list(x_coords)
                row2 = [len(contour)] + list(y_coords)

                if len(contour_matrix) == 0:
                    contour_matrix = [row1, row2]
                else:
                    contour_matrix[0].extend(row1)
                    contour_matrix[1].extend(row2)

    return np.array(contour_matrix) if contour_matrix else np.array([]).reshape(2, 0)


def plot_comparison(Z: np.ndarray, x: Optional[np.ndarray] = None,
                    y: Optional[np.ndarray] = None, levels: Optional[Union[int, np.ndarray]] = None):
    """
    Compare the two approaches visually.
    """
    m, n = Z.shape
    if x is None:
        x = np.arange(n)
    if y is None:
        y = np.arange(m)

    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original surface
    im = axes[0].contourf(X, Y, Z, 20, alpha=0.7)
    axes[0].contour(X, Y, Z, colors='black', linewidths=0.5)
    axes[0].set_title('Original Surface')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im, ax=axes[0])

    # Matplotlib approach
    M1 = contourc_matplotlib(Z, x, y, levels)
    axes[1].contourf(X, Y, Z, 20, alpha=0.3)

    # Plot contours from matplotlib matrix
    if M1.size > 0:
        plot_contour_matrix(M1, axes[1], 'red', 'Matplotlib')
    axes[1].set_title('Matplotlib Approach')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')

    # Skimage approach
    M2 = contourc_skimage(Z, x, y, levels)
    axes[2].contourf(X, Y, Z, 20, alpha=0.3)

    # Plot contours from skimage matrix
    if M2.size > 0:
        plot_contour_matrix(M2, axes[2], 'blue', 'Scikit-image')
    axes[2].set_title('Scikit-image Approach')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')

    plt.tight_layout()
    plt.show()

    return M1, M2


def plot_contour_matrix(M: np.ndarray, ax, color: str, label: str):
    """
    Plot contours from contour matrix format.
    """
    if M.size == 0:
        return

    i = 0
    while i < M.shape[1]:
        if i >= M.shape[1]:
            break

        level_val = M[0, i]
        num_vertices = int(M[1, i])

        if i + num_vertices >= M.shape[1]:
            break

        # Extract coordinates
        x_coords = M[0, i + 1:i + 1 + num_vertices]
        y_coords = M[1, i + 1:i + 1 + num_vertices]

        # Plot this contour line
        ax.plot(x_coords, y_coords, color=color, linewidth=1.5, alpha=0.8)

        i += 1 + num_vertices


def test_matlab_examples():
    """
    Test with examples from MATLAB documentation.
    """
    print("Testing MATLAB Examples")
    print("=" * 50)

    # Example 1: Basic paraboloid (matches MATLAB example)
    x = np.arange(-5, 5.6, 0.5)
    y = np.arange(-5, 5.6, 0.5)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2

    print("\nExample 1: Basic paraboloid")
    print("Z shape:", Z.shape)

    # Test without x,y coordinates (should use indices)
    M1_basic = contourc_matplotlib(Z)
    print(f"Matplotlib result shape: {M1_basic.shape}")
    if M1_basic.size > 0:
        print("First 5 columns:")
        print(M1_basic[:, :5])

    # Test with x,y coordinates
    M1_coords = contourc_matplotlib(Z, x, y)
    print(f"\nWith coordinates - Matplotlib result shape: {M1_coords.shape}")
    if M1_coords.size > 0:
        print("First 5 columns:")
        print(M1_coords[:, :5])

    # Test specific levels
    M1_levels = contourc_matplotlib(Z, x, y, [5, 10, 15, 20])
    print(f"\nSpecific levels - Matplotlib result shape: {M1_levels.shape}")
    if M1_levels.size > 0:
        print("First 5 columns:")
        print(M1_levels[:, :5])

    # Compare with skimage
    M2_coords = contourc_skimage(Z, x, y)
    print(f"\nSkimage result shape: {M2_coords.shape}")

    # Visual comparison
    print("\nGenerating comparison plots...")
    M1, M2 = plot_comparison(Z, x, y, [5, 10, 15, 20])

    return M1, M2


if __name__ == "__main__":
    # Run tests
    M1, M2 = test_matlab_examples()

    print("\nComparison Summary:")
    print("=" * 30)
    print(f"Matplotlib approach matrix shape: {M1.shape}")
    print(f"Scikit-image approach matrix shape: {M2.shape}")

    if M1.size > 0 and M2.size > 0:
        print("\nBoth approaches successfully generated contour matrices!")
        print("The formats should be compatible with MATLAB's contourc output.")
    else:
        print("One or both approaches failed to generate contours.")