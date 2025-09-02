import numpy as np
import matplotlib.pyplot as plt


# Simple debugging version
def debug_contourc_matplotlib(Z, x=None, y=None, levels=None):
    """Debug version with lots of print statements"""
    m, n = Z.shape

    if x is None:
        x = np.arange(n)
    if y is None:
        y = np.arange(m)

    X, Y = np.meshgrid(x, y)

    # Generate contours
    cs = plt.contour(X, Y, Z, levels)
    plt.close()

    print(f"Found {len(cs.levels)} contour levels: {cs.levels}")
    print(f"Number of allsegs: {len(cs.allsegs)}")

    # Debug: Check what's in allsegs
    for i, level in enumerate(cs.levels):
        if i < len(cs.allsegs):
            segments = cs.allsegs[i]
            print(f"Level {level}: {len(segments)} segments")
            for j, seg in enumerate(segments):
                if len(seg) > 0:
                    print(
                        f"  Segment {j}: {len(seg)} points, range x=[{seg[:, 0].min():.2f}, {seg[:, 0].max():.2f}], y=[{seg[:, 1].min():.2f}, {seg[:, 1].max():.2f}]")

    # Build matrix in correct MATLAB format
    result_row1 = []  # [level, x1, x2, ..., xn, level, x1, x2, ...]
    result_row2 = []  # [n_vertices, y1, y2, ..., yn, n_vertices, y1, y2, ...]

    total_segments = 0
    for level_idx, level_value in enumerate(cs.levels):
        if level_idx < len(cs.allsegs):
            segments = cs.allsegs[level_idx]

            for segment in segments:
                if len(segment) > 0:
                    vertices = np.array(segment)

                    # Add level and x-coordinates
                    result_row1.append(level_value)
                    result_row1.extend(vertices[:, 0].tolist())

                    # Add number of vertices and y-coordinates
                    result_row2.append(len(vertices))
                    result_row2.extend(vertices[:, 1].tolist())

                    total_segments += 1
                    print(f"Added segment: level={level_value}, {len(vertices)} vertices")

    print(f"Total segments processed: {total_segments}")
    print(f"Result row lengths: {len(result_row1)}, {len(result_row2)}")

    if len(result_row1) == 0:
        return np.array([]).reshape(2, 0)

    return np.array([result_row1, result_row2])


def debug_plot_contour_matrix(M, ax, color='red', label='Debug'):
    """Debug version of plotting function"""
    if M.size == 0:
        print("Matrix is empty!")
        return

    print(f"Matrix shape: {M.shape}")
    print(f"First 10 elements of each row:")
    print(f"Row 1: {M[0, :min(10, M.shape[1])]}")
    print(f"Row 2: {M[1, :min(10, M.shape[1])]}")

    i = 0
    segment_count = 0

    while i < M.shape[1]:
        try:
            level_val = M[0, i]
            num_vertices = int(M[1, i])

            print(f"Segment {segment_count}: level={level_val}, vertices={num_vertices}, starting at index {i}")

            if num_vertices <= 0:
                print(f"  Invalid vertex count: {num_vertices}")
                i += 1
                continue

            if i + num_vertices >= M.shape[1]:
                print(f"  Not enough data: need {num_vertices} more points, but only {M.shape[1] - i - 1} available")
                break

            # Extract coordinates
            x_coords = M[0, i + 1:i + 1 + num_vertices]
            y_coords = M[1, i + 1:i + 1 + num_vertices]

            print(f"  X range: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
            print(f"  Y range: [{y_coords.min():.2f}, {y_coords.max():.2f}]")

            # Plot this contour line
            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8,
                    label=f'{label} L={level_val:.1f}' if segment_count == 0 else "")

            segment_count += 1
            i += 1 + num_vertices

        except (IndexError, ValueError) as e:
            print(f"Error at index {i}: {e}")
            break

    print(f"Successfully plotted {segment_count} segments")
    return segment_count


# Test with simple data
print("Creating simple test data...")
x = np.linspace(-2, 2, 21)  # Smaller, simpler grid
y = np.linspace(-2, 2, 21)
X, Y = np.meshgrid(x, y)
Z = X ** 2 + Y ** 2  # Simple paraboloid

print(f"Data shape: {Z.shape}")
print(f"Data range: {Z.min():.2f} to {Z.max():.2f}")

# Test with just one level first
test_levels = [1.0, 2.0]  # Simple levels within data range
print(f"Testing with levels: {test_levels}")

# Generate matrix
print("\n" + "=" * 50)
print("GENERATING CONTOUR MATRIX")
print("=" * 50)
M = debug_contourc_matplotlib(Z, x, y, test_levels)

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Original with matplotlib
axes[0].contourf(X, Y, Z, 20, alpha=0.5)
cs_orig = axes[0].contour(X, Y, Z, test_levels, colors='black', linewidths=2)
axes[0].set_title('Original Matplotlib Contours')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].grid(True, alpha=0.3)

# Right: From our matrix
axes[1].contourf(X, Y, Z, 20, alpha=0.3)
print("\n" + "=" * 50)
print("PLOTTING FROM MATRIX")
print("=" * 50)
segments_plotted = debug_plot_contour_matrix(M, axes[1], 'red', 'Matrix')
axes[1].set_title('From Contour Matrix')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].grid(True, alpha=0.3)
if segments_plotted > 0:
    axes[1].legend()

plt.tight_layout()
plt.show()

print(f"\nSUMMARY:")
print(f"Matrix shape: {M.shape if M.size > 0 else 'Empty'}")
print(f"Segments plotted: {segments_plotted if 'segments_plotted' in locals() else 0}")