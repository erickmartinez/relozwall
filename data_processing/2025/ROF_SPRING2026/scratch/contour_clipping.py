import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Define your function
def f(x, y):
    return x**2 + y**2

# Create a grid
x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Plot contours
contours = ax.contour(X, Y, Z, levels=10, cmap='viridis')

# Define boundary using a set of coordinates
# Example: a square
coords = np.array([
    [-0.5, -0.5],
    [0.5, -0.5],
    [0.5, 0.5],
    [-0.5, 0.5],
    [-0.5, -0.5]  # Close the path
])

# Create a Path from coordinates
boundary_path = Path(coords)

# Clip contours to the boundary
for collection in contours.collections:
    collection.set_clip_path(PathPatch(boundary_path, transform=ax.transData))

# Visualize the boundary
ax.plot(coords[:, 0], coords[:, 1], 'k-', linewidth=2, label='Boundary')

ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)
ax.set_aspect('equal')
plt.title('Contours Clipped to Custom Boundary')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()