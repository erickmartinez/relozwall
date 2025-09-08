import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, splrep, splev
from scipy.interpolate import CubicSpline, make_smoothing_spline  # Available in scipy 1.7.1


def create_sample_data():
    """Generate sample noisy data for demonstration"""
    x = np.linspace(0, 10, 20)
    y = np.sin(x) + 0.1 * np.random.randn(len(x))  # Sine wave with noise
    return x, y


def method1_univariate_spline(x, y, smoothing_factor=None):
    """
    Method 1: UnivariateSpline with automatic smoothing
    Good for noisy data where you want automatic smoothing
    """
    # If smoothing_factor is None, scipy will choose automatically
    spline = UnivariateSpline(x, y, s=smoothing_factor)

    # Generate smooth curve
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_smooth = spline(x_smooth)

    return x_smooth, y_smooth, spline


def method2_interpolated_spline(x, y):
    """
    Method 2: InterpolatedUnivariateSpline (exact interpolation)
    Good when you want the spline to pass exactly through all points
    """
    spline = InterpolatedUnivariateSpline(x, y, k=3)  # k=3 for cubic spline

    # Generate smooth curve
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_smooth = spline(x_smooth)

    return x_smooth, y_smooth, spline


def method3_splrep_splev(x, y, smoothing_factor=0):
    """
    Method 3: Using splrep and splev (lower-level interface)
    More control over spline parameters
    """
    # Create spline representation
    tck = splrep(x, y, s=smoothing_factor)  # s=0 for interpolation, s>0 for smoothing

    # Evaluate spline
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_smooth = splev(x_smooth, tck)

    return x_smooth, y_smooth, tck


def method4_cubic_spline(x, y):
    """
    Method 4: CubicSpline (modern interface, available in scipy 1.7.1)
    Natural cubic spline with various boundary conditions
    """
    spline = CubicSpline(x, y, bc_type='natural')  # 'natural', 'clamped', or 'not-a-knot'

    # Generate smooth curve
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_smooth = spline(x_smooth)

    return x_smooth, y_smooth, spline


def plot_comparison(x, y):
    """Plot all methods for comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Smooth Spline Methods Comparison', fontsize=16)

    methods = [
        ("UnivariateSpline (Auto Smoothing)", method1_univariate_spline),
        ("InterpolatedUnivariateSpline", method2_interpolated_spline),
        ("splrep/splev", method3_splrep_splev),
        ("CubicSpline", method4_cubic_spline)
    ]

    for i, (title, method) in enumerate(methods):
        ax = axes[i // 2, i % 2]

        # Get spline results
        if method == method1_univariate_spline:
            x_smooth, y_smooth, spline = method(x, y, smoothing_factor=0.1)
        else:
            x_smooth, y_smooth, spline = method(x, y)

        # Plot original data and spline
        ax.scatter(x, y, color='red', alpha=0.7, label='Original data')
        ax.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Smooth spline')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demonstrate_smoothing_control():
    """Demonstrate how smoothing factor affects the result"""
    x, y = create_sample_data()

    plt.figure(figsize=(12, 8))

    # Different smoothing factors
    smoothing_factors = [0, 0.1, 0.5, 1.0, 5.0]

    plt.scatter(x, y, color='red', alpha=0.7, s=50, label='Original noisy data', zorder=5)

    for i, s in enumerate(smoothing_factors):
        x_smooth, y_smooth, spline = method1_univariate_spline(x, y, smoothing_factor=s)
        plt.plot(x_smooth, y_smooth, linewidth=2, label=f'Smoothing factor = {s}')

    plt.title('Effect of Smoothing Factor on UnivariateSpline')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    x_data, y_data = create_sample_data()

    print("Creating smooth splines using different methods...")

    # Method 1: UnivariateSpline with automatic smoothing
    print("\n1. UnivariateSpline (recommended for noisy data):")
    x_smooth1, y_smooth1, spline1 = method1_univariate_spline(x_data, y_data)
    print(f"   Smoothing factor used: {spline1.get_residual()}")

    # Method 2: Exact interpolation
    print("\n2. InterpolatedUnivariateSpline (exact interpolation):")
    x_smooth2, y_smooth2, spline2 = method2_interpolated_spline(x_data, y_data)

    # Method 3: Lower-level interface
    print("\n3. splrep/splev (lower-level control):")
    x_smooth3, y_smooth3, tck3 = method3_splrep_splev(x_data, y_data)

    # Method 4: Modern CubicSpline
    print("\n4. CubicSpline (modern interface):")
    x_smooth4, y_smooth4, spline4 = method4_cubic_spline(x_data, y_data)

    # Plot comparison
    plot_comparison(x_data, y_data)

    # Demonstrate smoothing control
    demonstrate_smoothing_control()

    print("\nAll methods completed. Check the plots to see the differences!")


# Additional utility functions for working with your own data
def spline_from_file(filename, x_col=0, y_col=1, delimiter=','):
    """Load data from file and create spline"""
    data = np.loadtxt(filename, delimiter=delimiter)
    x = data[:, x_col]
    y = data[:, y_col]
    return method1_univariate_spline(x, y)


def spline_derivatives(spline_obj, x_points, order=1):
    """Calculate derivatives of the spline at given points"""
    if hasattr(spline_obj, 'derivative'):
        # For UnivariateSpline and InterpolatedUnivariateSpline
        derivative_spline = spline_obj.derivative(order)
        return derivative_spline(x_points)
    elif isinstance(spline_obj, tuple):
        # For splrep/splev representation
        return splev(x_points, spline_obj, der=order)
    else:
        # For CubicSpline
        return spline_obj(x_points, order)