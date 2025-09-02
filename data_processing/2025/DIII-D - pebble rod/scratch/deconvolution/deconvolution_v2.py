import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize


def create_heaviside_kernel(width, sample_rate, duration):
    """Create a Heaviside (step) function kernel with specified width."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    kernel = np.heaviside(t, 0.5) - np.heaviside(t - width, 0.5)
    return kernel / np.sum(kernel)  # Normalize


# Method 1: scipy.signal.deconvolve (polynomial division)
def scipy_deconvolve_method(observed_signal, kernel):
    """
    Uses scipy.signal.deconvolve - works best when convolution is exact (no noise).
    This performs polynomial long division.
    """
    try:
        quotient, remainder = signal.deconvolve(observed_signal, kernel)
        return quotient
    except:
        return np.zeros_like(observed_signal)


# Method 2: scipy.signal.wiener (Wiener filtering)
def scipy_wiener_method(observed_signal, noise_power=None):
    """
    Uses scipy.signal.wiener for noise reduction.
    Note: This is primarily a noise reduction filter, not true deconvolution.
    """
    if noise_power is None:
        # Estimate noise power from high-frequency components
        noise_power = np.var(np.diff(observed_signal))

    return signal.wiener(observed_signal, mysize=None, noise=noise_power)


# Method 3: Using scipy.sparse for regularized deconvolution
def scipy_sparse_deconvolution(observed_signal, kernel, lambda_reg=1e-3):
    """
    Uses scipy.sparse matrices for Tikhonov regularized deconvolution.
    This creates a convolution matrix and solves the regularized least squares problem.
    """
    n = len(observed_signal)
    k = len(kernel)

    # Create convolution matrix (Toeplitz matrix)
    # Each row represents convolution with the kernel at different positions
    conv_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(min(k, n - i)):
            conv_matrix[i, (i + j) % n] = kernel[j]

    # Convert to sparse matrix for efficiency
    A_sparse = diags(conv_matrix.flatten(), shape=(n, n), format='csr')

    # Regularization matrix (identity for Tikhonov regularization)
    I = diags(np.ones(n), format='csr')

    # Solve regularized least squares: (A^T A + λI) x = A^T b
    AtA = A_sparse.T @ A_sparse
    Atb = A_sparse.T @ observed_signal
    regularized_matrix = AtA + lambda_reg * I

    deconvolved = spsolve(regularized_matrix, Atb)
    return deconvolved


# Method 4: Using scipy.optimize for constrained deconvolution
def scipy_optimize_deconvolution(observed_signal, kernel, lambda_reg=1e-3,
                                 non_negative=True, smooth_reg=1e-4):
    """
    Uses scipy.optimize to solve deconvolution as an optimization problem.
    Can include constraints like non-negativity and smoothness.
    """
    n = len(observed_signal)

    # Create convolution matrix
    conv_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(min(len(kernel), n - i)):
            conv_matrix[i, (i + j) % n] = kernel[j]

    # Objective function: ||Ax - b||^2 + λ||x||^2 + μ||Dx||^2
    def objective(x):
        # Data fidelity term
        data_term = np.sum((conv_matrix @ x - observed_signal) ** 2)

        # L2 regularization term
        l2_term = lambda_reg * np.sum(x ** 2)

        # Smoothness term (total variation)
        if smooth_reg > 0:
            smooth_term = smooth_reg * np.sum(np.diff(x) ** 2)
        else:
            smooth_term = 0

        return data_term + l2_term + smooth_term

    # Initial guess
    x0 = observed_signal.copy()

    # Constraints
    constraints = []
    if non_negative:
        # Non-negativity constraint
        bounds = [(0, None) for _ in range(n)]
    else:
        bounds = None

    # Solve optimization problem
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    return result.x if result.success else x0


# Method 5: scipy.signal.resample for frequency domain approach
def scipy_fft_deconvolution(observed_signal, kernel, regularization=1e-3):
    """
    Uses scipy's FFT functions for frequency domain deconvolution.
    """
    # Pad to avoid circular convolution
    n = len(observed_signal) + len(kernel) - 1

    # FFT
    S = np.fft.fft(observed_signal, n)
    K = np.fft.fft(kernel, n)

    # Regularized inverse filter
    K_conj = np.conj(K)
    K_power = np.abs(K) ** 2
    H = K_conj / (K_power + regularization * np.max(K_power))

    # Deconvolution
    deconv_fft = S * H
    deconvolved = np.real(np.fft.ifft(deconv_fft))

    return deconvolved[:len(observed_signal)]


# Method 6: Using scipy.signal.filtfilt for zero-phase filtering approach
def scipy_filtfilt_deconvolution(observed_signal, kernel):
    """
    Uses scipy.signal.filtfilt with an approximate inverse filter.
    This is more of a filtering approach than true deconvolution.
    """
    # Create approximate inverse filter (not perfect for step functions)
    # This is mainly for demonstration - works better with smooth kernels
    try:
        # Create a simple high-pass filter to counteract the step function
        b, a = signal.butter(4, 0.1, 'high', analog=False)
        deconvolved = signal.filtfilt(b, a, observed_signal)
        return deconvolved
    except:
        return observed_signal


# Example usage and comparison
if __name__ == "__main__":
    # Parameters
    sample_rate = 1000  # Hz
    duration = 2.0  # seconds
    heaviside_width = 0.01 # width of step function
    kernel_duration = 0.1

    # Time vector
    t = np.linspace(0, duration, int(duration * sample_rate))

    # Create original signal
    # original_signal = (np.sin(2 * np.pi * 5 * t) +
    #                    0.5 * np.sin(2 * np.pi * 15 * t) +
    #                    0.3 * np.sin(2 * np.pi * 25 * t))
    original_signal = np.exp(-np.power(t-1, 2) / (2 * np.power(0.2, 2)))

    # Create Heaviside kernel
    kernel = create_heaviside_kernel(heaviside_width, sample_rate, kernel_duration)

    # Convolve and add noise
    observed_signal = np.convolve(original_signal, kernel, mode='same')
    noise_level = 0.05
    observed_signal += noise_level * np.random.randn(len(observed_signal))

    # Apply different SciPy methods
    print("Applying SciPy deconvolution methods...")

    deconv_scipy_div = scipy_deconvolve_method(observed_signal, kernel)
    deconv_wiener = scipy_wiener_method(observed_signal)
    deconv_sparse = scipy_sparse_deconvolution(observed_signal, kernel, lambda_reg=1e-2)
    deconv_optimize = scipy_optimize_deconvolution(observed_signal, kernel, lambda_reg=1e-3)
    deconv_fft = scipy_fft_deconvolution(observed_signal, kernel, regularization=1e-3)
    deconv_filtfilt = scipy_filtfilt_deconvolution(observed_signal, kernel)

    # Plot comparison
    plt.figure(figsize=(10, 8))

    methods = [
        ('Original Signal', original_signal, 'blue'),
        ('Observed Signal', observed_signal, 'red'),
        ('scipy.signal.deconvolve', deconv_scipy_div[:len(t)], 'green'),
        ('scipy.signal.wiener', deconv_wiener, 'orange'),
        ('Sparse Matrix Method', deconv_sparse, 'purple'),
        ('Optimization Method', deconv_optimize, 'brown'),
        ('FFT Method', deconv_fft, 'pink'),
        ('Filtfilt Method', deconv_filtfilt, 'gray')
    ]

    for i, (name, data, color) in enumerate(methods):
        plt.subplot(4, 2, i + 1)
        if len(data) == len(t):
            plt.plot(t, data, color=color, linewidth=1.5)
        else:
            # Handle different lengths
            t_data = np.linspace(0, duration, len(data))
            plt.plot(t_data, data, color=color, linewidth=1.5)

        plt.title(name)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)

        if i > 1:  # Skip original and observed
            # Calculate MSE if same length as original
            if len(data) == len(original_signal):
                mse = np.mean((original_signal - data) ** 2)
                plt.title(f'{name}\nMSE: {mse:.4f}')

    plt.tight_layout()
    plt.show()

    # Summary of SciPy methods
    print("\nSciPy Deconvolution Methods Summary:")
    print("=" * 50)
    print("1. scipy.signal.deconvolve - Polynomial division (exact case)")
    print("2. scipy.signal.wiener - Wiener filtering (noise reduction)")
    print("3. scipy.sparse - Regularized least squares")
    print("4. scipy.optimize - Constrained optimization")
    print("5. scipy.fft - Frequency domain approach")
    print("6. scipy.signal.filtfilt - Zero-phase filtering")

    print("\nRecommendations:")
    print("- Use sparse matrix method for clean, regularized results")
    print("- Use optimization method when you need constraints (non-negativity, smoothness)")
    print("- Use FFT method for frequency domain control")
    print("- scipy.signal.deconvolve works best for noiseless, exact convolutions")