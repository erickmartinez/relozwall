import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq


def create_heaviside_kernel(width, sample_rate, duration):
    """
    Create a Heaviside (step) function kernel with specified width.

    Parameters:
    width (float): Width of the step function in time units
    sample_rate (float): Sampling rate in Hz
    duration (float): Total duration of the kernel in time units

    Returns:
    numpy.ndarray: Heaviside kernel
    """
    t = np.linspace(0, duration, int(duration * sample_rate))
    kernel = np.heaviside(t, 0.5) - np.heaviside(t - width, 0.5)
    return kernel / np.sum(kernel)  # Normalize


def wiener_deconvolution(signal_data, kernel, noise_power=0.01):
    """
    Perform Wiener deconvolution to recover original signal.

    Parameters:
    signal_data (numpy.ndarray): Convolved signal
    kernel (numpy.ndarray): Convolution kernel (Heaviside function)
    noise_power (float): Estimated noise power for regularization

    Returns:
    numpy.ndarray: Deconvolved signal
    """
    # Pad signals to avoid circular convolution artifacts
    n = len(signal_data) + len(kernel) - 1

    # FFT of signal and kernel
    S = fft(signal_data, n)
    K = fft(kernel, n)

    # Wiener filter
    K_conj = np.conj(K)
    K_power = np.abs(K) ** 2

    # Regularized inverse filter
    H = K_conj / (K_power + noise_power)

    # Deconvolution in frequency domain
    deconv_fft = S * H

    # Convert back to time domain
    deconvolved = np.real(ifft(deconv_fft))

    # Trim to original signal length
    return deconvolved[:len(signal_data)]


def richardson_lucy_deconvolution(signal_data, kernel, iterations=50):
    """
    Richardson-Lucy deconvolution algorithm.

    Parameters:
    signal_data (numpy.ndarray): Convolved signal
    kernel (numpy.ndarray): Convolution kernel
    iterations (int): Number of iterations

    Returns:
    numpy.ndarray: Deconvolved signal
    """
    # Initialize estimate as the observed signal
    estimate = signal_data.copy()

    # Flip kernel for correlation
    kernel_flipped = np.flip(kernel)

    for i in range(iterations):
        # Forward model: convolve estimate with kernel
        convolved = np.convolve(estimate, kernel, mode='same')

        # Avoid division by zero
        convolved[convolved == 0] = 1e-10

        # Correction factor
        correction = signal_data / convolved

        # Correlate with flipped kernel
        correction_conv = np.convolve(correction, kernel_flipped, mode='same')

        # Update estimate
        estimate = estimate * correction_conv

    return estimate


def simple_inverse_filter(signal_data, kernel, regularization=1e-3):
    """
    Simple inverse filtering with regularization.

    Parameters:
    signal_data (numpy.ndarray): Convolved signal
    kernel (numpy.ndarray): Convolution kernel
    regularization (float): Regularization parameter

    Returns:
    numpy.ndarray: Deconvolved signal
    """
    n = len(signal_data) + len(kernel) - 1

    S = fft(signal_data, n)
    K = fft(kernel, n)

    # Regularized inverse
    K_reg = K + regularization * np.max(np.abs(K))
    deconv_fft = S / K_reg

    deconvolved = np.real(ifft(deconv_fft))
    return deconvolved[:len(signal_data)]


# Example usage and demonstration
if __name__ == "__main__":
    # Parameters
    sample_rate = 1000  # Hz
    duration = 2.0  # seconds
    heaviside_width = 0.1  # width of step function in seconds
    kernel_duration = 0.3  # duration of kernel

    # Time vector
    t = np.linspace(0, duration, int(duration * sample_rate))

    # Create original signal (example: sum of sinusoids)
    # original_signal = (np.sin(2 * np.pi * 5 * t) +
    #                    0.5 * np.sin(2 * np.pi * 15 * t) +
    #                    0.3 * np.sin(2 * np.pi * 25 * t))
    original_signal = np.exp(-np.power(t - 1, 2) / (2 * np.power(0.02, 2)))

    # Create Heaviside kernel
    kernel = create_heaviside_kernel(heaviside_width, sample_rate, kernel_duration)

    # Convolve original signal with kernel to simulate the observed signal
    observed_signal = np.convolve(original_signal, kernel, mode='same')

    # Add some noise
    noise_level = 0.005
    observed_signal += noise_level * np.random.randn(len(observed_signal))

    # Perform deconvolution using different methods
    deconv_wiener = wiener_deconvolution(observed_signal, kernel, noise_power=noise_level ** 2)
    deconv_rl = richardson_lucy_deconvolution(observed_signal, kernel, iterations=30)
    deconv_inverse = simple_inverse_filter(observed_signal, kernel, regularization=1e-2)

    # Plot results
    plt.figure(figsize=(10, 8))

    # Original signal
    plt.subplot(3, 2, 1)
    plt.plot(t, original_signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Heaviside kernel
    plt.subplot(3, 2, 2)
    t_kernel = np.linspace(0, kernel_duration, len(kernel))
    plt.plot(t_kernel, kernel)
    plt.title(f'Heaviside Kernel (width = {heaviside_width} s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Observed (convolved) signal
    plt.subplot(3, 2, 3)
    plt.plot(t, observed_signal)
    plt.title('Observed Signal (Convolved + Noise)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Wiener deconvolution
    plt.subplot(3, 2, 4)
    plt.plot(t, deconv_wiener, label='Wiener Deconv', alpha=0.8)
    plt.plot(t, original_signal, label='Original', alpha=0.6, linestyle='--')
    plt.title('Wiener Deconvolution')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Richardson-Lucy deconvolution
    plt.subplot(3, 2, 5)
    plt.plot(t, deconv_rl, label='Richardson-Lucy', alpha=0.8)
    plt.plot(t, original_signal, label='Original', alpha=0.6, linestyle='--')
    plt.title('Richardson-Lucy Deconvolution')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Simple inverse filter
    plt.subplot(3, 2, 6)
    plt.plot(t, deconv_inverse, label='Inverse Filter', alpha=0.8)
    plt.plot(t, original_signal, label='Original', alpha=0.6, linestyle='--')
    plt.title('Simple Inverse Filter')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    # Calculate and print error metrics
    def calculate_mse(original, recovered):
        return np.mean((original - recovered) ** 2)


    print(f"Mean Squared Error:")
    print(f"Wiener: {calculate_mse(original_signal, deconv_wiener):.6f}")
    print(f"Richardson-Lucy: {calculate_mse(original_signal, deconv_rl):.6f}")
    print(f"Inverse Filter: {calculate_mse(original_signal, deconv_inverse):.6f}")