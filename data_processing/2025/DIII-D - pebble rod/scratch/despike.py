import numpy as np
from scipy import ndimage, signal
import matplotlib.pyplot as plt


def remove_spikes_zscore(spectrum, threshold=3, window_size=5):
    """
    Remove spikes using Z-score method with local statistics.

    Parameters:
    -----------
    spectrum : array-like
        Input spectrum/signal
    threshold : float, default=3
        Z-score threshold above which points are considered spikes
    window_size : int, default=5
        Size of the local window for calculating statistics

    Returns:
    --------
    cleaned_spectrum : ndarray
        Spectrum with spikes removed
    spike_mask : ndarray
        Boolean array indicating spike locations
    """
    spectrum = np.array(spectrum)
    cleaned_spectrum = spectrum.copy()

    # Calculate local median and MAD (Median Absolute Deviation)
    local_median = ndimage.median_filter(spectrum, size=window_size)
    mad = ndimage.median_filter(np.abs(spectrum - local_median), size=window_size)

    # Calculate modified Z-score using MAD
    with np.errstate(divide='ignore', invalid='ignore'):
        modified_z_score = 0.6745 * (spectrum - local_median) / mad

    # Identify spikes
    spike_mask = np.abs(modified_z_score) > threshold

    # Replace spikes with local median
    cleaned_spectrum[spike_mask] = local_median[spike_mask]

    return cleaned_spectrum, spike_mask


def remove_spikes_iterative(spectrum, threshold=2, max_iterations=10,
                            smoothing_window=5):
    """
    Iteratively remove spikes using derivative-based detection.

    Parameters:
    -----------
    spectrum : array-like
        Input spectrum/signal
    threshold : float, default=2
        Threshold for spike detection (in units of standard deviation)
    max_iterations : int, default=10
        Maximum number of iterations
    smoothing_window : int, default=5
        Window size for smoothing

    Returns:
    --------
    cleaned_spectrum : ndarray
        Spectrum with spikes removed
    total_spikes_removed : int
        Total number of spikes removed
    """
    spectrum = np.array(spectrum)
    cleaned_spectrum = spectrum.copy()
    total_spikes_removed = 0

    for iteration in range(max_iterations):
        # Calculate second derivative to find sharp peaks
        d2 = np.gradient(np.gradient(cleaned_spectrum))

        # Smooth the second derivative to reduce noise
        d2_smooth = ndimage.uniform_filter1d(d2, size=smoothing_window)

        # Find points with high second derivative (spikes)
        std_d2 = np.std(d2_smooth)
        spike_candidates = np.abs(d2_smooth) > threshold * std_d2

        # Additional check: spikes should be significantly higher than neighbors
        for i in np.where(spike_candidates)[0]:
            if i > 0 and i < len(cleaned_spectrum) - 1:
                neighbors = cleaned_spectrum[max(0, i - 2):min(len(cleaned_spectrum), i + 3)]
                neighbor_median = np.median(neighbors[neighbors != cleaned_spectrum[i]])

                # Check if point is significantly higher than neighbors
                if cleaned_spectrum[i] > neighbor_median + 2 * np.std(neighbors):
                    # Replace with interpolated value
                    if i > 0 and i < len(cleaned_spectrum) - 1:
                        cleaned_spectrum[i] = (cleaned_spectrum[i - 1] + cleaned_spectrum[i + 1]) / 2
                    total_spikes_removed += 1

        # Break if no spikes found in this iteration
        if not np.any(spike_candidates):
            break

    return cleaned_spectrum, total_spikes_removed


def remove_spikes_morphological(spectrum, structure_size=3, threshold_factor=1.5):
    """
    Remove spikes using morphological operations.

    Parameters:
    -----------
    spectrum : array-like
        Input spectrum/signal
    structure_size : int, default=3
        Size of the morphological structuring element
    threshold_factor : float, default=1.5
        Factor for spike detection threshold

    Returns:
    --------
    cleaned_spectrum : ndarray
        Spectrum with spikes removed
    spike_mask : ndarray
        Boolean array indicating spike locations
    """
    spectrum = np.array(spectrum)

    # Morphological opening to estimate baseline without spikes
    structure = np.ones(structure_size)
    opened = ndimage.grey_opening(spectrum, structure=structure)

    # Calculate difference between original and opened
    difference = spectrum - opened

    # Identify spikes based on threshold
    threshold = threshold_factor * np.std(difference)
    spike_mask = difference > threshold

    # Replace spikes with opened values
    cleaned_spectrum = spectrum.copy()
    cleaned_spectrum[spike_mask] = opened[spike_mask]

    return cleaned_spectrum, spike_mask


def demo_spike_removal():
    """
    Demonstration of spike removal methods.
    """
    # Generate synthetic Raman spectrum with spikes
    x = np.linspace(200, 3000, 1000)

    # Create a synthetic spectrum with multiple peaks
    spectrum = (1000 * np.exp(-(x - 1000) ** 2 / 10000) +
                500 * np.exp(-(x - 1500) ** 2 / 5000) +
                300 * np.exp(-(x - 2000) ** 2 / 8000) +
                100 * np.random.normal(0, 1, len(x)))  # Add noise

    # Add cosmic ray spikes
    spike_positions = [150, 300, 450, 600, 750]
    spike_intensities = [2000, 1500, 1800, 1200, 1600]

    spectrum_with_spikes = spectrum.copy()
    for pos, intensity in zip(spike_positions, spike_intensities):
        spectrum_with_spikes[pos:pos + 2] += intensity

    # Apply different spike removal methods
    cleaned_zscore, mask_zscore = remove_spikes_zscore(spectrum_with_spikes)
    cleaned_iterative, n_removed = remove_spikes_iterative(spectrum_with_spikes)
    cleaned_morph, mask_morph = remove_spikes_morphological(spectrum_with_spikes)

    # Plot results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(x, spectrum, 'g-', alpha=0.7, label='Original clean')
    plt.plot(x, spectrum_with_spikes, 'r-', alpha=0.7, label='With spikes')
    plt.title('Original vs Corrupted Spectrum')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(x, spectrum_with_spikes, 'r-', alpha=0.5, label='With spikes')
    plt.plot(x, cleaned_zscore, 'b-', label='Z-score method')
    plt.plot(x[mask_zscore], spectrum_with_spikes[mask_zscore], 'ro',
             markersize=4, label='Detected spikes')
    plt.title('Z-score Method')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(x, spectrum_with_spikes, 'r-', alpha=0.5, label='With spikes')
    plt.plot(x, cleaned_iterative, 'g-', label='Iterative method')
    plt.title(f'Iterative Method ({n_removed} spikes removed)')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(x, spectrum_with_spikes, 'r-', alpha=0.5, label='With spikes')
    plt.plot(x, cleaned_morph, 'm-', label='Morphological method')
    plt.plot(x[mask_morph], spectrum_with_spikes[mask_morph], 'mo',
             markersize=4, label='Detected spikes')
    plt.title('Morphological Method')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'original': spectrum,
        'with_spikes': spectrum_with_spikes,
        'cleaned_zscore': cleaned_zscore,
        'cleaned_iterative': cleaned_iterative,
        'cleaned_morph': cleaned_morph
    }


# Example usage
if __name__ == "__main__":
    # Run the demonstration
    results = demo_spike_removal()

    # Example of using the functions with your own data
    # your_spectrum = np.loadtxt('your_raman_data.txt')  # Load your data
    # cleaned_spectrum, spike_mask = remove_spikes_zscore(your_spectrum, threshold=3)
    # plt.plot(your_spectrum, label='Original')
    # plt.plot(cleaned_spectrum, label='Cleaned')
    # plt.legend()
    # plt.show()