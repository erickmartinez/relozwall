import numpy as np
from typing import List, Dict


def find_local_minima(signal: List[float], window_size: int = 3) -> Dict[str, List]:
    """
    Find local minima in a signal using a sliding window approach.

    Parameters:
    -----------
    signal : List[float]
        Input signal as a list of numerical values
    window_size : int, optional
        Size of the window to consider for finding local minima (default: 3)
        Must be odd number

    Returns:
    --------
    Dict with three keys:
        'minima_indices': List[int] - Indices where local minima occur in original signal
        'minima_values': List[float] - Values of the local minima
        'minima_pairs': List[Tuple[int, float]] - List of (index, value) pairs for compatibility

    Raises:
    -------
    ValueError
        If window_size is not odd or is larger than signal length
    """
    if len(signal) < window_size:
        raise ValueError("Signal length must be greater than window size")
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")

    # Convert to numpy array for easier processing
    signal_array = np.array(signal)
    half_window = window_size // 2

    # Initialize separate lists for indices and values
    minima_indices = []
    minima_values = []

    # Iterate through signal excluding edges
    for i in range(half_window, len(signal_array) - half_window):
        window = signal_array[i - half_window:i + half_window + 1]
        center_value = window[half_window]

        # Check if center point is minimum in window
        if center_value == np.min(window):
            # Ensure it's strictly less than at least one neighbor
            if np.sum(window == center_value) == 1:
                minima_indices.append(i)
                minima_values.append(center_value)

    # Return results in a dictionary for clarity
    return {
        'minima_indices': minima_indices,
        'minima_values': minima_values,
        'minima_pairs': list(zip(minima_indices, minima_values))
    }


# Example usage and visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate sample signal
    x = np.linspace(0, 10, 1000)
    signal = np.sin(x) + 0.1 * np.sin(10 * x)

    # Find minima
    minima_data = find_local_minima(signal.tolist(), window_size=5)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(x, signal, label='Signal')
    plt.plot(x[minima_data['minima_indices']],
             minima_data['minima_values'],
             'ro',
             label='Local Minima')
    plt.grid(True)
    plt.legend()
    plt.title('Signal with Local Minima')
    plt.show()