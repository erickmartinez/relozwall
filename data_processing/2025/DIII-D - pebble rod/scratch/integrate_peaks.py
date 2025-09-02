import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from numpy import trapz
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')


class PeakIntegrator:
    def __init__(self, time_data, signal_data):
        """
        Initialize the peak integrator with time and signal data.

        Parameters:
        time_data: array-like, time values
        signal_data: array-like, signal values (e.g., current)
        """
        self.time = np.array(time_data)
        self.signal = np.array(signal_data)
        self.baseline = None
        self.peaks_info = {}

    def estimate_baseline(self, method='linear_endpoints', window_size=1000):
        """
        Estimate baseline using various methods.

        Parameters:
        method: str, baseline estimation method
            - 'linear_endpoints': linear interpolation between start/end
            - 'rolling_minimum': rolling minimum with smoothing
            - 'polynomial': polynomial fit to low points
            - 'manual_points': specify baseline points manually
        window_size: int, window size for rolling operations
        """
        if method == 'linear_endpoints':
            # Simple linear baseline from start to end
            baseline_start = np.mean(self.signal[:100])  # Average first 100 points
            baseline_end = np.mean(self.signal[-100:])  # Average last 100 points
            self.baseline = np.linspace(baseline_start, baseline_end, len(self.signal))

        elif method == 'rolling_minimum':
            # Use rolling minimum with smoothing
            from scipy.ndimage import uniform_filter1d
            rolling_min = pd.Series(self.signal).rolling(window_size, center=True, min_periods=1).min()
            self.baseline = uniform_filter1d(rolling_min, size=window_size // 10)

        elif method == 'polynomial':
            # Fit polynomial to lower envelope
            # Find points that are likely baseline (lower 25th percentile in local windows)
            n_windows = len(self.signal) // window_size
            baseline_points_idx = []
            baseline_points_val = []

            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, len(self.signal))
                window_data = self.signal[start_idx:end_idx]
                threshold = np.percentile(window_data, 25)

                # Find points below threshold
                low_points = np.where(window_data <= threshold)[0] + start_idx
                if len(low_points) > 0:
                    baseline_points_idx.extend(low_points[::10])  # Subsample
                    baseline_points_val.extend(self.signal[low_points[::10]])

            # Fit polynomial to these points
            if len(baseline_points_idx) > 10:
                coeffs = np.polyfit(self.time[baseline_points_idx], baseline_points_val, deg=3)
                self.baseline = np.polyval(coeffs, self.time)
            else:
                # Fallback to linear method
                print("Not enough baseline points found, using linear method")
                self.estimate_baseline(method='linear_endpoints')

        return self.baseline

    def set_manual_baseline(self, baseline_points):
        """
        Set baseline manually using specified points.

        Parameters:
        baseline_points: list of tuples [(time1, value1), (time2, value2), ...]
        """
        times, values = zip(*baseline_points)
        f = interp1d(times, values, kind='linear', fill_value='extrapolate')
        self.baseline = f(self.time)
        return self.baseline

    def find_peaks(self, height_threshold=None, prominence_factor=0.1, width_threshold=None):
        """
        Find peaks in the baseline-corrected signal.

        Parameters:
        height_threshold: float, minimum peak height above baseline
        prominence_factor: float, prominence as fraction of signal range
        width_threshold: float, minimum peak width in time units
        """
        if self.baseline is None:
            self.estimate_baseline()

        # Baseline-corrected signal
        corrected_signal = self.signal - self.baseline

        # Auto-determine thresholds if not provided
        if height_threshold is None:
            height_threshold = np.std(corrected_signal) * 2

        signal_range = np.max(corrected_signal) - np.min(corrected_signal)
        prominence = prominence_factor * signal_range

        # Find peaks
        peaks, properties = find_peaks(
            corrected_signal,
            height=height_threshold,
            prominence=prominence,
            distance=int(len(corrected_signal) * 0.01)  # Min 1% of data length apart
        )

        # Calculate peak widths
        widths, width_heights, left_ips, right_ips = peak_widths(
            corrected_signal, peaks, rel_height=0.5
        )

        # Convert width indices to time units
        time_widths = widths * (self.time[1] - self.time[0])  # Assuming uniform sampling

        # Filter by width if specified
        if width_threshold is not None:
            width_mask = time_widths >= width_threshold
            peaks = peaks[width_mask]
            widths = widths[width_mask]
            time_widths = time_widths[width_mask]
            left_ips = left_ips[width_mask]
            right_ips = right_ips[width_mask]

        # Store peak information
        self.peaks_info = {
            'indices': peaks,
            'heights': corrected_signal[peaks],
            'widths': widths,
            'time_widths': time_widths,
            'left_ips': left_ips,
            'right_ips': right_ips,
            'corrected_signal': corrected_signal
        }

        return peaks, self.peaks_info

    def integrate_peak(self, peak_idx, method='trapz', integration_limits='auto'):
        """
        Integrate area under a specific peak.

        Parameters:
        peak_idx: int, index of peak in the peaks list
        method: str, integration method ('trapz', 'simpson')
        integration_limits: str or tuple
            - 'auto': use peak width boundaries
            - 'full_width': integrate to baseline intersections
            - (left_time, right_time): manual time limits

        Returns:
        dict with integration results
        """
        if not self.peaks_info:
            raise ValueError("Must find peaks first using find_peaks()")

        peak_position = self.peaks_info['indices'][peak_idx]
        corrected_signal = self.peaks_info['corrected_signal']

        # Determine integration limits
        if integration_limits == 'auto':
            left_idx = int(self.peaks_info['left_ips'][peak_idx])
            right_idx = int(self.peaks_info['right_ips'][peak_idx])
        elif integration_limits == 'full_width':
            # Find where signal returns to baseline
            left_idx = peak_position
            right_idx = peak_position

            # Search left for baseline crossing
            while left_idx > 0 and corrected_signal[left_idx] > 0:
                left_idx -= 1

            # Search right for baseline crossing
            while right_idx < len(corrected_signal) - 1 and corrected_signal[right_idx] > 0:
                right_idx += 1

        elif isinstance(integration_limits, tuple):
            left_time, right_time = integration_limits
            left_idx = np.argmin(np.abs(self.time - left_time))
            right_idx = np.argmin(np.abs(self.time - right_time))
        else:
            raise ValueError("Invalid integration_limits")

        # Ensure indices are within bounds and left < right
        left_idx = max(0, min(left_idx, len(self.time) - 1))
        right_idx = max(left_idx + 1, min(right_idx, len(self.time) - 1))

        # Extract integration region
        time_region = self.time[left_idx:right_idx + 1]
        signal_region = corrected_signal[left_idx:right_idx + 1]

        # Only integrate positive values (above baseline)
        signal_region = np.maximum(signal_region, 0)

        # Perform integration
        if method == 'trapz':
            area = trapz(signal_region, time_region)
        elif method == 'simpson':
            try:
                from scipy.integrate import simpson
                area = simpson(signal_region, time_region)
            except ImportError:
                # Fallback to scipy.integrate.simps for older versions
                from scipy.integrate import simps
                area = simps(signal_region, time_region)
        else:
            raise ValueError("Unknown integration method")

        # Calculate additional metrics
        peak_height = corrected_signal[peak_position]
        peak_time = self.time[peak_position]
        integration_width = time_region[-1] - time_region[0]

        return {
            'area': area,
            'peak_height': peak_height,
            'peak_time': peak_time,
            'integration_width': integration_width,
            'left_time': time_region[0],
            'right_time': time_region[-1],
            'left_idx': left_idx,
            'right_idx': right_idx
        }

    def integrate_all_peaks(self, method='trapz', integration_limits='auto'):
        """
        Integrate all found peaks and return results.
        """
        if not self.peaks_info:
            raise ValueError("Must find peaks first using find_peaks()")

        results = []
        for i in range(len(self.peaks_info['indices'])):
            peak_result = self.integrate_peak(i, method, integration_limits)
            peak_result['peak_index'] = i
            results.append(peak_result)

        return results

    def plot_results(self, figsize=(12, 8), show_baseline=True, show_integration_regions=True):
        """
        Plot the signal with baseline, peaks, and integration regions.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

        # Main plot
        ax1.plot(self.time, self.signal, 'b-', linewidth=1, label='Original Signal', alpha=0.7)

        if self.baseline is not None and show_baseline:
            ax1.plot(self.time, self.baseline, 'g--', linewidth=2, label='Baseline')
            corrected_signal = self.signal - self.baseline
            ax1.plot(self.time, corrected_signal + self.baseline.min(), 'r-',
                     linewidth=1, label='Baseline-corrected', alpha=0.8)

        # Mark peaks and integration regions
        if self.peaks_info:
            peak_indices = self.peaks_info['indices']
            peak_times = self.time[peak_indices]
            peak_values = self.signal[peak_indices]

            ax1.plot(peak_times, peak_values, 'ro', markersize=8, label='Detected Peaks')

            # Show integration regions if requested
            if show_integration_regions:
                results = self.integrate_all_peaks()
                for i, result in enumerate(results):
                    left_idx = result['left_idx']
                    right_idx = result['right_idx']

                    # Fill integration area
                    time_fill = self.time[left_idx:right_idx + 1]
                    signal_fill = np.maximum(self.peaks_info['corrected_signal'][left_idx:right_idx + 1], 0)
                    baseline_fill = self.baseline[left_idx:right_idx + 1]

                    ax1.fill_between(time_fill, baseline_fill, signal_fill + baseline_fill,
                                     alpha=0.3, label=f'Peak {i + 1} Area: {result["area"]:.3e}')

        ax1.set_ylabel('Signal Value')
        ax1.set_title('Peak Detection and Integration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residual plot
        if self.baseline is not None:
            ax2.plot(self.time, self.signal - self.baseline, 'purple', linewidth=1)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_ylabel('Residual')
            ax2.set_xlabel('Time')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Example usage with your data
def analyze_peaks_from_csv(filename):
    """
    Complete analysis pipeline for CSV data.
    """
    # Load data
    df = pd.read_csv(filename)
    time_data = df.iloc[:, 0].values  # First column (time)
    signal_data = df.iloc[:, 1].values  # Second column (signal)

    # Initialize integrator
    integrator = PeakIntegrator(time_data, signal_data)

    # Estimate baseline (try different methods)
    print("Estimating baseline...")
    baseline1 = integrator.estimate_baseline(method='linear_endpoints')

    # You can try other baseline methods:
    # baseline2 = integrator.estimate_baseline(method='rolling_minimum', window_size=2000)
    # baseline3 = integrator.estimate_baseline(method='polynomial', window_size=5000)

    # Find peaks
    print("Finding peaks...")
    peaks, peak_info = integrator.find_peaks(prominence_factor=0.05)
    print(f"Found {len(peaks)} peaks")

    # Integrate all peaks
    print("Integrating peaks...")
    results = integrator.integrate_all_peaks(method='trapz', integration_limits='auto')

    # Print results
    print("\nPeak Integration Results:")
    print("-" * 60)
    for i, result in enumerate(results):
        print(f"Peak {i + 1}:")
        print(f"  Area: {result['area']:.6e}")
        print(f"  Height: {result['peak_height']:.6e}")
        print(f"  Peak Time: {result['peak_time']:.2f}")
        print(f"  Integration Width: {result['integration_width']:.2f}")
        print(f"  Integration Range: [{result['left_time']:.2f}, {result['right_time']:.2f}]")
        print()

    # Plot results
    integrator.plot_results()

    return integrator, results


# To use with your data:
# integrator, results = analyze_peaks_from_csv('203780_voltage_and_rvsout.csv')

# Alternative: Manual baseline specification if automatic methods don't work well
def manual_baseline_example():
    """
    Example of how to set manual baseline points if needed.
    """
    # Load your data first
    df = pd.read_csv('203780_voltage_and_rvsout.csv')
    time_data = df.iloc[:, 0].values
    signal_data = df.iloc[:, 1].values

    integrator = PeakIntegrator(time_data, signal_data)

    # Specify baseline points manually: [(time1, value1), (time2, value2), ...]
    # You would determine these by visual inspection of your data
    baseline_points = [
        (1200, 0.0001),  # (time, baseline_value)
        (2000, 0.0002),
        (3000, 0.0001),
        (4000, 0.0003)
    ]

    integrator.set_manual_baseline(baseline_points)
    peaks, _ = integrator.find_peaks()
    results = integrator.integrate_all_peaks()
    integrator.plot_results()

    return integrator, results

if __name__ == '__main__':
    analyze_peaks_from_csv('203780_voltage_and_rvsout.csv')
