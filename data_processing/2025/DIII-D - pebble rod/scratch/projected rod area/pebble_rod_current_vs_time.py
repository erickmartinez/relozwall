import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def smooth_voltage_and_convert_current(csv_file, resistance=0.31, smoothing_method='savgol', **kwargs):
    """
    Smooth voltage data and convert to current using Ohm's law (I = V/R)

    Parameters:
    -----------
    csv_file : str
        Path to CSV file with columns 'voltage' and 't_s'
    resistance : float, default=0.31
        Resistance value in ohms for current conversion
    smoothing_method : str, default='savgol'
        Smoothing method: 'savgol', 'gaussian', 'moving_average', or 'butter'
    **kwargs : additional parameters for smoothing methods

    Returns:
    --------
    pandas.DataFrame with columns: t_s, voltage_raw, voltage_smooth, current
    """

    # Load the data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Ensure we have the required columns
    if 'voltage' not in df.columns or 't_s' not in df.columns:
        raise ValueError("CSV must contain 'voltage' and 't_s' columns")

    # Sort by time to ensure proper ordering
    df = df.sort_values('t_s').reset_index(drop=True)

    print(f"Data loaded: {len(df)} data points")
    print(f"Time range: {df['t_s'].min():.3f} to {df['t_s'].max():.3f} seconds")
    print(f"Voltage range: {df['voltage'].min():.6f} to {df['voltage'].max():.6f} V")

    # Apply smoothing based on selected method
    voltage_raw = df['voltage'].values

    if smoothing_method == 'savgol':
        # Savitzky-Golay filter (good for preserving peaks)
        window_length = kwargs.get('window_length', 51)  # Must be odd
        polyorder = kwargs.get('polyorder', 3)

        # Ensure window_length is odd and reasonable
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, len(voltage_raw))
        if window_length < polyorder + 1:
            window_length = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3

        voltage_smooth = signal.savgol_filter(voltage_raw, window_length, polyorder)
        print(f"Applied Savitzky-Golay filter (window={window_length}, poly_order={polyorder})")

    elif smoothing_method == 'gaussian':
        # Gaussian filter
        sigma = kwargs.get('sigma', 2.0)
        voltage_smooth = gaussian_filter1d(voltage_raw, sigma=sigma)
        print(f"Applied Gaussian filter (sigma={sigma})")

    elif smoothing_method == 'moving_average':
        # Simple moving average
        window = kwargs.get('window', 50)
        voltage_smooth = pd.Series(voltage_raw).rolling(window=window, center=True).mean().values
        # Fill NaN values at edges
        voltage_smooth = pd.Series(voltage_smooth).fillna(method='bfill').fillna(method='ffill').values
        print(f"Applied moving average (window={window})")

    elif smoothing_method == 'butter':
        # Butterworth low-pass filter
        cutoff = kwargs.get('cutoff', 0.1)  # Normalized frequency (0-1)
        order = kwargs.get('order', 4)
        b, a = signal.butter(order, cutoff, btype='low')
        voltage_smooth = signal.filtfilt(b, a, voltage_raw)
        print(f"Applied Butterworth filter (cutoff={cutoff}, order={order})")

    else:
        raise ValueError("smoothing_method must be 'savgol', 'gaussian', 'moving_average', or 'butter'")

    # Convert voltage to current using Ohm's law: I = V/R
    current_raw = voltage_raw / resistance
    current_smooth = voltage_smooth / resistance

    print(f"Converted to current using R = {resistance} ohms")
    print(f"Current range (raw): {current_raw.min():.6f} to {current_raw.max():.6f} A")
    print(f"Current range (smooth): {current_smooth.min():.6f} to {current_smooth.max():.6f} A")

    # Create result dataframe
    result_df = pd.DataFrame({
        't_s': df['t_s'],
        'voltage_raw': voltage_raw,
        'voltage_smooth': voltage_smooth,
        'current_raw': current_raw,
        'current_smooth': current_smooth
    })

    return result_df


def plot_results(df, title="Voltage Smoothing and Current Conversion Results"):
    """
    Plot the original and smoothed voltage/current data
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot voltage data
    ax1.plot(df['t_s'], df['voltage_raw'], 'b-', alpha=0.7, linewidth=0.5, label='Raw')
    ax1.plot(df['t_s'], df['voltage_smooth'], 'r-', linewidth=1.5, label='Smoothed')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Voltage vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot current data
    ax2.plot(df['t_s'], df['current_raw'], 'b-', alpha=0.7, linewidth=0.5, label='Raw')
    ax2.plot(df['t_s'], df['current_smooth'], 'r-', linewidth=1.5, label='Smoothed')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current (A)')
    ax2.set_title('Current vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Zoom in on a section for detail
    mid_idx = len(df) // 2
    zoom_range = slice(mid_idx - 1000, mid_idx + 1000)

    ax3.plot(df['t_s'].iloc[zoom_range], df['voltage_raw'].iloc[zoom_range],
             'b-', alpha=0.7, linewidth=0.8, label='Raw')
    ax3.plot(df['t_s'].iloc[zoom_range], df['voltage_smooth'].iloc[zoom_range],
             'r-', linewidth=2, label='Smoothed')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Voltage (V)')
    ax3.set_title('Voltage Detail View')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.plot(df['t_s'].iloc[zoom_range], df['current_raw'].iloc[zoom_range],
             'b-', alpha=0.7, linewidth=0.8, label='Raw')
    ax4.plot(df['t_s'].iloc[zoom_range], df['current_smooth'].iloc[zoom_range],
             'r-', linewidth=2, label='Smoothed')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Current (A)')
    ax4.set_title('Current Detail View')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def save_results(df, output_file="smoothed_current_data.csv"):
    """
    Save the processed data to a CSV file
    """
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Load and process the data
    csv_file = "data/VOLTAGE.203780.csv"  # Replace with your file path
    resistance = 0.31  # ohms

    # Try different smoothing methods:

    # 1. Savitzky-Golay filter (recommended for most cases)
    df_savgol = smooth_voltage_and_convert_current(
        csv_file,
        resistance=resistance,
        smoothing_method='savgol',
        window_length=201,  # Adjust for more/less smoothing
        polyorder=3
    )

    # 2. Gaussian filter (good for removing noise)
    # df_gaussian = smooth_voltage_and_convert_current(
    #     csv_file,
    #     resistance=resistance,
    #     smoothing_method='gaussian',
    #     sigma=3.0
    # )

    # 3. Moving average (simple but effective)
    # df_ma = smooth_voltage_and_convert_current(
    #     csv_file,
    #     resistance=resistance,
    #     smoothing_method='moving_average',
    #     window=100
    # )

    # 4. Butterworth filter (good for frequency domain filtering)
    # df_butter = smooth_voltage_and_convert_current(
    #     csv_file,
    #     resistance=resistance,
    #     smoothing_method='butter',
    #     cutoff=0.05,  # Lower values = more smoothing
    #     order=4
    # )

    # Plot results
    plot_results(df_savgol, "Savitzky-Golay Smoothed Data")

    # Save results
    save_results(df_savgol, "smoothed_current_savgol.csv")

    # Display statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    print(f"Original voltage std: {df_savgol['voltage_raw'].std():.6f} V")
    print(f"Smoothed voltage std: {df_savgol['voltage_smooth'].std():.6f} V")
    print(f"Noise reduction: {(1 - df_savgol['voltage_smooth'].std() / df_savgol['voltage_raw'].std()) * 100:.1f}%")
    print(f"Original current std: {df_savgol['current_raw'].std():.6f} A")
    print(f"Smoothed current std: {df_savgol['current_smooth'].std():.6f} A")