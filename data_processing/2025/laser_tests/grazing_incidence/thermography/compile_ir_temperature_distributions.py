import h5py
import numpy as np
from pathlib import Path
import os
import pandas as pd

PATH_TO_DISTRIBUTIONS = r'./data'


def load_distribution(h5file):
    with h5py.File(str(h5file), "r") as hf:
        mean_temperature_ds = hf.get('mean_temperature')
        mean_temperature = np.array(mean_temperature_ds).round(decimals=0)
        bin_centers_ds = hf.get('bin_centers')
        bin_centers = np.array(bin_centers_ds)
        histogram_matrix = np.array(hf.get('histogram')).squeeze()
        # print('bin_centers', bin_centers)
        # print('mean_temperature', mean_temperature)
    return mean_temperature, bin_centers, histogram_matrix


def main(path_to_distributions):
    path_to_distributions = Path(path_to_distributions)
    path_to_output_data = path_to_distributions / 'compiled'
    path_to_output_data.mkdir(parents=True, exist_ok=True)

    distribution_files = [file for file in os.listdir(str(path_to_distributions)) if file.endswith('.h5')]
    n_files = len(distribution_files)
    """
    Each file has a distribution with slightly different bins.
    The minimum bin center should be 310 K.
    The spacing between bins should be 20 K.
    Create a vector of bin centers from 310 K to the maximum bin center from all files.
    Create a histogram of bin centers from 310 K to the maximum bin center from all files.
    For histograms with bin centers within a smaller range, just add zeros.
    """
    bin_centers = []
    mean_temperatures = np.array([])
    mean_temperature_min = 1E10
    mean_temperature_max = -1E10
    bin_centers_min = 1E10
    bin_centers_max = -1E10
    for i, distribution_file in enumerate(distribution_files):
        path_to_distribution = path_to_distributions / distribution_file
        print(f'Reading {path_to_distribution}')
        mean_temperature, bin_centers, _ = load_distribution(path_to_distribution)
        bin_centers_min = min(bin_centers_min, np.min(bin_centers))
        bin_centers_max = max(bin_centers_max, np.max(bin_centers))
        mean_temperature_min = min(mean_temperature_min, mean_temperature.min())
        mean_temperature_max = max(mean_temperature_max, mean_temperature.max())
        mean_temperatures = np.concatenate((mean_temperatures, mean_temperature.astype(int)))


    # The spacing between bin centers (should be 20 K)
    bin_centers_spacing = bin_centers[1] - bin_centers[0]
    n_bins = int((bin_centers_max - bin_centers_min) // bin_centers_spacing) + 1
    # An array of bin centers considering a full range for all files
    bin_centers = bin_centers_min + np.arange(n_bins) * bin_centers_spacing
    n_temps = len(mean_temperatures)
    # A matrix of histograms with a shape compatible with all files
    histograms = np.zeros((n_temps, n_bins))
    print(f'n_bins: {n_bins}')
    # print(f'histograms.shape = {histograms.shape}')
    i0 = 0 # An offset equal to the number of histograms per file
    for i, distribution_file in enumerate(distribution_files):
        path_to_distribution = path_to_distributions / distribution_file
        # Load the data from the file
        mean_temperature, bin_centers_i, histogram_matrix = load_distribution(path_to_distribution)
        # Get the minimum and maximum bin_center:
        bin_center_min, bin_center_max = bin_centers_i[0], bin_centers_i[-1]
        idx1 = np.argmin(np.abs(bin_centers - bin_center_min))
        idx2 = np.argmin(np.abs(bin_centers - bin_center_max))
        n_temps = len(mean_temperature)
        for j in range(n_temps):
            histograms[j+i0, idx1:idx2+1] = histogram_matrix[j]
        i0 = len(histogram_matrix)


    """
    There are repeated temperature values in mean_temperatures with corresponding histograms.
    The following lines are aimed to performing an average per bin center
    """
    # print('mean_temperatures', mean_temperatures)
    # Make an array of mean temperatures from min_temperature to max temperature
    temperature_dimes_mean = np.arange(mean_temperature_min, mean_temperature_max+1, dtype=int)
    # print('temperature_dimes_mean', temperature_dimes_mean)
    # Count the number of repetead temperatures
    mean_temperature_repeats = np.zeros_like(temperature_dimes_mean)
    for i, mean_temperature in enumerate(temperature_dimes_mean):
        msk_mean_temperature = mean_temperature == mean_temperatures
        counts = np.sum(msk_mean_temperature)
        mean_temperature_repeats[i] = counts

    m = len(temperature_dimes_mean)
    histograms_mean = np.zeros((m, n_bins), dtype=float)

    for i, temperature in enumerate(temperature_dimes_mean):
        if temperature in mean_temperatures:
            msk_mean_temperature = int(temperature) == mean_temperatures.astype(int)
            counts = np.sum(msk_mean_temperature)
            idx_temps = np.argwhere(msk_mean_temperature)[:,0]
            histogram_i = np.zeros(n_bins, dtype=float)
            # print(f'Counts: {temperature}: {counts}')
            for j, idx_temp in enumerate(idx_temps):
                # print(f'idx_temp: {idx_temp}')
                histogram_i += histograms[idx_temp]
            histogram_i /= counts
            histograms_mean[i, :] = histogram_i

    with h5py.File(path_to_output_data / 'dimes_averaged_temperature_distributions.h5', 'w') as f:
        histograms_ds = f.create_dataset('histograms', data=histograms_mean, compression='gzip')
        bin_centers_ds = f.create_dataset('bin_centers', data=bin_centers, compression='gzip')
        dimes_mean_temperatures_ds = f.create_dataset('DiMES_mean_temperature', data=temperature_dimes_mean, compression='gzip')
        histograms_ds.attrs['units'] = '# of pixels'
        bin_centers_ds.attrs['units'] = 'K'
        dimes_mean_temperatures_ds.attrs['units'] = 'K'









    # with h5py.File(str(path_to_output_data / 'temperature_histograms.h5'), "w") as hf:
    #     mean_temperature_ds = hf.create_dataset('mean_temperature', data=np.array(mean_temperatures_sorted), compression="gzip")
    #     mean_temperature_ds.attrs['units'] = 'K'
    #     bin_centers_ds = hf.create_dataset('bin_centers', data=np.array(bin_centers_sorted), compression="gzip")
    #     bin_centers_ds.attrs['units'] = 'Temperature (K)'
    #     histograms_ds = hf.create_dataset('histogram', data=histograms_sorted, compression="gzip")
    #     histograms_ds.attrs['units'] = 'counts'

if __name__ == '__main__':
    main(path_to_distributions=PATH_TO_DISTRIBUTIONS)