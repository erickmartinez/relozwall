import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Dict
from TRIM_analysis import parse_sdtrimsp_output

PATH_TO_SIMULATION_RESULTS = r'./grazing angle/erosion_simulation_results'



def main(path_to_simulation_results):
    # Assume that the path contains only subfolders named with an integer value corresponding to the incident ion
    # energies

    energies_list = []
    paths_to_sdstrim_output = []
    for entry in os.listdir(path_to_simulation_results):
        full_path = os.path.join(path_to_simulation_results, entry)
        if os.path.isdir(full_path):
            path_to_sdtrimsp_out = os.path.join(full_path, 'output.dat')
            if os.path.exists(path_to_sdtrimsp_out):
                paths_to_sdstrim_output.append(path_to_sdtrimsp_out)
                energies_list.append(int(entry))

    n_energies = len(energies_list)
    sputtering_dtype = np.dtype([
        ('E0 (eV)', 'd'), ('Yield', 'd'), ('Mean sputtered energy (eV)', 'd')
    ])
    sputtering_results = np.empty(n_energies, dtype=sputtering_dtype)
    for i, e0, path_to_sdtrimsp_out in zip(range(n_energies), energies_list, paths_to_sdstrim_output):
        results = parse_sdtrimsp_output(path_to_sdtrimsp_out)
        calculated_total = results['sputtered_by_particle_1'] + results['sputtered_by_particle_2']
        total_projectiles = results['total_projectiles']
        sputtering_yield_1 = results['sputtering_yield_by_1']
        sputtering_yield_2 = results['sputtering_yield_by_2']
        total_sputtering_yield = results['total_sputtering_yield']
        sputtered_energy_1 = results['energy_per_sputtered_by_1']
        sputtered_energy_2 = results['energy_per_sputtered_by_2']
        total_sputtered_energy = (sputtering_yield_1 * sputtered_energy_1 +
                                   sputtering_yield_2 * sputtered_energy_2) / total_sputtering_yield
        total_energy_per_sputtered = results['total_energy_per_sputtered']
        diff_energy = np.abs(total_sputtered_energy - total_energy_per_sputtered) / total_energy_per_sputtered
        if diff_energy > 0.01:
            print(f'|total_sputtered_energy - total_energy_per_sputtered| = {diff_energy*100:.3f} %')

        sputtering_results[i] = (e0, total_sputtering_yield, total_energy_per_sputtered)

    results_df = pd.DataFrame(data=sputtering_results).sort_values(by=["E0 (eV)"]).reset_index(drop=True)
    results_df.to_csv(r'grazing angle/erosion_simulations.csv', index=False)

if __name__ == '__main__':
    main(path_to_simulation_results=PATH_TO_SIMULATION_RESULTS)

