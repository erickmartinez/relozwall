import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from data_processing.misc_utils.plot_style import load_plot_style
from typing import Tuple, Dict
from pathlib import Path
from scipy import integrate
import matplotlib.ticker as ticker


MODEL_PARAMS = r'./data/integrated_sweep/fit_results/L-mode_current_#1.txt'
CURRENT_DATA_DIR = r'./data/baselined'
MEAN_CURRENT_DATA_DIR = r'./data/mean_current'
FIGURES_DIR = r'./figures'
SHOT = 203781
BORON_DENSITY, BORON_DENSITY_DELTA = 2.1, 0.1
BORON_MOLAR_MASS = 10.811 # g / mol
PEBBLE_ROD_DETAILS_XLS = r'./pebble_rod_exposure.xlsx'

class MassLossFromCurrent(object):
    def __init__(
            self, shot, path_to_params=MODEL_PARAMS, experiment_details_xlsx=PEBBLE_ROD_DETAILS_XLS,
    ):
        self.shot = shot
        self.load_model_params(path_to_params)
        self.load_pebble_rod_details(shot=shot, xlsx=experiment_details_xlsx)


    def load_pebble_rod_details(self, shot, xlsx=PEBBLE_ROD_DETAILS_XLS) -> Dict[str, float]:
        """
        Return a pandas dataframe with the details of the pebble rod in the given shot.

        Parameters
        ----------
        shot: int, float
            The shot number.
        xlsx: str, Path
            Path to the excel file containing the pebble rod details.

        Returns
        -------
        Dict[str, float]:
            Dictionary with the details of the pebble rod in the given shot.
        """
        details_df = pd.read_excel(xlsx, sheet_name='pebble rod details')
        shots_df = pd.read_excel(xlsx, sheet_name='shots')

        df = pd.merge(details_df, shots_df, on=['sample id'], how='left')
        df = df[df['shot'] == shot].reset_index(drop=True)
        columns = df.columns.tolist()
        details = {}
        for column in columns:
            details[column] = df.loc[0, column]

        self._pebble_rod_details = details

    @property
    def pebble_rod_details(self) -> Dict[str, float]:
        return self._pebble_rod_details

    @property
    def model_params(self) -> Dict[str, float]:
        return self._model_params

    def load_model_params(self, path_to_file):
        with open(path_to_file, 'r') as f:
            file_content = f.read()

        # Regular expression pattern to match parameter lines
        # Matches: parameter_name = value -/+ uncertainty unit
        pattern = r'(b_[01])\s*=\s*([\d\.E\-\+]+)\s*-/\+\s*([\d\.E\-\+]+)\s*C/cm\^2/s'
        matches = re.finditer(pattern, file_content)
        params = {}
        for match in matches:
            param_name, value, uncertainty = match.groups()
            params[param_name] = {
                'value': float(value), 'uncertainty': float(uncertainty)
            }
        # Regular expression pattern to match the angle between the cylinder axis and the beam direction
        # Matches: theta: value deg
        pattern = re.compile(r'(theta):\s+(\d+\.?\d*)\sdeg')
        matches = pattern.findall(file_content, re.MULTILINE)
        if len(matches) > 0:
            params['theta'] = float(matches[0][1])
        self._model_params = params

    def current_to_mass_loss(self, current):
        b0, b1 = self.model_params['b_0']['value'], self.model_params['b_1']['value']
        b0_delta, b1_delta = self.model_params['b_1']['uncertainty'], self.model_params['b_1']['uncertainty']
        rho, rho_delta = self.pebble_rod_details['density (g/cm3)'], self.pebble_rod_details['density error (g/cm3)']
        diameter, diameter_delta = self.pebble_rod_details['diameter (cm)'], self.pebble_rod_details['diameter error (cm)']
        h0, h0_error = (0.1 * self.pebble_rod_details['protrusion (mm)'],
                                        0.1 * self.pebble_rod_details['protrusion error (mm)'])

        print(f'b0 = {b0:.3E}, b1 = {b1:.3E}')
        print(f'rho = {rho:.3E},-/+ {rho_delta:.3E}')
        print(f'diameter: {diameter:.3E},-/+ {diameter_delta:.3E}')
        print(f'Protrusion: {h0:.3E},-/+ {h0_error:.3E}')

        r, r_delta = 0.5*diameter, 0.5*diameter_delta
        theta = np.radians(self.model_params['theta'])
        print(f'theta: {self.model_params["theta"]:.3E}')
        sin_theta = abs(np.sin(theta))
        cos_theta = abs(np.cos(theta))

        dh =  - (b0/b1) * (cos_theta / sin_theta) * r + current / (b1 * np.pi * r * sin_theta)

        # Mass loss rate is negative
        # mass_loss_rate_g *= -1
        mass_loss_rate_g = rho * np.pi * r ** 2. * dh


        mass_loss_rate_g_delta = np.empty_like(current)
        ddh_dh0 = 1
        ddh_db0 = cos_theta / sin_theta * r

        for i in range(len(current)):
            ddh_db1 = -(b0 / b1 ** 2) * (cos_theta / sin_theta) * r + current[i] / ((b1 **2) * np.pi * r * sin_theta)
            ddh_dr = (b0 / b1) * (cos_theta / sin_theta) + current[i] / (b1 * np.pi * (r**2) * sin_theta)
            dh_error = np.array([ddh_dh0, ddh_db0, ddh_db1, ddh_dr]) @ np.array([h0_error, b0_delta, b1_delta, r_delta])
            dh_error = np.sqrt(dh_error)

            mass_loss_rate_g_delta[i] = mass_loss_rate_g[i] * np.linalg.norm([
                rho_delta / rho, dh_error / dh[i]
            ])

        # mass_loss_rate_g = (b0 / b1) * rho * (cos_theta / sin_theta) * np.pi * (r ** 3)
        # mass_loss_rate_g += h0 * rho * np.pi * (r ** 2)
        # mass_loss_rate_g -= r * rho * current / b1 / sin_theta



        return mass_loss_rate_g, mass_loss_rate_g_delta

def round_for_lim(value, factor):
    if value < 0:
        return np.floor(value * factor) / factor
    return np.ceil(value * factor) / factor

def main(shot, model_params, current_data_dir, boron_molar_mass, mean_current_data_dir, figures_dir):

    model:MassLossFromCurrent = MassLossFromCurrent(shot=shot, path_to_params=model_params)
    # Load the baselined current from the csv file
    csv_file = Path(current_data_dir) / f'{shot}_baselined_current.csv'
    current_df = pd.read_csv(csv_file).apply(pd.to_numeric, errors='coerce')
    t_s, current = 1E-3*current_df['t (ms)'].values, current_df['current (A)'].values
    # current -= current.min()

    # Get the positions of the peaks from the fit
    csv_peaks = Path(mean_current_data_dir) / f'{shot}_mean_current.csv'
    peaks_df = pd.read_csv(csv_peaks).apply(pd.to_numeric, errors='coerce')
    peak_left, peak_right = 1E-3 * peaks_df['Peak left (ms)'].values, 1E-3 * peaks_df['Peak right (ms)'].values
    msk = np.zeros_like(current,dtype=bool)
    for i in range(len(peaks_df)):
        interval_msk = (peak_left[i] <= t_s) & (t_s <= peak_right[i])
        msk = msk | interval_msk



    msk_positive = current >= 0
    t_s, current = t_s[msk], current[msk]

    N_A = 6.02214076e+23
    N_A = 6.02214076E1

    def grams_per_second_to_atoms_per_second(mass_g):
        return mass_g / boron_molar_mass * N_A

    def atoms_per_second_to_grams_per_second(atoms):
        return atoms / N_A * boron_molar_mass

    mass_loss_rate_g, mass_loss_rate_g_error = model.current_to_mass_loss(current)
    mass_loss_rage_atoms = grams_per_second_to_atoms_per_second(mass_loss_rate_g)
    mass_loss_rage_atoms_error = atoms_per_second_to_grams_per_second(mass_loss_rate_g + mass_loss_rate_g_error) - mass_loss_rage_atoms

    msk_positive = mass_loss_rate_g >= 0
    total_mass_loss = integrate.simpson(y=mass_loss_rate_g, x=t_s)
    print(f'Total mass loss: {total_mass_loss} g')

    load_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
    fig.set_size_inches(4.5, 5)

    ax_g = ax1.secondary_yaxis('right', functions=(atoms_per_second_to_grams_per_second, grams_per_second_to_atoms_per_second))

    ax1.plot(t_s, mass_loss_rage_atoms, color='C0')
    ax1.fill_between(
        t_s, (mass_loss_rage_atoms - mass_loss_rage_atoms_error), (mass_loss_rage_atoms + mass_loss_rage_atoms_error),
        color='C0', alpha=0.3
    )

    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useMathText=True)
    # ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)

    ax2.plot(t_s, current, color='C1')

    # ax.set_yscale('log')
    ax1.set_ylabel(r'{\sffamily x10\textsuperscript{22} B atoms/s}', usetex=True)
    ax_g.set_ylabel('g/s', usetex=False)
    # ax1.set_ylim(bottom=0)
    ax1.set_title(f'Total boron emission #{shot}')
    ax2.set_title('Current')
    ax2.set_ylabel('I (A)', usetex=False)

    y1lim2 =  round_for_lim(mass_loss_rage_atoms.max()*2, 5)
    ax1.set_ylim(0, y1lim2)

    y2lim1, y2lim2 = round_for_lim(current.min(), factor=5), round_for_lim(current.max()*1.1, factor=5)
    ax2.set_ylim(y2lim1, y2lim2)
    # ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    # ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

    ax_g.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax_g.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    ax1.text(
        0.025, 0.975, f'Total mass loss: {total_mass_loss:.3f} g',
        transform=ax1.transAxes, ha='left', va='top', fontsize=11,
        color='k'
    )

    for ax in [ax1, ax2]:
        ax.set_xlabel('Time (s)', usetex=False)
        xlim1, xlim2 = round_for_lim(t_s.min()*0.92, factor=4), round_for_lim(t_s.max(), factor=5)
        ax.set_xlim(xlim1, xlim2)
        xstep = round_for_lim((xlim2 - xlim1) / 5, 5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xstep))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xstep/2))

    path_to_figures = Path(figures_dir) / 'mass_loss_model'
    path_to_figures.mkdir(parents=True, exist_ok=True)

    fig.savefig(path_to_figures / f'{shot}_mass_loss_model.png', dpi=600)
    plt.show()








if __name__ == '__main__':
    main(
        shot=SHOT, model_params=MODEL_PARAMS, current_data_dir=CURRENT_DATA_DIR, boron_molar_mass=BORON_MOLAR_MASS,
        mean_current_data_dir=MEAN_CURRENT_DATA_DIR, figures_dir=FIGURES_DIR,
    )