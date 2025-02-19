import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, differential_evolution, OptimizeResult
import re

PATH_TO_SRIM_FILE = r'data/SRIM/stopping powers/Deuterium in Boron.txt'
DENSITY = 2.3502  # g/cm³


def read_stopping_data(filepath):
    """Read energy and stopping power data from SRIM output file.

    Returns:
    - energies: in eV (to match TRIMSP's E(IV))
    - electronic_stopping: in eV/Angstrom (to match TRIMSP's internal units)
    - nuclear_stopping: in eV/Angstrom (to match TRIMSP's internal units)
    """
    energies = []
    electronic_stopping = []
    nuclear_stopping = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find conversion factor from MeV/(mg/cm2) to eV/Angstrom
    conv_factor = None
    for line in lines:
        if "eV / Angstrom" in line:
            conv_factor = float(line.split()[0])
            break

    if conv_factor is None:
        raise ValueError("Could not find conversion factor in SRIM output")

    for line in lines:
        match = re.match(
            r'\s*(\d+\.?\d*(?:E[+-]\d+)?|\.\d+(?:E[+-]\d+)?)\s+(\S+)\s+(\d+\.\d+E[+-]\d+)\s+(\d+\.\d+E[+-]\d+)', line)
        if match:
            energy_str = match.group(1)
            unit = match.group(2)
            elec = float(match.group(3))
            nucl = float(match.group(4))

            # Convert energy to eV
            if unit.lower() == 'ev':
                energy = float(energy_str)
            elif unit.lower() == 'kev':
                energy = float(energy_str) * 1000
            elif unit.lower() == 'mev':
                energy = float(energy_str) * 1e6

            # Convert stopping powers to eV/Angstrom
            elec_converted = elec * conv_factor
            nucl_converted = nucl * conv_factor

            energies.append(energy)
            electronic_stopping.append(elec_converted)
            nuclear_stopping.append(nucl_converted)

    return np.array(energies), np.array(electronic_stopping), np.array(nuclear_stopping)


def stopping_power_model_kdee4(E, A1, A2, A3, A4, A5, M1=2.014):
    """
    Calculate stopping power using KDEE1=4 case from TRIMSP.

    Parameters:
    - E: energy in eV (matching TRIMSP's E(IV))
    - A1-A5: coefficients in TRIMSP's internal units
    - M1: projectile mass in amu

    Returns:
    - Stopping power in eV/Angstrom (matching TRIMSP's internal units)
    """
    # Convert E to keV/amu as done in TRIMSP:
    # EM(IV)=E(IV)*0.001D0/M1 where E(IV) is in eV
    EM = (E * 0.001) / M1  # Convert eV to keV/amu

    # Use EM (in keV/amu) for the threshold check
    mask_low = EM < 10.0  # Threshold is 10 keV/amu
    S = np.zeros_like(EM)

    # Low energy case
    S[mask_low] = A1 * np.sqrt(EM[mask_low])

    # High energy case
    high_E = EM[~mask_low]
    numerator = A2 * high_E ** 0.45 * (A3 / high_E) * np.log(1.0 + A4 / high_E + A5 * high_E)
    denominator = A2 * high_E ** 0.45 + (A3 / high_E) * np.log(1.0 + A4 / high_E + A5 * high_E)
    S[~mask_low] = numerator / denominator

    return S


def stopping_power_model_kdee5(E, A1, A2, A3, A4, A5, M1=2.014):
    """
    Calculate stopping power using KDEE1=5 case from TRIMSP.

    Parameters:
    - E: energy in eV (matching TRIMSP's E(IV))
    - A1-A5: coefficients in TRIMSP's internal units
    - M1: projectile mass in amu

    Returns:
    - Stopping power in eV/Angstrom (matching TRIMSP's internal units)
    """
    # Apply mass-dependent factor as in TRIMSP:
    # FHE=CVMGT(1.3333D0,1.D0,M1.LT.4.00D0)
    FHE = 1.3333 if M1 < 4.00 else 1.0

    # Convert energy as done in TRIMSP:
    # EM(IV)=E(IV)*0.001D0*FHE where E(IV) is in eV
    EM = E * 0.001 * FHE  # Convert eV to keV and apply FHE

    numerator = A1 * EM ** A2 * (A3 / (EM)) * np.log(1.0 + A4 / (EM) + A5 * EM)
    denominator = A1 * EM ** A2 + (A3 / (EM)) * np.log(1.0 + A4 / (EM) + A5 * EM)

    return numerator / denominator


def objective_function(params, E, S_data, model_type='kdee5', M1=2.014):
    """Objective function for least squares optimization.

    Parameters:
    - E: energy in eV (from SRIM output)
    - S_data: stopping power in eV/Angstrom
    - model_type: 'kdee4' or 'kdee5'
    - M1: projectile mass in amu
    """
    A1, A2, A3, A4, A5 = params
    if model_type == 'kdee4':
        S_model = stopping_power_model_kdee4(E, A1, A2, A3, A4, A5, M1)
    else:  # kdee5
        S_model = stopping_power_model_kdee5(E, A1, A2, A3, A4, A5, M1)
    return (S_model - S_data) / S_data

def objective_function_de(params, E, S_data, model_type='kdee5', M1=2.014):
    return 0.5 * np.linalg.norm(objective_function(params, E, S_data, model_type, M1))


def fit_stopping_power(filepath, model_type='kdee5', M1=2.014):
    """Main function to fit stopping power data.

    Parameters:
    - filepath: path to SRIM output file
    - model_type: 'kdee4' or 'kdee5'
    - M1: projectile mass in amu
    """
    # Read data
    energies, elec_stopping, nucl_stopping = read_stopping_data(filepath)
    total_stopping = elec_stopping + nucl_stopping

    # Initial parameter guess - in TRIMSP internal units
    if model_type == 'kdee4':
        # Adjusted initial parameters to better account for nuclear stopping
        x0 = [10.0, 5.0, 2.0, 0.1, 1e-4]  # [A1, A2, A3, A4, A5]
        # Set bounds to allow more flexibility in fitting
        bounds = ([0.1, 0.1, 0.1, 1e-5, 1e-6],  # lower bounds
                  [100.0, 50.0, 20.0, 1.0, 1e-2])  # upper bounds
    else:  # kdee5
        # Adjusted initial parameters for KDEE5
        x0 = [5.0, 0.3, 2.0, 0.5, 1e-4]  # [A1, A2, A3, A4, A5]
        # Set bounds for KDEE5
        bounds = ([0.1, 0.1, 0.1, 1e-3, 1e-6],  # lower bounds
                  [50.0, 1.0, 10.0, 5.0, 1e-2])  # upper bounds

    # Fit the model
    eps = float(np.finfo(np.float64).eps)
    # res_de: OptimizeResult = differential_evolution(
    #     func=objective_function_de,
    #     args=(energies, total_stopping, model_type, M1),
    #     x0=x0,
    #     bounds=[(0,1E3), (-1E3,1E3), (0, 1E3), (0,1E3),(0,1E3)],
    #     maxiter=10000 * len(x0),
    #     tol=eps,
    #     atol=eps,
    #     workers=-1,
    #     updating='deferred',
    #     recombination=0.2,
    #     strategy='best1bin',
    #     mutation=(0.2, 1.5),
    #     init='sobol',
    #     polish=False,
    #     disp=True
    # )

    result = least_squares(objective_function, x0,
                           args=(energies, elec_stopping, model_type, M1),
                           bounds=([0, -np.inf, 0, 0, 0],
                                   [np.inf, np.inf, np.inf, np.inf, np.inf]),
                           jac='3-point',
                           xtol=eps, gtol=eps, ftol=eps, max_nfev=10000*len(x0))

    # Get fitted parameters
    A1, A2, A3, A4, A5 = result.x

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data and fit
    ax.plot(energies, total_stopping, 'o', label='Data (Total)', alpha=0.6)
    ax.plot(energies, elec_stopping, 's', label='Electronic', alpha=0.6)
    ax.plot(energies, nucl_stopping, '^', label='Nuclear', alpha=0.6)

    # Generate smooth curve for fit
    E_fit = np.logspace(np.log10(energies.min()), np.log10(energies.max()), 1000)
    if model_type == 'kdee4':
        S_fit = stopping_power_model_kdee4(E_fit, A1, A2, A3, A4, A5, M1)
    else:  # kdee5
        S_fit = stopping_power_model_kdee5(E_fit, A1, A2, A3, A4, A5, M1)
    ax.plot(E_fit, S_fit, '-', label='Fit')

    # Set logarithmic scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Labels and title
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Stopping Power (eV/Å)')
    ax.set_title(f'Stopping Power vs Energy with TRIMSP {model_type.upper()} Fit')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()

    return fig, ax, result


if __name__ == "__main__":
    # Path to SRIM output file
    filepath = PATH_TO_SRIM_FILE

    # Set mass of deuterium
    M1 = 2.014  # amu

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Fit both models
    fig1, _, result1 = fit_stopping_power(filepath, model_type='kdee4', M1=M1)
    fig2, _, result2 = fit_stopping_power(filepath, model_type='kdee5', M1=M1)

    # Copy plots to the subplots
    ax1.clear()
    ax2.clear()
    for ax_from in fig1.axes:
        for line in ax_from.lines:
            ax1.plot(line.get_xdata(), line.get_ydata(),
                     label=line.get_label(), linestyle=line.get_linestyle(),
                     marker=line.get_marker(), alpha=line.get_alpha())
    for ax_from in fig2.axes:
        for line in ax_from.lines:
            ax2.plot(line.get_xdata(), line.get_ydata(),
                     label=line.get_label(), linestyle=line.get_linestyle(),
                     marker=line.get_marker(), alpha=line.get_alpha())

    # Set scales and labels
    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Stopping Power (eV/Å)')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()

    ax1.set_title('TRIMSP KDEE4 Model Fit')
    ax2.set_title('TRIMSP KDEE5 Model Fit')

    plt.tight_layout()
    plt.show()

    # Print fitted parameters for both models
    print("\nKDEE4 Fitted Parameters:")
    A1, A2, A3, A4, A5 = result1.x
    print(f"A1 = {A1:.3e}")
    print(f"A2 = {A2:.3e}")
    print(f"A3 = {A3:.3e}")
    print(f"A4 = {A4:.3e}")
    print(f"A5 = {A5:.3e}")

    print("\nKDEE5 Fitted Parameters:")
    A1, A2, A3, A4, A5 = result2.x
    print(f"A1 = {A1:.3e}")
    print(f"A2 = {A2:.3e}")
    print(f"A3 = {A3:.3e}")
    print(f"A4 = {A4:.3e}")
    print(f"A5 = {A5:.3e}")