import numpy as np
from scipy.optimize import least_squares, differential_evolution, OptimizeResult
import matplotlib.pyplot as plt
import re

PATH_TO_SRIM_FILE = r'data/SRIM/stopping powers/Deuterium in Boron.txt'
DENSITY = 2.3502  # g/cm³


def read_srim_data(filename):
    """Read energy and both electronic and nuclear stopping powers from SRIM output file."""
    with open(filename, 'r') as f:
        content = f.read()

    # Regular expression to match energy and stopping columns (electronic and nuclear)
    pattern = r'(\d+\.?\d*)\s+(eV|keV|MeV)\s+(\d\.\d+E[+-]\d+)\s+(\d\.\d+E[+-]\d+)'
    matches = re.finditer(pattern, content)

    energies = []
    electronic_stopping = []
    nuclear_stopping = []

    for match in matches:
        value = float(match.group(1))
        unit = match.group(2)

        # Convert all energies to eV
        if unit == 'keV':
            value *= 1e3
        elif unit == 'MeV':
            value *= 1e6

        energies.append(value)
        electronic_stopping.append(float(match.group(3)))  # dE/dx Elec. column
        nuclear_stopping.append(float(match.group(4)))  # dE/dx Nuclear column

    return np.array(energies), np.array(electronic_stopping), np.array(nuclear_stopping)


def nonlocal_inelastic_stopping(E, A, density=2.3502):
    """
    Calculate nonlocal inelastic stopping power using the TRIM.SP formula.

    Parameters:
    E : array_like
        Energy values in eV
    A : array_like
        Constants A1-A5 for the fit
        Constants ch1-ch5 for TRIM.SP electronic stopping
        ch1: Proportionality constant (for eV/Å output)
        ch2: Power law exponent for energy dependence
        ch3: Coefficient for denominator term
        ch4: Power law exponent for denominator
        ch5: Linear energy coefficient (in keV^-1)

    Returns:
    array_like
        Stopping power in the same units as SRIM output
    """
    A1, A2, A3, A4, A5 = A

    # Convert energy to keV for A&Z formula
    E_keV = np.array(E) / 1000.0

    # Calculate stopping power in eV/Å using TRIM.SP formula
    Se_eV_A = (A1 * E_keV ** A2) / (1.0 + A3 * E_keV ** A4 + A5 * E_keV)

    # Convert eV/Å to MeV/(mg/cm²)
    return Se_eV_A / (10 * density)


def nuclear_stopping(E, nk, density=2.3502):
    """
    Calculate nuclear stopping power using modified power law.

    Parameters:
    E : array_like
        Energy values in eV
    nk : array_like
        Constants nk1-nk3 for nuclear stopping

    Returns:
    array_like
        Nuclear stopping power in MeV/(mg/cm²) for comparison with SRIM
    """
    nk1, nk2, nk3 = nk
    density_boron = 2.3502  # g/cm³

    # Convert energy to keV for TRIM.SP formula
    E_keV = np.array(E) / 1000.0

    # Calculate nuclear stopping in eV/Å
    Sn_eV_A = nk1 * E_keV**nk2 / (1 + nk3 * E_keV)

    # Convert eV/Å to MeV/(mg/cm²)
    # First convert eV/Å to MeV/cm: (1e-6 MeV/eV) * (1e8 Å/cm)
    # Then divide by density to get per mass
    # Then convert g to mg (factor of 1000)
    return Sn_eV_A * 1e-6 * 1e8 / (density_boron * 1000)


def total_stopping(E, params):
    """Calculate total stopping power."""
    ch = params[:5]
    nk = params[5:]
    return nonlocal_inelastic_stopping(E, ch) + nuclear_stopping(E, nk)

def objective_function(A, E, S_exp):
    """Objective function for least squares optimization."""
    S_calc = total_stopping(E, A)
    return (S_calc - S_exp) / S_exp  # Relative error

def objective_function_de(A, E, S_exp):
    return 0.5 * np.linalg.norm(objective_function(A, E, S_exp))


def objective_function_separate(params, E, Se_exp, Sn_exp):
    """Objective function for least squares optimization."""
    ch = params[:5]  # First 5 parameters are for electronic stopping
    nk = params[5:]  # Last 3 parameters are for nuclear stopping

    Se_calc = nonlocal_inelastic_stopping(E, ch)
    Sn_calc = nuclear_stopping(E, nk)

    # Combine electronic and nuclear errors
    errors = np.concatenate([
        (Se_calc - Se_exp)/Se_exp,
        (Sn_calc - Sn_exp)/Sn_exp
    ])

    return errors

def objective_function_separate_de(A, E, Se_exp, Sn_exp):
    return 0.5 * np.linalg.norm(objective_function_separate(A, E, Se_exp, Sn_exp))


def main(path_to_file, density):
    energies, stopping_e, stopping_n = read_srim_data(path_to_file)
    stopping_total = stopping_e + stopping_n

    # Initial guess for parameters A1-A5
    # These values are chosen based on typical orders of magnitude
    A0 = [1e-1, 0.5, 1e-6, -0.5, 1e-7]

    # Initial guess for parameters based on A&Z values
    ch0 = [1.262, 1.44, 242.6, 12000.0, 0.1159]  # Electronic stopping, ch1 ~ A&Z value
    nk0 = [5.0, -0.5, 1e-3]  # Nuclear stopping
    params0 = np.concatenate([ch0, nk0])

    # Bounds for parameters

    bounds_lower = [0.8, 0.3, 1e-7, -1.0, 1e-8]    # ch1-ch5 lower bounds
    bounds_upper = [1E10, 1E1, 1e5, 1E6, 1e1]  # ch1-ch5 upper bounds
    # bounds_upper = [1.6, 0.7, 1e-3, 0.1, 1e-5]  # ch1-ch5 upper bounds
    bounds_lower.extend([0, -1.0, 1e-4])  # nk1-nk3 lower bounds
    bounds_upper.extend([10.0, -0.1, 1e-2])  # nk1-nk3 upper bounds

    # Perform optimization
    eps = float(np.finfo(np.float64).eps)
    res_de: OptimizeResult = differential_evolution(
        func=objective_function_separate_de,
        args=(energies, stopping_e, stopping_n),
        x0=params0,
        # bounds=[(0,1E7), (0,2E5), (0, 1E1), (-3,1E1),(0,1E3)],
        bounds=[(bl, bu) for bl, bu in zip(bounds_lower, bounds_upper)],
        maxiter=10000 * len(params0),
        tol=eps,
        atol=eps,
        workers=-1,
        updating='deferred',
        recombination=0.2,
        strategy='best1bin',
        mutation=(0.2, 1.5),
        init='sobol',
        polish=False,
        disp=True
    )

    result = least_squares(objective_function_separate, res_de.x,
                           args=(energies, stopping_e, stopping_n),
                           # bounds=(bounds_lower, bounds_upper),
                           xtol=eps,
                           ftol=eps,
                           gtol=eps,
                           method='trf',
                           max_nfev=1000*len(params0),
                           loss='linear',
                           verbose=2)

    # Get optimized parameters
    A_opt = result.x[:5]
    nk_opt = result.x[5:]

    # Calculate fitted stopping powers
    Se_fit = nonlocal_inelastic_stopping(energies, A_opt)
    Sn_fit = nuclear_stopping(energies, nk_opt)
    S_total_fit = Se_fit + Sn_fit

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
    fig.set_size_inches(6., 5.5)

    # Plot stopping powers
    ax1.loglog(energies, stopping_total, 'ko', label='Total (SRIM)', alpha=0.5, markersize=4)
    ax1.loglog(energies, S_total_fit, 'k-', label='Total (Fit)', linewidth=2)
    ax1.loglog(energies, stopping_e, 'bo', label='Electronic (SRIM)', alpha=0.3, markersize=4)
    ax1.loglog(energies, Se_fit, 'b--', label='Electronic (Fit)', linewidth=1)
    ax1.loglog(energies, stopping_n, 'ro', label='Nuclear (SRIM)', alpha=0.3, markersize=4)
    ax1.loglog(energies, Sn_fit, 'r--', label='Nuclear (Fit)', linewidth=1)
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Stopping (MeV/(mg/cm²))')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_title('Total and Component Stopping Powers')

    # Plot relative error of total stopping power
    rel_error_total = (S_total_fit - stopping_total) / stopping_total * 100
    ax2.semilogx(energies, rel_error_total, 'k-', label='Total', linewidth=2)
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Relative Error (%)')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_title('Relative Error of Total Stopping Power Fit')
    ax2.set_ylim(-20, 20)

    # Add text box with fitted parameters
    param_text = (f'Fitted Parameters:\n'
                  f'A1 = {A_opt[0]:.3e}\n'
                  f'A2 = {A_opt[1]:.3f}\n'
                  f'A3 = {A_opt[2]:.3e}\n'
                  f'A4 = {A_opt[3]:.3f}\n'
                  f'A5 = {A_opt[4]:.3e}')

    ax1.text(0.95, 0.05, param_text,
             transform=ax1.transAxes,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Print parameters to console
    print("\nFitted Parameters:")
    print(f"A1 = {A_opt[0]:.3e}")
    print(f"A2 = {A_opt[1]:.3f}")
    print(f"A3 = {A_opt[2]:.3e}")
    print(f"A4 = {A_opt[3]:.3f}")
    print(f"A5 = {A_opt[4]:.3e}")

    # Convert fitted parameters to TRIM.SP units
    ch1_trimsp = A_opt[0] * 10 * density  # Convert to eV/Å
    ch2_trimsp = A_opt[1]  # Exponent stays the same
    ch3_trimsp = A_opt[2]  # Dimensionless
    ch4_trimsp = A_opt[3]  # Exponent stays the same
    ch5_trimsp = A_opt[4]  # Already in keV^-1

    nk1_trimsp = nk_opt[0] * 10 * density  # Convert to eV/Å
    nk2_trimsp = nk_opt[1]
    nk3_trimsp = nk_opt[2]

    # Print parameters for TRIM.SP input
    print("\nTRIM.SP Electronic Stopping Coefficients:")
    print("----------------------------------------")
    print("Units: Energy in keV, Stopping in eV/Å")
    print(f"ch1 = {ch1_trimsp:.6e}  ! Proportionality constant [eV/Å]")
    print(f"ch2 = {ch2_trimsp:.6f}  ! Energy exponent [dimensionless]")
    print(f"ch3 = {ch3_trimsp:.6e}  ! Denominator coefficient [dimensionless]")
    print(f"ch4 = {ch4_trimsp:.6f}  ! Denominator exponent [dimensionless]")
    print(f"ch5 = {ch5_trimsp:.6e}  ! Linear energy term [keV^-1]")

    print("\nNuclear stopping parameters:")
    print(f"nk1 = {nk1_trimsp:.6e}  ! Nuclear stopping constant [eV/Å]")
    print(f"nk2 = {nk2_trimsp:.6f}  ! Energy exponent [dimensionless]")
    print(f"nk3 = {nk3_trimsp:.6e}  ! Energy term [keV^-1]")

    # Calculate and print fit quality metrics
    rmse = np.sqrt(np.mean((S_total_fit - stopping_total) ** 2))
    max_rel_error = np.max(np.abs(rel_error_total))
    print("\nFit Quality Metrics:")
    print(f"RMSE: {rmse:.2e} MeV/(mg/cm²)")
    print(f"Maximum Relative Error: {max_rel_error:.1f}%")

    return ch1_trimsp, fig

if __name__ == "__main__":
    filename = PATH_TO_SRIM_FILE
    params, plot = main(path_to_file=PATH_TO_SRIM_FILE, density=DENSITY)
    plt.show()
