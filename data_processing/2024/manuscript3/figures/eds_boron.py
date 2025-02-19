import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
import json
import re
from scipy.signal import find_peaks
import matplotlib.ticker as ticker
import hyperspy.api as hs
from exspy.signals import EDSSpectrum
from exspy.models import EDSSEMModel, EDSTEMModel
from hyperspy._components.gaussian import Gaussian as GaussianComponent
from hyperspy.component import Parameter as hsparam
from scipy.stats.distributions import t
import itertools as itt

path_to_pure_boron_csv = r'../data/EDX/20241122/hp_b_rod/Base(2).emsa'
path_to_pc_pebble_rod_csv = r'../data/EDX/20241122/pc_bp_rod/Base(1).emsa'

"""
k_BO = 0.15 -/+ 0.01
k_BN = 2.03 -/+ 0.08
"""
k_BO, k_BO_delta = 0.15,  0.01
k_BN, k_BN_delta = 2.08,  0.08

def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)
    mpl.rcParams['text.latex.preamble'] = (r'\usepackage{mathptmx}'
                                           r'\usepackage{xcolor}'
                                           r'\usepackage{helvet}'
                                           r'\usepackage{siunitx}'
                                           r'\usepackage{amsmath, array, makecell}')

line_energies = {
    r'$\mathregular{B_{K_a}}$': 0.1833,
    r'$\mathregular{C_{K_a}}$': 0.2774,
    r'$\mathregular{N_{K_a}}$': 0.3924,
    r'$\mathregular{O_{K_a}}$': 0.5249
}
pattern_id = re.compile(r".*?PEAKLAB\s+\:\s+(\d+\.?\d*)\s+(\w+)\s+(\w+)\s+(\d+)")

def quantify_bno(i_b, i_n, i_o, i_b_delta=None, i_n_delta=None, i_o_delta=None, k_BN=2.08, k_OB=0.152, k_BN_delta=0.08, k_OB_delta=0.01):
    a22 = -k_BN * (i_b / i_n)
    a31 = k_OB * (i_o / i_b)
    a = np.array([[1., 1., 1.], [1., a22, 0.], [a31, 0., -1.]])
    b = np.array([1., 0., 0.])
    if (i_b_delta is None) or (i_n_delta is None) or (i_o_delta is None):
        return np.linalg.solve(a, b), None
    d_22 = a22 * np.linalg.norm([k_BN_delta/k_BN, i_b_delta/i_b, i_n_delta/i_n])
    d_31 = a31 * np.linalg.norm([k_OB_delta/k_OB_delta, i_o_delta/i_o, i_b_delta/i_b])
    a_e = np.array([[0., 0., 0.], [0., d_22, 0.], [d_31, 0., 0.]])
    x, dx = propagate_matrix_uncertainty(A=a, A_uncertainties=a_e, b=b, method='linear_propagation')
    return x, dx



def propagate_matrix_uncertainty(A, A_uncertainties, b, b_uncertainties=None, method='monte_carlo', n_samples=10000):
    """
    Propagate uncertainties in a linear system Ax = b where A has uncertainties.

    Parameters:
    -----------
    A : ndarray
        Coefficient matrix (n x n)
    A_uncertainties : ndarray
        Matrix of uncertainties in A coefficients (n x n)
    b : ndarray
        Right-hand side vector (n)
    b_uncertainties : ndarray, optional
        Uncertainties in b vector
    method : str
        'monte_carlo' or 'linear_propagation'
    n_samples : int
        Number of Monte Carlo samples

    Returns:
    --------
    x_mean : ndarray
        Mean solution vector
    x_uncertainties : ndarray
        Uncertainties in solution vector
    """
    n = len(A)

    if method == 'monte_carlo':
        # Monte Carlo simulation
        x_samples = np.zeros((n_samples, n))

        for i in range(n_samples):
            # Generate random perturbations for A
            A_perturbed = A + np.random.normal(0, 1, size=A.shape) * A_uncertainties

            # Generate random perturbations for b if uncertainties provided
            if b_uncertainties is not None:
                b_perturbed = b + np.random.normal(0, 1, size=b.shape) * b_uncertainties
            else:
                b_perturbed = b

            # Solve perturbed system
            try:
                x_samples[i] = np.linalg.solve(A_perturbed, b_perturbed)
            except np.linalg.LinAlgError:
                # Skip singular matrices
                continue

        # Calculate statistics
        x_mean = np.mean(x_samples, axis=0)
        x_uncertainties = np.std(x_samples, axis=0)

    elif method == 'linear_propagation':
        # First-order linear uncertainty propagation
        x_nominal = np.linalg.solve(A, b)

        # Initialize Jacobian matrices
        J_A = np.zeros((n, n * n))

        # Calculate partial derivatives numerically
        eps = 1e-8
        for i, j in itt.product(range(n), range(n)):
            A_perturbed = A.copy()
            A_perturbed[i, j] += eps
            x_perturbed = np.linalg.solve(A_perturbed, b)
            J_A[:, i * n + j] = (x_perturbed - x_nominal) / eps

        # Construct covariance matrix for A
        cov_A = np.diag(A_uncertainties.flatten() ** 2)

        # Propagate uncertainties
        cov_x = J_A @ cov_A @ J_A.T

        x_mean = x_nominal
        x_uncertainties = np.sqrt(np.diag(cov_x))

    else:
        raise ValueError("Method must be 'monte_carlo' or 'linear_propagation'")

    return x_mean, x_uncertainties

def get_element_by_energy(energy):
    global line_energies
    elements, energies = list(line_energies), np.array(list(line_energies.values()))
    idx_energy = np.argmin(np.abs(energies-energy))
    return elements[idx_energy]


def get_identified_elements(path_to_emsa):
    global pattern_id
    identified = []
    with open(path_to_emsa, 'r') as f:
        for line in f:
            if line.startswith('#'):
                m = pattern_id.match(line)
                if m:
                    element = m.group(2)
                    energy = float(m.group(1))
                    transition = m.group(3)
                    identified.append({
                        'element': element, 'energy': energy, 'transition':transition
                    })
            else:
                break
    return identified


def get_tval(counts):
    n = len(counts)  # The number of datapoints in the spectrum
    p = 3 * 4  # 3 gaussian parameters per element
    dof = max(n - p, 0)
    confidence_level = 0.95
    alpha = 1. - confidence_level
    tval = t.ppf(1. - alpha / 2., dof)
    se_factor = tval / np.sqrt(n)
    return se_factor, tval

def main():
    global path_to_pure_boron_csv, path_to_pc_pebble_rod_csv, line_energies
    global k_BO, k_BO_delta, k_BN, k_BN_delta

    # Load the spectra using hyperspy and read the metadata
    s_bp: EDSSpectrum = hs.load(path_to_pc_pebble_rod_csv, signal_type='EDS_SEM')  # load the eds spectrum of hBN
    s_br: EDSSpectrum = hs.load(path_to_pure_boron_csv, signal_type='EDS_SEM')  # load the eds spectrum of the boron rod
    energy_keV_bp = s_bp.axes_manager[0].axis
    t_bp = s_bp.metadata.Acquisition_instrument.SEM.Detector.EDS.real_time  # The acquisition real time for the hBN spectrum
    i_bp = s_bp.metadata.Acquisition_instrument.SEM.beam_current  # Get the current for the hBN EDS spectrum
    i_bp_u = s_bp.metadata.Acquisition_instrument.SEM.beam_current_units  # Get the units of the current for the hBN spectrum

    energy_keV_br = s_br.axes_manager[0].axis
    t_br = s_br.metadata.Acquisition_instrument.SEM.Detector.EDS.real_time  # The acquisition real time for the hBN spectrum
    i_br = s_br.metadata.Acquisition_instrument.SEM.beam_current  # Get the current for the hBN EDS spectrum
    i_br_u = s_br.metadata.Acquisition_instrument.SEM.beam_current_units  # Get the units of the current for the hBN spectrum

    counts_bp = s_bp.data
    cpspna_bp = counts_bp / t_bp / i_bp

    counts_br = s_br.data
    cpspna_br = counts_br / t_br / i_br

    tval_bp, se_factor_bn = get_tval(counts_bp)


    # Refer to exspy to build a model to esimate the intensities of the peaks
    s_bp.set_elements(['B', 'C', 'N', 'O'])

    m_bp: EDSSEMModel = s_bp.create_model()
    m_bp.print_current_values()
    m_bp.calibrate_energy_axis(calibrate='resolution')
    m_bp.calibrate_xray_lines('energy', ['O_Ka'], bound=10)
    m_bp.calibrate_xray_lines('width', ['O_Ka'], bound=10)
    m_bp.fit()
    m_bp.fit_background()

    result_bn = m_bp.get_lines_intensity(plot_result=True)
    B_Ka_component_bp: GaussianComponent = m_bp.active_components[1]
    N_Ka_component_bp: GaussianComponent = m_bp.active_components[3]
    O_Ka_component_bp: GaussianComponent = m_bp.active_components[4]
    B_Ka_height_bp: hsparam = B_Ka_component_bp.parameters[0]
    N_Ka_height_bp: hsparam = N_Ka_component_bp.parameters[0]
    O_Ka_height_bp: hsparam = O_Ka_component_bp.parameters[0]

    I_B_Ka_bp = B_Ka_height_bp.value
    I_B_Ka_bp_delta = B_Ka_height_bp.std * se_factor_bn

    I_N_Ka_bp = N_Ka_height_bp.value
    I_N_Ka_bp_delta = N_Ka_height_bp.std * se_factor_bn

    I_O_Ka_bp = O_Ka_height_bp.value
    I_O_Ka_bp_delta = O_Ka_height_bp.std * se_factor_bn

    atomic_concentrations, concentration_uncertainties = quantify_bno(
        i_b=I_B_Ka_bp, i_n=I_N_Ka_bp, i_o=I_O_Ka_bp, i_b_delta=I_B_Ka_bp_delta, i_n_delta=I_N_Ka_bp_delta,
        i_o_delta=I_O_Ka_bp_delta
    )

    """
    Monte Carlo uncertainties
    -------------------------
    c_B = 0.477 -/+ 0.053
    c_N = 0.432 -/+ 0.049
    c_O = 0.090 -/+ 0.096
    
    Linear propagation
    ------------------
    c_B = 0.473 -/+ 0.050
    c_N = 0.426 -/+ 0.046
    c_O = 0.101 -/+ 0.091
    
    """
    subscripts = ['B', 'N', 'O']

    concentrations_txt = ""
    for i, xi, dxi in zip(range(len(atomic_concentrations)), atomic_concentrations, concentration_uncertainties):
        print(f"c_{subscripts[i]} = {xi:.3f} -/+ {dxi:.3f}")
        concentrations_txt += rf"$c_{{\mathrm{{{subscripts[i]}}}}} = ({100*xi:.0f} \pm {100*dxi:.0f})~\%$"
        if i < 3:
            concentrations_txt += '\\\\ \n'

    col_names = ['Energy (keV)', 'Counts']
    # boron_df = pd.read_csv(path_to_pure_boron_csv, comment='#', header=None, names=col_names, usecols=[0,1]).apply(pd.to_numeric)
    # pc_bp_df = pd.read_csv(path_to_pc_pebble_rod_csv, comment='#', header=None, names=col_names, usecols=[0,1]).apply(pd.to_numeric)

    boron_df = pd.DataFrame(data={
        'Energy (keV)': energy_keV_br,
        'Counts': counts_br,
        f'Counts/s/{i_br_u}': cpspna_br
    })

    pc_bp_df = pd.DataFrame(data={
        'Energy (keV)': energy_keV_bp,
        'Counts': counts_bp,
        f'Counts/s/{i_bp_u}': cpspna_bp
    })


    boron_df = boron_df[boron_df['Energy (keV)'] <= 2.5].reset_index(drop=True)
    pc_bp_df = pc_bp_df[pc_bp_df['Energy (keV)'] <= 2.5].reset_index(drop=True)
    # boron_rod_identified = get_identified_elements(path_to_pure_boron_csv)
    # pc_boron_pebble_identified = get_identified_elements(path_to_pc_pebble_rod_csv)

    energy_b_rod = boron_df['Energy (keV)'].values
    energy_pc_b_rod = pc_bp_df['Energy (keV)'].values

    intensity_b_rod = boron_df[f'Counts/s/{i_br_u}'].values * 1E-3  # kCounts
    intensity_pc_b_pebble_rod = pc_bp_df[f'Counts/s/{i_bp_u}'].values * 1E-3 # kCounts
    # intensity_pc_b_pebble_rod = cpspna_bp[energy_keV_bp <= 2.5]

    peaks_b_rod, _ = find_peaks(intensity_b_rod, threshold=0.01)
    peaks_pc_b_rod, _ = find_peaks(intensity_pc_b_pebble_rod, threshold=0.01)

    peaks_b_rod = energy_b_rod[peaks_b_rod]
    peaks_b_rod = np.append(peaks_b_rod, [0.524])

    peaks_pc_b_rod = energy_pc_b_rod[peaks_pc_b_rod]

    # print(peaks_b_rod)
    # print()
    # print(energy_pc_b_rod[peaks_pc_b_rod])

    load_plot_style()

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True)
    fig.set_size_inches(4.5, 4.75)

    ax1.plot(energy_b_rod, intensity_b_rod, color='C0', label='B rod')
    ax2.plot(energy_pc_b_rod, intensity_pc_b_pebble_rod, color='C1', label='poly-B pebble rod')

    for peak_energy in  peaks_b_rod:
        idx_energy = np.argmin(np.abs(energy_b_rod - peak_energy))
        intensity = intensity_b_rod[idx_energy]
        element = get_element_by_energy(peak_energy)
        ax1.plot(
            [peak_energy], [intensity], marker='|', ms=10, color='tab:red', ls='none', mew=1.5
        )

        ax1.annotate(
            text=element,
            xy=(peak_energy, intensity), xytext=(0, 15),
            xycoords='data', textcoords='offset pixels',
            ha='center', va='bottom'
        )

    for peak_energy in peaks_pc_b_rod:
        idx_energy = np.argmin(np.abs(energy_pc_b_rod - peak_energy))
        intensity = intensity_pc_b_pebble_rod[idx_energy]
        element = get_element_by_energy(peak_energy)
        ax2.plot(
            [peak_energy], [intensity], marker='|', ms=10, color='tab:red', ls='none', mew=1.5
        )

        ax2.annotate(
            text=element,
            xy=(peak_energy, intensity), xytext=(0, 15),
            xycoords='data', textcoords='offset pixels',
            ha='center', va='bottom'
        )

    ax1.set_ylim(0, 2)
    ax2.set_ylim(0, 1)
    for ax in (ax1, ax2):
        ax.set_xlim(0, 2.5)
        # ax.yaxis.set_major_formatter(ticker.EngFormatter)
        ax.legend(loc='upper right', frameon=True, fontsize=10)

    ax2.set_xlabel('Energy (keV)')

    ax1.set_title('Boron rod')
    ax2.set_title('Boron pebble rod')


    fig.supylabel(f'Counts/s/nA (x1000)')

    for i, axi in enumerate([ax1, ax2]):
        axi.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        axi.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        # panel_label = chr(ord('`') + i + 1) # starts from a
        panel_label = chr(ord('`') + i + 1)
        axi.text(
            -0.12, 1.05, f'({panel_label})', transform=axi.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='right'
        )

    ax2.text(
        0.95, 0.75, concentrations_txt, ha='right', va='top', transform=ax2.transAxes, fontsize=12, color='k',
        usetex=True
    )
    fig.savefig('fig_eds_boron.svg', dpi=600)
    fig.savefig('fig_eds_boron.png', dpi=600)
    plt.show()



if __name__ == '__main__':
    main()