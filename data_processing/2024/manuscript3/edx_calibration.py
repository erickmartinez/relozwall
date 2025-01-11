"""
This code is used to obtain the Cliff-Lorime (k-) Sensitivity Factors for elemental quantification using
two known references hBN (McMaster 95% purity), and H3BO3 (Fisher (99% purity).

It uses hyperspy and exspy to load the EDX spectra saved from the Fisher Pathfinder software in .emsa file format.
The advantage of these python libraries is that they can easily read metadata in the files and contain basic functions
to estimate the intensities of the peaks for the required transitions.
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import hyperspy.api as hs
from exspy.signals import EDSSpectrum
from exspy.models import EDSSEMModel, EDSTEMModel
from matplotlib import ticker
from hyperspy._components.gaussian import Gaussian as GaussianComponent
from hyperspy.component import Parameter as hsparam
from scipy.stats.distributions import t
import itertools as itt

"""
See: 
https://exspy.readthedocs.io/en/latest/user_guide/eds.html
https://www.globalsino.com/EM/page4624.html
"""

path_to_hbn_file = r'./data/EDX/20241202/hBN/Base(1).emsa'
path_to_h3bo3_file = r'./data/EDX/20241210/H3BO3/EDS/Base(1).emsa'

line_energies = {
    'B': 0.1833,
    'C': 0.2774,
    'N': 0.3924,
    'O': 0.5249
}

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

def get_tval(counts):
    n = len(counts)  # The number of datapoints in the spectrum
    p = 3 * 4  # 3 gaussian parameters per element
    dof = max(n - p, 0)
    confidence_level = 0.95
    alpha = 1. - confidence_level
    tval = t.ppf(1. - alpha / 2., dof)
    se_factor = tval / np.sqrt(n)
    return se_factor, tval


def quantify_bno(i_b, i_n, i_o, i_b_delta=None, i_n_delta=None, i_o_delta=None, k_BN=2.03, k_OB=0.152, k_BN_delta=0.22, k_OB_delta=0.005):
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

def main():
    global path_to_hbn_file, path_to_h3bo3_file
    s_bn: EDSSpectrum = hs.load(path_to_hbn_file, signal_type='EDS_SEM') # load the eds spectrum of hBN
    t_bn = s_bn.metadata.Acquisition_instrument.SEM.Detector.EDS.real_time # The acquisition real time for the hBN spectrum
    i_bn = s_bn.metadata.Acquisition_instrument.SEM.beam_current # Get the current for the hBN EDS spectrum
    i_bn_u = s_bn.metadata.Acquisition_instrument.SEM.beam_current_units # Get the units of the current for the hBN spectrum

    # print("******** hBN spectrum metadata ********")
    # print(s_bn.metadata)

    # The data from the spectrum
    energy_keV_bn = s_bn.axes_manager[0].axis
    counts_bn = s_bn.data

    tval_bn, se_factor_bn = get_tval(counts_bn)

    # Refer to exspy to build a model to esimate the intensities of the peaks
    s_bn.set_elements(['B', 'C', 'N', 'O'])

    m_bn: EDSSEMModel = s_bn.create_model()
    m_bn.print_current_values()
    m_bn.calibrate_energy_axis(calibrate='resolution')
    m_bn.calibrate_xray_lines('energy', ['N_Ka'], bound=10)
    m_bn.calibrate_xray_lines('width', ['N_Ka'], bound=10)
    m_bn.fit()
    m_bn.fit_background()


    load_plot_style()

    result_bn = m_bn.get_lines_intensity(plot_result=True)
    B_Ka_component: GaussianComponent = m_bn.active_components[1]
    N_Ka_component: GaussianComponent = m_bn.active_components[3]
    O_Ka_component_bn: GaussianComponent = m_bn.active_components[4]
    B_Ka_height: hsparam = B_Ka_component.parameters[0]
    N_Ka_height: hsparam = N_Ka_component.parameters[0]
    O_Ka_height_bn: hsparam = O_Ka_component_bn.parameters[0]

    I_B_Ka = B_Ka_height.value
    I_B_Ka_delta = B_Ka_height.std * se_factor_bn

    I_N_Ka = N_Ka_height.value
    I_N_Ka_delta = N_Ka_height.std * se_factor_bn

    I_O_Ka_bn = O_Ka_height_bn.value
    I_O_Ka_delta_bn = O_Ka_height_bn.std * se_factor_bn

    print("******* FITTED INTENSITIES OF B_Ka AND N_Ka (hBN) *********")
    print(f"B_Ka at {B_Ka_component.parameters[1].value:>6.4f} keV: Intensity: {I_B_Ka:>5.4E} -/+ {I_B_Ka_delta:>5.4E}")
    print(f"N_Ka at {N_Ka_component.parameters[1].value:>6.4f} keV: Intensity: {I_N_Ka:>5.4E} -/+ {I_N_Ka_delta:>5.4E}")



    """
    Get the ratio I_N_Ka/I_B_Ka to estimate the k-parameter according to
    https://www.globalsino.com/EM/page4624.html
    The boron reference has 95% boron nitride (BN) and 5% boron trioxide (BO3)
    0.95 * (0.5B, 0.5 N) + 0.05 * (0.25 B, 0.75 O) = (0.95*0.5 + 0.05 *0.25) B, 0.95*0.5 N, 0.05*0.75 O
    
    """
    C_B, C_N, C_O = 0.485, 0.475, 0.0375
    k_BN = (C_B / I_B_Ka) * (I_N_Ka / C_N)
    # Assume that the reference hBN is 99 % pure (error in concentration is 1%)
    # Assume the error in the total BN purity (1%) expands equally to both concentrations
    dC_B = 0.01 * C_B
    dC_N = 0.01 * C_N
    dC_O = 0.01 * C_O
    dk_BN = k_BN * np.linalg.norm([dC_B/C_B, dC_N/C_N, I_B_Ka_delta/I_B_Ka, I_N_Ka_delta/I_N_Ka])

    k_OB_bn = (C_O / C_B) * (I_B_Ka / I_O_Ka_bn)
    dk_OB_bn = k_OB_bn * np.linalg.norm([dC_B / C_B, dC_O / C_O, I_B_Ka_delta / I_B_Ka, I_O_Ka_delta_bn / I_O_Ka_bn])

    print("******* ESTIMATED k_factors ********")
    print(f"k_BN = {k_BN:.2f} -/+ {dk_BN:.2f}")
    print(f"k_OB = {k_OB_bn:.2f} -/+ {dk_OB_bn:.2f}")


    fig_bn, ax_bn = plt.subplots(1, 1, constrained_layout=True)
    fig_bn.set_size_inches(4.5, 3.)
    # ax.plot(energy_keV, counts)

    m_bn.plot(True,fig=fig_bn, xray_lines=['B_Ka', 'C_Ka', 'N_Ka', 'O_Ka'])

    axes = fig_bn.get_axes()
    axes[0].remove()

    axes = fig_bn.get_axes()
    ax = axes[0]
    # hs.plot.plot_spectra(s_bn, fig=fig, ax=ax)
    #
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Counts')
    ax.set_ylim(top=counts_bn.max()*1.3)

    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((-3,3))
    ax.yaxis.set_major_formatter(xfmt)
    ax.ticklabel_format(useMathText=True)

    ax.set_xlim(0, 2)
    ax.set_title('hBN reference')


    k_BN_txt = fr"$k_{{\mathrm{{BN}}}} = {k_BN:.2f} \pm {dk_BN:.2f}$"
    ax.text(
        0.95, 0.95, k_BN_txt, ha='right', va='top', transform=ax.transAxes,
        fontsize=11, usetex=True
    )

    """
    Process the H3BO3 spectrum
    """
    s_ba: EDSSpectrum = hs.load(path_to_h3bo3_file, signal_type='EDS_SEM')  # load the eds spectrum of H3BO3
    t_ba = s_ba.metadata.Acquisition_instrument.SEM.Detector.EDS.real_time  # The acquisition real time for the H3BO3 spectrum
    i_ba = s_ba.metadata.Acquisition_instrument.SEM.beam_current  # Get the current for the H3BO3 EDS spectrum
    i_ba_u = s_ba.metadata.Acquisition_instrument.SEM.beam_current_units  # Get the units of the current for the H3BO3 spectrum
    # print("******** H3BO3 spectrum metadata ********")
    # print(s_ba.metadata)

    energy_keV_ba = s_ba.axes_manager[0].axis
    counts_ba = s_ba.data

    tval_ba, se_factor_ba = get_tval(counts_ba)

    s_ba.set_elements(['B', 'C', 'N', 'O'])

    m_ba: EDSSEMModel = s_ba.create_model()
    m_ba.print_current_values()
    m_ba.calibrate_energy_axis(calibrate='resolution')
    m_ba.calibrate_xray_lines('energy', ['O_Ka'], bound=10)
    m_ba.calibrate_xray_lines('width', ['O_Ka'], bound=10)
    m_ba.fit()
    m_ba.fit_background()

    result_ba = m_ba.get_lines_intensity(plot_result=True)

    # print(m_ba.active_components)
    B_Ka_component_ba: GaussianComponent = m_ba.active_components[1]
    N_Ka_component_ba: GaussianComponent = m_ba.active_components[3]
    O_Ka_component_ba: GaussianComponent = m_ba.active_components[4]
    B_Ka_height_ba: hsparam = B_Ka_component_ba.parameters[0]
    N_Ka_height_ba: hsparam = N_Ka_component_ba.parameters[0]
    O_Ka_height_ba: hsparam = O_Ka_component_ba.parameters[0]

    # print(O_Ka_component_ba.parameters)

    I_B_Ka_ba = B_Ka_height_ba.value
    I_B_Ka_ba_delta = B_Ka_height_ba.std * se_factor_ba

    I_N_Ka_ba = N_Ka_height_ba.value
    I_N_Ka_ba_delta = N_Ka_height_ba.std * se_factor_ba

    I_O_Ka_ba = O_Ka_height_ba.value
    I_O_Ka_ba_delta = O_Ka_height_ba.std * se_factor_ba

    print("******* FITTED INTENSITIES OF B_Ka AND O_Ka (H3BO3) *********")
    print(f"B_Ka at {B_Ka_component_ba.parameters[1].value:>6.4f} keV: Intensity: {I_B_Ka_ba:>5.4E} -/+ {I_B_Ka_ba_delta:>5.4E}")
    print(f"O_Ka at {O_Ka_component_ba.parameters[1].value:>6.4f} keV: Intensity: {I_O_Ka_ba:>5.4E} -/+ {I_O_Ka_ba_delta:>5.4E}")

    """
    Get the ratio I_B_Ka/I_O_Ka to estimate the k-parameter according to
    https://www.globalsino.com/EM/page4624.html
    """
    C_B, C_O = 0.25, 0.75
    k_OB_ba = (C_O / C_B) * (I_B_Ka_ba / I_O_Ka_ba)
    # Assume that the reference hBN is 99.5 % pure (error in concentration is 0.5%)
    # Assume the error in the total H3BO3 purity (0.5%) expands equally to the
    # concentration of C_B and C_O
    dC_B = 0.005 * C_B
    dC_O = 0.005 * C_O
    dk_OB_ba = k_OB_ba * np.linalg.norm([dC_B / C_B, dC_O / C_O, I_B_Ka_ba_delta / I_B_Ka_ba, I_O_Ka_ba_delta / I_O_Ka_ba])

    k_OB = 0.5 * (k_OB_bn + k_OB_ba)
    dk_OB = np.linalg.norm([dk_OB_bn, dk_OB_ba]) / np.sqrt(2.)
    # k_OB = k_OB_ba
    # dk_OB = dk_OB_ba

    print("******* ESTIMATED K_OB ********")
    print(f"k_OB = {k_OB_ba:.2f} -/+ {dk_OB_ba:.2f}")
    print("******* Mean K_OB ********")
    print(f"k_OB = {k_OB:.3f} -/+ {dk_OB:.3f}")
    print("******* Using K_OB from H3BO3 ********")
    k_OB = k_OB_ba
    dk_OB = dk_OB_ba


    fig_ba, ax_ba = plt.subplots(1, 1, constrained_layout=True)
    fig_ba.set_size_inches(4.5, 3.)
    # ax.plot(energy_keV, counts)

    m_ba.plot(True, fig=fig_ba, xray_lines=['B_Ka', 'C_Ka', 'N_Ka', 'O_Ka'])

    axes2 = fig_ba.get_axes()
    axes2[0].remove()

    axes2 = fig_ba.get_axes()
    ax2 = axes2[0]
    # hs.plot.plot_spectra(s_bn, fig=fig, ax=ax)
    #
    ax2.set_xlabel('Energy (keV)')
    ax2.set_ylabel('Counts')
    ax2.set_ylim(top=counts_ba.max() * 1.3)

    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((-3, 3))
    ax2.yaxis.set_major_formatter(xfmt)
    ax2.ticklabel_format(useMathText=True)

    ax2.set_xlim(0, 2)
    ax2.set_title(r'H$_{\mathregular{3}}$BO$_{\mathregular{3}}$ reference')

    k_BO_txt = fr"$k_{{\mathrm{{OB}}}} = {k_OB:.3f} \pm {dk_OB:.3f}$"
    ax2.text(
        0.95, 0.95, k_BO_txt, ha='right', va='top', transform=ax2.transAxes,
        fontsize=11, usetex=True
    )


    # Estimate the concentration of B, N, and O in the hBN sample
    print("======================================================================")
    print(f"Estimation of concentration of O and B in the hBN sample using k_OB:")
    print("======================================================================")

    x, x_err = quantify_bno(
        i_b=I_B_Ka, i_n=I_N_Ka, i_o=I_O_Ka_bn,
        i_b_delta=I_O_Ka_delta_bn, i_n_delta=I_N_Ka_delta, i_o_delta=I_O_Ka_delta_bn,
        k_BN=k_BN, k_OB=k_OB, k_BN_delta=dk_BN, k_OB_delta=dk_OB
    )
    subscripts = ['B', 'N', 'O']
    for i, xi in enumerate(x):
        print(f"c_{subscripts[i]} = {xi:.3f} -/+ {x_err[i]:.3f}")

    # Estimate the concentration of O, N, and B in the H3BO3 sample
    c_b_bn = 1. / (1. + (1./k_BN) * (I_N_Ka_ba / I_B_Ka_ba)  + k_OB * (I_O_Ka_ba / I_B_Ka_ba))
    c_o_bn = k_OB * (I_O_Ka_ba / I_B_Ka_ba) * c_b_bn
    c_n_bn = 1. - c_b_bn - c_o_bn
    print("======================================================================")
    print(f"Estimation of concentration of O and B in the H3BO3 sample using k_OB:")
    print("======================================================================")
    print(f"C_B = {c_b_bn:.3f}, C_N = {c_n_bn:.3f}, C_O = {c_o_bn:.3f}")

    print("")
    print("******** Estimation of the concentration of B, O, and N in H3BO3 using a linear solver ***********")
    # a = np.array([[1., 1., 1.], [1., -k_BN * (I_B_Ka_ba / I_N_Ka_ba), 0.], [k_OB*(I_O_Ka_ba/I_B_Ka_ba), 0., -1.]])
    # print(a)
    # b = np.array([1., 0., 0.])
    # x = np.linalg.solve(a, b)
    x, x_err = quantify_bno(
        i_b=I_B_Ka_ba, i_n=I_N_Ka_ba, i_o=I_O_Ka_ba,
        i_b_delta=I_B_Ka_ba_delta, i_n_delta=I_N_Ka_ba_delta, i_o_delta=I_O_Ka_ba_delta,
        k_BN=k_BN, k_OB=k_OB, k_BN_delta=dk_BN, k_OB_delta=dk_OB
    )
    subscripts = ['B', 'N', 'O']
    for i, xi in enumerate(x):
        print(f"c_{subscripts[i]} = {xi:.3f} -/+ {x_err[i]:.3f}")

    fig_bn.savefig(r'./figures/edx_calibration_hbn.png', dpi=600)
    fig_ba.savefig(r'./figures/edx_calibration_h3bo3.png', dpi=600)

    plt.show()


if __name__ == '__main__':
    main()


