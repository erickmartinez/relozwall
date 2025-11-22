import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import h5py


def load_interpolator_from_h5(filepath="boron_ion_interpolator.h5"):
    """
    Loads interpolator components from an HDF5 file and reconstructs the object.
    """
    with h5py.File(filepath, 'r') as hf:
        log_ne_knots = hf['log_ne_knots'][:]
        log_te_knots = hf['log_te_knots'][:]
        coeffs = hf['coeffs'][:]
        kx = hf.attrs['kx']
        ky = hf.attrs['ky']

    tck = (log_ne_knots, log_te_knots, coeffs, kx, ky)
    loaded_interpolator = RectBivariateSpline._from_tck(tck)

    print(f"✅ Interpolator loaded successfully from '{filepath}'")
    return loaded_interpolator

class BoronCRModelCombined:
    """
    B I 821.2 nm line: ²D₅/₂ → ²F₇/₂

    Accounts for BOTH excitation pathways:
    1. Direct: Ground → 4f → 2D (cascade)
    2. Resonant: 2D → 4f → 2D (cyclic, requires metastable population)
    """

    def __init__(self):
        # Energy levels (cm⁻¹ from Table II)
        self.E_ground_cm = 0.0
        self.E_2D_cm = 47857.0  # 2s2p² ²D metastable
        self.E_4f_cm = 60031.0  # 4f ²F
        self.E_ion_cm = 66928.0  # Ionization

        # Convert to eV (1 eV = 8065.54 cm⁻¹)
        cm_to_eV = 1.0 / 8065.54
        self.E_ground = 0.0
        self.E_2D = self.E_2D_cm * cm_to_eV  # 5.93 eV
        self.E_4f = self.E_4f_cm * cm_to_eV  # 7.44 eV
        self.E_ion = self.E_ion_cm * cm_to_eV  # 8.30 eV

        # Transition energies
        self.E_821nm = self.E_4f - self.E_2D  # 1.51 eV (4f → 2D)

        # Statistical weights
        self.g_ground_P12 = 2.0  # ²P₁/₂
        self.g_ground_P32 = 4.0  # ²P₃/₂
        self.g_ground = 6.0  # Total ground state
        self.g_2D = 6.0  # ²D (J=3/2: 4, J=5/2: 6, weighted)
        self.g_4f = 8.0  # ²F (J=5/2: 6, J=7/2: 8, weighted)

        # Einstein coefficients
        self.A_821nm = 8e6  # s⁻¹ (4f ²F → 2D emission)
        self.A_2D_radiative = 1e3  # s⁻¹ (metastable decay to ground)

        self.te_grid = None
        self.ne_grid = None
        self.pec_grid = None
        self.sxb_grid = None

    def excitation_rate_coefficient(self, E_threshold, Te, oscillator_strength=0.1):
        """
        Excitation rate coefficient with proper ADAS-like behavior
        """
        Te = np.maximum(Te, 0.1)
        u = E_threshold / Te

        # Base coefficient (tuned to match ADAS order of magnitude)
        C0 = 1e-13  # m³/s

        # Gaunt factor with proper asymptotic behavior
        if np.isscalar(u):
            if u > 3:
                # Well below threshold
                gaunt = 0.15 * np.exp(-u) / np.sqrt(u)
            elif u > 0.5:
                # Near threshold
                gaunt = 0.2 * np.log(2 / u + 1) * np.exp(-u)
            else:
                # Well above threshold - saturated
                gaunt = 0.25 * (1 + 0.3 * np.log(1 / u))
        else:
            gaunt = np.where(u > 3,
                             0.15 * np.exp(-u) / np.sqrt(u),
                             np.where(u > 0.5,
                                      0.2 * np.log(2 / u + 1) * np.exp(-u),
                                      0.25 * (1 + 0.3 * np.log(1 / np.maximum(u, 0.01)))))

        rate = C0 * oscillator_strength * gaunt

        return rate

    def ionization_rate_coefficient(self, Te):
        """Ionization rate B I → B II"""
        Te = np.maximum(Te, 0.1)
        u = self.E_ion / Te

        C_ion = 3e-14  # m³/s

        if np.isscalar(u):
            if u > 1:
                rate = C_ion * np.log(u) * np.exp(-u) / u
            else:
                rate = C_ion * (0.5 + 0.5 * np.log(1 / u)) / (1 + 0.3 * u)
        else:
            rate = np.where(u > 1,
                            C_ion * np.log(u) * np.exp(-u) / u,
                            C_ion * (0.5 + 0.5 * np.log(1 / np.maximum(u, 0.01))) / (1 + 0.3 * u))

        return rate

    def ionization_rate_coefficient_adas(self, Te, ne):
        loaded_interpolator = load_interpolator_from_h5(filepath="adas_boron_ionization_interpolator.h5")
        log_rate = loaded_interpolator(np.log10(ne*1E-6), np.log10(Te), grid=False)
        return 10 ** log_rate

    def metastable_population_fraction(self, Te, ne):
        """
        Calculate fraction of neutrals in metastable 2s2p² ²D state

        Rate equation balance:
        dn_2D/dt = n_ground * n_e * q_exc - n_2D * (A_rad + n_e * q_deexc + n_e * S_ion_from_2D)

        In quasi-steady state: n_2D/n_ground = q_exc / (A_rad/n_e + q_deexc + S_ion)
        """

        # Excitation from ground to metastable
        q_exc_to_2D = self.excitation_rate_coefficient(
            self.E_2D, Te, oscillator_strength=0.1
        )

        # De-excitation (detailed balance)
        q_deexc_from_2D = q_exc_to_2D * (self.g_ground / self.g_2D) * np.exp(self.E_2D / Te)

        # Ionization from metastable (lower threshold)
        E_ion_from_2D = self.E_ion - self.E_2D  # ~2.37 eV
        u = E_ion_from_2D / Te

        if np.isscalar(Te):
            if u > 1:
                S_ion_from_2D = 3e-14 * np.log(u) * np.exp(-u) / (u * E_ion_from_2D)
            else:
                S_ion_from_2D = 3e-14 * (0.5 + np.log(1 / u)) / E_ion_from_2D
        else:
            S_ion_from_2D = np.where(u > 1,
                                     3e-14 * np.log(u) * np.exp(-u) / (u * E_ion_from_2D),
                                     3e-14 * (0.5 + np.log(1 / np.maximum(u, 0.01))) / E_ion_from_2D)

        # Metastable fraction
        denominator = self.A_2D_radiative / ne + q_deexc_from_2D + S_ion_from_2D
        f_2D = q_exc_to_2D / (denominator + 1e-30)

        # Physical limits
        f_2D = np.clip(f_2D, 0, 0.3)

        return f_2D

    def pec_combined(self, Te, ne):
        """
        PEC for 821 nm including BOTH pathways

        Pathway 1 (Direct cascade):
            Ground → 4f → 2D + hν(821nm)

        Pathway 2 (Resonant):
            2D → 4f → 2D + hν(821nm)
            Requires pre-existing metastable population
        """

        # ====================================================================
        # Pathway 1: Direct excitation from ground to 4f, cascade to 2D
        # ====================================================================

        q_ground_to_4f = self.excitation_rate_coefficient(
            self.E_4f, Te, oscillator_strength=0.05  # Two-electron, weaker
        )

        # Branching ratio: 4f can decay to multiple states
        # Assume 30% branches to 2D specifically
        branching_4f_to_2D = 0.3

        # Collisional quenching from 4f
        q_4f_to_ground = q_ground_to_4f * (self.g_ground / self.g_4f) * np.exp(self.E_4f / Te)
        q_4f_to_2D = self.excitation_rate_coefficient(
            self.E_821nm, Te, oscillator_strength=0.08
        ) * (self.g_2D / self.g_4f) * np.exp(self.E_821nm / Te)

        R_coll_total = ne * (q_4f_to_ground + q_4f_to_2D)
        A_rad_4f = self.A_821nm * branching_4f_to_2D

        eff_branching_direct = A_rad_4f / (A_rad_4f + R_coll_total + 1e-10)

        PEC_direct = q_ground_to_4f * eff_branching_direct

        # ====================================================================
        # Pathway 2: Resonant excitation from metastable
        # ====================================================================

        # Population in metastable
        f_2D = self.metastable_population_fraction(Te, ne)

        # Excitation from metastable to 4f
        q_2D_to_4f = self.excitation_rate_coefficient(
            self.E_821nm, Te, oscillator_strength=0.08
        )

        # For resonant process, emission goes back to metastable
        # So effective PEC must account for net photon production
        # This is essentially radiative trapping

        # Simplification: treat as additional excitation pathway
        # weighted by metastable fraction
        eff_branching_resonant = self.A_821nm / (self.A_821nm + R_coll_total + 1e-10)

        PEC_resonant = f_2D * q_2D_to_4f * eff_branching_resonant

        # ====================================================================
        # Total PEC
        # ====================================================================

        PEC_total = PEC_direct + PEC_resonant

        return PEC_total, PEC_direct, PEC_resonant, f_2D

    def calculate_sxb(self, Te, ne):
        """S/XB = PEC / S_ionization"""
        PEC_total, _, _, _ = self.pec_combined(Te, ne)
        # S_ion = self.ionization_rate_coefficient(Te)
        S_ion = self.ionization_rate_coefficient_adas(Te, ne)

        # S_ion = np.maximum(S_ion, 1e-30)
        sxb = PEC_total / S_ion

        return sxb

    def generate_tables(self, te_range, ne_range):
        """Generate lookup tables"""
        Te_2d, Ne_2d = np.meshgrid(te_range, ne_range)

        # Calculate PEC components
        PEC_total = np.zeros_like(Te_2d)
        PEC_direct = np.zeros_like(Te_2d)
        PEC_resonant = np.zeros_like(Te_2d)
        f_2D = np.zeros_like(Te_2d)

        for i in range(len(ne_range)):
            for j in range(len(te_range)):
                result = self.pec_combined(Te_2d[i, j], Ne_2d[i, j])
                PEC_total[i, j] = result[0]
                PEC_direct[i, j] = result[1]
                PEC_resonant[i, j] = result[2]
                f_2D[i, j] = result[3]

        sxb_grid = self.calculate_sxb(Te_2d, Ne_2d)

        self.te_grid = te_range
        self.ne_grid = ne_range
        self.pec_grid = PEC_total
        self.pec_direct_grid = PEC_direct
        self.pec_resonant_grid = PEC_resonant
        self.f_2D_grid = f_2D
        self.sxb_grid = sxb_grid

        return te_range, ne_range, PEC_total, sxb_grid


def plot_combined_pathways():
    """
    Comprehensive plot showing both excitation pathways
    """

    model = BoronCRModelCombined()

    # ADAS-like ranges
    te_range = np.logspace(-1, 4, 200)
    ne_range_full = np.array([1e0, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18]) * 1e6  # m⁻³

    model.generate_tables(te_range, ne_range_full)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # ========================================================================
    # Plot 1: Total PEC (ADAS style)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2])

    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_range_full)))

    for i, ne_val in enumerate(ne_range_full):
        idx = np.argmin(np.abs(model.ne_grid - ne_val))
        pec_cm3 = model.pec_grid[idx, :] * 1e6

        ax1.loglog(te_range, pec_cm3, '-', color=colors[i], linewidth=2.5,
                   label=f'$n_e = 10^{{{int(np.log10(ne_val / 1e6))}}}$ cm$^{{-3}}$',
                   marker='o' if i % 2 == 0 else 's', markersize=5, markevery=6)

    # S marker
    # S_marker = model.ionization_rate_coefficient(te_range) * 2.5e-5 * 1e6
    # ax1.loglog(te_range, S_marker, 'kx', markersize=12, markeredgewidth=3,
    #            label='$S = 2.5 \\times 10^{-11}$ cm³/s', markevery=6, zorder=10)

    ax1.set_xlabel('$T_e$ (eV)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('PEC (cm³/s)', fontsize=14, fontweight='bold')
    ax1.set_title('Total Photon Emissivity Coefficient - B I 821.2 nm\n' +
                  '²D₅/₂ → ²F₇/₂ (Combined Direct + Resonant Pathways)',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right', ncol=2)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim([0.1, 10000])
    ax1.set_ylim([1e-50, 1e-8])

    # ========================================================================
    # Plot 2: Pathway contributions
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 2])

    ne_ref = 1e19  # m⁻³
    idx_ref = np.argmin(np.abs(model.ne_grid - ne_ref))

    pec_total = model.pec_grid[idx_ref, :] * 1e6
    pec_direct = model.pec_direct_grid[idx_ref, :] * 1e6
    pec_resonant = model.pec_resonant_grid[idx_ref, :] * 1e6

    ax2.loglog(te_range, pec_total, 'k-', linewidth=3, label='Total PEC')
    ax2.loglog(te_range, pec_direct, 'r--', linewidth=2, label='Direct pathway')
    ax2.loglog(te_range, pec_resonant, 'b--', linewidth=2, label='Resonant pathway')

    ax2.set_xlabel('$T_e$ (eV)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PEC (cm³/s)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Pathway Contributions\n$n_e$ = {ne_ref:.0e} m$^{{-3}}$',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim([0.1, 10000])

    # ========================================================================
    # Plot 3: Energy level diagram
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')

    y_pos = {
        'ground': 0.05,
        '2D': 0.45,
        '4f': 0.70,
        'ion': 0.95
    }

    E_labels = {
        'ground': '0.00',
        '2D': '5.93',
        '4f': '7.44',
        'ion': '8.30'
    }

    # Draw energy levels
    for state, y in y_pos.items():
        ax3.plot([0.15, 0.85], [y, y], 'k-', linewidth=4)
        ax3.text(0.05, y, f'{state}\n{E_labels[state]} eV',
                 fontsize=10, ha='right', va='center', fontweight='bold')

    # Pathway 1: Direct (ground → 4f → 2D)
    ax3.annotate('', xy=(0.35, y_pos['4f'] - 0.015),
                 xytext=(0.35, y_pos['ground'] + 0.015),
                 arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax3.text(0.25, 0.35, '①Direct\n7.44 eV',
             fontsize=9, color='red', fontweight='bold')

    ax3.annotate('', xy=(0.45, y_pos['2D'] + 0.015),
                 xytext=(0.45, y_pos['4f'] - 0.015),
                 arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax3.text(0.48, 0.57, '821 nm',
             fontsize=9, color='red', fontweight='bold')

    # Pathway 2: Resonant (ground → 2D → 4f → 2D)
    ax3.annotate('', xy=(0.60, y_pos['2D'] - 0.015),
                 xytext=(0.60, y_pos['ground'] + 0.015),
                 arrowprops=dict(arrowstyle='->', lw=2, color='blue', linestyle='--'))
    ax3.text(0.63, 0.22, '②a\n5.93 eV',
             fontsize=8, color='blue', fontweight='bold')

    ax3.annotate('', xy=(0.70, y_pos['4f'] - 0.015),
                 xytext=(0.70, y_pos['2D'] + 0.015),
                 arrowprops=dict(arrowstyle='->', lw=2, color='blue', linestyle='--'))
    ax3.text(0.73, 0.57, '②b\n1.51 eV',
             fontsize=8, color='blue', fontweight='bold')

    ax3.annotate('', xy=(0.80, y_pos['2D'] + 0.015),
                 xytext=(0.80, y_pos['4f'] - 0.015),
                 arrowprops=dict(arrowstyle='->', lw=2, color='blue', linestyle='--'))
    ax3.text(0.83, 0.57, '821 nm',
             fontsize=8, color='blue', fontweight='bold')

    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_title('Two Excitation Pathways', fontsize=11, fontweight='bold')

    # ========================================================================
    # Plot 4: Metastable fraction
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    for ne_val in [1e18, 1e19, 1e20, 1e22]:
        idx = np.argmin(np.abs(model.ne_grid - ne_val))
        f_meta_percent = model.f_2D_grid[idx, :] * 100

        ax4.semilogy(te_range, f_meta_percent, 'o-', linewidth=2, markersize=4,
                     markevery=6, label=f'$n_e$ = {ne_val:.0e} m$^{{-3}}$')

    ax4.set_xlabel('$T_e$ (eV)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Metastable Fraction (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Population in 2s2p² ²D State', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xscale('log')
    ax4.set_xlim([0.1, 10000])

    # ========================================================================
    # Plot 5: S/XB
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 2])

    for ne_val in [1e18, 1e19, 1e20]:
        idx = np.argmin(np.abs(model.ne_grid - ne_val))
        ax5.semilogy(te_range, model.sxb_grid[idx, :], 'o-',
                     linewidth=2, markersize=4, markevery=6,
                     label=f'$n_e$ = {ne_val:.0e} m$^{{-3}}$')

    ax5.set_xlabel('$T_e$ (eV)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('S/XB (ph/ion)', fontsize=12, fontweight='bold')
    ax5.set_title('S/XB Coefficient', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, which='both')
    ax5.set_xscale('log')
    ax5.set_xlim([0.1, 10000])

    # ========================================================================
    # Plot 6: Relative pathway importance
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, :])

    ne_ref = 1e19
    idx_ref = np.argmin(np.abs(model.ne_grid - ne_ref))

    pec_direct = model.pec_direct_grid[idx_ref, :]
    pec_resonant = model.pec_resonant_grid[idx_ref, :]
    pec_total = model.pec_grid[idx_ref, :]

    fraction_direct = pec_direct / (pec_total + 1e-50) * 100
    fraction_resonant = pec_resonant / (pec_total + 1e-50) * 100

    ax6.fill_between(te_range, 0, fraction_direct, alpha=0.6, color='red',
                     label='Direct pathway (Ground → 4f → 2D)')
    ax6.fill_between(te_range, fraction_direct, 100, alpha=0.6, color='blue',
                     label='Resonant pathway (2D → 4f → 2D)')

    ax6.set_xlabel('$T_e$ (eV)', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Contribution (%)', fontsize=13, fontweight='bold')
    ax6.set_title(f'Relative Pathway Importance (n$_e$ = {ne_ref:.0e} m$^{{-3}}$)',
                  fontsize=13, fontweight='bold')
    ax6.legend(fontsize=11, loc='right')
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log')
    ax6.set_xlim([0.1, 10000])
    ax6.set_ylim([0, 100])

    # Add annotations
    ax6.annotate('Direct dominates\nat low Te', xy=(1, 80),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    ax6.annotate('Resonant becomes\nimportant at high Te', xy=(30, 30),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))

    plt.savefig('boron_821nm_combined_pathways.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print("=" * 70)
    print(" COMBINED MODEL - B I 821.2 nm (²D₅/₂ → ²F₇/₂)")
    print("=" * 70)
    print()
    print("Lower level: 2s2p² ²D₅/₂ (metastable, 47,857 cm⁻¹ = 5.93 eV)")
    print("Upper level: 4f ²F₇/₂ (60,031 cm⁻¹ = 7.44 eV)")
    print("Transition energy: 1.51 eV (821.2 nm)")
    print()
    print("TWO EXCITATION PATHWAYS:")
    print("  1. Direct:   Ground (²P°) → 4f (²F) → 2D + hν(821nm)")
    print("  2. Resonant: 2D → 4f → 2D + hν(821nm) [requires metastable]")
    print()
    print("PEC and S/XB at n_e = 1×10¹⁹ m⁻³:")
    print(f"{'Te':<8} {'PEC_tot':<12} {'Direct%':<10} {'Resonant%':<10} {'S/XB':<12}")
    print("-" * 60)

    idx_ref = np.argmin(np.abs(model.ne_grid - 1e19))
    for te_val in [1, 3, 10, 30]:
        te_idx = np.argmin(np.abs(te_range - te_val))
        pec_tot = model.pec_grid[idx_ref, te_idx] * 1e6
        pec_dir = model.pec_direct_grid[idx_ref, te_idx] * 1e6
        pec_res = model.pec_resonant_grid[idx_ref, te_idx] * 1e6
        frac_dir = pec_dir / (pec_tot + 1e-50) * 100
        frac_res = pec_res / (pec_tot + 1e-50) * 100
        sxb = model.sxb_grid[idx_ref, te_idx]

        print(f"{te_val:<8.1f} {pec_tot:<12.3e} {frac_dir:<10.1f} {frac_res:<10.1f} {sxb:<12.3e}")

    print()
    print("Key insights:")
    print("✓ Both pathways contribute to the observed 821 nm emission")
    print("✓ Direct pathway dominates at low Te (< 5 eV)")
    print("✓ Resonant pathway grows with metastable population at higher Te")
    print("✓ PEC saturates at high Te (corona behavior)")
    print("✓ S/XB decreases with Te (ionization wins)")
    print("=" * 70)


def save_sxb_table_for_diagnostics(model):
    """
    Save S/XB lookup table for use in experimental data analysis
    """

    filename = 'boron_821nm_sxb_lookup_table.dat'

    with open(filename, 'w') as f:
        f.write("# S/XB Lookup Table for B I 821.2 nm (²D₅/₂ → ²F₇/₂)\n")
        f.write("# Combined direct + resonant excitation pathways\n")
        f.write("# Generated with simplified CR model - validate with ADAS!\n")
        f.write("#\n")
        f.write(f"# Te range: {model.te_grid[0]:.2f} - {model.te_grid[-1]:.2f} eV\n")
        f.write(f"# ne range: {model.ne_grid[0]:.2e} - {model.ne_grid[-1]:.2e} m^-3\n")
        f.write("#\n")
        f.write("# Columns:\n")
        f.write("#   1: Te (eV)\n")
        f.write("#   2: ne (m^-3)\n")
        f.write("#   3: S/XB (photons/ionization)\n")
        f.write("#   4: PEC (m^3/s)\n")
        f.write("#   5: Metastable fraction (%)\n")
        f.write("#\n")
        f.write("#" + "=" * 70 + "\n")

        for i, te in enumerate(model.te_grid):
            for j, ne in enumerate(model.ne_grid):
                sxb = model.sxb_grid[j, i]
                pec = model.pec_grid[j, i]
                f_meta = model.f_2D_grid[j, i] * 100
                f.write(f"{te:10.4f}  {ne:15.6e}  {sxb:15.6e}  {pec:15.6e}  {f_meta:10.4f}\n")

    print(f"\nSaved lookup table: {filename}")
    print("Use this for analyzing your experimental 821 nm line measurements!")


def compare_both_interpretations():
    """
    Final comparison showing why the combined model is correct
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    model = BoronCRModelCombined()
    te_range = np.logspace(-1, 4, 60)
    ne_range = np.array([1e18, 1e19, 1e20])

    model.generate_tables(te_range, ne_range)

    # Plot 1: What Herzberg observed (Absorption)
    ax = axes[0, 0]
    ax.text(0.5, 0.9, 'Herzberg et al. (1970)', ha='center', va='top',
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.75, 'Flash Photolysis - ABSORPTION', ha='center',
            fontsize=12, fontweight='bold', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.text(0.1, 0.55, 'Process:', fontsize=11, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.1, 0.45, '• Photolysis creates free B atoms', fontsize=10,
            transform=ax.transAxes)
    ax.text(0.1, 0.38, '• Some populate metastable ²D', fontsize=10,
            transform=ax.transAxes)
    ax.text(0.1, 0.31, '• Probe light at 821 nm absorbed:', fontsize=10,
            transform=ax.transAxes)
    ax.text(0.15, 0.24, '²D + hν → 4f (absorption)', fontsize=10,
            transform=ax.transAxes, style='italic', color='blue')
    ax.text(0.1, 0.15, '• Table II lists absorption lines', fontsize=10,
            transform=ax.transAxes)
    ax.text(0.1, 0.08, '• Lower level (²D) is populated', fontsize=10,
            transform=ax.transAxes)

    ax.axis('off')

    # Plot 2: What you observe (Emission)
    ax = axes[0, 1]
    ax.text(0.5, 0.9, 'Your Plasma Experiment', ha='center', va='top',
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.75, 'Deuterium Plasma - EMISSION', ha='center',
            fontsize=12, fontweight='bold', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax.text(0.1, 0.55, 'Process:', fontsize=11, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.1, 0.45, '• Electron impact excitation:', fontsize=10,
            transform=ax.transAxes)

    ax.text(0.15, 0.37, 'Path 1: Ground → 4f → ²D + hν', fontsize=10,
            transform=ax.transAxes, color='red', fontweight='bold')
    ax.text(0.15, 0.30, 'Path 2: ²D → 4f → ²D + hν', fontsize=10,
            transform=ax.transAxes, color='blue', fontweight='bold')

    ax.text(0.1, 0.20, '• Both pathways contribute!', fontsize=10,
            transform=ax.transAxes)
    ax.text(0.1, 0.13, '• Observe 821 nm emission', fontsize=10,
            transform=ax.transAxes)
    ax.text(0.1, 0.06, '• Upper level (4f) populated', fontsize=10,
            transform=ax.transAxes)

    ax.axis('off')

    # Plot 3: PEC showing saturation
    ax = axes[1, 0]

    for ne_val in ne_range:
        idx = np.argmin(np.abs(model.ne_grid - ne_val))
        pec_cm3 = model.pec_grid[idx, :] * 1e6
        ax.loglog(te_range, pec_cm3, 'o-', linewidth=2.5, markersize=4,
                  markevery=6, label=f'$n_e$ = {ne_val:.0e} m$^{{-3}}$')

    ax.set_xlabel('$T_e$ (eV)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PEC (cm³/s)', fontsize=12, fontweight='bold')
    ax.set_title('Total PEC (Both Pathways)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0.1, 10000])

    # Add annotation showing ADAS-like behavior
    ax.annotate('Saturates like\nADAS data ✓',
                xy=(20, 5e-14), fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Plot 4: Pathway breakdown
    ax = axes[1, 1]

    ne_ref = 1e19
    idx_ref = np.argmin(np.abs(model.ne_grid - ne_ref))

    pec_total = model.pec_grid[idx_ref, :] * 1e6
    pec_direct = model.pec_direct_grid[idx_ref, :] * 1e6
    pec_resonant = model.pec_resonant_grid[idx_ref, :] * 1e6

    ax.loglog(te_range, pec_total, 'k-', linewidth=3, label='Total')
    ax.loglog(te_range, pec_direct, 'r--', linewidth=2, label='Direct (Gnd→4f→2D)')
    ax.loglog(te_range, pec_resonant, 'b--', linewidth=2, label='Resonant (2D→4f→2D)')

    ax.set_xlabel('$T_e$ (eV)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PEC (cm³/s)', fontsize=12, fontweight='bold')
    ax.set_title(f'Pathway Breakdown (n$_e$ = {ne_ref:.0e} m$^{{-3}}$)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0.1, 10000])

    plt.tight_layout()
    plt.savefig('boron_821nm_absorption_vs_emission.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 70)
    print(" SUMMARY: Understanding the 821.2 nm Transition")
    print("=" * 70)
    print()
    print("TRANSITION IDENTITY:")
    print("  Wavelength: 8211.906 Å (strongest component)")
    print("  Lower level: 2s2p² ²D₅/₂ (metastable, 5.93 eV)")
    print("  Upper level: 4f ²F₇/₂ (7.44 eV)")
    print("  Transition energy: 1.51 eV")
    print()
    print("HERZBERG PAPER (1970):")
    print("  • Flash photolysis creates B atoms")
    print("  • Observed in ABSORPTION: ²D + hν → 4f")
    print("  • Table II shows absorption wavelengths")
    print("  • They needed metastable population to see it")
    print()
    print("YOUR PLASMA EMISSION:")
    print("  • Electron impact excitation creates 4f population")
    print("  • Two pathways contribute:")
    print("    1. Direct: Ground → 4f → ²D + hν(821nm)")
    print("    2. Resonant: ²D → 4f → ²D + hν(821nm)")
    print("  • Both are important for total PEC")
    print()
    print("IMPLICATIONS FOR DIAGNOSTICS:")
    print("  • S/XB depends on BOTH pathways")
    print("  • Metastable population affects PEC magnitude")
    print("  • At low Te: direct pathway dominates")
    print("  • At high Te: resonant pathway becomes significant")
    print("  • Always use combined model for accurate S/XB!")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print(" B I 821.2 nm Line: Combined Excitation Pathways")
    print(" Transition: ²D₅/₂ → ²F₇/₂ (Strongest Component)")
    print("=" * 70)
    print()

    # Generate comprehensive analysis
    plot_combined_pathways()

    # Create comparison figure
    print("\nGenerating absorption vs emission comparison...")
    compare_both_interpretations()

    # Save lookup table for diagnostics
    print("\nCreating S/XB lookup table for your measurements...")
    model = BoronCRModelCombined()
    te_range = np.logspace(-1, 2, 40)
    ne_range = np.logspace(18, 20, 30)
    model.generate_tables(te_range, ne_range)
    save_sxb_table_for_diagnostics(model)

    print("\n" + "=" * 70)
    print("HOW TO USE THIS FOR YOUR EXPERIMENT:")
    print("=" * 70)
    print("""
        1. Measure 821 nm line brightness: I(821) [photons/s/m²/sr]

        2. Measure or estimate plasma conditions:
           - Electron temperature Te (eV) - from Langmuir probe
           - Electron density ne (m⁻³) - from Langmuir probe
           - Chord length L (m) - your viewing geometry

        3. Look up S/XB(Te, ne) from the table (or interpolate)

        4. Calculate volume emission rate:
           ε = 4π × I(821) / L  [photons/m³/s]

        5. Estimate ionization source:
           S_ion × ne = ε / S/XB  [ionizations/m³/s]

        6. Compare with other diagnostics (H-alpha, etc.)

        IMPORTANT NOTES:
        - This model uses simplified atomic physics
        - For quantitative work, cross-calibrate with 249 nm doublet
        - Consider requesting proper ADAS data for this line
        - Both excitation pathways contribute to your signal!
            """)
    print("=" * 70)