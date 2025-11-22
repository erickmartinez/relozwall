"""
Boron B I 821.2 nm S/XB Calculator - FIXED VERSION
Standalone implementation for 2s2p² ²D → 4f ²F transition
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from dataclasses import dataclass

@dataclass
class AtomicData:
    """Container for atomic data"""
    E_ion: float = 8.298  # B I ionization energy (eV)
    E_meta: float = 5.93  # Ground to 2s2p² ²D metastable (eV)
    E_4f: float = 1.51    # Metastable to 4f ²F (eV) - from paper: 60031-47857 cm⁻¹
    A_spontaneous: float = 1e7  # Einstein A coefficient (s⁻¹) - estimate
    wavelength: float = 821.2  # nm

class BoronCRModel:
    """
    Simplified Collisional-Radiative model for B I 821.2 nm line
    """

    def __init__(self):
        self.atomic = AtomicData()
        self.te_grid = None
        self.ne_grid = None
        self.sxb_grid = None

    def maxwell_rate_coefficient(self, E_threshold, Te, sigma_0=1e-20):
        """
        Calculate rate coefficient using simplified Maxwell-averaged cross section

        Parameters:
        -----------
        E_threshold : float
            Threshold energy (eV)
        Te : array
            Electron temperature (eV)
        sigma_0 : float
            Cross section scale (m²)

        Returns:
        --------
        rate : array
            Rate coefficient (m³/s)
        """
        # Avoid division by zero
        Te = np.maximum(Te, 0.1)

        # Thermal velocity
        v_th = np.sqrt(2 * 1.602e-19 * Te / 9.109e-31)  # m/s

        # Rate coefficient with Arrhenius form
        rate = sigma_0 * v_th * np.exp(-E_threshold / Te)

        return rate

    def ionization_rate(self, Te):
        """
        Ionization rate coefficient for B I → B II
        Uses fit similar to ADAS but simplified
        """
        E_ion = self.atomic.E_ion
        Te = np.maximum(Te, 0.1)

        # Simplified ionization rate
        A = 3e-14  # Fitting parameter (m³/s)
        S_ion = A * (Te / E_ion) * np.exp(-E_ion / Te)

        return S_ion

    def excitation_rate_ground_to_meta(self, Te, ne):
        """
        Excitation from ground state to metastable 2s2p² ²D
        """
        sigma_0 = 5e-21  # m²
        rate = self.maxwell_rate_coefficient(self.atomic.E_meta, Te, sigma_0)
        return rate

    def excitation_rate_meta_to_4f(self, Te):
        """
        Excitation from metastable 2s2p² ²D to 4f ²F
        """
        sigma_0 = 3e-21  # m²
        rate = self.maxwell_rate_coefficient(self.atomic.E_4f, Te, sigma_0)
        return rate

    def metastable_fraction(self, Te, ne):
        """
        Estimate fraction of atoms in metastable state
        """
        S_exc = self.excitation_rate_ground_to_meta(Te, ne)
        A_rad = 1e6  # s⁻¹

        # Collisional de-excitation
        g_ratio = 2.0 / 6.0
        S_deexc = S_exc * g_ratio * np.exp(self.atomic.E_meta / Te)

        # Metastable fraction
        f_meta = S_exc / (A_rad / ne + S_deexc)
        f_meta = np.clip(f_meta, 0, 0.1)

        return f_meta

    def photon_emissivity_coefficient(self, Te, ne):
        """
        Calculate PEC for 821.2 nm line (photons⋅m³/s)
        """
        f_meta = self.metastable_fraction(Te, ne)
        S_exc_4f = self.excitation_rate_meta_to_4f(Te)

        # Collisional quenching from 4f state
        A_coll_quench = ne * 1e-21 * np.sqrt(2 * 1.602e-19 * Te / 9.109e-31)

        # Effective branching including collisional losses
        eff_branching = self.atomic.A_spontaneous / (self.atomic.A_spontaneous + A_coll_quench)

        # Total PEC
        PEC = f_meta * S_exc_4f * eff_branching

        return PEC

    def calculate_sxb(self, Te, ne):
        """
        Calculate S/XB coefficient

        S/XB = PEC / S_ionization
        """
        PEC = self.photon_emissivity_coefficient(Te, ne)
        S_ion = self.ionization_rate(Te)
        S_ion = np.maximum(S_ion, 1e-20)

        sxb = PEC / S_ion

        return sxb

    def generate_sxb_table(self, te_range, ne_range):
        """
        Generate S/XB lookup table over Te, ne grid
        """
        Te_2d, Ne_2d = np.meshgrid(te_range, ne_range)

        sxb_grid = self.calculate_sxb(Te_2d, Ne_2d)

        self.te_grid = te_range
        self.ne_grid = ne_range
        self.sxb_grid = sxb_grid

        return te_range, ne_range, sxb_grid

    def interpolate_sxb(self, Te_query, ne_query):
        """
        Interpolate S/XB at arbitrary Te, ne points

        Parameters:
        -----------
        Te_query : float or array
            Electron temperature (eV)
        ne_query : float or array
            Electron density (m⁻³)

        Returns:
        --------
        sxb : float or array
            Interpolated S/XB value(s)
        """
        if self.sxb_grid is None:
            raise ValueError("Must call generate_sxb_table first")

        # Store original input type
        Te_scalar = np.isscalar(Te_query)
        ne_scalar = np.isscalar(ne_query)

        # Convert to arrays
        Te_query = np.atleast_1d(Te_query)
        ne_query = np.atleast_1d(ne_query)

        # Create interpolator
        interp = RectBivariateSpline(
            np.log10(self.ne_grid),
            np.log10(self.te_grid),
            self.sxb_grid,
            kx=3, ky=3
        )

        # Interpolate
        result = interp(np.log10(ne_query), np.log10(Te_query), grid=False)

        # Return scalar if inputs were scalar
        if Te_scalar and ne_scalar and result.size == 1:
            return float(result)

        return result


def plot_sxb_results(model, save_fig=True):
    """
    Create comprehensive plots of S/XB behavior
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    Te_grid, Ne_grid = np.meshgrid(model.te_grid, model.ne_grid)

    # 1. Main contour plot
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    levels = np.logspace(np.log10(np.nanmin(model.sxb_grid) + 1e-10),
                         np.log10(np.nanmax(model.sxb_grid)), 30)

    cs = ax1.contourf(Te_grid, Ne_grid / 1e18, model.sxb_grid,
                      levels=levels, cmap='plasma',
                      norm=plt.matplotlib.colors.LogNorm())
    ax1.contour(Te_grid, Ne_grid / 1e18, model.sxb_grid,
                levels=levels, colors='white', alpha=0.2, linewidths=0.5)

    ax1.set_xlabel('Electron Temperature (eV)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Electron Density (10$^{18}$ m$^{-3}$)', fontsize=13, fontweight='bold')
    ax1.set_title('S/XB for B I 821.2 nm (2s2p² ²D → 4f ²F)',
                  fontsize=14, fontweight='bold', pad=10)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')

    cbar = plt.colorbar(cs, ax=ax1)
    cbar.set_label('S/XB (photons/ionization)', fontsize=12, fontweight='bold')

    # 2. S/XB vs Te at different densities
    ax2 = fig.add_subplot(gs[0, 2])

    ne_samples = [1e18, 3e18, 1e19, 3e19]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ne_samples)))

    for ne_val, color in zip(ne_samples, colors):
        idx = np.argmin(np.abs(model.ne_grid - ne_val))
        ax2.semilogy(model.te_grid, model.sxb_grid[idx, :], 'o-',
                     color=color, linewidth=2, markersize=4,
                     label=f'{ne_val:.1e} m$^{{-3}}$')

    ax2.set_xlabel('Te (eV)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('S/XB', fontsize=11, fontweight='bold')
    ax2.set_title('Fixed Density', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=8, title='$n_e$', title_fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. S/XB vs ne at different temperatures
    ax3 = fig.add_subplot(gs[1, 2])

    te_samples = [5, 10, 20, 30]
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(te_samples)))

    for te_val, color in zip(te_samples, colors):
        idx = np.argmin(np.abs(model.te_grid - te_val))
        ax3.loglog(model.ne_grid / 1e18, model.sxb_grid[:, idx], 'o-',
                   color=color, linewidth=2, markersize=4,
                   label=f'{te_val} eV')

    ax3.set_xlabel('$n_e$ (10$^{18}$ m$^{-3}$)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('S/XB', fontsize=11, fontweight='bold')
    ax3.set_title('Fixed Temperature', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, title='$T_e$', title_fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Metastable fraction
    ax4 = fig.add_subplot(gs[2, 0])

    f_meta = model.metastable_fraction(Te_grid, Ne_grid)
    cs4 = ax4.contourf(Te_grid, Ne_grid / 1e18, f_meta * 100,
                       levels=20, cmap='coolwarm')
    ax4.set_xlabel('Te (eV)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('$n_e$ (10$^{18}$ m$^{-3}$)', fontsize=11, fontweight='bold')
    ax4.set_title('Metastable Fraction (%)', fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    plt.colorbar(cs4, ax=ax4, label='%')

    # 5. Ionization rate
    ax5 = fig.add_subplot(gs[2, 1])

    S_ion = model.ionization_rate(Te_grid)
    cs5 = ax5.contourf(Te_grid, Ne_grid / 1e18, S_ion * 1e14,
                       levels=20, cmap='viridis')
    ax5.set_xlabel('Te (eV)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('$n_e$ (10$^{18}$ m$^{-3}$)', fontsize=11, fontweight='bold')
    ax5.set_title('Ionization Rate (10$^{-14}$ m$^3$/s)', fontsize=12, fontweight='bold')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    plt.colorbar(cs5, ax=ax5)

    # 6. PEC
    ax6 = fig.add_subplot(gs[2, 2])

    PEC = model.photon_emissivity_coefficient(Te_grid, Ne_grid)
    cs6 = ax6.contourf(Te_grid, Ne_grid / 1e18, PEC * 1e14,
                       levels=20, cmap='magma')
    ax6.set_xlabel('Te (eV)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('$n_e$ (10$^{18}$ m$^{-3}$)', fontsize=11, fontweight='bold')
    ax6.set_title('PEC (10$^{-14}$ ph⋅m$^3$/s)', fontsize=12, fontweight='bold')
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    plt.colorbar(cs6, ax=ax6)

    if save_fig:
        plt.savefig('boron_821nm_sxb_comprehensive.png', dpi=300, bbox_inches='tight')
        print("Saved: boron_821nm_sxb_comprehensive.png")

    plt.show()


def create_sxb_table_file(model, filename='boron_821nm_sxb_table.txt'):
    """
    Save S/XB table in format suitable for experimental analysis
    """
    with open(filename, 'w') as f:
        f.write("# S/XB Table for B I 821.2 nm (2s2p² ²D → 4f ²F)\n")
        f.write("# Generated by standalone CR model\n")
        f.write("# WARNING: Uses simplified atomic physics - validate with ADAS\n")
        f.write("#\n")
        f.write(f"# Te range: {model.te_grid[0]:.2f} - {model.te_grid[-1]:.2f} eV\n")
        f.write(f"# ne range: {model.ne_grid[0]:.2e} - {model.ne_grid[-1]:.2e} m^-3\n")
        f.write("#\n")
        f.write("# Format: Te(eV)  ne(m^-3)  S/XB(ph/ion)\n")
        f.write("#" + "="*60 + "\n")

        for i, te in enumerate(model.te_grid):
            for j, ne in enumerate(model.ne_grid):
                f.write(f"{te:8.3f}  {ne:12.5e}  {model.sxb_grid[j,i]:12.5e}\n")

    print(f"Saved: {filename}")


def example_analysis(model, Te_plasma=10.0, ne_plasma=5e18):
    """
    Example: analyze measured line intensity
    """
    print("\n" + "="*70)
    print("EXAMPLE: Estimating ionization from line intensity measurement")
    print("="*70)

    # Get S/XB at plasma conditions
    sxb = model.interpolate_sxb(Te_plasma, ne_plasma)

    print(f"\nPlasma conditions:")
    print(f"  Te = {Te_plasma} eV")
    print(f"  ne = {ne_plasma:.2e} m^-3")
    print(f"\nS/XB = {sxb:.3e} photons/ionization")

    # Example measurement
    measured_brightness = 1e15  # photons/s/m²/sr
    chord_length = 0.05  # m

    # Volume emission rate
    volume_emission = 4 * np.pi * measured_brightness / chord_length
    print(f"\nMeasured brightness: {measured_brightness:.2e} ph/s/m²/sr")
    print(f"Chord length: {chord_length} m")
    print(f"Volume emission: {volume_emission:.2e} ph/m³/s")

    # Ionization rate
    S_ion_ne = volume_emission / sxb
    S_ion_local = S_ion_ne / ne_plasma

    print(f"\nIonization source:")
    print(f"  S_ion × ne = {S_ion_ne:.2e} ionizations/m³/s")
    print(f"  S_ion = {S_ion_local:.2e} m³/s")

    # Particle balance estimate
    ion_flux = S_ion_ne * chord_length
    print(f"  Integrated ion flux: {ion_flux:.2e} ions/m²/s")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("="*70)
    print(" Boron B I 821.2 nm S/XB Calculator")
    print(" Standalone CR Model (No external dependencies)")
    print("="*70)
    print()
    print("Transition: 2s2p² ²D → 4f ²F")
    print("Reference: Edlén et al., JOSA 60, 889 (1970)")
    print()

    # Initialize model
    model = BoronCRModel()

    # Define grid
    te_range = np.logspace(np.log10(2), np.log10(50), 40)
    ne_range = np.logspace(18, 19.5, 35)

    print(f"Computing S/XB over grid:")
    print(f"  Te: {te_range[0]:.1f} - {te_range[-1]:.1f} eV ({len(te_range)} points)")
    print(f"  ne: {ne_range[0]:.2e} - {ne_range[-1]:.2e} m^-3 ({len(ne_range)} points)")
    print()

    # Generate S/XB table
    te_grid, ne_grid, sxb_grid = model.generate_sxb_table(te_range, ne_range)

    print("S/XB calculation complete!")
    print()
    print("Sample values:")
    print("-"*70)
    print(f"{'Te (eV)':>8} {'ne (m^-3)':>12} {'S/XB':>12} {'f_meta (%)':>12}")
    print("-"*70)

    for te_val in [5, 10, 20]:
        for ne_val in [1e18, 5e18, 1e19]:
            sxb = model.interpolate_sxb(te_val, ne_val)  # FIXED: No [0]
            f_meta = model.metastable_fraction(te_val, ne_val) * 100
            print(f"{te_val:8.1f} {ne_val:12.2e} {sxb:12.3e} {f_meta:12.2f}")

    print()

    # Create plots
    print("Generating comprehensive plots...")
    plot_sxb_results(model, save_fig=True)

    # Save table
    create_sxb_table_file(model)

    # Example analysis
    example_analysis(model, Te_plasma=10.0, ne_plasma=5e18)

    print("\n" + "="*70)
    print("IMPORTANT CAVEATS:")
    print("="*70)
    print("""
1. This model uses SIMPLIFIED atomic physics:
   - Approximate cross sections
   - Simplified metastable kinetics
   - Estimated Einstein A coefficient
   
2. For QUANTITATIVE work, you MUST:
   - Obtain proper ADAS atomic data
   - Use detailed CR model with all relevant states
   - Cross-calibrate with 249 nm doublet
   
3. Uncertainties in this model: ~factor of 2-5

4. To improve accuracy:
   - Measure I(821nm)/I(249nm) ratio in your plasma
   - Use well-known S/XB for 249 nm to calibrate
   - Contact ADAS team for metastable-resolved data
    """)

    print("\nADAS Contact: https://open.adas.ac.uk/contact")
    print("="*70)