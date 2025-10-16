import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from dataclasses import dataclass
from boron_sxb_calculator_v0 import BoronCRModel


@dataclass
class AtomicData:
    """Container for atomic data"""
    E_ion: float = 8.298  # B I ionization energy (eV)
    E_meta: float = 5.93  # Ground to 2s2p² ²D metastable (eV)
    E_4f: float = 1.51  # Metastable to 4f ²F (eV)
    A_spontaneous_4f: float = 8.2e6  # Einstein A for 4f->2D (s⁻¹) - from literature
    A_spontaneous_meta: float = 1e3  # Forbidden transition, very slow (s⁻¹)
    wavelength: float = 821.2  # nm
    g_ground: float = 2.0  # 2P_1/2 statistical weight
    g_meta: float = 6.0  # 2D statistical weight (2J+1 = 2*2.5+1)
    g_4f: float = 8.0  # 2F statistical weight


class BoronCRModelImproved:
    """
    IMPROVED Collisional-Radiative model for B I 821.2 nm line
    with better physics
    """

    def __init__(self):
        self.atomic = AtomicData()
        self.te_grid = None
        self.ne_grid = None
        self.sxb_grid = None

    def excitation_rate_coefficient(self, E_threshold, Te, g_lower, g_upper,
                                    oscillator_strength=0.1):
        """
        Improved excitation rate using van Regemorter formula

        For allowed transitions: <σv> ∝ (πa₀²/E_threshold) * (Ry/Te) * f * g(E/Te)
        where g(E/Te) is the Gaunt factor
        """
        Te = np.maximum(Te, 0.1)

        # Constants
        a0 = 5.29e-11  # Bohr radius (m)
        Ry = 13.6  # Rydberg energy (eV)

        # Gaunt factor approximation: g ≈ 0.2 for E/Te >> 1, approaches const for E/Te << 1
        u = E_threshold / Te
        gaunt_factor = np.where(u > 1,
                                0.2 / u * np.log(1.0 + u),
                                0.1 + 0.2 * u)

        # van Regemorter formula
        rate = (np.pi * a0 ** 2) * np.sqrt(8 * 1.602e-19 / (np.pi * 9.109e-31 * Te)) * \
               (Ry / E_threshold) * oscillator_strength * gaunt_factor * \
               np.exp(-E_threshold / Te) * (g_upper / g_lower)

        return rate

    def ionization_rate_improved(self, Te):
        """
        Improved ionization rate using proper scaling
        Includes contributions from excited states
        """
        E_ion = self.atomic.E_ion
        Te = np.maximum(Te, 0.1)

        # Ground state ionization (Lotz formula)
        u = E_ion / Te

        # Lotz: S = 4.5e-14 * (1/E_ion) * ln(u) / u * exp(-u)  [cm³/s]
        # But this is for u >> 1

        # Better approximation combining regimes:
        if np.isscalar(Te):
            if u > 1:
                S_ground = 4.5e-20 * (1 / E_ion) * np.log(u) / u * np.exp(-u)
            else:
                S_ground = 4.5e-20 * (1 / E_ion) * (1 - u / 2) * np.exp(-u)
        else:
            S_ground = np.where(u > 1,
                                4.5e-20 * (1 / E_ion) * np.log(u) / u * np.exp(-u),
                                4.5e-20 * (1 / E_ion) * (1 - u / 2) * np.exp(-u))

        # Add contribution from metastable state ionization
        # (lower threshold from excited state)
        E_ion_from_meta = E_ion - self.atomic.E_meta  # ~2.37 eV
        u_meta = E_ion_from_meta / Te

        if np.isscalar(Te):
            if u_meta > 1:
                S_meta = 4.5e-20 * (1 / E_ion_from_meta) * np.log(u_meta) / u_meta * np.exp(-u_meta)
            else:
                S_meta = 4.5e-20 * (1 / E_ion_from_meta) * (1 - u_meta / 2) * np.exp(-u_meta)
        else:
            S_meta = np.where(u_meta > 1,
                              4.5e-20 * (1 / E_ion_from_meta) * np.log(u_meta) / u_meta * np.exp(-u_meta),
                              4.5e-20 * (1 / E_ion_from_meta) * (1 - u_meta / 2) * np.exp(-u_meta))

        # Weight metastable contribution by its population
        # (will be calculated later, but assume ~10% max)
        f_meta_typical = 0.05

        S_total = S_ground + f_meta_typical * S_meta

        return S_total

    def metastable_population_improved(self, Te, ne):
        """
        Improved metastable population calculation
        Uses proper rate equation balance
        """
        # Excitation rate ground -> metastable
        S_exc = self.excitation_rate_coefficient(
            self.atomic.E_meta, Te,
            self.atomic.g_ground, self.atomic.g_meta,
            oscillator_strength=0.05  # Estimated for this transition
        )

        # De-excitation rate (detailed balance)
        S_deexc = S_exc * (self.atomic.g_ground / self.atomic.g_meta) * \
                  np.exp(self.atomic.E_meta / Te)

        # Radiative decay (metastable, so very slow)
        A_rad = self.atomic.A_spontaneous_meta

        # Ionization from metastable
        E_ion_from_meta = self.atomic.E_ion - self.atomic.E_meta
        u = E_ion_from_meta / Te
        S_ion_meta = 4.5e-20 * (1 / E_ion_from_meta) * np.log(np.maximum(u, 1.0)) / \
                     np.maximum(u, 1.0) * np.exp(-u)

        # Population balance: n_meta/n_ground = S_exc / (S_deexc + A_rad/ne + S_ion_meta)
        denominator = S_deexc + A_rad / ne + S_ion_meta
        f_meta = ne * S_exc / (ne * denominator + 1e-30)  # Add small number to avoid div/0

        # Physical limits
        f_meta = np.clip(f_meta, 0, 0.2)  # Max 20% in metastable

        return f_meta

    def photon_emissivity_coefficient_improved(self, Te, ne):
        """
        Improved PEC calculation
        """
        # Metastable fraction
        f_meta = self.metastable_population_improved(Te, ne)

        # Excitation from metastable to 4f
        S_exc_4f = self.excitation_rate_coefficient(
            self.atomic.E_4f, Te,
            self.atomic.g_meta, self.atomic.g_4f,
            oscillator_strength=0.08  # Two-electron transition, weaker
        )

        # Cascade and branching
        # The 4f state can decay to multiple lower states
        # For simplicity, assume 30% branches to 2D (our lower level)
        branching_ratio = 0.3

        # Collisional quenching from 4f
        # Uses approximate collisional de-excitation
        S_quench = 1e-13 * np.sqrt(Te / self.atomic.E_4f)  # Rough estimate
        A_total = self.atomic.A_spontaneous_4f + ne * S_quench

        # Effective emission rate
        eff_emission = self.atomic.A_spontaneous_4f * branching_ratio / A_total

        # Total PEC (per unit neutral density)
        PEC = f_meta * S_exc_4f * eff_emission

        return PEC

    def calculate_sxb_improved(self, Te, ne):
        """
        Calculate S/XB with improved physics
        """
        PEC = self.photon_emissivity_coefficient_improved(Te, ne)
        S_ion = self.ionization_rate_improved(Te)

        S_ion = np.maximum(S_ion, 1e-25)  # Avoid division by zero

        sxb = PEC / S_ion

        return sxb

    def generate_sxb_table(self, te_range, ne_range):
        """Generate lookup table"""
        Te_2d, Ne_2d = np.meshgrid(te_range, ne_range)

        sxb_grid = self.calculate_sxb_improved(Te_2d, Ne_2d)

        self.te_grid = te_range
        self.ne_grid = ne_range
        self.sxb_grid = sxb_grid

        return te_range, ne_range, sxb_grid

    def interpolate_sxb(self, Te_query, ne_query):
        """Interpolate S/XB"""
        if self.sxb_grid is None:
            raise ValueError("Must call generate_sxb_table first")

        Te_scalar = np.isscalar(Te_query)
        ne_scalar = np.isscalar(ne_query)

        Te_query = np.atleast_1d(Te_query)
        ne_query = np.atleast_1d(ne_query)

        interp = RectBivariateSpline(
            np.log10(self.ne_grid),
            np.log10(self.te_grid),
            self.sxb_grid,
            kx=3, ky=3
        )

        result = interp(np.log10(ne_query), np.log10(Te_query), grid=False)

        if Te_scalar and ne_scalar and result.size == 1:
            return float(result)

        return result


def compare_models():
    """
    Compare old and new models side by side
    """
    # Import old model
    from dataclasses import dataclass as dc_old

    # Create both models
    model_old = BoronCRModel()  # Your original
    model_new = BoronCRModelImproved()

    # Same grid
    te_range = np.logspace(np.log10(2), np.log10(50), 40)
    ne_range = np.logspace(18, 19.5, 35)

    print("Calculating old model...")
    model_old.generate_sxb_table(te_range, ne_range)

    print("Calculating improved model...")
    model_new.generate_sxb_table(te_range, ne_range)

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ne_samples = [1e18, 5e18, 1e19]
    colors = ['blue', 'green', 'red']

    # S/XB vs Te comparison
    for i, ne_val in enumerate(ne_samples):
        ax = axes[0, i]

        idx = np.argmin(np.abs(model_old.ne_grid - ne_val))
        ax.semilogy(model_old.te_grid, model_old.sxb_grid[idx, :], 'o--',
                    color=colors[i], label='Original', alpha=0.7)

        idx = np.argmin(np.abs(model_new.ne_grid - ne_val))
        ax.semilogy(model_new.te_grid, model_new.sxb_grid[idx, :], 's-',
                    color=colors[i], label='Improved', linewidth=2)

        ax.set_xlabel('Te (eV)', fontweight='bold')
        ax.set_ylabel('S/XB', fontweight='bold')
        ax.set_title(f'n$_e$ = {ne_val:.0e} m$^{{-3}}$', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    # Metastable fraction comparison
    Te_2d, Ne_2d = np.meshgrid(te_range, ne_range)

    for i, ne_val in enumerate(ne_samples):
        ax = axes[1, i]

        idx = np.argmin(np.abs(ne_range - ne_val))

        # Old model
        f_old = model_old.metastable_fraction(te_range, ne_val)
        ax.semilogy(te_range, f_old * 100, 'o--', color=colors[i],
                    label='Original', alpha=0.7)

        # New model
        f_new = model_new.metastable_population_improved(te_range, ne_val)
        ax.semilogy(te_range, f_new * 100, 's-', color=colors[i],
                    label='Improved', linewidth=2)

        ax.set_xlabel('Te (eV)', fontweight='bold')
        ax.set_ylabel('Metastable Fraction (%)', fontweight='bold')
        ax.set_title(f'n$_e$ = {ne_val:.0e} m$^{{-3}}$', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nTypical S/XB values:")
    print("=" * 70)
    print(f"{'Model':<12} {'Te=5eV':<12} {'Te=10eV':<12} {'Te=20eV':<12}")
    print("-" * 70)

    for te in [5, 10, 20]:
        sxb_old = model_old.interpolate_sxb(te, 1e19)
        sxb_new = model_new.interpolate_sxb(te, 1e19)
        print(f"{'Original':<12} {sxb_old:.3e}    ", end="")
        if te == 10:
            print(f"{sxb_old:.3e}    {sxb_old:.3e}")

    print()
    for te in [5, 10, 20]:
        sxb_new = model_new.interpolate_sxb(te, 1e19)
        print(f"{'Improved':<12} {sxb_new:.3e}    ", end="")
        if te == 10:
            print(f"{sxb_new:.3e}    {sxb_new:.3e}")


if __name__ == "__main__":
    print("=" * 70)
    print(" Model Comparison: Original vs Improved Physics")
    print("=" * 70)
    print()

    # Need to paste the original BoronCRModel class here for comparison
    # Or import it

    compare_models()