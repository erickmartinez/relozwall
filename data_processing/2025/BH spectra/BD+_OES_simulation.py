import numpy as np
import matplotlib.pyplot as plt
from data_processing.misc_utils.plot_style import load_plot_style


# Molecular constants for BD+ (scaled from BH+ using isotope relations)
# From Almy & Horsfall (1937) BH+ constants, scaled for BD+

# BH+ constants from the paper:
# B₀'' = 12.374 cm⁻¹, B₀' = 11.565 cm⁻¹
# D₀'' = -0.00125 cm⁻¹, D₀' = -0.00124 cm⁻¹
# ν₀₀ = 26376.2 cm⁻¹, λ = 1.21

# For BD+, we scale by the reduced mass ratio σ² = μ(BH)/μ(BD)
sigma_squared = 0.5424  # From Thunberg's isotope analysis

# ²Σ state (lower state) - scaled from BH+
B0_sigma_plus = 12.374 * sigma_squared  # ≈ 6.71 cm⁻¹
B1_sigma_plus = B0_sigma_plus - 0.2  # estimated α
D0_sigma_plus = 0.00125 * sigma_squared ** 2  # ≈ 0.000368
D1_sigma_plus = D0_sigma_plus * 1.1

# ²Π state (upper state) - scaled from BH+
B0_pi_plus = 11.565 * sigma_squared  # ≈ 6.27 cm⁻¹
B1_pi_plus = B0_pi_plus - 0.25  # estimated α
D0_pi_plus = 0.00124 * sigma_squared ** 2  # ≈ 0.000365
D1_pi_plus = D0_pi_plus * 1.15

# Electronic coupling constant (scaled)
lambda_bd_plus = 1.21 * sigma_squared  # A/B ratio should scale

# Band origins (cm⁻¹) - scaled from BH+
# BH+: ν₀₀ = 26376.2 cm⁻¹ → BD+: similar (electronic transition)
nu_00_plus = 26376.2  # Electronic transition energy doesn't change much
nu_11_plus = nu_00_plus - 180  # Vibrational spacing difference
nu_01_plus = nu_00_plus + 600  # Hot band
nu_10_plus = nu_00_plus - 600  # Sequence band

print("BD+ Molecular Constants (estimated from BH+ scaling):")
print(f"²Σ state: B₀'' = {B0_sigma_plus:.3f} cm⁻¹, D₀'' = {D0_sigma_plus:.6f} cm⁻¹")
print(f"²Π state: B₀' = {B0_pi_plus:.3f} cm⁻¹, D₀' = {D0_pi_plus:.6f} cm⁻¹")
print(f"Coupling: λ = {lambda_bd_plus:.3f}")
print(f"Main band origin: ν₀₀ = {nu_00_plus:.1f} cm⁻¹ ({1e7 / nu_00_plus:.1f} nm)")
print()


def rotational_term(B, D, J):
    """Calculate rotational term F(J) = BJ(J+1) - DJ²(J+1)²"""
    return B * J * (J + 1) - D * (J * (J + 1)) ** 2


def doublet_splitting(lambda_val, J, J_prime):
    """Calculate ²Π doublet splitting for BD+"""
    # For ²Π state, F₁(J) and F₂(J) differ by lambda-doubling
    if J_prime == J + 0.5:  # F₁ component (J = K + 1/2)
        return lambda_val * (J + 1)
    elif J_prime == J - 0.5:  # F₂ component (J = K - 1/2)
        return -lambda_val * J
    else:
        return 0


def line_position_doublet(nu_band, B_upper, D_upper, B_lower, D_lower,
                          J_upper, J_lower, lambda_val, component='F1'):
    """Calculate line position for ²Π → ²Σ transition"""
    F_lower = rotational_term(B_lower, D_lower, J_lower)
    F_upper = rotational_term(B_upper, D_upper, J_upper)

    # Add lambda-doubling for ²Π state
    if component == 'F1':
        lambda_correction = lambda_val * (J_upper + 0.5)
    else:  # F2
        lambda_correction = -lambda_val * (J_upper - 0.5) if J_upper > 0.5 else 0

    return nu_band + F_upper + lambda_correction - F_lower


def intensity_factor_doublet(J, branch_type, component='F1', temperature=3000):
    """Calculate intensity factor for doublet transitions"""
    kT = 0.695 * temperature / 8065.5  # kT in cm⁻¹

    # Boltzmann factor for lower state
    boltzmann = (2 * J + 1) * np.exp(-rotational_term(6.7, 0, J) / kT)

    # Hönl-London factors for ²Π → ²Σ transition
    if branch_type == 'P':  # ΔJ = -1
        if component == 'F1':
            honl_london = (J + 0.5) / (2 * J + 1) if J > 0 else 0
        else:  # F2
            honl_london = (J - 0.5) / (2 * J + 1) if J > 0.5 else 0
    elif branch_type == 'Q':  # ΔJ = 0 (forbidden for ²Π → ²Σ)
        honl_london = 0.0
    elif branch_type == 'R':  # ΔJ = +1
        if component == 'F1':
            honl_london = (J + 1.5) / (2 * J + 1)
        else:  # F2
            honl_london = (J + 0.5) / (2 * J + 1)
    else:
        honl_london = 1.0

    return boltzmann * honl_london


def generate_doublet_band(nu_origin, B_up, D_up, B_low, D_low, lambda_val, max_J=20):
    """Generate all lines for a ²Π → ²Σ band (doublet structure)"""
    lines = []

    for J_low in range(max_J):
        # P branch: ΔJ = -1
        if J_low > 0:
            J_up = J_low - 1
            # F1 component
            nu_f1 = line_position_doublet(nu_origin, B_up, D_up, B_low, D_low,
                                          J_up, J_low, lambda_val, 'F1')
            intensity_f1 = intensity_factor_doublet(J_low, 'P', 'F1')
            lines.append((nu_f1, intensity_f1, f'P₁({J_low})'))

            # F2 component
            if J_up > 0:
                nu_f2 = line_position_doublet(nu_origin, B_up, D_up, B_low, D_low,
                                              J_up, J_low, lambda_val, 'F2')
                intensity_f2 = intensity_factor_doublet(J_low, 'P', 'F2')
                lines.append((nu_f2, intensity_f2, f'P₂({J_low})'))

        # R branch: ΔJ = +1
        J_up = J_low + 1
        # F1 component
        nu_f1 = line_position_doublet(nu_origin, B_up, D_up, B_low, D_low,
                                      J_up, J_low, lambda_val, 'F1')
        intensity_f1 = intensity_factor_doublet(J_low, 'R', 'F1')
        lines.append((nu_f1, intensity_f1, f'R₁({J_low})'))

        # F2 component
        nu_f2 = line_position_doublet(nu_origin, B_up, D_up, B_low, D_low,
                                      J_up, J_low, lambda_val, 'F2')
        intensity_f2 = intensity_factor_doublet(J_low, 'R', 'F2')
        lines.append((nu_f2, intensity_f2, f'R₂({J_low})'))

    return lines


def cm_to_nm(wavenumber):
    """Convert wavenumber (cm⁻¹) to wavelength (nm)"""
    return 1e7 / wavenumber


def generate_bd_plus_spectrum(wavelength_range=(280, 4500), resolution=0.1):
    """Generate complete BD+ emission spectrum"""

    all_lines = []

    # (0,0) band - strongest
    lines_00 = generate_doublet_band(nu_00_plus, B0_pi_plus, D0_pi_plus,
                                     B0_sigma_plus, D0_sigma_plus, lambda_bd_plus)
    for nu, intensity, label in lines_00:
        all_lines.append((nu, intensity * 1.0, f'(0,0) {label}'))

    # (1,1) band - weaker due to vibrational population
    lines_11 = generate_doublet_band(nu_11_plus, B1_pi_plus, D1_pi_plus,
                                     B1_sigma_plus, D1_sigma_plus, lambda_bd_plus)
    for nu, intensity, label in lines_11:
        all_lines.append((nu, intensity * 0.4, f'(1,1) {label}'))

    # (1,0) band - hot band
    lines_10 = generate_doublet_band(nu_10_plus, B1_pi_plus, D1_pi_plus,
                                     B0_sigma_plus, D0_sigma_plus, lambda_bd_plus)
    for nu, intensity, label in lines_10:
        all_lines.append((nu, intensity * 0.2, f'(1,0) {label}'))

    # (0,1) band - sequence
    lines_01 = generate_doublet_band(nu_01_plus, B0_pi_plus, D0_pi_plus,
                                     B1_sigma_plus, D1_sigma_plus, lambda_bd_plus)
    for nu, intensity, label in lines_01:
        all_lines.append((nu, intensity * 0.15, f'(0,1) {label}'))

    # Convert to wavelengths and filter
    wavelength_lines = []
    for nu, intensity, label in all_lines:
        wavelength = cm_to_nm(nu)
        if wavelength_range[0] <= wavelength <= wavelength_range[1]:
            wavelength_lines.append((wavelength, intensity, label))

    # Create spectrum
    wavelengths = np.arange(wavelength_range[0], wavelength_range[1], resolution)
    spectrum = np.zeros_like(wavelengths)

    # Add lines with Gaussian broadening
    linewidth = 0.3  # nm, doublet structure requires better resolution
    for wl, intensity, label in wavelength_lines:
        profile = intensity * np.exp(-((wavelengths - wl) / linewidth) ** 2)
        spectrum += profile

    return wavelengths, spectrum, wavelength_lines


# Generate BD+ spectrum
print("Generating BD+ emission spectrum...")
wavelengths, spectrum, line_data = generate_bd_plus_spectrum(wavelength_range=(280,1000), resolution=0.01)

load_plot_style()
# Create the plot
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 9), constrained_layout=True, sharex=False)

# Main spectrum
# plt.subplot(3, 1, 1)
axes[0].plot(wavelengths, spectrum, 'r-', linewidth=0.8, label=r'BD$^{\mathregular{+}}$ emission')
axes[0].set_xlabel('Wavelength (nm)')
axes[0].set_ylabel('Intensity (arbitrary units)')
axes[0].set_title(r'\textbf{\sffamily BD\textsuperscript{+} (Boron Deuteride Ion) Emission Spectrum \\ $^2\Pi$ → $^2\Sigma$ Electronic Transition (scaled from BH\textsuperscript{+})}', usetex=True)
axes[0].grid(True, alpha=0.3)
axes[0].legend()
# axes[0].set_ylim(top=max(spectrum)*1.2)
# axes[0].set_xlim(300, 900)

# Mark major band heads
band_heads = [
    (cm_to_nm(nu_00_plus), '(0,0)'),
    (cm_to_nm(nu_11_plus), '(1,1)'),
    (cm_to_nm(nu_10_plus), '(1,0)'),
    (cm_to_nm(nu_01_plus), '(0,1)')
]

for wl, label in band_heads:
    if 280 <= wl <= 4500:
        axes[1].axvline(x=wl, color='red', linestyle='--', alpha=0.75, lw=1.5)
        axes[1].text(wl, max(spectrum), label, rotation=90,
                 verticalalignment='top', ha='left', fontsize=10)

# Zoom on main band showing doublet structure
# plt.subplot(3, 1, 2)
zoom_center = cm_to_nm(nu_00_plus)
zoom_range = 30
zoom_mask = (wavelengths >= zoom_center - zoom_range) & (wavelengths <= zoom_center + zoom_range)

axes[1].plot(wavelengths[zoom_mask], spectrum[zoom_mask], 'r-', linewidth=1.2)
axes[1].set_xlabel('Wavelength (nm)')
axes[1].set_ylabel('Intensity')
axes[1].set_title(r'\textbf{\sffamily BD\textsuperscript{+} (0,0) Band Detail - Doublet Structure from $^2\Pi$ State}', usetex=True)
axes[1].grid(True, alpha=0.3)

# Mark F1 and F2 components
for wl, intensity, label in line_data:
    if zoom_center - zoom_range <= wl <= zoom_center + zoom_range and '(0,0)' in label:
        if 'R₁(' in label or 'P₁(' in label:
            axes[1].axvline(x=wl, color='blue', linestyle=':', alpha=0.6, linewidth=0.8)
        elif 'R₂(' in label or 'P₂(' in label:
            axes[1].axvline(x=wl, color='green', linestyle=':', alpha=0.6, linewidth=0.8)

# Ultra-zoom showing individual doublets
# plt.subplot(3, 1, 3)
ultra_zoom = 1
ultra_mask = (wavelengths >= zoom_center - ultra_zoom) & (wavelengths <= zoom_center + ultra_zoom)

axes[2].plot(wavelengths[ultra_mask], spectrum[ultra_mask], 'r-', linewidth=1.5)
axes[2].set_xlabel('Wavelength (nm)')
axes[2].set_ylabel('Intensity')
axes[2].set_title(r'\textbf{\sffamily Ultra-zoom: Individual F\textsubscript{1}/F\textsubscript{2} Doublets in BD\textsuperscript{+}', usetex=True)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(wavelengths[ultra_mask].min(), wavelengths[ultra_mask].max())

# Label individual lines
count = 0
for wl, intensity, label in line_data:
    if zoom_center - ultra_zoom <= wl <= zoom_center + ultra_zoom and '(0,0)' in label:
        if count < 10:  # Limit labels to avoid crowding
            if 'R₁(' in label:
                plt.axvline(x=wl, color='blue', linestyle='-', alpha=0.8)
                plt.text(wl, intensity * 0.8, label.split('(0,0) ')[1].replace('₁', r'$_{\mathregular{1}}$'), rotation=90,
                         fontsize=8, color='blue')
            elif 'R₂(' in label:
                plt.axvline(x=wl, color='green', linestyle='-', alpha=0.8)
                plt.text(wl, intensity * 0.6, label.split('(0,0) ')[1].replace('₂', r'$_{\mathregular{2}}$'), rotation=90,
                         fontsize=8, color='green')
            count += 1

# plt.tight_layout()
fig.savefig(r'./figures/BD+_OES_spectrum.png', dpi=600)
plt.show()

# Statistics
print(f"\nBD+ Spectrum Statistics:")
print(f"Total lines generated: {len(line_data)}")
print(f"Main band at: {cm_to_nm(nu_00_plus):.1f} nm")
print(f"Expected doublet splitting: ~{lambda_bd_plus:.2f} cm⁻¹")

# Show some prominent lines
print(f"\nStrongest BD+ lines in near-UV/visible (350-450 nm):")
visible_lines = [(wl, intensity, label) for wl, intensity, label in line_data
                 if 350 <= wl <= 450]
visible_lines.sort(key=lambda x: x[1], reverse=True)

for i, (wl, intensity, label) in enumerate(visible_lines[:15]):
    component = "F₁" if "₁" in label else "F₂"
    print(f"{i + 1:2d}. {wl:6.1f} nm - {label:20s} - {component} (I={intensity:.3f})")

print(f"\nNote: This BD+ spectrum is estimated by scaling BH+ constants.")
print(f"The doublet structure arises from the ²Π electronic state spin-orbit coupling.")
print(f"Compare with the neutral BD spectrum to see the ionization effect!")