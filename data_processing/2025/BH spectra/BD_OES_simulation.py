import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from data_processing.misc_utils.plot_style import load_plot_style

# Molecular constants for BD from Thunberg (1936)
# ¹Σ state (ground state)
B0_sigma = 6.449  # cm⁻¹
B1_sigma = 6.285  # cm⁻¹
alpha_sigma = 0.166  # cm⁻¹
D0_sigma = 0.000351  # cm⁻¹

# ²Π state (excited state)
B0_pi = 6.513  # cm⁻¹
B1_pi = 6.233  # cm⁻¹
alpha_pi = 0.280  # cm⁻¹
D0_pi = 0.000420  # cm⁻¹

# Band origins (cm⁻¹) from Tables
nu_00 = 23098.75  # (0,0) band
nu_11 = 22989.57  # (1,1) band

# Additional estimated origins for other bands
nu_01 = nu_00 + 800  # Approximate (0,1) band
nu_10 = nu_00 - 800  # Approximate (1,0) band


def rotational_term(B, D, J):
    """Calculate rotational term F(J) = BJ(J+1) - DJ²(J+1)²"""
    return B * J * (J + 1) - D * (J * (J + 1)) ** 2


def line_position(nu_band, B_upper, D_upper, B_lower, D_lower, J_upper, J_lower):
    """Calculate line position for a transition"""
    F_upper = rotational_term(B_upper, D_upper, J_upper)
    F_lower = rotational_term(B_lower, D_lower, J_lower)
    return nu_band + F_upper - F_lower


def intensity_factor(J, branch_type, temperature=3000):
    """Calculate intensity factor including Boltzmann population and Hönl-London factors"""
    kT = 0.695 * temperature / 8065.5  # kT in cm⁻¹ (T in K)

    # Boltzmann factor
    boltzmann = (2 * J + 1) * np.exp(-rotational_term(6.4, 0, J) / kT)

    # Hönl-London factors for ²Π - ¹Σ transition
    if branch_type == 'P':  # ΔJ = -1
        honl_london = J / (2 * J + 1) if J > 0 else 0
    elif branch_type == 'Q':  # ΔJ = 0
        honl_london = 1.0
    elif branch_type == 'R':  # ΔJ = +1
        honl_london = (J + 1) / (2 * J + 1)
    else:
        honl_london = 1.0

    return boltzmann * honl_london


def generate_band(nu_origin, B_up, D_up, B_low, D_low, max_J=25):
    """Generate all lines for a single band"""
    lines = []

    for J_low in range(max_J):
        # P branch: ΔJ = -1 (J_up = J_low - 1)
        if J_low > 0:
            J_up = J_low - 1
            nu = line_position(nu_origin, B_up, D_up, B_low, D_low, J_up, J_low)
            intensity = intensity_factor(J_low, 'P')
            lines.append((nu, intensity, f'P({J_low})'))

        # Q branch: ΔJ = 0 (not allowed for ²Π - ¹Σ, but included for completeness)
        # J_up = J_low
        # nu = line_position(nu_origin, B_up, D_up, B_low, D_low, J_low, J_low)
        # intensity = intensity_factor(J_low, 'Q') * 0.1  # Weak forbidden transition
        # lines.append((nu, intensity, f'Q({J_low})'))

        # R branch: ΔJ = +1 (J_up = J_low + 1)
        J_up = J_low + 1
        nu = line_position(nu_origin, B_up, D_up, B_low, D_low, J_up, J_low)
        intensity = intensity_factor(J_low, 'R')
        lines.append((nu, intensity, f'R({J_low})'))

    return lines


def cm_to_nm(wavenumber):
    """Convert wavenumber (cm⁻¹) to wavelength (nm)"""
    return 1e7 / wavenumber


def generate_spectrum(wavelength_range=(280, 4500), resolution=0.1):
    """Generate complete BD emission spectrum"""

    # Generate lines for different bands
    all_lines = []

    # (0,0) band - strongest
    lines_00 = generate_band(nu_00, B0_pi, D0_pi, B0_sigma, D0_sigma)
    for nu, intensity, label in lines_00:
        all_lines.append((nu, intensity * 1.0, f'(0,0) {label}'))

    # (1,1) band - medium strength
    lines_11 = generate_band(nu_11, B1_pi, D0_pi, B1_sigma, D0_sigma)
    for nu, intensity, label in lines_11:
        all_lines.append((nu, intensity * 0.6, f'(1,1) {label}'))

    # (1,0) band - weaker
    lines_10 = generate_band(nu_10, B1_pi, D0_pi, B0_sigma, D0_sigma)
    for nu, intensity, label in lines_10:
        all_lines.append((nu, intensity * 0.3, f'(1,0) {label}'))

    # (0,1) band - weaker
    lines_01 = generate_band(nu_01, B0_pi, D0_pi, B1_sigma, D0_sigma)
    for nu, intensity, label in lines_01:
        all_lines.append((nu, intensity * 0.3, f'(0,1) {label}'))

    # Convert to wavelengths and filter by range
    wavelength_lines = []
    for nu, intensity, label in all_lines:
        wavelength = cm_to_nm(nu)
        if wavelength_range[0] <= wavelength <= wavelength_range[1]:
            wavelength_lines.append((wavelength, intensity, label))

    # Create wavelength grid
    wavelengths = np.arange(wavelength_range[0], wavelength_range[1], resolution)
    spectrum = np.zeros_like(wavelengths)

    # Add each line with Gaussian profile
    linewidth = 0.5  # nm, instrumental broadening
    for wl, intensity, label in wavelength_lines:
        # Gaussian profile
        profile = intensity * np.exp(-((wavelengths - wl) / linewidth) ** 2)
        spectrum += profile

    return wavelengths, spectrum, wavelength_lines


# Generate the spectrum
print("Generating BD emission spectrum...")
wavelengths, spectrum, line_data = generate_spectrum(wavelength_range=(280, 4000), resolution=0.05)

load_plot_style()
# Create the plot
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 7), constrained_layout=True)

# Main spectrum plot
# plt.subplot(2, 1, 1)
axes[0].plot(wavelengths, spectrum, 'b-', linewidth=0.8)
axes[0].set_xlabel('Wavelength (nm)')
axes[0].set_ylabel('Intensity (arbitrary units)')
axes[0].set_title('BD (Boron Deuteride) Emission Spectrum\nBased on Thunberg (1936) molecular constants')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(280, 4500)

# Add band labels for major features
major_bands = [
    (cm_to_nm(nu_00), '(0,0)'),
    (cm_to_nm(nu_11), '(1,1)'),
    (cm_to_nm(nu_10), '(1,0)'),
    (cm_to_nm(nu_01), '(0,1)')
]

for wl, label in major_bands:
    if 280 <= wl <= 4500:
        plt.axvline(x=wl, color='red', linestyle='--', alpha=0.7)
        plt.text(wl, max(spectrum) * 0.8, label, rotation=90,
                 verticalalignment='bottom', fontsize=10)

# Zoomed view of the main (0,0) band region
plt.subplot(2, 1, 2)
zoom_center = cm_to_nm(nu_00)
zoom_range = 20  # nm
zoom_mask = (wavelengths >= zoom_center - zoom_range) & (wavelengths <= zoom_center + zoom_range)

axes[1].plot(wavelengths[zoom_mask], spectrum[zoom_mask], 'b-', linewidth=1.0)
axes[1].set_xlabel('Wavelength (nm)')
axes[1].set_ylabel('Intensity (arbitrary units)')
axes[1].set_title(f'Zoomed view: (0,0) band region around {zoom_center:.1f} nm')
axes[1].grid(True, alpha=0.3)

# Add individual line markers in zoom
for wl, intensity, label in line_data:
    if zoom_center - zoom_range <= wl <= zoom_center + zoom_range and '(0,0)' in label:
        if 'P(' in label or 'R(' in label:  # Only P and R branches
            plt.axvline(x=wl, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)

# fig.tight_layout()
fig.savefig(r'./figures/BD_OES_spectrum.png', dpi=600)
plt.show()

# Print some statistics
print(f"\nSpectrum Statistics:")
print(f"Total number of lines: {len(line_data)}")
print(f"Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
print(f"Main (0,0) band head at: {cm_to_nm(nu_00):.1f} nm")
print(f"Main (1,1) band head at: {cm_to_nm(nu_11):.1f} nm")

# Show strongest lines
print(f"\nStrongest lines in visible range (400-700 nm):")
visible_lines = [(wl, intensity, label) for wl, intensity, label in line_data
                 if 400 <= wl <= 700]
visible_lines.sort(key=lambda x: x[1], reverse=True)

for i, (wl, intensity, label) in enumerate(visible_lines[:10]):
    print(f"{i + 1:2d}. {wl:6.1f} nm - {label:15s} (I = {intensity:.3f})")