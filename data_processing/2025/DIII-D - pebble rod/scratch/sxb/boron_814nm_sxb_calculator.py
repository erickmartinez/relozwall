import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import h5py
from data_processing.misc_utils.plot_style import load_plot_style

PEC_KNOT_FILE = '../../mds_spectra/sxb/boron_pec_interpolator.h5'
S_KNOT_FILE = '../../mds_spectra/sxb/boron_ionization_interpolator.h5'


class Boron814SXBCalculator:
    def __init__(self, pec_interpolator_knots_h5, ionization_interpolator_knots_h5):
        self.interpolator_pec = self.load_interpolator_from_h5(filepath=pec_interpolator_knots_h5)
        self.interpolator_ionization = self.load_interpolator_from_h5(filepath=ionization_interpolator_knots_h5)

    @staticmethod
    def load_interpolator_from_h5(filepath):
        """
        Loads interpolator components from an HDF5 file and reconstructs the object.

        Args:
            filepath (str): The path to the HDF5 file.

        Returns:
            RectBivariateSpline: The reconstructed interpolator object.
        """
        with h5py.File(filepath, 'r') as hf:
            log_te_knots = hf['log_te_knots'][:]
            log_ne_knots = hf['log_ne_knots'][:]
            coeffs = hf['coeffs'][:]
            k_te = hf.attrs['k_te']
            k_ne = hf.attrs['k_ne']

        # The TCK tuple must be in (tx, ty, c, kx, ky) order for reconstruction.
        # tx -> te_knots, ty -> ne_knots, kx -> k_te, ky -> k_ne
        tck = (log_te_knots, log_ne_knots, coeffs, k_te, k_ne)
        loaded_interpolator = RectBivariateSpline._from_tck(tck)

        print(f"✅ Interpolator loaded successfully from '{filepath}'")
        return loaded_interpolator

    def ionization_rate_coefficient_adas(self, Te, ne):
        log_rate = self.interpolator_ionization(np.log10(Te), np.log10(ne), grid=False)
        return 10 ** log_rate

    def excitation_rate_coefficient_adas(self, Te, ne):
        log_rate = self.interpolator_pec(np.log10(Te), np.log10(ne), grid=False)
        return 10 ** log_rate

    def calculate_sxb(self, T_e, ne):
        """S/XB = PEC / S_ionization"""
        S = self.ionization_rate_coefficient_adas(Te=T_e, ne=ne)
        PEC = self.excitation_rate_coefficient_adas(Te=T_e, ne=ne)
        sxb = S / PEC
        return sxb


def main(pec_knot_file=PEC_KNOT_FILE, ionization_file=S_KNOT_FILE):
    calculator = Boron814SXBCalculator(pec_interpolator_knots_h5=pec_knot_file, ionization_interpolator_knots_h5=ionization_file)

    Te_range = np.logspace(0, 4, 500)
    n_e_range = np.array([1E10, 1E11, 1E12, 1E13, 1E15]) # cm^{-3}
    load_plot_style()
    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True, sharex=True)
    fig.set_size_inches(4, 8)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    print(f'PEC at 5 eV, 2E11/cm^3: {calculator.excitation_rate_coefficient_adas(Te=5, ne=2E11)}')
    for i, n_e in enumerate(n_e_range):
        pec = calculator.excitation_rate_coefficient_adas(Te=Te_range, ne=n_e)
        s = calculator.ionization_rate_coefficient_adas(Te=Te_range, ne=n_e)
        ne_split = f'{n_e:.0E}'.split('E')
        exponent = int(ne_split[1])
        lbl = f'n$_{{\mathregular{{e}}}}$ = 10$^{{\mathregular{{{exponent:d}}}}}$ (cm$^{{\mathregular{{-3}}}}$)'
        sxb = s/pec
        if i == 0:
            axes[0].plot(Te_range, s, color='k')
        axes[1].plot(Te_range, pec, color=colors[i], label=lbl)
        axes[2].plot(Te_range, sxb, color=colors[i])


    axes[1].legend(loc='best')
    # axes[1].legend(loc='best')
    # axes[2].legend(loc='best')
    axes[0].set_ylabel(r'{\sffamily S (cm\textsuperscript{-3}/s)}', usetex=True)
    axes[1].set_ylabel(r'{\sffamily PEC (cm\textsuperscript{-3}/s)}', usetex=True)
    axes[2].set_ylabel(r'{\sffamily S/XB (ionizations/photon)}', usetex=True)
    axes[2].set_xlabel(r'{\sffamily T\textsubscript{e}} (K)', usetex=True)
    axes[0].set_title(r'{\sffamily ADAS B $\to$ B\textsuperscript{+1}}', usetex=True, fontweight='bold')
    axes[1].set_title('ADAS 814.82 nm')

    axes[0].set_xlim(1, 1E4)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')

    fig.savefig('boron_814nm_sxb.png', dpi=600)
    plt.show()

if __name__ == '__main__':
    main()

