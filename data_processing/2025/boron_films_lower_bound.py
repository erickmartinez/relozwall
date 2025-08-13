"""
Estimate the lower bound of thickness of boron films based on estimated uncertainty of
transmission measurements
"""
import numpy as np

# These values come from the measurements of optical transmission on a blank glass slide
REF_SIGNAL_V = 0.408811659320735
REF_SIGNAL_ERR_V = 0.0000113553064660624
SLIDE_SIGNAL_V = 0.406683940440001
SLIDE_SIGNAL_ERR_V = 0.0000108828133624986
BACKGROUND_SIGNAL_V = 0.000168
BACKGROUND_SIGNAL_ERR_V = 0.000000092443
ALPHA_BORON_650 = 3.057E+04 # 1/cm

def get_absorbance(I, I0, I_err, I0_err):
    """
    Determine absorbance according to A = -log10(T) = -log10(I/I0)
    Parameters
    ----------
    I: float
        The intensity of transmitted light
    I0: float
        The intensity of incident light
    I_err: float
        The error in the measurement of the transmitted light
    I_0_err: float
        The error in the measurement of the incident light

    Returns
    -------
    tuple:
        The absorbance and its associated error
    """
    absorbance = -np.log10(I / I0)
    uncertainty = np.linalg.norm([I_err/I, I0_err/I0])/np.log(10)
    return absorbance, uncertainty

def main(
    ref_signal, ref_signal_error, slide_signal, slide_signal_error, background_signal, background_signal_error, alpha_boron
):
    I0 = ref_signal - background_signal
    I0_err = np.linalg.norm([ref_signal_error, background_signal_error])
    I = slide_signal - background_signal
    I_err = float(np.linalg.norm([slide_signal_error, background_signal_error]))
    print(f"I_0: {I0:.3E} -/+ {I0_err:.3E} V")
    print(f"I: {I:.3E} -/+ {I_err:.3E} V")
    print(f"T = {I/I0:.3f}")
    absorbance, absorbance_error = get_absorbance(I=I, I0=I0, I_err=I_err, I0_err=I0_err)
    print(f'Absorbance: {absorbance:.3E} -/+ {absorbance_error:.4E}')
    thickness = absorbance / alpha_boron * 1E7 # nm
    thickness_error = thickness * np.abs(absorbance_error/absorbance)
    print(f'Thickness upper limit: {thickness_error:.3f} -/+ {thickness_error:.4f} nm')

if __name__ == '__main__':
    main(
        ref_signal=REF_SIGNAL_V,
        ref_signal_error=REF_SIGNAL_ERR_V,
        slide_signal=SLIDE_SIGNAL_V,
        slide_signal_error=SLIDE_SIGNAL_ERR_V,
        background_signal=BACKGROUND_SIGNAL_V,
        background_signal_error=BACKGROUND_SIGNAL_ERR_V,
        alpha_boron=ALPHA_BORON_650
    )

