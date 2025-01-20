import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.fft import fft
import os
import json
import re
from scipy.optimize import least_squares, OptimizeResult, minimize, differential_evolution
from matplotlib.lines import Line2D

import data_processing.confidence as cf
from scipy.stats.distributions import t

data_file = "./data/PA_probe/20241031/gamma_ivdata0011.raw"

# https://webbook.nist.gov/cgi/cbook.cgi?ID=C14464472&Units=SI
m_d1 = 2.0135531979
# https://webbook.nist.gov/cgi/cbook.cgi?Formula=D2%2B&MatchIso=on&Units=SI
m_d2 = 4.0276549757
# https://webbook.nist.gov/cgi/cbook.cgi?ID=C12595969&Mask=8
m_d3 = 6.0417567535

w_d1 = 0.41 # The concentration of the d+ ions relative to n_e
w_d2 = 0.22 # The concentration of the d2+ ions relative to n_e
w_d3 = 0.37 # The concentration of the d3+ ions relative to n_e


# ***************** Problem parameters **********************************

# Pisces-A probe geometry
# AreaP = .085 # probe tip area [cm^2], corresponds to 4*R*L, two direction projected area
# AreaP = .025 # probe tip area [cm^2], corresponds to 2*R*L, single direction projected area for probe near target
AreaP = .049 # probe tip area [cm^2] that Daisuke is using at present (10/2024) for gamma probe
Vscale = 100. # voltage division done in Langmuir probe box :  Vprobe = dataV(:,1)*Vscale
Rc = 10. # current resistor [Ohms] in Langmuir probe box : Iprobe = dataV(:,2)/Rc. Usually 2 Ohm or 5 Ohm
# xScale = 1.56 # probe plunge position conversion [cm/V]
xScale = 1.53 # probe plunge position conversion [cm/V] that Daisuke is using (10/2024)


Amag = 0.61 # Use Amag = 0.5 for magnetized plasma (rci^2 << Ap) ; 0.61 for unmagnetized plasma
# mi = 1.9995 # mi is ion mass in units of hydrogen mass. Typically use 4 for He+ and 2 for D+
mi = w_d1 * m_d1 + w_d2 * m_d2 + w_d3 * m_d3
mi /= 1.00727647 # Divide by the mass of the proton to match the units
Zion = 1. # ion charge state
Tescale = 0.67 # scale factor on Te for estimating correct plasma potential on manual fits

iRampType = 3 # set to 1 for rising sawtooth; 2 for triangle wave; 3 for sine wave
iLeak = 1 # set to 1 to correct for leakage current
iSheathExp = 1 # allow empirical correction for sheath expansion factor in numerical fit
iFitEsat = 1 # set to 1 if interested in fitting esat; otherwise just fit isat region
Ti = 0.5 # ion temperature
IpAtten = 1 # attenuation on Ip signal from isolator

ee = 1.6e-19 # electron charge. 
rpe = 1836.2 # proton over electron mass ratio
Jfac = np.sqrt(rpe*mi/(2.*np.pi))/(2.*Amag)


loss = 'soft_l1'
f_scale = 0.1 # The scaling factor for the outliers in the optimizer for the fit
"""
x0 = np.array([JsatFit[iramp], TeFit[iramp], VsFit[iramp], eslopeFit[iramp], VesatFit[iramp], JslopeFit[iramp]])
"""
bounds_fit=(
    [-1E4, 0.1, -100., 0., -np.inf, 0.],
    [1E4, 40., 100., np.inf, 100., 1E10]
) # The bounds for the fit

#### Global parameters needed for manual fits

TeMan = []
neMan = []
VsMan = []
VsManC = []
JsatMan = []

def characterize_sawtooth(v, ax):
    """
    Characterize probe voltage sawtooth

    Parameters
    ----------
    v: np.ndarray
        An array containing the voltage

    Returns
    -------

    """
    global iRampType
    n_points = len(v) # total number of points in the sawtooth
    v_off = v - v.mean() # center Vpv vertically to remove k=0 Fourier component
    fVp = fft(v_off)    # Fourier transform sawtooth
    km, im = np.max(fVp[0:int(0.5*n_points)]), np.argmax(fVp[0:int(0.5*n_points)]) # locate largest peak in Fourier spectrum
    lam = int(n_points/im) # wavelength of sawtooth [points]
    fVp_im = fVp[im]
    # print(fVp_im)
    if iRampType == 1:
        Vpp = 6.34 * abs(fVp_im) / n_points # peak-peak amplitude of sawtooth [V]
    else:
        Vpp = max(v[0:2*lam]) - min(v[0:2*lam])
    lam0 = lam * ( np.pi/2. + np.arctan2(np.imag(fVp_im),np.real(fVp_im)) )
    # lam0 = lam * (np.pi / 2. + np.arctan(np.imag(fVp_im) / np.real(fVp_im)))
    # lam0 = lam * np.angle(fVp_im)
    lam0 = int(lam0)
    if lam0 > lam:
        lam0 -= lam
    iStart = lam0
    if iRampType > 1:
        kp, iStart = np.max(v[0:lam]), np.argmax(v[0:lam])

    # here, define the phase as the distance to the bottom of the first complete sawtooth ramp [points]
    # in the case of a triangle ramp, it's the distance to the top of the first "V"
    n_ramps = 0 # initialize number of good ramps to 0
    n_ramps_max = int(n_points / lam) # maximum number of ramps
    iStartRamp = []
    iEndRamp = []

    for iramp in range(n_ramps_max):
        ilo = iStart - 10 # anticipate about 100 points per ramp, so search +/- 10 points around peak
        ihi = iStart + 10
        ilo = max(ilo, 0)
        ihi = min(ihi, n_points-1)
        # ilo, ihi = int(ilo), int(ihi)

        if iRampType == 1:
            idum = np.argmin(v[ilo:ihi]) # iStart is beginning of ramp "/" for sawtooth or sine wave
            dum = v[idum]
        else:
            idum = np.argmax(v[ilo:ihi]) # iStart is beginning of "V" for triangle
            dum = v[idum]
        iStart = ilo + idum
        iEnd = iStart + lam # iEnd is end of ramp for sawtooth
        if iRampType > 1:
            iEnd = int(iStart + lam / 2.) # iEnd is bottom of "V" for triangle
        ilo = iEnd - 10
        ihi = iEnd + 10
        ilo = max(ilo, 0)
        ihi = min(ihi, n_points-1)
        if iRampType == 1:
            idum = np.argmax(v[ilo:ihi])
        else:
            idum = np.argmin(v[ilo:ihi])
        iEnd = ilo + idum
        vpp_meas = v[iEnd] - v[iStart] # amplitude of this sawtooth
        if iRampType > 1:
            vpp_meas *= -1.
        if (vpp_meas > Vpp * 0.5) and (vpp_meas < 1.5 * Vpp): # check to make sure this ramp looks reasonable
            n_ramps += 1
            iStartRamp.append(iStart) # characterize ramps by their start and end indices
            iEndRamp.append(iEnd)
            # print(f"{iramp:>5d}: {iStart:>5d}, {iEnd:>5d}")

        if iEnd > (n_points - lam + 5): # exit loop
            break
        if iRampType == 1:
            iStart = iEnd + 5 # prepare loop for next ramp
        else:
            iStart += lam

    # plot out ramp data and lines for each ramp
    ax.plot(v, 'o', color='C0', mfc='none', label='Data')
    for iramp in range(n_ramps):
        iStart = iStartRamp[iramp]
        iEnd = iEndRamp[iramp]
        if iRampType == 1:
            ax.plot([iStart, iEnd], [v[iStart], v[iEnd]], ls='-', color='tab:red', lw=1.25)
        elif iRampType == 2:
            ax.plot([iStart, iEnd], [v[iStart], v[iEnd]], ls='-', color='tab:red', lw=1.25,)
            ax.plot([iStart, iStart+lam], [v[iStart], v[iEnd+lam]], ls='-', color='tab:red', lw=1.25)
        else:
            vpp = v[iStart] - v[iEnd]
            voff = 0.5*(v[iStart] + v[iEnd])
            ax.plot(
                np.arange(iStart, iStart+lam),
                voff + 0.5 * vpp * np.cos((np.arange(iStart, iStart+lam)-iStart) * 2. * np.pi / lam),
                color = 'tab:red',
                lw = 1.25,
            )

    ax.set_xlabel('Array index')
    ax.set_ylabel('V$_{\mathregular{probe}}$', usetex=False)
    # plt.draw()
    return np.array(iStartRamp), np.array(iEndRamp), lam, n_ramps

pattern_input_idx = re.compile("\D*(\d+)\D+(\d+)\D*")
def parse_input_indices(attempts=1):
    global pattern_input_idx
    input_txt = input("iStart, iEnd:\n")
    m = pattern_input_idx.match(input_txt)
    if m:
        try:
            i_start = int(m.group(1))
            i_end = int(m.group(2))
            return i_start, i_end
        except Exception as e:
            print('Error parsing iStart, iEnd input:')
            print(e)
            if attempts <= 10:
                print(f"Attempt {attempts} of 10. Try again...")
                attempts +=1
                parse_input_indices(attempts)
            else:
                raise ValueError(f"Incorrect input for iStart, iEnd")
    else:
        print('Error parsing iStart, iEnd input.')
        if attempts <= 10:
            print(f"Attempt {attempts} of 10. Try again...")
            attempts += 1
            parse_input_indices(attempts)

def model_poly(x, b) -> np.ndarray:
    n = len(b)
    r = np.zeros(len(x))
    for i in range(n):
        r += b[i] * x ** i
    return r


def res_poly(b, x, y, w=1.):
    return (model_poly(x, b) - y) * w


def jac_poly(b, x, y, w=1):
    n = len(b)
    r = np.zeros((len(x), n))
    for i in range(n):
        r[:, i] = w * x ** i
    return r


eps = float(np.finfo(np.float64).eps)
def fit_leakage_current(jp, start_ramp_ids, leak_start_idx, leak_end_idx, lam, ax, degree=6):
    global eps
    jp_leak = np.zeros(lam)
    n = 0
    for iramp in range(leak_start_idx-1, leak_end_idx-1):
        iStart = start_ramp_ids[iramp]
        for ii in range(lam):
            jp_leak[ii] += jp[ii-1+iStart]
        n += 1
    x = np.arange(lam)
    jp_leak /= n
    tol = eps
    ls_res = least_squares(
        res_poly,
        loss='soft_l1', f_scale=1.,
        x0 = [0.01 ** k for k in range(degree+1)],
        args=(x, jp_leak),
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=2,
        x_scale='jac',
        max_nfev=10000 * degree
    )
    popt = ls_res.x
    jp_leak_fit = model_poly(x, popt)
    ax.plot(jp_leak, 'x', color='C0', mfc='none', label='Data')
    ax.plot(jp_leak_fit, color='tab:red', ls='-', lw=1.25, label='Fit')
    ax.set_xlabel('Array index')
    ax.set_ylabel('J$_{\mathregular{p,leak}}$')
    ax.set_title('Fit to leakage current')
    ax.legend(loc='best', frameon=True, fontsize=10)
    plt.show()
    return jp_leak_fit

pick_events = 0
x_picked = np.empty(4, dtype=np.float64)
y_picked = np.empty(4, dtype=np.float64)
te_fit_l2d = None
js_fit_l2d = None
popt_te_onpick = np.ones(2)
jsat_onpick = None

"""
cs = ((Z*k_B*T_e + k_B*T_i)/M_i)^(1/2) = ((Z*k_B*T_e + k_B*T_i)/(M_p * m_i)^(1/2)

where M_i is the mass of the in kg, m_i = M_i / M_p, M_p is the mass of the proton,
k_B being Boltzmann's constant, T is in eV.
  
1 eV = 1.6021E-19 J 

1 KT = 1.38E-23 J/K * T[K] = 1.38E-23 J/K * T[K] x (1 eV / 1.6021E-19 J) = (k_B / q_e) [eV]

=> 1 [eV] = q_e / k_B

The prefactor for the ion speed of sound is then

cs_prefactor = (k_B T [K] / m_p)^0.5 = (q_e / m_p)^(1/2) x10^2 cm/s/eV^(1/2)



"""
m_p = 1.67262192369 # x1E-27 kg
e_v = 1.602176634 # x1E-19 J
cs_factor = np.power(e_v / m_p, 0.5) * 1E2


def on_mouse_move(event, horizontal_line, vertical_line, fig):
    # Check if the mouse is inside the plot axes
    if event.inaxes is not None:
        # Make lines visible and update their positions
        horizontal_line.set_visible(True)
        vertical_line.set_visible(True)
        horizontal_line.set_ydata([event.ydata, event.ydata])  # Changed to list format
        vertical_line.set_xdata([event.xdata, event.xdata])    # Changed to list format
        fig.canvas.draw()  # Use draw() instead of draw_idle()
    else:
        # Hide lines when mouse leaves plot area
        horizontal_line.set_visible(False)
        vertical_line.set_visible(False)
        fig.canvas.draw()

def on_pick(event, vp_fit, y_j_fit, jmin0, id_ramp, fig, ax):
    global pick_events, x_picked, y_picked, popt_te_onpick, jsat_onpick
    global te_fit_l2d, js_fit_l2d
    global Zion, Amag, Ti, mi, ee, Tescale
    global cs_factor # ~9.79E5 cm/s
    global TeMan, neMan, VsMan, VsManC, JsatMan
    global bounds_fit

    # if isinstance(event.artist, Line2D):
        # thisline = event.artist
        # xdata = thisline.get_xdata()
        # ydata = thisline.get_ydata()
        # ind = event.ind
        # x_mean = np.mean(xdata[ind])
        # y_mean = np.mean(ydata[ind])
        # x_picked[pick_events] = x_mean
        # y_picked[pick_events] = y_mean
    x_picked[pick_events] = event.xdata
    y_picked[pick_events] = event.ydata
    pick_events += 1
    # fig = event.artist.get_figure()
    if pick_events == 2:
        popt_te_onpick = fit_te_range_man(vp_fit, y_j_fit, x_picked[0:2])
        te_fit_l2d, = ax.plot(x_picked[0:2], model_poly(x_picked[0:2], popt_te_onpick), color='red')
        ax.set_title('Mouse select ion saturation current region', color='red')
        fig.canvas.draw()
        fig.canvas.flush_events()

    if pick_events == 4:
        jsat_onpick, yJave = fit_jsat_range_man(vp_fit, y_j_fit, x_picked[2:4], jmin0)
        js_fit_l2d, = ax.plot(x_picked[2:4], [yJave]*2, color='red')
        fig.canvas.draw()
        fig.canvas.flush_events()
        # plt.draw()
        # Calculate plasma parameters resulting from fit
        Te = 1. / popt_te_onpick[1] # popt[1] is the slope of the fit, d(logJ)/dV ~= 1/Te. Te is in [eV]
        Cs = 9.79E5 * np.sqrt((Zion*Te + Ti)/mi) # ion sound speed [cm/s]
        ne = np.abs(jsat_onpick)/Amag/Cs/ee # electron density [cm^-3]
        vf = (np.log(jmin0)-popt_te_onpick[0])/popt_te_onpick[1] # estimate plasma floating potential [V] from zero-crossing
        vs = vf + 2.3 * Te
        vsc = vf + 2.3 * Te * Tescale
        print("Results of manual fit:")
        print(f"Te: {Te:7.3g} eV, ne: {ne:5.3g} cm^3, Vs: {vs:7.3g} V")
        success_txt = input("Are the results reasonable (y/n): ")
        if success_txt.lower() == 'y':
            JsatMan[id_ramp] = jsat_onpick
            TeMan[id_ramp] = Te
            neMan[id_ramp] = ne
            VsMan[id_ramp] = vs
            VsManC[id_ramp] = vsc
            for line in ax.lines:
                line.remove()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.close(fig)
        else:
            lines = ax.lines
            lines[-1].remove()
            lines[-1].remove()
            ax.set_title('Mouse select range for T$_{\mathregular{e}}$ slope finder', color='red')
            fig.canvas.draw()
            fig.canvas.flush_events()
        pick_events = 0
        return


def fit_te_range_man(vp_fit, y_j_probe, vrange):
    global eps
    v_min = np.min(vrange)
    v_max = np.amax(vrange)
    msk_tf = (v_min <= vp_fit) & (vp_fit <= v_max)
    vp_t_fit = vp_fit[msk_tf]
    yj_t_fit = y_j_probe[msk_tf]
    tol = eps# ** (2./3.)
    ls_res = least_squares(
        res_poly,
        loss='soft_l1', f_scale=0.5,
        x0=[vp_t_fit[0], 10],
        args=(vp_t_fit, yj_t_fit),
        bounds=([-np.inf, 0.], [np.inf, np.inf]),
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=0,
        jac=jac_poly,
        # x_scale='jac',
        max_nfev=100000
    )
    popt = ls_res.x
    return popt

def fit_jsat_range_man(vp_fit, y_j_probe, vrange, jmin0):
    v_min = np.min(vrange)
    v_max = np.amax(vrange)
    msk_tf = (v_min <= vp_fit) & (vp_fit <= v_max)
    VpvJfit = vp_fit[msk_tf]
    yJJfit = y_j_probe[msk_tf]
    yJave = np.mean(yJJfit)
    j_sat = np.exp(yJave) - jmin0
    return j_sat, yJave

def fit_characeteristic_manually(
        v_pv, jp, start_ramp_ids, end_ramp_ids, fit_start_idx, fit_end_idx, lam,
        n_ramps
):
    global iRampType
    global TeMan, neMan, VsMan, VsManC, JsatMan
    """
    zero initial guesses for all vectors
    vectors corresponding to Te, ne, and Vs obtained from manual fits to data
    """
    n_ramps_to_fit = fit_end_idx - fit_start_idx

    id_ramp = 0
    # step through each ramp and do manual fit to characteristic
    iramp = fit_start_idx
    ax = None
    for i in range(n_ramps_to_fit+1): # cycle through manual fits
        iStart = start_ramp_ids[iramp]
        iEnd = end_ramp_ids[iramp]
        # print(f"Fitting man ramp indices [{iStart}:{iEnd}]")
        if iRampType > 1:
            iStart = end_ramp_ids[iramp] #  manual fit to second leg of "V"
            iEnd = start_ramp_ids[iramp] + lam

        Vpfit = v_pv[iStart:iEnd]
        Jfit = jp[iStart:iEnd]

        Jmin = Jfit.min()
        Jmin0 = 1.1 * np.abs(Jmin) # offset probe current by 1.1*Jmin
        JpOffset = Jfit + Jmin0 # jp_offset is jp offset to be positive-definite
        yjprobe = np.log(JpOffset)
        print(f"Probe data for sweep {iramp+1}")
        if not ax is None:
            for line in ax.lines:
                line.clear()
                fig.canvas.draw()
                fig.canvas.flush_events()
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        fig.set_size_inches(7.5, 4.5)
        line, = ax.plot(Vpfit, yjprobe, 'o', ms=4, color='C0', mfc='none', label='Data', picker=True, pickradius=3)
        ax.set_xlabel(r"V$_{\mathregular{probe}}$")
        ax.set_ylabel(r"log J$_{\mathregular{probe}}$")
        ax.set_title(
            fr'Mouse select range for T$_{{\mathregular{{e}}}}$ slope finder ({iramp + 1}/{fit_end_idx + 1})',
            color='red'
        )
        ax.set_ylim(-5,1.5)
        # fig.canvas.mpl_connect(
        #     'pick_event',
        #     lambda event: on_pick(event, Vpfit, yjprobe, Jmin0, iramp, fig, ax)
        # )
        fig.canvas.mpl_connect(
            'button_press_event',
            lambda event: on_pick(event, Vpfit, yjprobe, Jmin0, iramp, fig, ax)
        )
        # Initialize crosshairs
        horizontal_line = ax.axhline(y=0, color='k', ls='--', alpha=0.5, visible=False)
        vertical_line = ax.axvline(x=0, color='k', ls='--', alpha=0.5, visible=False)

        fig.canvas.mpl_connect('motion_notify_event', lambda event: on_mouse_move(event, horizontal_line, vertical_line, fig))
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=True)
        iramp += 1

    # return TeMan, neMan, VsMan, VsManC, JsatMan


def model_jfit(b, v_probe):
    global Jfac, iSheathExp, iFitEsat
    Jsat = b[0]
    Te = b[1]
    Vs = b[2]
    eslope = b[3]
    Vesat = b[4]
    JslopeFit = b[5]
    if iSheathExp == 1: # allow slope in Jsat
        Jfit = (Jsat + v_probe*JslopeFit) * (1. - Jfac * np.exp((v_probe-Vs)/Te))
    else:
        Jfit = Jsat *  (1. - Jfac * np.exp((v_probe-Vs)/Te))

    if iFitEsat == 1: # include esat correction
        Jesat = Jsat * (1. - Jfac * np.exp((Vesat - Vs)/Te) )
        # electron saturation current region
        msk_escr = v_probe >= Vesat
        Jfit[msk_escr] = Jesat + eslope*(v_probe[msk_escr]-Vesat) # allow some arbitrary slope here

    return Jfit

def res_jfit(b, v_probe, Jp):
    global iFitEsat
    Jfit = model_jfit(b, v_probe)
    Jsat = b[0]
    Te = b[1]
    Vs = b[2]
    eslope = b[3]
    Vesat = b[4]
    JslopeFit = b[5]
    chi = Jfit - Jp

    if iFitEsat == 1:
        return chi
    Jsort = Jp.copy()
    Jsort.sort()
    Jsatguess = Jsort[0:10].mean()
    msk_guess = Jp <= Jsatguess

    # turn up error a lot for unphysical solutions
    if (Te <= 0.1) or (abs(Jsat) <= 1E-10) or (np.abs(Vs) > 100) or (eslope < 10) or (JslopeFit < 0) or (Vesat > 1E3):
        chi *= 1000

    return chi[msk_guess]

def res_jfit_scalar(b, Vpv, Jp):
    res = res_jfit(b, Vpv, Jp)
    # return np.sum(np.abs(res))
    return 0.5*np.linalg.norm(res)

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

def estimate_ne_uncertainty(
    n_e:np.ndarray, c_s:np.ndarray, j_sat: np.ndarray, c_s_uncertainty: np.ndarray, j_sat_uncertainty:np.ndarray
) -> np.ndarray:
    return np.abs(n_e * np.linalg.norm(np.column_stack([j_sat_uncertainty/j_sat, c_s_uncertainty/c_s]), axis=1))

def estimate_cs_uncertainty(c_s: np.ndarray, T_e_uncertainty: np.ndarray, mu_i: float=mi) -> np.ndarray:
    return np.abs(9.759E11 * T_e_uncertainty / (2. * mu_i * c_s))

def estimate_jsat_uncertainty(
    v_s_fit: np.ndarray, T_e_fit: np.ndarray, J_slope_fit: np.ndarray,
    j_sat_fit_delta: np.ndarray, v_s_fit_delta: np.ndarray, T_e_fit_delta: np.ndarray, J_slope_fit_delta: np.ndarray
) -> np.ndarray:
    s = np.column_stack([j_sat_fit_delta, j_sat_fit_delta*(v_s_fit - 3.*T_e_fit), T_e_fit_delta*(3.*J_slope_fit)])
    return np.abs(np.linalg.norm(s, axis=1))

def main():
    global data_file, ee, rpe, Jfac, xScale, AreaP, Rc, Vscale, IpAtten
    global iLeak
    global iSheathExp, iFitEsat
    global Amag
    global eps
    global iRampType
    global TeMan, neMan, VsMan, VsManC, JsatMan
    global f_scale, loss
    data_df = pd.read_csv(data_file, header=None, sep='\t', names=['V', 'J', 'x'])
    Vpv = Vscale * data_df['V'].values
    Jp = -IpAtten * data_df['J'].values / (AreaP * Rc)
    xV = xScale * data_df['x'].values

    # plt.ion()
    load_plot_style()
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig1.set_size_inches(4.5, 3.5)
    iStartRamp, iEndRamp, lam, n_ramps = characterize_sawtooth(Vpv, ax1)
    # """
    # zero initial guesses for all vectors
    # vectors corresponding to Te, ne, and Vs obtained from manual fits to data
    # """
    TeMan = np.ones(n_ramps)
    neMan = np.zeros(n_ramps)
    VsMan = np.zeros(n_ramps)
    VsManC = np.zeros(n_ramps)
    JsatMan = np.zeros(n_ramps)
    """
    vectors corresponding to Te, ne, Vs, slope of J in eSat region, probe V at which eSat region begins, 
    esat to isat ratio, Jsat, and slope of Isat region of characteristic.
    
    Jsat is JsatFit corrected for possible slope in JsatFit
    
    JsatR is simple, robust fit to Jsat just averaging data in Isat region
    """
    TeFit = np.ones(n_ramps)
    neFit = np.zeros(n_ramps)
    VsFit = np.zeros(n_ramps)
    eslopeFit = np.zeros(n_ramps)
    VesatFit = np.zeros(n_ramps)
    esatoisat = np.zeros(n_ramps)
    JsatFit = np.zeros(n_ramps)
    JslopeFit = np.zeros(n_ramps)
    Jsat = np.zeros(n_ramps)
    JsatR = np.zeros(n_ramps)
    VsManC = np.zeros(n_ramps)

    TeFit_err = np.ones(n_ramps)
    neFit_err = np.zeros(n_ramps)
    VsFit_err = np.zeros(n_ramps)
    eslopeFit_err = np.zeros(n_ramps)
    VesatFit_err = np.zeros(n_ramps)
    esatoisat_err = np.zeros(n_ramps)
    JsatFit_err = np.zeros(n_ramps)
    JslopeFit_err = np.zeros(n_ramps)
    Jsat_err = np.zeros(n_ramps)
    JsatR_err = np.zeros(n_ramps)
    VsManC_err = np.zeros(n_ramps)

    # for simple Jsat calc, assume we're in Jsat for V < VJsatRmax
    VJsatRmax = 0.75 * np.min(Vpv)

    if iLeak:
        print(f"Data consists of {n_ramps} voltage sweeps")
        print("Input indices of sweeps to be fit for leakage current in the format 'index_start, index_end'")
        print("These should be in a region with no plasma signal")
        i_ramps_leak = parse_input_indices()
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        fig2.set_size_inches(4.5, 3.5)
        jp_leak_fit = fit_leakage_current(
            jp=Jp, start_ramp_ids=iStartRamp, leak_start_idx=i_ramps_leak[0],
            leak_end_idx=i_ramps_leak[1], lam=lam, ax=ax2
        )
        for iramp in range(n_ramps-1):
            iStart = iStartRamp[iramp]
            iEnd = iStart + lam
            for ii in range(iStart, iEnd):
                Jp[ii] = Jp[ii] - jp_leak_fit[ii-iStart]

    """
    Manual fit to characteristic
    """
    print(f"Data consists of {n_ramps} voltage sweeps")
    print("Input indices of sweeps to be fit manually in the format 'index_start, index_end'")
    i_ramps_man = parse_input_indices()

    fit_characeteristic_manually(
        v_pv=Vpv, jp=Jp, start_ramp_ids=iStartRamp, end_ramp_ids=iEndRamp, fit_start_idx=i_ramps_man[0]-1,
        fit_end_idx=i_ramps_man[1]-1, lam=lam, n_ramps=n_ramps
    )
    # print(neMan[i_ramps_man[0]-1:i_ramps_man[1]-1])

    """
    **************** Numerical fit to characteristic *******************
    Choose which voltage ramps are to be fit numerically
    """
    print(f"Data consists of {n_ramps} voltage sweeps")
    print("Input indices of sweeps to be fit numerically, in the format 'index_start, index_end'")
    i_ramps_fit = parse_input_indices()
    i_ramps_fit_start, i_ramps_fit_end = i_ramps_fit[0], i_ramps_fit[1]
    i_ramps_fit_start -= 1
    i_ramps_fit_end -= 1
    i_ramps_fit_start = max(i_ramps_fit_start,2)
    i_ramps_fit_end = min(i_ramps_fit_end, n_ramps-1)

    # Assume that the point i_ramps_man[0] is in the range i_ramps_fit_start:ii_ramps_fit_end
    # First, scan down in iramp from i_ramps_man[1] to i_ramps_fit_start
    tol = eps ** (2/3)
    # tol = eps ** 0.5
    # f_scale = 0.01
    for iramp in range(i_ramps_man[0]-1, i_ramps_fit_start-1, -1):
        if JsatMan[iramp] != 0: # an initial guess exists for this point
            # use manual fit to data as initial guess for numerical fit
            TeFit[iramp] = TeMan[iramp]
            JsatFit[iramp] = JsatMan[iramp]
            VsFit[iramp] = VsMan[iramp]
            VesatFit[iramp] = VsMan[iramp]
        else:
            # otherwise, use solution from previous point as initial guess
            TeFit[iramp] = TeFit[iramp+1]
            JsatFit[iramp] = JsatFit[iramp+1]
            VsFit[iramp] = VsFit[iramp+1]
            eslopeFit[iramp] = eslopeFit[iramp+1]
            VesatFit[iramp] = VesatFit[iramp+1]
            JslopeFit[iramp] = JslopeFit[iramp+1]

        if (TeFit[iramp] < 0.01) or (TeFit[iramp] > 30.):
            TeFit[iramp] = 1.

        # Now, fit presumed functional form for probe characteristic
        x0 = np.array([JsatFit[iramp], TeFit[iramp], VsFit[iramp], eslopeFit[iramp], VesatFit[iramp], JslopeFit[iramp]])
        iStart = iStartRamp[iramp]
        iEnd = iEndRamp[iramp]
        if iRampType > 1:
            iEnd = iStart + lam - 1
        # eliminate end points from fit because of occasional Jp spikes here
        Vpfit = Vpv[iStart+1:iEnd-1]
        JFit = Jp[iStart+1:iEnd-1]

        # bnds = [(-1E10, 1E10), (0.1, 40),(-100,100), (eps, 1E50), (-1E50, 1E50),(eps, 1E50)]
        # ls_res1 = differential_evolution(
        #     res_jfit_scalar, bnds, args=(Vpfit, JFit),
        #     updating='deferred', workers=-1, tol=eps**0.5, maxiter=10000
        # )

        ls_res1 = least_squares(
            fun=res_jfit, args=(Vpfit, JFit),
            x0=x0,
            loss=loss, f_scale=f_scale,
            bounds=bounds_fit,
            xtol=tol,
            ftol=tol,
            gtol=tol,
            verbose=0,
            x_scale='jac',
            jac='3-point',
            max_nfev=10000 * len(x0)
        )

        popt_jp1 = ls_res1.x
        ci = cf.confidence_interval(res=ls_res1)
        popt_jp1_delta = np.abs(ci[:,1] - popt_jp1)
        VsFit[iramp] = popt_jp1[2]
        TeFit[iramp] = popt_jp1[1]
        JsatFit[iramp] = popt_jp1[0]
        eslopeFit[iramp] = popt_jp1[3]
        VesatFit[iramp] = popt_jp1[4]
        JslopeFit[iramp] = popt_jp1[5]

        VsFit_err[iramp] = popt_jp1_delta[2]
        TeFit_err[iramp] = popt_jp1_delta[1]
        JsatFit_err[iramp] = popt_jp1_delta[0]
        eslopeFit_err[iramp] = popt_jp1_delta[3]
        VesatFit_err[iramp] = popt_jp1_delta[4]
        JslopeFit_err[iramp] = popt_jp1_delta[5]

        print(f"Numerically fitting iramp = {iramp:>4d}")
        print(f"Te = {TeFit[iramp]:>6.3g} eV, Jsat = {JsatFit[iramp]:>6.3g} A/cm^2, Vs = {VsFit[iramp]:>6.3g} V, "
              f"Vesat = {VesatFit[iramp]:>6.3g} V")

    # Then, scan up in iramp + 1 from iRampsMan(1) to iRampsFit(2)
    for iramp in range(i_ramps_man[0], i_ramps_fit_end+1):
        if neMan[iramp] != 0.: # if available, use manual fit to data as initial guess for numerical fit
            TeFit[iramp] = TeMan[iramp]
            JsatFit[iramp] = JsatMan[iramp]
            VsFit[iramp] = VsMan[iramp]
            VesatFit[iramp] = VsMan[iramp]
        else: # otherwise, use solution from previous point as initial guess
            TeFit[iramp] = TeFit[iramp - 1]
            JsatFit[iramp] = JsatFit[iramp - 1]
            VsFit[iramp] = VsFit[iramp - 1]
            eslopeFit[iramp] = eslopeFit[iramp - 1]
            VesatFit[iramp] = VesatFit[iramp - 1]
            JslopeFit[iramp] = JslopeFit[iramp - 1]

        if (TeFit[iramp] < 0.01) or (TeFit[iramp] > 30.):
            TeFit[iramp] = 1.

        # Now, fit presumed functional form for probe characteristic
        x0 = np.array([JsatFit[iramp], TeFit[iramp], VsFit[iramp], eslopeFit[iramp], VesatFit[iramp], JslopeFit[iramp]])
        iStart = iStartRamp[iramp]
        iEnd = iEndRamp[iramp]
        if iRampType > 1:
            iEnd = iStart + lam - 1
        # eliminate end points from fit because of occasional Jp spikes here
        Vpfit = Vpv[iStart + 1:iEnd - 1]
        JpFit = Jp[iStart + 1:iEnd - 1]

        # bnds = [(-1E10, 1E10), (0.1, 40), (-100, 100), (eps, 1E50), (-1E50, 1E50), (eps, 1E50)]
        # ls_res2 = differential_evolution(
        #     res_jfit_scalar, bnds, args=(Vpfit, JpFit),
        #     updating='deferred', workers=-1, tol=eps ** 0.5, maxiter=10000
        # )

        ls_res2 = least_squares(
            fun=res_jfit, args=(Vpfit, JpFit),
            x0=x0,
            loss=loss, f_scale=f_scale,
            bounds=bounds_fit,
            xtol=tol,
            ftol=tol,
            gtol=tol,
            verbose=0,
            x_scale='jac',
            jac='3-point',
            max_nfev=10000 * len(x0)
        )

        popt_jp2 = ls_res2.x
        ci = cf.confidence_interval(res=ls_res2)
        popt_jp2_delta = np.abs(ci[:, 1] - popt_jp2)

        VsFit[iramp] = popt_jp2[2]
        TeFit[iramp] = popt_jp2[1]
        JsatFit[iramp] = popt_jp2[0]
        eslopeFit[iramp] = popt_jp2[3]
        VesatFit[iramp] = popt_jp2[4]
        JslopeFit[iramp] = popt_jp2[5]

        VsFit_err[iramp] = popt_jp2_delta[2]
        TeFit_err[iramp] = popt_jp2_delta[1]
        JsatFit_err[iramp] = popt_jp2_delta[0]
        eslopeFit_err[iramp] = popt_jp2_delta[3]
        VesatFit_err[iramp] = popt_jp2_delta[4]
        JslopeFit_err[iramp] = popt_jp2_delta[5]


        print(f"Numerically fitting iramp = {iramp:>4d}")
        print(f"Te = {TeFit[iramp]:>6.3g} eV, Jsat = {JsatFit[iramp]:>6.3g} A/cm^2, Vs = {VsFit[iramp]:>6.3g} V, "
              f"Vesat = {VesatFit[iramp]:>6.3g} V")

    fig_fit, ax_fit = plt.subplots()
    fig_fit.set_size_inches(4.5, 3.5)
    ax_fit.set_xlabel('index')
    ax_fit.set_ylabel('J$_{\mathregular{p}}$')
    ax_fit.set_title('Probe current and fit')
    ax_fit.plot(Jp, marker='o', ms=6, mfc='none', mew=1.25, ls='none')
    for iramp in range(i_ramps_fit_start, i_ramps_fit_end+1):
        iStart = iStartRamp[iramp]
        iEnd = iEndRamp[iramp]

        count = 0
        JsatR[iramp] = 0. # average over region with sufficiently negative tip voltage
        for ii in range(iStart, iEnd):
            if Vpv[ii] <= VJsatRmax:
                count += 1
                JsatR[iramp] += Jp[ii]
        JsatR[iramp] /= count
        if iRampType > 1:
            iEnd = iStart + lam - 1
        Vpfit = Vpv[iStart:iEnd]

        popt_iramp = [JsatFit[iramp], TeFit[iramp], VsFit[iramp], eslopeFit[iramp], VesatFit[iramp], JslopeFit[iramp]]
        Jp_pred = model_jfit(popt_iramp, Vpfit)
        ax_fit.plot(np.arange(iStart, iEnd)+1, Jp_pred, color='red', ls='-', lw=1.25)

    ax_fit.set_xlim(1, len(Jp))
    ax_fit.set_ylim(Jp.min(), Jp.max())
    plt.show()

    # calculate density and average Jsat
    for iramp in range(i_ramps_fit_start, i_ramps_fit_end):
        Csfit = 9.79E5 * np.sqrt((Zion*TeFit[iramp]+Ti)/mi) # ion sound speed [cm/s]
        Csfit_err = estimate_cs_uncertainty(c_s=Csfit, T_e_uncertainty=TeFit_err[iramp])
        Jsat[iramp] = JsatFit[iramp] + JslopeFit[iramp] * (VsFit[iramp] - 3. * TeFit[iramp]) # use value well away from curve
        Jsat_err[iramp] = estimate_jsat_uncertainty(
            v_s_fit=VsFit[iramp], T_e_fit=TeFit[iramp], J_slope_fit=JslopeFit[iramp],
            j_sat_fit_delta=JsatFit_err[iramp], v_s_fit_delta=VsFit_err[iramp], T_e_fit_delta=TeFit_err[iramp],
            J_slope_fit_delta=JslopeFit_err[iramp]
        )
        neFit[iramp] = -Jsat[iramp] / (Amag * ee * Csfit) # density [cm^-3]
        neFit[iramp] = max(neFit[iramp], 0.)
        neFit_err[iramp] = estimate_ne_uncertainty(
            n_e=neFit[iramp], c_s=Csfit, j_sat=Jsat[iramp], c_s_uncertainty=Csfit_err, j_sat_uncertainty=JsatFit_err[iramp]
        )
        if iFitEsat == 1:
            esatoisat[iramp] = Jfac * np.exp((VesatFit[iramp]-VsFit[iramp])/TeFit[iramp]) - 1.
    # make vector xFit which has average position at each iramp
    xFit = np.zeros(n_ramps, dtype=float)
    xFit_err = np.zeros(n_ramps, dtype=float)
    confidence_level = 0.95
    alpha = 1. - confidence_level
    for iramp in range(n_ramps):
        iMean = int(round( (iStartRamp[iramp] + iEndRamp[iramp])/2. ))
        n_x = iEndRamp[iramp] - iStartRamp[iramp]
        dof = n_x = 1
        tval = t.ppf(1 - alpha/2., dof)
        x_std = np.std(xV[iStartRamp[iramp]:iEndRamp[iramp]], ddof=1)
        xFit[iramp] = xV[iMean]
        xFit_err[iramp] = x_std * tval / np.sqrt(n_x)

    # plot out plasma parameters vs. position
    iStart, iEnd = i_ramps_fit_start, i_ramps_fit_end
    iStartMan, iEndMan = i_ramps_man[0]-1, i_ramps_man[1] - 1
    idx = np.arange(n_ramps)
    msk_fit = np.where((iStart <= idx) & (idx <= iEnd))
    msk_man = np.where((iStartMan <= idx) & (idx <= iEndMan))
    xFitS = xFit[iStart:iEnd]
    neFitS = neFit[iStart:iEnd]
    TeFitS = TeFit[iStart:iEnd]
    JsatS = Jsat[iStart:iEnd]
    VsFitS = VsFit[iStart:iEnd]
    JsatRS = JsatR[iStart:iEnd]

    xFitS_err = xFit_err[iStart:iEnd]
    neFitS_err = neFit_err[iStart:iEnd]
    TeFitS_err = TeFit_err[iStart:iEnd]
    JsatS_err = Jsat_err[iStart:iEnd]
    VsFitS_err = VsFit_err[iStart:iEnd]

    xManS = xFit[msk_man]
    JManS = JsatMan[msk_man]
    neManS = neMan[msk_man]
    TeManS = TeMan[msk_man]
    VsManS = VsMan[msk_man]
    VsManCS = VsManC[msk_man]

    out_df = pd.DataFrame(data={
        'x (cm)': xFitS,
        'n_e (cm^{-3})': neFitS,
        'T_e (eV)': TeFitS,
        'J_sat (A/cm^2)': JsatS,
        'V_sc (eV)': VsFitS,
        'J_sat_robust (A/cm^2)': JsatRS,
        'x error (cm)': xFitS_err,
        'n_e error (cm^{-3})': neFitS_err,
        'T_e error (eV)': TeFitS_err,
        'J_sat error (A/cm^2)': JsatS_err,
        'V_sc error (eV)': VsFitS_err,
    })

    fig_pp, axes_pp = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    fig_pp.set_size_inches(7.5, 6.0)
    # numerical fit
    axes_pp[0, 0].errorbar(
        xFitS, neFitS / 1e11, xerr=xFitS_err, yerr=neFitS_err / 1E11, marker='o', color='C0', ls='none', mfc='none', mew=1.25,
        capsize=2.75,  ms=10, elinewidth=1.25,
    )
    axes_pp[0, 0].plot(xManS,neManS/1e11, marker='x', color='tab:red', ls='none', mfc='none', mew=1.25) # manual fit
    axes_pp[0, 0].set_xlabel('Position (cm)')
    axes_pp[0, 0].set_ylabel('$n_e$ (x 10$^{\mathregular{11}}$ cm$^{\mathregular{-3}}$)')
    axes_pp[0, 0].set_title('Density')

    axes_pp[0, 1].plot(xFitS, TeFitS, marker='o', color='C0', ls='none', mfc='none', mew=1.25) # numerical fit
    axes_pp[0, 1].plot(xManS, TeManS, marker='x', color='tab:red', ls='none', mfc='none', mew=1.25) # manual fit
    axes_pp[0, 1].set_xlabel('Position (cm)')
    axes_pp[0, 1].set_ylabel('$T_e$ (eV)')
    axes_pp[0, 1].set_title('Temperature')

    axes_pp[1, 0].plot(xFitS, VsFitS, marker='o', color='C0', ls='none', mfc='none', mew=1.25) # numerical fit
    axes_pp[1, 0].plot(xManS, VsManS, marker='x', color='tab:red', ls='none', mfc='none', mew=1.25) # manual fit
    axes_pp[1, 0].set_xlabel('Position (cm)')
    axes_pp[1, 0].set_ylabel('$V_{\mathregular{s}}$ (eV)')
    axes_pp[1, 0].set_title('Space-charge potential')

    axes_pp[1, 1].plot(xFitS, JsatS, marker='o', color='C0', ls='none', mfc='none', mew=1.25)  # numerical fit to Jsat
    axes_pp[1, 1].plot(xFitS, JsatRS, marker='o', color='tab:green', ls='none', mfc='none', mew=1.25) # robust Jsat fit
    axes_pp[1, 1].plot(xManS, JManS, marker='x', color='tab:red', ls='none', mfc='none', mew=1.25) # manual fit
    axes_pp[1, 1].set_xlabel('Position (cm)')
    axes_pp[1, 1].set_ylabel('$J_{\mathregular{sat}}$ (A/cm$^{\mathregular{2}}$)')
    axes_pp[1, 1].set_title('Ion saturation current density')

    """
    Save data and figures
    """
    file_basename = os.path.basename(data_file) # The name of the file
    file_tag = os.path.splitext(file_basename)[0]
    file_dir = os.path.dirname(data_file)
    folder = os.path.basename(file_dir)
    output_data_dir = os.path.join(file_dir, 'langprobe_results')
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    out_df.to_csv(os.path.join(output_data_dir, 'lang_results_'+file_tag+'.csv'), index=False, lineterminator='\n')
    fig_pp.savefig(os.path.join(output_data_dir, 'lang_results_'+file_tag+'.png'), dpi=600)

    plt.show()










if __name__ == '__main__':
    main()