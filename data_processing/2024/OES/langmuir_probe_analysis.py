import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from astropy.utils.metadata.utils import dtype
from dask.array import block
from scipy.fft import fft
import os
import json
import re
from scipy.optimize import least_squares, OptimizeResult
from matplotlib.lines import Line2D

data_file = "./data/PA_probe/20240815/gamma_ivdata0007.raw"


# ***************** Problem parameters **********************************

# Pisces-A probe geometry
# AreaP = .085 # probe tip area [cm^2], corresponds to 4*R*L, two direction projected area
# AreaP = .025 # probe tip area [cm^2], corresponds to 2*R*L, single direction projected area for probe near target
AreaP = .049 # probe tip area [cm^2] that Daisuke is using at present (10/2024) for gamma probe
Vscale = 100. # voltage division done in Langmuir probe box :  Vprobe = dataV(:,1)*Vscale
Rc = 20. # current resistor [Ohms] in Langmuir probe box : Iprobe = dataV(:,2)/Rc. Usually 2 Ohm or 5 Ohm
# xScale = 1.56 # probe plunge position conversion [cm/V]
xScale = 1.53 # probe plunge position conversion [cm/V] that Daisuke is using (10/2024)


Amag = 0.61 # Use Amag = 0.5 for magnetized plasma (rci^2 << Ap) ; 0.61 for unmagnetized plasma
mi = 2. # mi is ion mass in units of hydrogen mass. Typically use 4 for He+ and 2 for D+
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
        Vpp = np.max(v[0:2*lam]) - np.min(v[0:2*lam])
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
        ilo = max(ilo, 1)
        ihi = min(ihi, n_points)
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
        ilo = max(ilo, 1)
        ihi = min(ihi, n_points)
        if iRampType == 1:
            idum = np.argmax(v[ilo:ihi])
        else:
            idum = np.argmin(v[ilo:ihi])
        iEnd = ilo + idum
        vpp_meas = v[iEnd] - v[iStart] # amplitude of this sawtooth
        if iRampType > 1:
            vpp_meas *= -1.
        if vpp_meas > Vpp * 0.5 and vpp_meas < 1.5 * Vpp: # check to make sure this ramp looks reasonable
            iStartRamp.append(iStart) # characterize ramps by their start and end indices
            iEndRamp.append(iEnd)
            # print(f"{iramp:>5d}: {iStart:>5d}, {iEnd:>5d}")

            n_ramps += 1
        if iEnd > n_points - lam + 5: # exit loop
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
def fit_leakage_current(jp, start_ramp_ids, leak_start_idx, leak_end_idx, lam, ax, degree=10):
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
        loss='cauchy', f_scale=0.1,
        x0 = [0.01 ** k for k in range(degree+1)],
        args=(x, jp_leak),
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=2,
        x_scale='jac',
        max_nfev=10000 * len(x)
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

def on_pick(event, vp_fit, y_j_fit, jmin0, TeMan, JsatMan, neMan, VsMan, VsManC, id_ramp):
    global pick_events, x_picked, y_picked, popt_te_onpick, jsat_onpick
    global te_fit_l2d, js_fit_l2d
    global Zion, Amag, Ti, mi, ee, Tescale
    if isinstance(event.artist, Line2D):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        x_mean = np.mean(xdata[ind])
        y_mean = np.mean(ydata[ind])
        x_picked[pick_events] = x_mean
        y_picked[pick_events] = y_mean
        pick_events += 1
        fig = event.artist.get_figure()
        ax = fig.axes[0]
        if pick_events == 2:
            popt_te_onpick = fit_te_range_man(vp_fit, y_j_fit, x_picked[0:2])
            te_fit_l2d, = ax.plot(x_picked[0:2], model_poly(x_picked[0:2], popt_te_onpick), color='red')
            ax.set_title('Mouse select ion saturation current region', color='red')
            fig.canvas.draw()
            fig.canvas.flush_events()

        if pick_events == 4:
            jsat_onpick = fit_jsat_range_man(vp_fit, y_j_fit, x_picked[2:4], jmin0)
            js_fit_l2d, = ax.plot(x_picked[2:4], y_picked[2:4], color='red')
            fig.canvas.draw()
            fig.canvas.flush_events()
            # plt.draw()
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
                plt.close(fig)
            else:
                lines = ax.lines
                lines[-1].remove()
                lines[-1].remove()
                ax.set_title('Mouse select range for T$_{\mathregular{e}}$ slope finder', color='red')
                fig.canvas.draw()
                fig.canvas.flush_events()
            pick_events = 0


def fit_te_range_man(vp_fit, y_j_probe, vrange):
    global eps
    v_min = np.min(vrange)
    v_max = np.amax(vrange)
    msk_tf = (v_min <= vp_fit) & (vp_fit <= v_max)
    vp_t_fit = vp_fit[msk_tf]
    yj_t_fit = y_j_probe[msk_tf]
    tol = eps ** (1./3.)
    ls_res = least_squares(
        res_poly,
        loss='soft_l1', f_scale=0.1,
        x0=[vp_t_fit[0], 10],
        args=(vp_t_fit, yj_t_fit),
        xtol=tol,
        ftol=tol,
        gtol=tol,
        verbose=0,
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
    JsatMan = np.exp(yJave) - jmin0
    return JsatMan


def fit_characeteristic_manually(
        v_pv, jp, start_ramp_ids, end_ramp_ids, fit_start_idx, fit_end_idx, lam,
        n_ramps
):
    global iRampType

    """
    zero initial guesses for all vectors
    vectors corresponding to Te, ne, and Vs obtained from manual fits to data
    """
    n_ramps_to_fit = fit_end_idx - fit_start_idx
    TeMan = np.ones(n_ramps_to_fit)
    neMan = np.zeros(n_ramps_to_fit)
    VsMan = np.zeros(n_ramps_to_fit)
    VsManC = np.zeros(n_ramps_to_fit)
    JsatMan = np.zeros(n_ramps_to_fit)

    id_ramp = 0
    # step through each ramp and do manual fit to characteristic
    iramp = fit_start_idx
    for id_ramp in range(n_ramps_to_fit): # cycle through manual fits
        iStart = start_ramp_ids[iramp]
        iEnd = end_ramp_ids[iramp]
        if iRampType > 1:
            iStart = end_ramp_ids[iramp] #  manual fit to second leg of "V"
            iEnd = start_ramp_ids[iramp] + lam

        Vpfit = v_pv[iStart:iEnd]
        Jfit = jp[iStart:iEnd]

        Jmin = Jfit.min()
        Jmin0 = 1.1 * np.abs(Jmin) # offset probe current by 1.1*Jmin
        JpOffset = Jfit + Jmin0 # jp_offset is jp offset to be positive-definite
        yjprobe = np.log(JpOffset)
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        fig.set_size_inches(4.5, 3.5)
        line, = ax.plot(Vpfit, yjprobe, 'o', color='C0', mfc='none', label='Data', picker=True, pickradius=5)
        ax.set_xlabel(r"V$_{\mathregular{probe}}$")
        ax.set_ylabel(r"log J$_{\mathregular{probe}}$")
        ax.set_title(fr'Mouse select range for T$_{{\mathregular{{e}}}}$ slope finder ({id_ramp}/{n_ramps_to_fit})', color='red')
        fig.canvas.mpl_connect(
            'pick_event',
            lambda event: on_pick(event, Vpfit, yjprobe, Jmin0, TeMan, JsatMan, neMan, VsMan, VsManC, id_ramp)
        )
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=True)
    return TeMan, neMan, VsMan, VsManC, JsatMan


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

def main():
    global data_file, ee, rpe, Jfac, xScale, AreaP, Rc, Vscale, IpAtten
    global iLeak
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
    # TeMan = np.ones(n_ramps)
    # neMan = np.zeros(n_ramps)
    # VsMan = np.zeros(n_ramps)
    # VsManC = np.zeros(n_ramps)
    # JsatMan = np.zeros(n_ramps)
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

    # for simple Jsat calc, assume we're in Jsat for V < VJsatRmax
    VJsatRmax = 0.75 * np.min(Vpv)

    if iLeak:
        print(f"Data consists of {n_ramps} voltage sweeps")
        print("Input indices of sweeps to be fit for leakage current")
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
    print("Input indices of sweeps to be fit manually")
    i_ramps_man = parse_input_indices()
    TeMan, neMan, VsMan, VsManC, JsatMan = fit_characeteristic_manually(
        v_pv=Vpv, jp=Jp, start_ramp_ids=iStartRamp, end_ramp_ids=iEndRamp, fit_start_idx=i_ramps_man[0]-1,
        fit_end_idx=i_ramps_man[1]-1, lam=lam, n_ramps=n_ramps
    )
    print(neMan)






if __name__ == '__main__':
    main()