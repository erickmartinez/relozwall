import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, OptimizeResult
from data_processing.utils import get_experiment_params, latex_float, lighten_color
from scipy.stats.distributions import t
import data_processing.confidence as cf

chamber_volume = 31.57  # L
chamber_cube_length = 12. * 2.54E-2
gauge_position = 0.5 * chamber_cube_length

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC'
data_csv = 'LCT_R4N55_100PCT_2023-03-16_1.csv'
pumpdown_data_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\laser_chamber_pumping_speed\turbo\DEGASSING_TRURBO_PUMPDOWN_2023-03-09_1.csv'
out_dir = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\manuscripts\paper1\submission_JAP\rev1\figures'
sample_diameter = 0.9
beam_diameter = 0.8164

min_pressure_mt = 1.0


def aperture_factor(beam_diameter, sample_diameter):
    wz = 0.5 * beam_diameter
    r = 0.5 * sample_diameter
    return 1. - np.exp(-2. * (r / wz) ** 2.)


def load_plot_style():
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)



def main():
    # Create a file tag
    file_tag = os.path.splitext(data_csv)[0]
    # Read the pressure data during laser exposure
    data_df = pd.read_csv(os.path.join(base_path, data_csv), comment='#').apply(pd.to_numeric)
    laser_power = data_df['Laser output peak power (W)'].values
    time_sample = data_df['Measurement Time (s)'].values
    pressure_sample = 1000. * data_df['Pressure (Torr)'].values
    # Just take data after laser has been turned on
    msk_on = laser_power > 0.
    t_on = time_sample[msk_on]
    t0 = t_on.min()
    idx_t0 = np.abs(time_sample - t0).argmin()
    time_sample = time_sample[idx_t0::] - t0
    pressure_sample = pressure_sample[idx_t0::]
    laser_power = laser_power[idx_t0::]

    # change sample minimum pressure to the gauge detection limit:
    pg0 = pressure_sample[pressure_sample > 0]
    pressure_sample[pressure_sample == 0] = pg0.min()

    sample_area = 0.25 * np.pi * sample_diameter ** 2.
    laser_mean_power = laser_power[laser_power > 0].mean()
    heat_load = laser_power * aperture_factor(beam_diameter, sample_diameter) / sample_area / 100.
    mean_laser_heat_load = laser_mean_power * aperture_factor(beam_diameter, sample_diameter) / sample_area / 100.
    print(f'Mean laser heat load: {mean_laser_heat_load:.1f} MW/m^2')


    """
    Get the chamber pumpdown data
    """
    pumpdown_df = pd.read_csv(pumpdown_data_csv, comment='#').apply(pd.to_numeric)
    measurement_time_pumpdown = pumpdown_df['Measurement Time (h)'].values * 3600.0
    pressure_pumpdown = pumpdown_df['Pressure (Torr)'].values * 1000.

    # Remove pumpdown data equal or lower than zero Torr
    idx_pump_down_gt0 = pressure_pumpdown > 0.
    measurement_time_pumpdown = measurement_time_pumpdown[idx_pump_down_gt0]
    pressure_pumpdown = pressure_pumpdown[idx_pump_down_gt0]
    # Find the range of measurements before the vacuum pump started running
    p0 = pressure_pumpdown[0]
    idx_p0 = len(pressure_pumpdown) - len(pressure_pumpdown[pressure_pumpdown < p0]) - 1
    print(f"Starting index: {idx_p0}")
    t0 = measurement_time_pumpdown[idx_p0]
    measurement_time_pumpdown = measurement_time_pumpdown[idx_p0::] - t0
    pressure_pumpdown = pressure_pumpdown[idx_p0::]
    # Interpolate the pressure-time curve
    f_p = interp1d(pressure_pumpdown, measurement_time_pumpdown)
    f_t = interp1d(measurement_time_pumpdown, pressure_pumpdown)

    """
    Model to find the best t_pumpdown that matches the equilibrium portion of the pressure plot
    """
    def model(t, b):
        return f_t(t+b[0])

    def fobj(b, t, p):
        return np.log(model(t, b)) - np.log(p)

    # Get the the pressure from the laser experiments at times t = 1.5 and 4 s
    t_test1, t_test2 = 1.5, 4.2  # seconds
    idx1, idx2 = (np.abs(time_sample - t_test1)).argmin(), (np.abs(time_sample - t_test2)).argmin()
    # Get the pressure at t_test1 and t_test2
    p_t1, p_t2 = pressure_sample[idx1], pressure_sample[idx2]
    # find the corresponding time it takes for the chamber to pumpdown to that pressure:
    t_pd1, t_pd2 = f_p(p_t1), f_p(p_t2)
    # Find the time at which the laser shuts off
    msk_off = ~(laser_power > 0)
    t_off = time_sample[msk_off]
    t1 = t_off.min()
    t_pd_1i = t_pd1 - (t_test1 - t1)  # The initial time for pumpdown curve 1
    t_pd_2i = t_pd2 - (t_test2 - t1)  # The initial time for pumpdown curve 2
    t_pd_1f = t_pd1 + time_sample.max() - t_test1  # The final time for pumpdown curve 1
    t_pd_2f = t_pd2 + time_sample.max() - t_test2  # The final time for pumpdown curve 2
    t_pd_pred1 = np.linspace(t_pd_1i, t_pd_1f, 500)
    t_pd_pred2 = np.linspace(t_pd_2i, t_pd_2f, 500)
    t_pred = np.linspace(t1, time_sample.max(), 500)
    p_pd_pred1 = f_t(t_pd_pred1)
    p_pd_pred2 = f_t(t_pd_pred2)

    # Estimate outgassing rates for both cases:
    xy0 = np.array([0, t1])
    xy1 = np.array([pressure_sample[0], p_pd_pred1[0]])
    xy2 = np.array([pressure_sample[0], p_pd_pred2[0]])
    # dpdt1 = np.diff(xy1)[0] / t1
    # dpdt2 = np.diff(xy2)[0] / t1
    # # multiply dp/dt by volume divide by area an convert mTorr to Torr:
    # area = 0.25 * np.pi * sample_diameter ** 2.
    # q_out1 = chamber_volume * dpdt1 * area * 10.
    # q_out2 = chamber_volume * dpdt2 * area * 10.
    #
    # print(f'Outgassing using point at {t_test1:.1f}s: {q_out1:.0f} Torr-L/m^2-s')
    # print(f'Outgassing using point at {t_test2:.1f}s: {q_out2:.0f} Torr-L/m^2-s')

    """
    Fit the model to a range around t_test2
    """
    t_test3, t_test4 = 4.1, time_sample.max()
    idx3, idx4 = (np.abs(time_sample - t_test3)).argmin(), (np.abs(time_sample - t_test4)).argmin()
    # Get the pressure at t_test1 and t_test2
    p_t3, p_t4 = pressure_sample[idx3], pressure_sample[idx4]
    # find the corresponding time it takes for the chamber to pumpdown to that pressure:
    t_pd3, t_pd4 = f_p(p_t3), f_p(p_t4)
    tfit = time_sample[idx3:idx4+1]
    pfit = pressure_sample[idx3:idx4+1]
    n = len(tfit)
    print(f'Number of points to be fitted: {n}')

    all_tol = np.finfo(np.float64).eps
    res: OptimizeResult = least_squares(
        fun=fobj, x0=[t_pd3], args=(tfit, pfit), jac='3-point', xtol=all_tol, gtol=all_tol, ftol=all_tol,
        verbose=2, max_nfev=10000*n, x_scale='jac',
        # bounds=(0.001 * t_pd3, 100. * t_pd4),
        # loss='soft_l1', f_scale=0.1,
        diff_step=all_tol,
    )

    popt = res.x
    pcov = cf.get_pcov(res)
    var = pcov[0, 0]
    rr = fobj(popt, tfit, pfit)
    SS_res = np.dot(rr.T, rr)
    s = np.sqrt((SS_res / (n - 2.)))
    conf = 0.95
    alpha = 1.0 - conf  # significance
    q = t.ppf(1.0 - alpha / 2.0, n - 2)
    ypred = model(tfit, popt)

    t_pred_fit = np.linspace(t_off.min(), time_sample.max(), 500)
    p_pred_fit = model(t_pred_fit, popt)
    # t_pred_bar = t_pred_fit.mean()
    # t_pred_s = t_pred_fit.std()
    # dy = q * s * np.sqrt(1. + 1./n + ((t_pred_fit - t_pred_bar) ** 2.) / ((n - 1.) * t_pred_s) + var**2.)
    # Stdev of an individual measurement
    var_n = 2.
    se = np.sqrt(np.sum((pfit - ypred) ** 2) / (n - var_n) )
    # Auxiliary definitions
    sx = (t_pred_fit - tfit.mean()) ** 2
    sxd = np.sum((tfit - tfit.mean()) ** 2)
    # Prediction band
    dy = q * se * np.sqrt(1.0 + (1.0 / n) + (sx / sxd))
    # print('t_pred.shape:', t_pred.shape)
    # print('p_pred_fit.shape:', p_pred_fit.shape)
    # print('dy.shape:', dy.shape)
    print(f't_pd2: {t_pd2:.3E}')
    print(f'popt[0]: {popt[0]:.3E}')
    print(f's: {s:.3E}')
    print(f't_student: {q:.3f}')

    # Estimate outgassing rates for both cases:
    xy0 = np.array([0, t1])
    xy1 = np.array([pressure_sample[0], p_pd_pred1[0]])
    xy2 = np.array([pressure_sample[0], p_pred_fit[0]])
    dpdt1 = np.diff(xy1)[0] / t1
    dpdt2 = np.diff(xy2)[0] / t1
    # multiply dp/dt by volume divide by area an convert mTorr to Torr:
    area = 0.25 * np.pi * sample_diameter ** 2.
    q_out1 = chamber_volume * dpdt1 * area * 10.
    q_out2 = chamber_volume * dpdt2 * area * 10.

    dA = area * 0.5
    dt = 0.03

    dq_out2 = np.linalg.norm(np.array([
        (q_out2/t1)*dt, (q_out2/area)*dA, (chamber_volume/area/t1) * dy[0]
    ]))


    print(f'Outgassing using point at {t_test1:.1f}s: {q_out1:.0f} Torr-L/m^2-s')
    print(f'Outgassing using point at {t_test2:.1f}s: {q_out2:.0f}±{dq_out2:.1f} Torr-L/m^2-s')
    error_pct = 100. * dq_out2 / q_out2
    print(f'Outgassing error percentage: {error_pct:.2f} %')



    load_plot_style()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 2.5)

    ax_laser = ax.twinx()


    ax.fill_between(t_pred_fit, p_pred_fit - dy, p_pred_fit + dy, color=lighten_color('tab:purple', 0.5), zorder=0)
    pd_line2 = ax.plot(t_pred_fit, p_pred_fit, color='tab:purple', ls='--', lw=1.25)
    pr_line2 = ax.plot(xy0, xy2, color='tab:purple', ls=':', lw=1.25)
    ax_laser.plot(time_sample, heat_load, color='r', lw=1., ls='--', zorder=0)

    ax.plot(time_sample, pressure_sample, zorder=1, c='C0')
    ax.plot(tfit, ypred, zorder=2, c='tab:green', lw=1.0)

    scp20 = ax.plot([t1], [p_pred_fit[0]], ls='none', marker='o', mfc='none', c='k', ms=6)
    # scp21 = ax.plot([t_test2], [p_t2], ls='none', marker='o', mfc='k', c='k', ms=6)



    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure (mTorr)')
    ax_laser.set_ylabel('Heat load (MW/m$^{\\mathregular{2}}$)', color='red')
    ax_laser.tick_params(axis='y', labelcolor='r')

    q_out1_txt = rf'${{\mathregular{{Q_{{\mathrm{{out}} }} }} }}$ = {q_out1:.0f} Torr-L/m$^{{\mathregular{{2}}}}$-s'
    q_out2_txt = rf'${{\mathregular{{Q_{{\mathrm{{out}} }} }} }}$ = {q_out2:.0f} Torr-L/m$^{{\mathregular{{2}}}}$-s'
    correction_factor = q_out2 / q_out1
    print(f'Correction factor: {correction_factor:.3f}x')

    txt2 = ax.text(
        t1, p_pd_pred2[0] * 1.1, q_out2_txt,
        va='bottom', ha='left', fontsize=11, color='tab:purple',
        # transform=ax.transAxes,
    )

    ax.set_xlim(0, time_sample.max())
    ax.set_ylim(0, 25)

    # Put heat load plot behind pressure plot
    ax.set_zorder(ax.get_zorder() + 1)
    ax_laser.set_zorder(ax_laser.get_zorder() - 1)
    ax.patch.set_visible(False)

    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))

    fig.savefig(os.path.join(out_dir, 'laser_heat_outgassing.png'), dpi=600)
    fig.savefig(os.path.join(out_dir, 'laser_heat_outgassing.pdf'), dpi=600)
    fig.savefig(os.path.join(out_dir, 'laser_heat_outgassing.svg'), dpi=600)

    pd_line1 = ax.plot(t_pred, p_pd_pred1, color='tab:brown')
    pr_line1 = ax.plot(xy0, xy1, color='tab:brown', ls=':', lw=1.25)
    scp10 = ax.plot([t1], [p_pd_pred1[0]], ls='none', marker='o', mfc='none', c='k', ms=6)
    scp11 = ax.plot([t_test1], [p_t1], ls='none', marker='o', mfc='k', c='k', ms=6)

    txt1 = ax.text(
        t1, p_pd_pred1[0] * 0.8, q_out1_txt,
        va='top', ha='left', fontsize=11, color='tab:brown',
        # transform=ax.transAxes,
    )

    # Save figure with justification on the factor for the editor
    fig.savefig(os.path.join(out_dir, 'laser_heat_outgassing_editor.png'), dpi=600)
    fig.savefig(os.path.join(out_dir, 'laser_heat_outgassing_editor.svg'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
