import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from data_processing.utils import get_experiment_params, latex_float
from scipy.optimize import differential_evolution, least_squares
from scipy.special import voigt_profile
import matplotlib.ticker as ticker

chamber_volume = 31.57  # L
chamber_cube_length = 12. * 2.54E-2
gauge_position = 0.5 * chamber_cube_length

data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC'
data_csv = 'LCT_R4N55_100PCT_2023-03-16_1.csv'
pumpdown_data_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\laser_chamber_pumping_speed\turboDEGASSING_TRURBO_PUMPDOWN_2023-03-09_1'
sample_diameter = 0.9
beam_diameter = 0.8164

SF = 1. / np.sqrt(2. * np.pi)
pump_speed = 20.0


def aperture_factor(beam_diameter, sample_diameter):
    wz = 0.5 * beam_diameter
    r = 0.5 * sample_diameter
    return 1. - np.exp(-2. * (r / wz) ** 2.)


def multi_gaussian_model(t, peak_centers, peak_heights, peak_sigmas):
    n = len(peak_centers)
    y = np.zeros((len(t)), dtype=float)
    for i, c, h, s in zip(range(n), peak_centers, peak_heights, peak_sigmas):
        y += gaussian_model(t, c, h, s)
    return y


def gaussian_model(t, center, height, sigma):
    return np.array((height * SF / sigma) * np.exp(-0.5 * ((t - center) / sigma) ** 2.))


def shifted_mb(t, v0, amplitude, mass_amu, temperature, d):
    # 1 Dalton (amu) = 1.660539066E-27 kg
    # Kb = 1.380649E-23 J/K
    # (1/2) x (1.660539066E-27/1.380649E-23) =  0.601361774788E-4
    arg = (0.60136178E-4 / temperature) * mass_amu * (((d - v0*t)/t) ** 2.)
    g = (d / t) ** 2.
    return np.array(amplitude * g * np.exp(-arg))


def smb_derivative(v, v0, amplitude, mass_amu, temperature):
    """
    df/dv = Av^3 x (-m(v-v0)/kT) x exp(-m(v-v0)^2/2kT) + 3Av^2 x exp(-m(v-v0)^2/2kT)
    df/dv = (3/v -m(v-v0)/kT) x f
    """
    r = (3. / v - (1.20272E-4 / temperature) * mass_amu * (v - v0)) * shifted_mb(v, v0, amplitude, mass_amu, temperature)
    return r


def multi_smb_model(t, v0_values, amplitudes, masses, temperatures, d):
    n = len(v0_values)
    y = np.zeros((len(t)), dtype=float)
    for i, v0, a, m, T in zip(range(n), v0_values, amplitudes, masses, temperatures):
        y += shifted_mb(t, v0, a, m, T, d)
    return y


def gaussian_derivative(t, center, height, sigma):
    return np.array(- ((t - center) / (sigma ** 2.)) * gaussian_model(t, center, height, sigma))


def multi_voigt_model(t, peak_centers, peak_heights, peak_sigmas, peak_gammas):
    n = len(peak_centers)
    y = np.zeros((len(t)), dtype=float)
    for i, c, h, s, g in zip(range(n), peak_centers, peak_heights, peak_sigmas, peak_gammas):
        y += voigt_model(t, c, h, s, g)
    return y


def voigt_model(t, center, height, sigma, gamma):
    return height * voigt_profile(t - center, sigma, gamma)


def fobj(b, t, p):
    peak_centers = b[0::3]
    peak_heights = b[1::3]
    peak_sigmas = b[2::3]
    # peak_gammas = b[3::4]
    # print("Peak centers: ", peak_centers)
    y_model = multi_gaussian_model(
        t, peak_centers=peak_centers, peak_heights=peak_heights, peak_sigmas=peak_sigmas,  # peak_gammas=peak_gammas
    )
    return y_model - p


def fobj_smb(b, t, p, d):
    centers = b[0::4]
    amplitudes = b[1::4]
    masses = np.exp(b[2::4])
    temperatures = b[3::4]
    # print("Peak centers: ", peak_centers)
    y_model = multi_smb_model(
        t=t, v0_values=centers, amplitudes=amplitudes, masses=masses, temperatures=temperatures, d=d
    )
    return y_model - p


def loss(b, t, p):
    r = fobj(b, t, p)
    return 0.5 * np.dot(r.T, r)

def loss_smb(b, t, p, d):
    r = fobj_smb(b, t, p, d)
    return 0.5 * np.dot(r.T, r)


def main():
    file_tag = os.path.splitext(data_csv)[0]
    params = get_experiment_params(data_path, file_tag)
    data_df = pd.read_csv(os.path.join(data_path, data_csv), comment='#').apply(pd.to_numeric)
    laser_power = data_df['Laser output peak power (W)'].values
    time_s = data_df['Measurement Time (s)'].values
    pressure = 1000. * data_df['Pressure (Torr)'].values
    # Just take data after laser has been turned on
    msk_on = laser_power > 0.
    t_on = time_s[msk_on]
    t0 = t_on.min()
    idx_t0 = np.abs(time_s - t0).argmin()
    time_s = time_s[idx_t0::] - t0
    pressure = pressure[idx_t0::]
    laser_power = laser_power[idx_t0::]
    velocity = np.zeros_like(time_s)
    velocity[1::] = gauge_position / time_s[1::]

    sample_area = 0.25 * np.pi * sample_diameter ** 2.
    laser_mean_power = laser_power[laser_power > 0].mean()
    heat_load = laser_power * aperture_factor(beam_diameter, sample_diameter) / sample_area / 100.
    n = len(time_s)

    all_tol = np.finfo(np.float64).eps
    b0 = [1., pressure.max() * 0.1, 0.5, 0.5, 2.9, pressure.max() * 0.0125, 0.5, 0.5]
    b0 = [1., pressure.max() * 0.1, 0.5, 2.9, pressure.max() * 0.0125, 0.5, ]

    t01, t02 = 1.1, 2.9
    p0_1, p0_2 = 19.96, 4.75
    a01, a02 = p0_1*(t01/gauge_position)**2., p0_2*(t02/gauge_position)**2.
    v01, v02 = 0.1*gauge_position/t01, 0.1*gauge_position/t02
    b0_smb = [v01, a01, -4., 3500., v02, a02, 1., 2500.]

    print(f'Velocity range: [{velocity[1::].min():.3E},{velocity.max():.3E}] m/s')


    bounds_de = (
        (0., time_s.max()),
        (0., pressure.max()),
        (0., time_s.max()),
        # (0., 1000.),
        (0., time_s.max()),
        (0., pressure.max()),
        (0., time_s.max()),
        # (0., 1000.)
    )

    # bounds_de_smb = (
    #     (v01*0.9, v01*1.1),
    #     (a01*0.001, a01*10.),
    #     (-5, np.log(12.)),
    #     (80., 100000),
    #     (v02*0.9, v02*1.1),
    #     (a02*0.001, a02*10.),
    #     (1., np.log(56.)),
    #     (80., 100000),
    # )
    #
    # bounds_ls_smb = (
    #     (v01*0.9, a01*0.001,   -5, 80.,    v02*0.9, a02*0.001,   1., 80.),
    #     (v01*1.1, a01*10., np.log(12.), 100000., v02*1.1, a02*10., np.log(56.), 100000.)
    # )

    res0 = differential_evolution(
        func=loss, x0=b0,
        args=(time_s, pressure),
        bounds=bounds_de,
        maxiter=n * 1000000,
        tol=all_tol,
        atol=all_tol,
        workers=-1,
        updating='deferred',
        strategy='currenttobest1bin'
    )

    res = least_squares(
        fobj,
        res0.x,
        loss='soft_l1', f_scale=0.1,
        jac='3-point',
        args=(time_s, pressure),
        bounds=(
            [0, 0, 0, 0, 0, 0],
            [time_s.max(), pressure.max(), time_s.max(), time_s.max(), pressure.max(), time_s.max()]),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        diff_step=all_tol,
        max_nfev=10000 * n,
        method='trf',
        x_scale='jac',
        verbose=2
    )

    # res0_smb = differential_evolution(
    #     func=loss_smb, x0=b0_smb,
    #     args=(time_s[1::], pressure[1::], gauge_position),
    #     bounds=bounds_de_smb,
    #     maxiter=n * 1000000,
    #     tol=all_tol ** 0.5,
    #     atol=all_tol ** 0.5,
    #     workers=-1,
    #     updating='deferred',
    #     strategy='best1exp'
    # )

    # res_smb = least_squares(
    #     fobj_smb,
    #     res0_smb.x,
    #     # loss='soft_l1', f_scale=0.1,
    #     # jac='3-point',
    #     args=(time_s[1::], pressure[1::], gauge_position),
    #     bounds=bounds_ls_smb,
    #     xtol=all_tol,
    #     ftol=all_tol,
    #     gtol=all_tol,
    #     diff_step=all_tol,
    #     max_nfev=10000 * n,
    #     method='trf',
    #     x_scale='jac',
    #     verbose=2
    # )

    popt = res.x
    # popt_smb = res0_smb.x

    # popt_smb[1::4] = np.exp(popt_smb[1::4])
    # popt_smb[2::4] = np.exp(popt_smb[2::4])
    # print('Optimized values for res_smb:')
    # for i, pi in enumerate(popt_smb):
    #     # if i == 1 or i == 5 :
    #     #     popt_smb[i] = np.exp(popt_smb[i])
    #     print(f'popt[{i}] = {popt_smb[i]:.3E}')

    n_gaussians = int(len(b0) / 3)
    print('n_gaussians:', n_gaussians)
    ypred = np.zeros((n, n_gaussians), dtype=float)
    yderiv = np.zeros_like(ypred)

    n_smbs = int(len(b0_smb) / 4)
    print('n_smbs:', n_smbs)

    tpred = time_s[1::]
    ypred_smb = np.zeros((len(tpred), 2), dtype=float)
    # yderiv_smb = np.zeros((len(tpred), 2), dtype=float)


    for i in range(n_gaussians):
        k = 3 * i
        ypred[:, i] = gaussian_model(t=time_s, center=popt[k], height=popt[k + 1], sigma=popt[k + 2])
        yderiv[:, i] = gaussian_derivative(t=time_s, center=popt[k], height=popt[k + 1], sigma=popt[k + 2])

    # for i in range(2):
    #     k = 4 * i
    #     # shifted_mb(t, v0, amplitude, mass_amu, temperature, d)
    #     ypred_smb[:, i] = shifted_mb(
    #         t=tpred, v0=popt_smb[k], amplitude=popt_smb[k + 1], mass_amu=popt_smb[k + 2], temperature=popt_smb[k + 3],
    #         d=gauge_position
    #     )
        # yderiv_smb[:, i] = smb_derivative(
        #     v=tpred, v0=popt_smb[k], amplitude=popt_smb[k + 1], mass_amu=popt_smb[k + 2], temperature=popt_smb[k + 3]
        # )


    # print(ypred_smb)

    yderiv *= 1E-3 * chamber_volume * 1E4 / sample_area

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, axes = plt.subplots(ncols=1, nrows=2, constrained_layout=True)
    fig.set_size_inches(4.5, 5.5)

    fig_outgassing, axes_outgassing = plt.subplots(ncols=1, nrows=2, constrained_layout=True)
    fig_outgassing.set_size_inches(4., 5.0)

    colors = ['C0', 'tab:red']

    peak_colors = ['cadetblue', 'slateblue']

    axes[0].plot(
        time_s, pressure, c=colors[0], label='Pressure'
    )

    axes_outgassing[0].plot(
        time_s, pressure, c=colors[0], label='Pressure'
    )

    axes[0].plot(
        time_s, ypred[:, 0], c=peak_colors[0], lw=1.25, ls='-', label='Gaussian profile 1'
    )

    axes[0].plot(
        time_s, ypred[:, 1], c=peak_colors[1], lw=1.25, ls='-', label='Gaussian profile 2'
    )

    # axes[0].plot(
    #     time_s, np.sum(ypred[:], axis=1), c='tab:gray', lw=1.25, ls='-', label='Fitted pressure'
    # )
    #
    # axes_smb[0].plot(
    #     tpred, ypred_smb[:, 0], c=peak_colors[0], lw=1.25, ls='-', label='SMB profile 1'
    # )
    #
    # axes_smb[0].plot(
    #     tpred, ypred_smb[:, 1], c=peak_colors[1], lw=1.25, ls='-', label='SMB profile 2'
    # )

    q_out = pressure * pump_speed * 1E1 / sample_area
    axes_outgassing[1].plot(
        time_s, q_out, color='C1', label='Outgassing rate'
    )

    axes_outgassing[0].plot(
        time_s, ypred[:, 0], c=peak_colors[0], lw=1.25, ls=':', label='Gaussian profile 1'
    )

    axes_outgassing[0].plot(
        time_s, ypred[:, 1], c=peak_colors[1], lw=1.25, ls=':', label='Gaussian profile 2'
    )

    axes_outgassing[0].plot(
        time_s, np.sum(ypred,axis=1), c='tab:grey', lw=1.25, ls='-', label='Fit'
    )

    q_out1 = ypred[:, 0] * pump_speed * 1E1 / sample_area
    q_out2 = ypred[:, 1] * pump_speed * 1E1 / sample_area

    print(f'Q_out1 max: {q_out1.max():.0f} Torr-L/m^2-s')
    print(f'Q_out2 max: {q_out2.max():.0f} Torr-L/m^2-s')




    ax_laser = axes[0].twinx()

    ax_dp2 = axes[1].twinx()
    ax_laser.plot(time_s, heat_load, color=colors[1], lw=1., ls='--')

    ax_laser_outgassing = axes_outgassing[0].twinx()
    ax_laser_outgassing.plot(time_s, heat_load, color=colors[1], lw=1., ls='--')

    axes[1].plot(
        time_s, yderiv[:, 0], c=peak_colors[0], lw=1.25, ls='-', label='Gaussian profile 1'
    )

    ax_dp2.plot(
        time_s, yderiv[:, 1], c=peak_colors[1], lw=1.25, ls='-', label='Gaussian profile 2'
    )

    # axes[1].plot(
    #     time_s, np.sum(yderiv, axis=1), c='k', lw=1.25, ls='-', label='Sum'
    # )

    for ax in axes:
        ax.set_xlabel('Time (s)')
        ax.set_xlim(0, 4.5)

    for i, ax in enumerate(axes_outgassing):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.set_xlim(0, 4.5)
        ax.set_xlabel('Time (s)')

    axes[0].set_ylabel('$p$ (mTorr)', color=colors[0])
    axes[0].tick_params(axis='y', labelcolor=colors[0])
    ax_laser.set_ylabel('Heat load (MW/m$^{\mathregular{2}}$)', color=colors[1])
    ax_laser.tick_params(axis='y', labelcolor=colors[1])
    axes[0].set_title('Chamber pressure')
    axes[1].set_title('Outgassing rate')
    axes[1].ticklabel_format(axis='y', useMathText=True, scilimits=(-3,3))
    axes[1].set_ylim(-20000, 20000)
    ax_dp2.set_ylim(-2000, 2000.)
    ax_dp2.set_ylabel('$Q_{\mathrm{out, p2}}$ (Torr-L/m$^{\mathregular{2}}$-s)', color=peak_colors[1])
    ax_dp2.tick_params(axis='y', labelcolor=peak_colors[1])
    axes[1].tick_params(axis='y', labelcolor=peak_colors[0])

    axes[1].set_ylabel('$Q_{\mathrm{out}, p1}$ (Torr-L/m$^{\mathregular{2}}$-s)')
    # axes[1].set_yscale('symlog')
    axes[0].legend(
        loc='best', frameon=True
    )

    axes_outgassing[0].set_ylabel('$p$ (mTorr)', color=colors[0])
    axes_outgassing[0].tick_params(axis='y', labelcolor=colors[0])
    ax_laser_outgassing.set_ylabel('Heat load (MW/m$^{\mathregular{2}}$)', color=colors[1])
    ax_laser_outgassing.tick_params(axis='y', labelcolor=colors[1])
    ax_laser_outgassing.set_ylim(0, 50)

    axes_outgassing[0].set_title('Chamber pressure')
    axes_outgassing[1].set_title(f'Outgassing rate (S={pump_speed:.1f} L/s)')

    axes_outgassing[0].set_ylim(0, 25)
    axes_outgassing[1].ticklabel_format(axis='y', useMathText=True)
    axes_outgassing[1].set_ylabel('$Q_{\mathrm{out}}$ (Torr-L/m$^{\mathregular{2}}$-s)')
    axes_outgassing[1].ticklabel_format(axis='y', useMathText=True, scilimits=(-2,2))
    axes_outgassing[1].set_ylim(0, 7E3)
    axes_outgassing[0].yaxis.set_major_locator(ticker.MultipleLocator(5))
    axes_outgassing[0].yaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    axes_outgassing[1].yaxis.set_major_locator(ticker.MultipleLocator(2000))
    axes_outgassing[1].yaxis.set_minor_locator(ticker.MultipleLocator(1000))

    axes_outgassing[0].legend(
        loc='best', frameon=True, fontsize=10
    )

    fig.savefig(os.path.join(data_path, file_tag + '_outgassing_gaussians.png'), dpi=600)
    fig.savefig(os.path.join(data_path, file_tag + '_outgassing_gaussians.svg'), dpi=600)

    fig_outgassing.savefig(os.path.join(data_path, file_tag + '_outgassing_pumping_speed.png'), dpi=600)
    fig_outgassing.savefig(os.path.join(data_path, file_tag + '_outgassing_pumping_speed.svg'), dpi=600)

    plt.show()


if __name__ == '__main__':
    main()
