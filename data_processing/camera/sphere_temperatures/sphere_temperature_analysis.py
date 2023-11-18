"""
This code plots the individual sphere temperature vs time for a pebble sample subject to a laser heat load
"""
import pandas as pd
import numpy as np
from data_processing.utils import get_experiment_params, latex_float, lighten_color
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import json
from scipy.optimize import least_squares, OptimizeResult, differential_evolution
from mpl_toolkits.axes_grid1 import make_axes_locatable

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC'
save_dir = 'R4N85_stats'
tracking_csv = r'LCT_R4N85_manual_tracking.xlsx'
sheet_name = 'R4N85'
info_csv = r'LCT_R4N85_ROW375_100PCT_2023-08-18_1.csv'
calibration_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\CALIBRATION_20231010\calibration_20231010_4us.csv'

pixel_size = 20.4215  # pixels/mm
px2mm = 1. / pixel_size
px2cm = 0.1 * px2mm
center_mm = np.array([12.05, 30.78])

sensor_pixel_size_cm = 3.45E-4

CAMERA_ANGLE = 18.
CAMERA_DISTANCE = 5.

DEG2RAD = np.pi / 180.
RAD2DEG = 1. / DEG2RAD
GO2 = 0.5 * 9.82E2

X_MIN, X_MAX = -0.5, 0.5
Y_MIN, Y_MAX = -0.5, 0.5
THETA_MAX = 90.

OBJECTIVE_F_CM = 9.  # The focal length of the objective
WD_CM = 34.  # The working distance

all_tol = np.finfo(np.float64).eps
frame_rate = 200.


def load_plot_style():
    with open('../../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['thinLinePlotStyle']
    mpl.rcParams.update(plot_style)


def load_calibration(calibration_csv=calibration_csv):
    df = pd.read_csv(calibration_csv).apply(pd.to_numeric)
    return df['Temperature [K]'].values


def convert_to_temperature(adc, cal):
    if (type(adc) == list) or (type(adc) == np.ndarray):
        r = []
        for a in adc:
            r.append(cal[int(a)])
        return np.array(r)
    return cal[int(adc)]


class DictClass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def rotation_y(x: float, y: float, z: float, angle: float, radians: bool = True) -> DictClass:
    if not radians:
        angle = angle * DEG2RAD
    rot_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    zx = rot_matrix.dot(np.array([[z], [x]]))
    return DictClass(x=zx[1, 0], y=y, z=zx[0, 0])


def perspective(x, y, z, f: float) -> DictClass:
    pc = 10. * pixel_size
    ps = 1. / sensor_pixel_size_cm
    pspc = ps / pc
    wd = f * (1. + pspc)
    # m = f / (wd - f - z)
    m = f / (f * pspc - z) * pspc
    return DictClass(x=m * x, y=m * y, z=z)


def inverse_perspective(x, y, z, f) -> DictClass:
    a = f / (f - z)
    m = np.array([
        [f]
    ])


def get_particle_position(t: np.ndarray, initial_state: DictClass) -> DictClass:
    vs = initial_state.v0 * np.sin(initial_state.theta0)
    v0x = vs * np.cos(initial_state.phi0)
    v0y = vs * np.sin(initial_state.phi0)
    v0z = initial_state.v0 * np.cos(initial_state.theta0)
    x = initial_state.x0 + v0x * t
    z = v0z * t
    y = initial_state.y0 + v0y * t - GO2 * (t ** 2.)
    return DictClass(x=x, y=y, z=z)


def project_trajectory(particle_position: DictClass, angle: float, radians: bool = True) -> DictClass:
    if not radians:
        angle *= DEG2RAD
    x, y, z = particle_position.x, particle_position.y, particle_position.z
    # x -= 0.1*center_mm[0]
    n_particles = len(x)
    new_x, new_y, new_z = np.empty(n_particles), np.empty(n_particles), np.empty(n_particles)
    for i, xi, yi, zi in zip(range(n_particles), x, y, z):
        new_xyz = rotation_y(xi, yi, zi, angle)
        new_x[i] = new_xyz.x  # + 0.1*center_mm[0]
        new_y[i] = new_xyz.y
        new_z[i] = new_xyz.z
    return DictClass(x=new_x, y=new_y, z=new_z)


def model(t, v0, theta0, phi0, x0, y0):
    # x0 -= 0.1 * center_mm[0]
    rotated_initial = project_trajectory(DictClass(x=[x0], y=[y0], z=[0.]), angle=-CAMERA_ANGLE * DEG2RAD)
    initial_params = DictClass(v0=v0, theta0=theta0, phi0=phi0, x0=rotated_initial.x, y0=rotated_initial.y)
    tsim = t - t[0]
    position = get_particle_position(tsim, initial_params)
    rotated_position = project_trajectory(particle_position=position, angle=CAMERA_ANGLE * DEG2RAD)
    projected_position = perspective(x=rotated_position.x, y=rotated_position.y, z=rotated_position.z, f=OBJECTIVE_F_CM)
    # rotated_position.x += 0.1 * center_mm[0]
    return projected_position


def fobj(b, t, x0, y0, xy):
    xy_model = model(t=t, v0=10 ** b[0], theta0=b[1], phi0=b[2], x0=x0, y0=y0)
    res = (np.vstack([xy_model.x, xy_model.y]).T - xy)
    return res.flatten()


def de_loss(b, t, x0, y0, xy):
    r = fobj(b, t, x0, y0, xy)
    n = len(t)
    r = np.array([r[ii] * np.exp((ii) / n) for ii in range(n)])
    # return np.dot(r, r)
    return 0.5 * np.linalg.norm(r)


def fit_trajectory(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> OptimizeResult:
    x0, y0 = x[0] * np.cos(DEG2RAD * CAMERA_ANGLE), y[0]
    x0, y0 = x[0], y[0]
    xy = np.vstack([x, y]).T
    v_guess = np.linalg.norm(xy[1] - xy[0]) * frame_rate * 0.5
    # v_guess = 72.
    p0 = 45.
    last_x, last_y = x[-1], y[-1]
    if last_x > 0.:
        p0 = 45. if last_y > 0. else 315.
    if last_x < 0.:
        p0 = 135. if last_y > 0. else 225.
    q0 = 30. * DEG2RAD if y.max() * x.max() > 0. else 0.
    p0 *= DEG2RAD
    q0, p0 = 0., 0.
    b0 = np.array([np.log10(v_guess), q0, p0])
    n = len(x)
    tol = all_tol ** 0.5
    res0 = differential_evolution(
        func=de_loss,
        args=(t, x0, y0, xy),
        x0=b0,
        bounds=((-2., np.log10(200)), (0., THETA_MAX * DEG2RAD), (0., 360. * DEG2RAD)),
        maxiter=n * 1000000,
        tol=tol,
        atol=tol,
        workers=-1,
        updating='deferred',
        recombination=0.5,
        strategy='best1bin',
        mutation=(0.5, 1.5),
        init='sobol',
        polish=False,
        # disp=True
    )
    res = least_squares(
        fobj,
        res0.x,
        # loss='soft_l1', f_scale=0.1,
        jac='3-point',
        args=(t[::], x0, y0, xy[::]),
        bounds=([-3., 0., 0.], [3, THETA_MAX * DEG2RAD, 360. * DEG2RAD]),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        diff_step=all_tol,
        max_nfev=10000 * n,
        method='trf',
        x_scale='jac',
        verbose=0
    )
    return res0, res


def main():
    cal = load_calibration()
    file_tag = os.path.splitext(info_csv)[0]
    params = get_experiment_params(relative_path=base_path, filename=file_tag)
    pulse_length = float(params['Emission Time']['value'])
    sample_name = params['Sample Name']['value']
    tracking_df: pd.DataFrame = pd.read_excel(io=os.path.join(base_path, tracking_csv), sheet_name=sheet_name).apply(
        pd.to_numeric)
    pids = tracking_df['PID'].unique()
    n = len(pids)
    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=0, vmax=n)
    norm_v = mpl.colors.Normalize(vmin=0, vmax=1000)
    norm_v = mpl.colors.Normalize(vmin=0, vmax=1000)
    cmap_v = plt.get_cmap('jet')
    ejection_data = np.empty(n, dtype=np.dtype([
        ('PID', 'i'), ('t_0 [s]', 'd'), ('t_ejection [s]', 'd'), ('dt [s]', 'd'),
        ('T_0 [K]', 'd'), ('T_ejection [K]', 'd'), ('dT [K]', 'd'),
        ('v_ejection [cm/s]', 'd')
    ]))

    cooling_data = pd.DataFrame(columns=[
        'PID', 't [s]', 'x [cm]', 'y [cm]', 'T [K]', 'T_lb [K]', 'T_ub [K]', 'x_fit [cm]', 'y_fit [cm]'
    ])

    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
    fig.set_size_inches(3.5, 6.0)
    fig2, axes2 = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig2.set_size_inches(3.5, 4.25)
    fig3, axes3 = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig3.set_size_inches(3.5, 4.25)
    fig_t, ax_t = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig_t.set_size_inches(4.75, 3.)
    fig_th, ax_th = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    fig_th.set_size_inches(4.75, 3.)
    for k, p in enumerate(pids):
        pid_df = tracking_df.loc[tracking_df['PID'] == p]
        t = pid_df['t (s)'].values
        t0 = t[0]
        adc_raw = pid_df['Mean gray'].values
        adc_corrected = pid_df['Corrected gray'].values
        adc_delta = pid_df['95% corrected delta'].values
        adc_lb, adc_ub = adc_corrected - adc_delta, adc_corrected + adc_delta
        temperature_raw = convert_to_temperature(adc=adc_raw, cal=cal)
        temperature = convert_to_temperature(adc=adc_corrected, cal=cal)
        temperature_lb = convert_to_temperature(adc=adc_lb, cal=cal)
        temperature_ub = convert_to_temperature(adc=adc_ub, cal=cal)

        # for i in range(len(temperature)):
        #     print(f'T_lb: {temperature_lb[i]:>3.0f}, T: {temperature[i]:>3.0f}, T_ub: {temperature_ub[i]:>3.0f}')

        x = pid_df['x'].values
        y = pid_df['y'].values

        s = pid_df['d_proj (px)'].values
        msk_takeoff = s < 5.

        s_takeoff = s[msk_takeoff]
        t_takeoff = t[msk_takeoff]
        temp_takeoff = temperature[msk_takeoff]
        s_takeoff = s_takeoff[-1]
        t_takeoff = t_takeoff[-1]
        temp_takeoff = temp_takeoff[-1]

        # Get the position and temperature after the pebble has been ejected
        t_threshold = t_takeoff + 0. * 0.005
        msk_ejected = t >= t_threshold
        t_ejected = t[msk_ejected]
        x_ejected = x[msk_ejected]
        y_ejected = y[msk_ejected]
        temp_ejected = temperature[msk_ejected]
        temp_ejected_lb = temperature_lb[msk_ejected]
        temp_ejected_ub = temperature_ub[msk_ejected]

        v_ejection = np.nan

        if len(t_ejected) > 3:
            x_vector = px2cm * x_ejected - center_mm[0] * 0.1
            y_vector = px2cm * (1080 - y_ejected) - center_mm[1] * 0.1
            res_de, res_lq = fit_trajectory(t=(t_ejected - t_threshold), x=x_vector, y=y_vector)
            popt_de, popt_lq = res_de.x, res_lq.x
            xy_fit = model(t=t_ejected - t_threshold, v0=10. ** popt_lq[0], theta0=popt_lq[1], phi0=popt_lq[2],
                           x0=x_vector[0], y0=y_vector[0])
            v_ejection = 10. ** popt_lq[0]
            # for j in range(len(t_ejected)):
            cooling_data = pd.concat([
                cooling_data,
                pd.DataFrame(data={
                    'PID': [p for ii in range(len(t_ejected))],
                    't [s]': t_ejected - t_takeoff,
                    'x [cm]': x_vector,
                    'y [cm]': y_vector,
                    'T [K]': temp_ejected,
                    'T_lb [K]': temp_ejected_lb,
                    'T_ub [K]': temp_ejected_ub,
                    'x_fit [cm]': xy_fit.x,
                    'y_fit [cm]': xy_fit.y,
                })
            ])
            x_cm = px2cm * x - center_mm[0] * 0.1
            y_cm = px2cm * (1080 - y) - center_mm[1] * 0.1

            ax_t.plot(x_cm, y_cm, ls='-', color='tab:grey', lw=1.0)
            ax_t.plot(x_cm, y_cm, marker='o', color=cmap_v(norm_v(v_ejection)), fillstyle='full', ms=5, ls='none')
            ax_t.plot(xy_fit.x, xy_fit.y, marker='s', color=cmap_v(norm_v(v_ejection)), fillstyle='none', ms=5,
                      mew=0.75)
            cost_lq = np.linalg.norm(res_lq.fun)
            print(
                f'PID: {p:>2d}, v0: {10. ** popt_lq[0]:>7.1f} cm/s, theta0: {popt_lq[1] * RAD2DEG:>4.1f} deg, '
                f'phi0 = {popt_lq[2] * RAD2DEG:>5.1f} deg, points: {len(t_ejected):>2d}, cost: {cost_lq:>5.2E}'
            )

        ejection_data[k] = (
            p, t[0], t_takeoff, t_takeoff - t[0], temperature[0], temp_takeoff, temp_takeoff - temperature[0],
            v_ejection
        )
        axes[0].plot(t, adc_raw, lw=1.5, c='olive')
        axes[0].plot(t, adc_corrected, lw=1.5, c='C0')
        axes[1].fill_between(t, temperature_lb, temperature_ub, color=lighten_color(color='C0', amount=0.5))
        axes[1].plot(t, temperature, lw=1.5, color='C0')
        axes[1].plot([t_takeoff], [temp_takeoff], marker='o', fillstyle='none', color='k', mew=1.5)
        axes[2].plot(t, s, color='tab:purple')
        axes[2].plot([t_takeoff], [s_takeoff], marker='o', fillstyle='none', color='k', mew=1.5)

        axes2[0].fill_between(t - t0, temperature_lb, temperature_ub,
                              color=lighten_color(color=cmap(norm(k)), amount=0.5))
        axes2[0].plot(t - t0, temperature, lw=1.5, c=cmap(norm(k)))
        axes2[0].plot([t_takeoff - t0], [temp_takeoff], marker='o', fillstyle='none', color='k', mew=1.5)
        axes2[1].plot([t_takeoff - t0], [s_takeoff], marker='o', fillstyle='none', color='k', mew=1.5)
        axes2[1].plot(t - t0, s, color=cmap(norm(k)))

        yerr_neg = np.array([max(yy, 0.) for yy in (temp_ejected - temp_ejected_lb)])
        yerr_pos = temp_ejected_ub - temp_ejected
        axes3[0].errorbar(
            x=t_ejected - t_takeoff,
            y=temp_ejected,
            yerr=(yerr_neg, yerr_pos),
            ls='-', color=cmap(norm(k)), marker='o', ms=7, mew=1.25, mfc='none',
            capsize=2.75, elinewidth=1.25, lw=1.5,
        )

    # Plot the ejection temperature histogram
    num_bins = 5
    T_e, T_std = ejection_data['T_ejection [K]'].mean(), np.std(ejection_data['T_ejection [K]'])
    print(f'T_e: {T_e:.1f}, T_std: {T_std:.1f}')
    n, bins, patches = ax_th.hist(ejection_data['T_ejection [K]'], num_bins, density=True)
    # add a 'best fit' line
    v_fit = ((1 / (np.sqrt(2 * np.pi) * T_std)) * np.exp(-0.5 * (1 / T_std * (bins - T_e)) ** 2))
    ax_th.plot(bins, v_fit, '--')
    ax_th.set_xlabel('T [K]')
    ax_th.set_ylabel('Counts')
    ax_th.set_title('Ejection temperature')
    
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    for i, ax in enumerate(axes2):
        ax.set_xlim(0, 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    axes[0].set_ylabel('ADC')
    axes[1].set_ylabel('K')
    axes[2].set_ylabel(r'$\Delta$px')

    axes[0].set_ylim(0, 160)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(32))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(16))

    axes[1].set_ylim(2000, 3600)
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(400))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(200))

    axes2[0].set_ylim(2000, 3600)
    axes2[0].yaxis.set_major_locator(ticker.MultipleLocator(400))
    axes2[0].yaxis.set_minor_locator(ticker.MultipleLocator(200))

    axes[0].set_title('Gray value')
    axes[1].set_title('Pebble temperature')
    axes[2].set_title('Pebble instantaneous displacement')

    axes2[0].set_ylabel('K')
    axes2[1].set_ylabel(r'$\Delta$px')

    axes2[0].set_title('Pebble temperature')
    axes2[1].set_title('Pebble instantaneous displacement')
    axes[2].set_xlabel('Time [s]')
    axes2[1].set_xlabel('Time [s]')

    ax_t.set_xlabel('x [cm]')
    ax_t.set_ylabel('y [cm]')
    ax_t.set_title('Fitted trajectories')
    axes3[0].set_xlabel(r'$\Delta$t [s]')
    axes3[0].set_ylabel('T [K]')
    axes3[0].set_title('Pebble cooling rate')
    axes3[1].set_xlabel('T$_{\mathregular{ejected}}$ [K]')
    axes3[1].set_ylabel('v [cm/s]')
    axes3[1].set_title('Ejection velocity')
    ed = np.sort(ejection_data, order='dt [s]')
    axes3[1].plot(ed['T_ejection [K]'], ed['v_ejection [cm/s]'], marker='o', fillstyle='none', ls='none')

    ax_t.set_aspect('equal', adjustable='datalim', anchor='C')
    ax1_divider = make_axes_locatable(ax_t)
    cax1 = ax1_divider.append_axes("right", size="7%", pad="1%")
    sm = mpl.cm.ScalarMappable(norm=norm_v, cmap=cmap_v)
    cbar = fig_t.colorbar(sm, cax=cax1)
    cbar.set_label('\n$v_0$ [cm/s]', size=9)
    cbar.ax.set_ylim(0, 1000.)

    legend_elements0 = [Line2D([0], [0], color='olive', lw=1.5, label='Raw'),
                        Line2D([0], [0], color='C0', lw=1.5, label='Corrected')]
    legend_elements1 = [Line2D([0], [0], color='C0', label='Temperature'),
                        Patch(facecolor=lighten_color('C0', 0.5), edgecolor='C0', lw=1.5,
                              label='95% CI')]

    axes[0].legend(handles=legend_elements0, loc='lower right', fontsize=8)
    axes[1].legend(handles=legend_elements1, loc='lower right', fontsize=8)

    fig.savefig(os.path.join(base_path, save_dir, 'temperature_fulltime.png'), dpi=600)
    fig2.savefig(os.path.join(base_path, save_dir, 'temperature_relatvive.png'), dpi=600)
    fig3.savefig(os.path.join(base_path, save_dir, 'temperature_released.png'), dpi=600)
    fig_t.savefig(os.path.join(base_path, save_dir, 'trajectories.png'), dpi=600)
    fig_th.savefig(os.path.join(base_path, save_dir, 'ejection_temperatures_hist.png'), dpi=600)

    print(cooling_data)

    ejection_data_df = pd.DataFrame(data=ejection_data)
    ejection_data_df.to_csv(os.path.join(base_path, save_dir, 'ejection_data.csv'), index=False, encoding='utf-8-sig')
    cooling_data.to_csv(os.path.join(base_path, save_dir, 'cooling_data.csv'), index=False, encoding='utf-8-sig')

    plt.show()


if __name__ == '__main__':
    load_plot_style()
    main()
