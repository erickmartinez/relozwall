import json
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, OptimizeResult, differential_evolution
from data_processing.utils import get_experiment_params
import re
import matplotlib.ticker as ticker

"""
Particles tracked with MTracJ:

https://imagescience.org/meijering/software/mtrackj/manual/
"""
import platform
from pathlib import Path
drive_path = r'/Users/erickmartinez/OneDrive' if platform.system() == 'Darwin' else r'C:\Users\erick\OneDrive'
data_path = r'Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC'
info_csv = r'LCT_R4N55_100PCT_2023-03-16_1.csv'
tracking_points_csv = r'LCT_R4N55_100PCT_2023-03-16_1_trackpoints_revised.csv'
frame_rate = 200.
pixel_size = 20.4215  # pixels/mm
p = re.compile(r'.*?-(\d+)\.jpg')
nmax = 200
calibration_csv = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\calibration\adc_calibration_curve.csv'
px2mm = 1. / pixel_size
px2cm = 0.1 * px2mm
center_mm = np.array([18.41, 26.77])
just_plot = True
data_dir = r'Documents\ucsd\Postdoc\research\data\firing_tests\SS_TUBE\GC\LCT_R4N55_100PCT_2023-03-16_1_fitting_results'
parameters_csv = r'fitted_params.csv'
trajectories_csv = r'fitted_trajectories.csv'

exclude_trajectories = []#[34, 36]
sensor_pixel_size_cm = 3.45E-4

CAMERA_ANGLE = 18.
CAMERA_DISTANCE = 5.

DEG2RAD = np.pi / 180.
RAD2DEG = 1. / DEG2RAD
GRAVITY = 9.82E2

X_MIN, X_MAX = -0.5, 0.5
Y_MIN, Y_MAX = -0.5, 0.5
THETA_MAX = 90

OBJECTIVE_F_CM = 9.  # The focal length of the objective
WD_CM = 34.  # The working distance

all_tol = np.finfo(np.float64).eps

if platform.system() == 'Darwin':
    data_dir = data_dir.replace("\\", os.path.sep)
    data_path = data_path.replace("\\", os.path.sep)

class DictClass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load_tracking_data(path_to_data=data_path):
    data_path = Path(drive_path).joinpath(path_to_data)
    df = pd.read_csv(
        os.path.join(data_path, tracking_points_csv), usecols=['TID', 'PID', 'x [pixel]', 'y [pixel]', 't [sec]']
    )
    return df.apply(pd.to_numeric)


def get_file_list(base_dir: str, tag: str):
    files = []
    for f in os.listdir(base_dir):
        if f.startswith(tag) and f.endswith('.jpg'):
            files.append(f)
    return files


def frame_id_time(list_of_files: list) -> np.ndarray:
    times_dict = {}
    pattern = re.compile(r'.*?-(\d+)-(\d+)\.jpg')
    for i, f in enumerate(list_of_files):
        m = pattern.match(f)
        fn = int(m.group(1))
        ts = float(m.group(2))
        times_dict[fn] = ts
        # print(f'f: {f}, fid: {fn:>3d}, ts: {ts}, t: {times_dict[fn]:5.3f} s')

    # sort list of files
    frame_keys = list(times_dict.keys())
    frame_keys.sort()

    time_from_timestamp = np.array([times_dict[i] for i in frame_keys]) * 1E-9

    t0 = time_from_timestamp[0]
    time_from_timestamp -= t0
    # print(time_from_timestamp)

    return time_from_timestamp


def get_time_from_images(fid: list, t_from_timestamp):
    return np.array([t_from_timestamp[int(i)] for i in fid])


def rotation_y(x: float, y: float, z: float, angle: float, radians: bool = True) -> DictClass:
    if not radians:
        angle = angle * DEG2RAD
    # rot_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    # zx = rot_matrix.dot(np.array([[z], [x]]))
    # return DictClass(x=zx[1, 0], y=y, z=zx[0, 0])
    rot_matrix = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0, np.cos(angle)]])
    xyz_prime = np.squeeze(rot_matrix @ np.array([x, y, z]))
    return DictClass(x=xyz_prime[0], y=xyz_prime[1], z=xyz_prime[2])



def perspective(x, y, z, f: float, wd=WD_CM, theta:float = CAMERA_ANGLE) -> DictClass:
    # pc = 10. * pixel_size # Multiply px/mm x 10 to get px/cm
    # ps = 1. / sensor_pixel_size_cm # <  2898.6 /cm
    # pspc = ps/pc # This is 1/M (M: magnification)
    M = pixel_size * 10 * sensor_pixel_size_cm / np.cos(np.radians(theta))# Magnification will be higher for objects closer to the lens
    m = f / (f/M - z) # Do not consider height inversion (already inverted by software)
    return DictClass(x=m * x, y=m * y, z=0)

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
    y = initial_state.y0 + v0y * t - 0.5 * 9.8E2 * (t ** 2.)
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
    rotated_initial = project_trajectory(DictClass(x=[x0], y=[y0], z=[0.]), angle=-CAMERA_ANGLE*DEG2RAD)
    initial_params = DictClass(v0=v0, theta0=theta0, phi0=phi0, x0=rotated_initial.x, y0=rotated_initial.y)
    tsim = t - t[0]
    position = get_particle_position(tsim, initial_params)
    rotated_position = project_trajectory(particle_position=position, angle=CAMERA_ANGLE * DEG2RAD)
    projected_position = perspective(x=rotated_position.x, y=rotated_position.y, z=rotated_position.z, f=OBJECTIVE_F_CM)
    # rotated_position.x += 0.1 * center_mm[0]
    return rotated_position


def fobj(b, t, x0, y0, xy):
    xy_model = model(t=t, v0=b[0], theta0=b[1], phi0=b[2], x0=x0, y0=y0)
    res = (np.vstack([xy_model.x, xy_model.y]).T - xy)
    return res.flatten()


def loss(b, t, x0, y0, xy):
    res = fobj(b, t, x0, y0, xy)
    return np.linalg.norm(res)


def fit_trajectory(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> OptimizeResult:
    x0, y0 = x[0], y[0]
    xy = np.vstack([x, y]).T
    # v_guess = np.linalg.norm(xy[1] - xy[0])  / (t[1] - t[0])
    v_guess = 1E-3
    p0 = 45.
    last_x, last_y = x[-1], y[-1]
    if last_x > 0.:
        p0 = 45. if last_y > 0. else 315.
    if last_x < 0.:
        p0 = 135. if last_y > 0. else 225.
    q0 = 90. * DEG2RAD #if y.max() * x.max() > 0. else 0.
    p0 *= DEG2RAD
    # q0, p0 = 0, 0.
    b0 = np.array([v_guess, q0, p0])
    n = len(x)
    tol = all_tol  # ** 0.5
    res0 = differential_evolution(
        func=loss,
        args=(t[::], x0, y0, xy[::]),
        x0=b0,
        bounds=((0., 100000.), (0., THETA_MAX * DEG2RAD), (0., 360. * DEG2RAD)),
        maxiter=n * 10000,
        tol=all_tol**0.5,
        # atol=all_tol,
        workers=-1,
        updating='deferred',
        recombination=0.2,
        popsize=150,
        strategy='best1bin',
        mutation=(0.5, 1.5),
        init='sobol',
        polish=False,
        # disp=True,
        # strategy='currenttobest1bin'
    )
    res = least_squares(
        fobj,
        res0.x,
        loss='soft_l1', f_scale=0.1,
        jac='3-point',
        args=(t[::], x0, y0, xy[::]),
        bounds=([0., 0., 0.], [100000., THETA_MAX * DEG2RAD, 360. * DEG2RAD]),
        xtol=tol**0.5,
        ftol=tol**0.5,
        gtol=tol**0.5,
        diff_step=2*all_tol,
        max_nfev=1000 * len(b0),
        method='trf',
        # x_scale='jac',
        # verbose=2
    )
    return res0, res


def main(data_path=data_path):
    data_path = Path(drive_path).joinpath(data_path)
    experiment_params = get_experiment_params(relative_path=data_path, filename=os.path.splitext(info_csv)[0])
    sample_id = experiment_params['Sample Name']['value']
    img_prefix = sample_id + '_IMG'
    trajectories_df = load_tracking_data()
    fitted_trajectories_df = pd.DataFrame(data={'FID':[], 'TID':[], 'PID':[], 't (s)':[], 'x (cm)':[], 'y (cm)':[]})
    max_time = trajectories_df['t [sec]'].max()
    trajectory_ids = trajectories_df['TID'].unique()
    v0_list = []
    theta0_list = []
    phi0_list = []
    loss_list = []
    v0_list_lq = []
    theta0_list_lq = []
    phi0_list_lq = []
    loss_list_lq = []
    tid_list = []
    dt = 1. / frame_rate
    file_tag = os.path.splitext(info_csv)[0]
    base_dir = os.path.join(data_path, file_tag + '_images')
    list_of_files = get_file_list(base_dir=base_dir, tag=img_prefix)
    t_from_timestamp = frame_id_time(list_of_files=list_of_files)
    # print('t_from_timestamp', t_from_timestamp.T)
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig_t, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig_t.set_size_inches(4.75, 3.)
    ax1_divider = make_axes_locatable(ax)
    cax1 = ax1_divider.append_axes("right", size="7%", pad="1%")


    # print(f'Max time: {max_time:.3f} s')

    # if just_plot:
    #     trajectories_df = pd.read_csv(os.path.join(data_dir, trajectories_csv)).apply(pd.to_numeric)
    #     fitted_params_df = pd.read_csv(os.path.join(data_dir, parameters_csv)).apply(pd.to_numeric)

    vmin, vmax = 1E3, -1E3
    for tid in trajectory_ids:
        points_df = trajectories_df[trajectories_df['TID'] == tid]
        # points_df = points_df[1::]  # first frame is usually just expansion (no kinematics here)
        x_vector = px2cm * points_df['x [pixel]'].values - center_mm[0] * 0.1
        n_points = len(x_vector)
        if n_points > 3 and tid not in exclude_trajectories:
            y_vector = px2cm * (1080 - points_df['y [pixel]'].values) - center_mm[1] * 0.1
            frame_id = np.array(points_df['t [sec]'].values * frame_rate, dtype=int)
            t = get_time_from_images(frame_id, t_from_timestamp=t_from_timestamp)
            res_de, res_lq = fit_trajectory(t=t, x=x_vector, y=y_vector)
            popt_de, popt_lq = res_de.x, res_lq.x
            xy_fit = model(t=t, v0=popt_lq[0], theta0=popt_lq[1], phi0=popt_lq[2], x0=x_vector[0], y0=y_vector[0])
            trajectory_df = pd.DataFrame(data={
                'FID': frame_id, 'TID': [tid for i in range(n_points)], 'PID': points_df['PID'],
                'x (cm)': xy_fit.x, 'y (cm)': xy_fit.y, 't (s)': t
            })
            fitted_trajectories_df = pd.concat((fitted_trajectories_df,trajectory_df), ignore_index=True).reset_index(drop=True)
            v0_list.append(popt_de[0])
            theta0_list.append(popt_de[1])
            phi0_list.append(popt_de[2])
            cost_de = np.linalg.norm(res_de.fun)
            loss_list.append(cost_de)

            v0_list_lq.append(popt_lq[0])
            theta0_list_lq.append(popt_lq[1])
            phi0_list_lq.append(popt_de[2])
            cost_lq = np.linalg.norm(res_lq.fun)
            loss_list_lq.append(cost_de)
            tid_list.append(tid)

            vmin = min(vmin, popt_lq[0])
            vmax = max(vmax, popt_lq[0])

            print(
                f'TID: {tid:>2d}, v0: {popt_lq[0]:>7.1f} cm/s, theta0: {popt_lq[1] * RAD2DEG:>4.1f} deg, '
                f'phi0 = {popt_lq[2] * RAD2DEG:>5.1f} deg, points: {n_points:>2d}, cost: {cost_lq:>5.2E}')
            # print(
            #     f'TID (DE): {tid:>2d}, v0: {popt_de[0]:>7.1f} cm/s, theta0: {popt_de[1] * RAD2DEG:>4.1f} deg, '
            #     f'phi0 = {popt_de[2] * RAD2DEG:>5.1f} deg, points: {n_points:>2d}, cost: {cost_de:>5.2E}')

            # alpha_factor = 1. / t.max()
            # if tid not in exclude_trajectories:
            #     ax.plot(x_vector, y_vector, ls='-', lw=1.0, color='tab:gray', alpha=0.5)
            #     for ti, xi, yi in zip(t, x_vector, y_vector):
            #         h1, = ax.plot(xi, yi, marker='s', ms=8, fillstyle='none', ls=':', lw=1.0, color=cmap(norm(popt_lq[0])),
            #                 alpha=alpha_factor * ti, label='Data')
            #
            #     for idx, ti, xi, yi in zip(range(len(t)), t, xy_fit.x, xy_fit.y):
            #         h2, = ax.plot(xi, yi, marker='o', ms=5, fillstyle='full', ls=':', lw=1.0, color=cmap(norm(popt_lq[0])),
            #                 mew=0.75, label='Fit')
            #         if idx == 0:
            #             ax.legend(loc='upper left', handles=[h1, h2], frameon=True)



    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title('Pebble trajectories')

    # ax.set_xlim(-2., 5.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    # ax.set_ylim(0, 5.5)
    ax.set_aspect('equal', adjustable='box', anchor='C')


    v0_list = np.array(v0_list, dtype=float)
    theta0_list = np.array(theta0_list)
    phi0_list = np.array(phi0_list)
    v0_list_lq = np.array(v0_list_lq)
    theta0_list_lq = np.array(theta0_list_lq)
    phi0_list_lq = np.array(phi0_list_lq)
    mean_velocity = v0_list_lq.mean()
    mean_theta = theta0_list_lq.mean() * RAD2DEG
    mean_phi = phi0_list_lq.mean() * RAD2DEG
    # print(f'Mean velocity: {mean_velocity:.1f} cm/s')
    fitted_params_df = pd.DataFrame(data={
        'TID': tid_list,
        'v0 de (cm/s)': v0_list,
        'theta0 de (deg)': theta0_list * RAD2DEG,
        'phi0 de (deg)': phi0_list,
        'cost de': cost_de,
        'v0 lq (cm/s)': v0_list_lq,
        'theta0 lq (deg)': theta0_list_lq * RAD2DEG,
        'phi0 lq (deg)': phi0_list_lq,
        'cost lq': cost_lq
    })
    print(fitted_params_df)

    norm = mpl.colors.Normalize(vmin=np.floor(vmax/50)*50, vmax=np.ceil(vmax/50)*50)
    cmap = plt.cm.jet

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig_t.colorbar(sm, cax=cax1)
    cbar.set_label('\n$v_0$ (cm/s)', size=9)
    cbar.ax.set_ylim(np.floor(vmax/50)*50, np.ceil(vmax/50)*50)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100.))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(50.))
    cbar.ax.tick_params(labelsize=8)

    for tid in trajectory_ids:
        if tid in fitted_trajectories_df['TID'].unique():
            trajectory_row = fitted_trajectories_df[fitted_trajectories_df['TID'] == tid]
            trajectory_params = fitted_params_df[fitted_params_df['TID'] == tid]
            points_df = trajectories_df[trajectories_df['TID'] == tid]
            x_data = px2cm * points_df['x [pixel]'].values - center_mm[0] * 0.1
            y_data = px2cm * (1080 - points_df['y [pixel]'].values) - center_mm[1] * 0.1
            t = trajectory_row['t (s)'].values
            x_fit = trajectory_row['x (cm)'].values
            y_fit = trajectory_row['y (cm)'].values
            v0 = trajectory_params['v0 lq (cm/s)'].values
            color = cmap(norm(v0))
            ax.plot(x_fit, y_fit, ls='-', lw=1.0, color=color, alpha=0.5)
            for i, xi, yi, ti in zip(range(len(x_data)), x_data, y_data, t):
                h1, = ax.plot(
                    xi, yi, marker='s', ms=8, fillstyle='none', ls='none', lw=1.0, color=color,
                    alpha=ti / t.max(), label='Data'
                )
            for i, xi, yi, ti in zip(range(len(x_fit)), x_fit, y_fit, t):
                h2, = ax.plot(
                    xi, yi, marker='o', ms=5, fillstyle='full', ls='-', lw=1.0, color=color,
                    alpha=ti / t.max(), label='Fit'
                )
                if  i == 0 :
                    ax.legend(loc='upper left', handles=[h1, h2], frameon=True)


    save_path = os.path.join(data_path, file_tag + '_fitting_results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fitted_trajectories_df['x (px)'] = fitted_trajectories_df['x (cm)'] * 10. * pixel_size + center_mm[0] * pixel_size
    fitted_trajectories_df['y (px)'] = 1080 - (10. * pixel_size * fitted_trajectories_df['y (cm)'] + center_mm[1] * pixel_size)
    fitted_trajectories_df.to_csv(os.path.join(save_path, 'fitted_trajectories.20250929.csv'), index=False)
    fitted_params_df.to_csv(os.path.join(save_path, 'fitted_params.20250929.csv'), index=False)

    fig, axes = plt.subplots(ncols=3, nrows=1, constrained_layout=True, frameon=True)
    fig.set_size_inches(8., 2.5)

    count_v, bins_v, _ = axes[0].hist(v0_list_lq, bins=50, density=True)
    count_q, bins_q, _ = axes[1].hist(theta0_list_lq * RAD2DEG, bins=50, density=True)
    count_p, bins_p, _ = axes[2].hist(phi0_list_lq * RAD2DEG, bins=50, density=True)

    center_v = (bins_v[:-1] + bins_v[1:]) / 2
    center_q = (bins_q[:-1] + bins_q[1:]) / 2
    center_p = (bins_p[:-1] + bins_p[1:]) / 2
    # print('len(count_v):', len(count_v))
    # print('count_v:')
    # print(count_v)
    # print('len(bins_v):', len(bins_v))
    # print('bins_v:')
    # print(bins_v)
    # print('len(center_v):', len(center_v))
    # print('center_v:')
    # print(center_v)
    idx_mode_v = count_v == count_v.max()
    idx_mode_q = count_q == count_q.max()
    idx_mode_p = count_p == count_p.max()
    mode_v = center_v[idx_mode_v][0]
    mode_q = center_q[idx_mode_q][0]
    mode_p = center_p[idx_mode_p][0]

    axes[0].set_xlabel('$v_0$ (cm/s)')
    axes[0].set_ylabel('Probability density')
    axes[0].set_title('Pebble ejection velocity')
    # axes[0].set_xlim(0, 1000)
    # axes[0].xaxis.set_major_locator(ticker.MultipleLocator(250))
    # axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(50))

    axes[1].set_xlabel('$\\theta_0$ (deg)')
    axes[1].set_ylabel('Probability density')
    axes[1].set_title('Azimuthal angle')
    axes[1].set_xlim(0, 90.)
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(30))
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(10))

    axes[2].set_xlabel('$\\varphi_0$ (deg)')
    axes[2].set_ylabel('Probability density')
    axes[2].set_title('Polar angle')
    axes[2].set_xlim(0, 360)
    axes[2].xaxis.set_major_locator(ticker.MultipleLocator(90))
    axes[2].xaxis.set_minor_locator(ticker.MultipleLocator(15))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # print(st.mode(v0_list))
    # results_txt = f'Average: {mean_velocity:.1f} cm/s\nMode: {st.mode(v0_list).mode[0]:.5f} cm/s'
    results_v_txt = f'Average: {mean_velocity:.1f} cm/s\nMode: {mode_v:.1f} cm/s'
    axes[0].text(
        0.95,
        0.95,
        results_v_txt,
        fontsize=9,
        # color='tab:green',
        transform=axes[0].transAxes,
        va='top', ha='right',
        bbox=props
    )

    results_q_txt = f'Average: {mean_theta:.1f} deg\nMode: {mode_q:.1f} deg'
    axes[1].text(
        0.95,
        0.95,
        results_q_txt,
        fontsize=9,
        # color='tab:green',
        transform=axes[1].transAxes,
        va='top', ha='right',
        bbox=props
    )

    results_p_txt = f'Average: {mean_phi:.1f} deg\nMode: {mode_p:.1f} deg'
    axes[2].text(
        0.95,
        0.95,
        results_p_txt,
        fontsize=9,
        # color='tab:green',
        transform=axes[2].transAxes,
        va='top', ha='right',
        bbox=props
    )

    fig.savefig(os.path.join(save_path, file_tag + '_velocity_histogram.20250929.png'), dpi=600)
    fig.savefig(os.path.join(save_path, file_tag + '_velocity_histogram.20250929.svg'), dpi=600)
    fig.savefig(os.path.join(save_path, file_tag + '_velocity_histogram.pdf'), dpi=600)

    fig_t.savefig(os.path.join(save_path, file_tag + '_particle_trajectories.20250929.png'), dpi=600)
    fig_t.savefig(os.path.join(save_path, file_tag + '_particle_trajectories.20250929.svg'), dpi=600)
    fig_t.savefig(os.path.join(save_path, file_tag + '_particle_trajectories.20250929.pdf'), dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
