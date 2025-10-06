import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as manimation

data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\pebble_simulations'
file_tag = 'simulated_trajectories'

n_particles = 200
t_max = 1.
dt = 0.005

DEG2RAD = np.pi / 180.
GO2 = 0.5 * 9.82E2

X_MIN, X_MAX = -0.5, 0.5
Y_MIN, Y_MAX = -0.5, 0.5
THETA_MAX = 60.

CAMERA_ANGLE = 18.
center = np.array([15.0, 4.5])

pixel_size = 20.8252  # pixels/mm
px2mm = 1. / pixel_size

w = 1440
h = 1080


class DictClass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def rotation_y(x: float, y: float, z: float, angle: float, radians: bool = True) -> DictClass:
    if not radians:
        angle = angle * DEG2RAD
    rot_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    zx = rot_matrix.dot(np.array([[z], [x]]))
    return DictClass(x=zx[1, 0], y=y, z=zx[0, 0])


def generate_initial_ensemble(n_particles: int, center_x: float, center_y: float, **kwargs) -> list:
    theta_max = kwargs.get('theta_min', THETA_MAX) * DEG2RAD
    v_mean = kwargs.get('v_mean', 60.)  # cm/s
    v_std = kwargs.get('v_std', 200.)  # cm/s
    sigma_x = (X_MAX - X_MIN) / 3.
    sigma_y = (Y_MAX - Y_MIN) / 3.
    x0_list = np.random.normal(loc=center_x, scale=sigma_x, size=n_particles)
    y0_list = np.random.normal(loc=center_y, scale=sigma_y, size=n_particles)
    theta0 = np.random.uniform(low=0., high=theta_max, size=n_particles)
    phi0 = np.random.uniform(low=0., high=360., size=n_particles) * DEG2RAD
    v0 = np.abs(np.random.normal(loc=v_mean, scale=v_std, size=n_particles))

    ensamble = DictClass(x0=x0_list, y0=y0_list, v0=v0, theta0=theta0, phi0=phi0)
    return ensamble


def get_ensamble_position(t: float, ensemble: DictClass) -> DictClass:
    vs = ensemble.v0 * np.sin(ensemble.theta0)
    v0x = vs * np.cos(ensemble.phi0)
    v0y = vs * np.sin(ensemble.phi0)
    v0z = ensemble.v0 * np.cos(ensemble.theta0)
    x = ensemble.x0 + v0x * t
    z = v0z * t
    y = ensemble.y0 + v0y * t - GO2 * t ** 2.
    return DictClass(x=x, y=y, z=z)


def rotate_ensemble(ensamble_position: DictClass, angle: float, radians: bool = True) -> DictClass:
    if not radians:
        angle *= DEG2RAD
    x, y, z = ensamble_position.x, ensamble_position.y, ensamble_position.z
    n_particles = len(x)
    new_x, new_y, new_z = np.empty(n_particles), np.empty(n_particles), np.empty(n_particles)
    for i, xi, yi, zi in zip(range(n_particles), x, y, z):
        new_xyz = rotation_y(xi, yi, zi, -angle)
        new_x[i] = new_xyz.x
        new_y[i] = new_xyz.y
        new_z[i] = new_xyz.z
    return DictClass(x=new_x, y=new_y, z=new_z)


def update_line(n: int, ln: list, particle_ensemble: DictClass, t_sim: float):
    t = t_sim[n]
    t_txt = f'{t:05.4f} s'
    data_1x, data_1y = ln[1].get_xdata(), ln[1].get_ydata()
    data_2x, data_2y = ln[2].get_xdata(), ln[2].get_ydata()
    ln[0].set_data(data_1x, data_1y)
    ln[1].set_data(data_2x, data_2y)
    current_positions = get_ensamble_position(t, particle_ensemble)
    current_positions = rotate_ensemble(ensamble_position=current_positions, angle=CAMERA_ANGLE * DEG2RAD)
    ln[2].set_data(current_positions.x, current_positions.y)
    ln[3].set_text(t_txt)
    return ln


def main(n_particles: int, t_max: float, w: int, h: float):
    particle_ensemble = generate_initial_ensemble(n_particles=n_particles, center_x=center[0], center_y=center[1])
    positions = get_ensamble_position(t=0, ensemble=particle_ensemble)

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(4.5, 3.)
    rotated_xyz0 = rotate_ensemble(ensamble_position=positions, angle=CAMERA_ANGLE * DEG2RAD)
    ms = 6
    ln1, = ax.plot(
        rotated_xyz0.x, rotated_xyz0.y, marker='o', color='b', alpha=0.1,
        fillstyle='none', ms=ms, ls='none'
    )

    time_steps = int(t_max / dt) + 1
    t_sim = dt * np.arange(0, time_steps)

    ln2, = ax.plot(
        rotated_xyz0.x, rotated_xyz0.y, marker='o', color='b', alpha=0.5,
        fillstyle='none', ms=ms, ls='none'
    )

    ln3, = ax.plot(
        rotated_xyz0.x, rotated_xyz0.y, marker='o', color='b', alpha=1.0,
        fillstyle='none', ms=ms, ls='none'
    )

    w = w * px2mm
    h = h * px2mm
    ax.set_xlim(0, w)
    ax.set_ylim(center[1] - 0.5*h, 0.5*h + center[1])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    # plt.axis('off')
    time_txt = ax.text(
        0.95, 0.95, '0.0000 s',
        horizontalalignment='right',
        verticalalignment='top',
        color='r',
        transform=ax.transAxes,
        fontsize=8
    )
    line = [ln1, ln2, ln3, time_txt]



    metadata = dict(title=f'Pebble release simulation', artist='Erick',
                    comment=f'frame rate: {1./dt}')
    writer = FFMpegWriter(fps=10, metadata=metadata)

    ani = manimation.FuncAnimation(
        fig, update_line, interval=100,
        repeat=True, frames=np.arange(0, time_steps, 1),
        fargs=(line, particle_ensemble, t_sim)
    )

    fig_h, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    fig_h.set_size_inches(10.0, 3.0)
    axes[0].hist(particle_ensemble.v0, bins=20)
    axes[0].set_xlabel('$v$ (cm/s)')
    axes[0].set_ylabel('Counts')
    axes[0].set_title('Pebble ejection velocity')
    axes[0].set_xlim(0, 500)

    axes[1].hist(particle_ensemble.theta0*180./np.pi, bins=20)
    axes[1].set_xlabel('$\\theta$ (deg)')
    axes[1].set_ylabel('Counts')
    axes[1].set_title('Pebble ejection azimuthal angle')
    axes[1].set_xlim(0, 60.)

    axes[2].hist(particle_ensemble.phi0*180./np.pi, bins=20)
    axes[2].set_xlabel('$\\varphi$ (deg)')
    axes[2].set_ylabel('Counts')
    axes[2].set_title('Pebble ejection polar angle')
    axes[2].set_xlim(0, 360.)
    # fig_h.savefig(os.path.join(data_path, file_tag + '_histograms.png'), dpi=600)

    plt.show()

    ft = file_tag + '_movie.mp4'
    save_dir = data_path  # os.path.dirname(base_path)
    # ani.save(os.path.join(save_dir, ft), writer=writer, dpi=200)  # dpi=pixel_size*25.4)


if __name__ == '__main__':
    main(n_particles=n_particles, t_max=t_max, w=w, h=h)
