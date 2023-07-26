import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as manimation

base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\pebble_simulations'
file_tag = 'simulated_projection'

rectangle_wh_cm = np.array([12., 1.])
rectangle_origin = np.array([0., 1., 0.])

DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi
GO2 = 0.5 * 9.82E2

X_MIN, X_MAX = -0.5, 0.5
Y_MIN, Y_MAX = -0.5, 0.5
THETA_MAX = 60.

CAMERA_ANGLE = 18.
OBJECTIVE_F_CM = 9.  # The focal length of the objective


# pixel_size = 20.8252  # pixels/mm
pixel_size = 20.4215  # pixels/mm
px2mm = 1. / pixel_size

w = 1440
h = 1080

OBJECTIVE_F_CM = 9.  # The focal length of the objective
WD_CM = 33.3  # The working distance

sensor_pixel_size_cm = 3.45E-4


class DictClass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def perspective(x, y, z, f: float) -> DictClass:
    pc = 10. * pixel_size
    ps = 1. / sensor_pixel_size_cm
    dz = f * (1. + ps / pc)
    # m = f / (dz - f - z)
    m = f / (f - z)  # / sensor_pixel_size_cm / (pixel_size * 10.)
    return DictClass(x=m * x, y=m * y, z=z)


def rotation_y(x: float, y: float, z: float, angle: float, radians: bool = True) -> DictClass:
    if not radians:
        angle = angle * DEG2RAD
    rot_matrix = np.array([
        [np.cos(angle), 0., np.sin(angle)],
        [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)],
    ])
    rotated = rot_matrix.dot(np.array([x, y, z]).reshape(3, 1))
    return DictClass(x=rotated[0], y=rotated[1], z=rotated[2])


def get_rectangle_vertex(origin, rectangle_dimensions) -> np.ndarray:
    w, h = rectangle_dimensions
    w = np.array([0., 0., w])
    h = np.array([0., h, 0.])
    p1 = origin
    p2 = origin + w
    p3 = origin + w + h
    p4 = p1 + h
    return np.array([p1, p2, p3, p4])


def main():
    vertices = get_rectangle_vertex(origin=10. * rectangle_origin, rectangle_dimensions=10. * rectangle_wh_cm)
    rotated_vertices = np.empty_like(vertices)
    print('Shape of \'vertices\':', vertices.shape)
    print('Shape of \'rotated_vertices\':', rotated_vertices.shape)

    angle = CAMERA_ANGLE * DEG2RAD
    for i, v in enumerate(vertices):
        rv = rotation_y(x=v[0], y=v[1], z=v[2], angle=angle)
        # pv = perspective(x=rv.x, y=rv.y, z=rv.z, f=OBJECTIVE_F_CM)
        rotated_vertices[i] = np.array([rv.x, rv.y, rv.z]).reshape((3,))

    print('Vertices:', vertices)
    print('Projected vertices:', rotated_vertices)

    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    fig, axes = plt.subplots(ncols=1, nrows=3, constrained_layout=True)
    fig.set_size_inches(4., 7.)

    w, h = 10. * rectangle_wh_cm
    rv_origin = [rotated_vertices[0, 2], rotated_vertices[0, 1]]
    rw = rotated_vertices[1, 0] - rotated_vertices[0, 0]
    rh = rotated_vertices[2, 1] - rotated_vertices[1, 1]
    rz = rotated_vertices[1, 2] - rotated_vertices[0, 2]
    z1_zx = np.array([0., np.sin(angle) * rw])
    x1_zx = np.array([0., np.cos(angle) * rw])
    z2_zx = np.array([0., np.cos(angle) * rz])
    x2_zx = np.array([0., -np.sin(angle) * rz])

    axes[1].plot(
        [0, w], [0, 0], c='tab:blue'
    )

    axes[1].plot(
        z1_zx, x1_zx, c='tab:green', ls='-', lw=1.25
    )

    axes[1].plot(
        z2_zx, x2_zx, c='tab:green', ls=':', lw=1.25
    )

    rect_zy = mpl.patches.Rectangle(
        10. * rectangle_origin, w, h, lw=1., ec='b', fc='tab:blue', alpha=0.5
    )
    axes[0].add_patch(rect_zy)

    rect_zy2 = mpl.patches.Rectangle(
        rv_origin, rw, rh, lw=1., ec='tab:green', fc='tab:green', alpha=0.5
    )
    axes[2].add_patch(rect_zy2)

    ms = 6
    for i, vr, v in zip(range(len(rotated_vertices)), rotated_vertices, vertices):
        axes[1].plot(
            v[2], v[0], marker='s', color='b',  # alpha=1,
            fillstyle='none', ms=ms, ls='none', label='Original'
        )
        axes[0].plot(
            v[2], v[1], marker='s', color='b',  # alpha=1,
            fillstyle='none', ms=ms, ls='none', label='Original'
        )

        axes[2].plot(
            vr[0], vr[1], marker='o', color='r',  # alpha=1,
            fillstyle='none', ms=ms, ls='none', label='Projected'
        )

        # ax.plot(
        #     [rotated_vertices[i:i+1].x, rotated_vertices[i:i+1].y, marker='o', color='tab:gray',  # alpha=1,
        #     fillstyle='none', ms=ms, ls='none', label='Projected'
        # )

        # ax.plot(
        #     vertices[i:i + 1,0], vertices[i:i + 1,1], marker='o', color='tab:gray',  # alpha=1,
        #     fillstyle='none', ms=ms, ls='none', label='Projected'
        # )

    axes[1].set_title('ZX')
    axes[1].set_xlabel('z (mm)')
    axes[1].set_ylabel('x (mm)')
    axes[0].set_title('ZY')
    axes[0].set_xlabel('z (mm)')
    axes[0].set_ylabel('y (mm)')
    axes[0].set_ylim(0, 60)
    axes[2].set_title(f'XY (rotated $\\theta$ = {angle * RAD2DEG:.1f} deg)')
    axes[2].set_xlabel('x\' (mm)')
    axes[2].set_ylabel('y (mm)')
    axes[2].set_ylim(0, 60)

    axes[2].set_xlim(axes[0].get_xlim())

    for i, ax in enumerate(axes):
        ax.axis('equal')

    fig.savefig(os.path.join(base_path, file_tag + '.svg'), dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
