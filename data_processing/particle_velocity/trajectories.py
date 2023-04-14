import numpy as np

DEG2RAD = np.pi / 180.
GO2 = 0.5 * 9.82E-2


class DictClass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def cartesian_to_spherical(x: float, y: float, z: float) -> DictClass:
    if x == 0. and y == 0. and z == 0.:
        return DictClass(rho=np.nan, theta=np.nan, phi=np.nan)
    rho = np.linalg.norm([x, y, z])
    r = np.linalg.norm([x, y])
    rz = r / z
    yx = y / x
    theta, phi = np.nan, np.nan
    if z < 0.:
        theta = np.pi + np.arctan(rz)
    elif z == 0.:
        theta = 0.5 * np.pi
    elif z > 0.:
        theta = np.arctan(rz)

    if x < 0.:
        phi = np.arctan(yx) - np.pi if y < 0. else np.arctan(yx) + np.pi
    elif x == 0.:
        if y < 0.:
            phi = -np.pi
        elif y == 0.:
            raise Warning(f'Underfined value of phi for x = 0 and y = 0.')
        elif y > 0.:
            phi = np.pi
    elif x > 0.:
        phi = np.arctan(yx)

    return DictClass(rho=rho, theta=theta, phi=phi)


def spherical_to_cartesian(rho: float, theta: float, phi: float, radians: bool = True) -> DictClass:
    if not radians:
        theta *= DEG2RAD
        phi *= DEG2RAD
    rs = rho * np.sin(theta)
    x = rs * np.cos(phi)
    y = rs * np.sin(phi)
    z = rho * np.cos(phi)
    return DictClass(x=x, y=y, z=z)


def rotation_y(x: float, y: float, z: float, angle: float, radians: bool = True) -> DictClass:
    if not radians:
        angle = angle * DEG2RAD
    rot_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    zx = rot_matrix.dot(np.array([[z], [x]]))
    return DictClass(x=zx[1, 0], y=y, z=zx[0, 0])


def particle_position(t: float, x0: float, y0: float, z0: float, v0: float, theta: float, phi: float,
                      radians: bool = True) -> DictClass:
    if not radians:
        theta *= DEG2RAD
        phi *= DEG2RAD
    vs = v0 * np.sin(theta)
    v0x = vs * np.cos(phi)
    v0y = vs * np.sin(phi)
    v0z = v0 * np.cos(theta)
    x = x0 + v0x * t
    z = z0 + v0z * t
    y = y0 + v0y * t - GO2 * t ** 2.
    return DictClass(x=x, y=y, z=z)
