import numpy as np


def rot_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rot_z(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def euler_to_rotation(phi, theta, psi):
    """
    Body-to-world rotation matrix with ZYX Euler convention.
    """
    return rot_z(psi) @ rot_y(theta) @ rot_x(phi)


def euler_rate_matrix(phi, theta):
    """
    Maps body angular velocity [p, q, r] to Euler angle rates.
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    tth = np.tan(theta)

    if abs(cth) < 1e-3:
        cth = 1e-3 if cth >= 0.0 else -1e-3

    return np.array([
        [1.0, sphi * tth, cphi * tth],
        [0.0, cphi, -sphi],
        [0.0, sphi / cth, cphi / cth],
    ])


def rotation_to_euler(R):
    """
    Extract ZYX Euler angles [phi, theta, psi] from a rotation matrix.
    """
    R = np.asarray(R, dtype=float)

    theta = np.arcsin(-R[2, 0])

    if abs(np.cos(theta)) < 1e-6:
        phi = 0.0
        psi = np.arctan2(-R[0, 1], R[1, 1])
    else:
        phi = np.arctan2(R[2, 1], R[2, 2])
        psi = np.arctan2(R[1, 0], R[0, 0])

    return np.array([phi, theta, psi], dtype=float)