import numpy as np


def clamp(value, low, high):
    return np.minimum(np.maximum(value, low), high)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


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
    Body-to-world rotation matrix using ZYX Euler angles:
        R = Rz(psi) @ Ry(theta) @ Rx(phi)
    """
    return rot_z(psi) @ rot_y(theta) @ rot_x(phi)


def euler_rate_matrix(phi, theta):
    """
    Maps body rates [p, q, r] to Euler angle rates [phi_dot, theta_dot, psi_dot].
    Valid away from theta = +/- pi/2.
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


def quad_dynamics(state, control, params, disturbance_model=None, t=0.0):
    """
    6-DoF rigid-body quadcopter dynamics.

    State:
        [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

    Control:
        [T, tau_x, tau_y, tau_z]

    Disturbance model:
        object with method get_force_torque(t, state) -> (force_world, torque_body)
    """
    m = params.mass
    g = params.g
    J = params.J

    pos = state[0:3]
    vel = state[3:6]
    phi, theta, psi = state[6:9]
    omega = state[9:12]

    T = control[0]
    tau = np.asarray(control[1:4], dtype=float)

    disturbance_force_world = np.zeros(3)
    disturbance_torque_body = np.zeros(3)

    if disturbance_model is not None:
        disturbance_force_world, disturbance_torque_body = disturbance_model.get_force_torque(t, state)

    R = euler_to_rotation(phi, theta, psi)

    # Translational dynamics
    thrust_world = R @ np.array([0.0, 0.0, T])
    gravity_world = np.array([0.0, 0.0, -g])

    pos_dot = vel
    vel_dot = (thrust_world + disturbance_force_world) / m + gravity_world

    # Rotational dynamics
    E = euler_rate_matrix(phi, theta)
    euler_dot = E @ omega

    omega_dot = np.linalg.solve(
        J,
        tau + disturbance_torque_body - np.cross(omega, J @ omega)
    )

    return np.concatenate([pos_dot, vel_dot, euler_dot, omega_dot])


def rk4_step(f, x, u, dt, params, disturbance_model=None, t=0.0):
    """
    One RK4 integration step.
    """
    k1 = f(x, u, params, disturbance_model=disturbance_model, t=t)
    k2 = f(x + 0.5 * dt * k1, u, params, disturbance_model=disturbance_model, t=t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, u, params, disturbance_model=disturbance_model, t=t + 0.5 * dt)
    k4 = f(x + dt * k3, u, params, disturbance_model=disturbance_model, t=t + dt)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Keep yaw wrapped for nicer plots
    x_next[8] = wrap_to_pi(x_next[8])

    return x_next


def simulate_step(state, control, dt, params, disturbance_model=None, t=0.0, enforce_ground=True):
    """
    Convenience wrapper around RK4 + simple ground contact.
    """
    next_state = rk4_step(
        f=quad_dynamics,
        x=state,
        u=control,
        dt=dt,
        params=params,
        disturbance_model=disturbance_model,
        t=t,
    )

    if enforce_ground and next_state[2] < 0.0:
        next_state[2] = 0.0
        if next_state[5] < 0.0:
            next_state[5] = 0.0

    return next_state