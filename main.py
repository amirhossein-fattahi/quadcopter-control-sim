"""
Minimal first version of a quadcopter simulation project in Python.

Features:
- 6-DoF quadcopter dynamics
- Cascaded position/attitude controller
- Waypoint tracking
- Time-history plots
- 3D animation

This version is intentionally compact and learnable.
It uses Euler angles, so it is best for moderate roll/pitch angles.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from trajectories.waypoint import get_default_waypoint_trajectory
from trajectories.hover import get_default_hover_trajectory
from trajectories.circle import get_default_circular_trajectory
from environments.indoor import get_default_indoor_environment
from environments.outdoor import get_windy_outdoor_environment
from config.drone_params import get_default_drone_params
from config.sim_params import get_default_sim_params
from config.controller_params import get_default_controller_gains
from controllers.pid import CascadedPIDController
from dynamics.rigid_body import (
    rk4_step,
    quad_dynamics,
    wrap_to_pi,
    euler_to_rotation,
)


# =========================
# Helpers
# =========================
def clamp(value, low, high):
    return np.minimum(np.maximum(value, low), high)

'''
### REMOVE LATER ###
def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
'''

def rot_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rot_z(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

'''
### REMOVE LATER ###
def euler_to_rotation(phi, theta, psi):
    """
    Body-to-world rotation matrix using ZYX convention:
    R = Rz(psi) * Ry(theta) * Rx(phi)
    """
    return rot_z(psi) @ rot_y(theta) @ rot_x(phi)
'''

def euler_rate_matrix(phi, theta):
    """
    Maps body rates [p, q, r] to Euler angle rates [phi_dot, theta_dot, psi_dot].
    Valid away from theta = +/- pi/2.
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    tth = np.tan(theta)

    # Avoid numerical issues near singularities
    if abs(cth) < 1e-3:
        cth = np.sign(cth) * 1e-3 if cth != 0 else 1e-3

    return np.array([
        [1.0, sphi * tth, cphi * tth],
        [0.0, cphi, -sphi],
        [0.0, sphi / cth, cphi / cth]
    ])

"""
### REMOVE LATER ###
def rk4_step(f, x, u, dt, params):
    """One Runge-Kutta 4 integration step."""
    k1 = f(x, u, params)
    k2 = f(x + 0.5 * dt * k1, u, params)
    k3 = f(x + 0.5 * dt * k2, u, params)
    k4 = f(x + dt * k3, u, params)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
"""

# =========================
# Parameters
# =========================
@dataclass
class QuadcopterParams:
    mass: float = 1.0
    g: float = 9.81
    arm_length: float = 0.22

    # Inertia matrix
    Jx: float = 0.02
    Jy: float = 0.02
    Jz: float = 0.04

    # Actuator limits
    thrust_min: float = 0.0
    thrust_max: float = 20.0
    tau_x_max: float = 1.0
    tau_y_max: float = 1.0
    tau_z_max: float = 0.5

    # Safety / controller limits
    max_tilt_rad: float = np.deg2rad(25.0)

    @property
    def J(self):
        return np.diag([self.Jx, self.Jy, self.Jz])

    @property
    def tau_max(self):
        return np.array([self.tau_x_max, self.tau_y_max, self.tau_z_max])


@dataclass
class ControllerGains:
    # Outer loop (position)
    kp_pos: np.ndarray
    kd_pos: np.ndarray

    # Inner loop (attitude)
    kp_att: np.ndarray
    kd_att: np.ndarray


# =========================
# Reference trajectory
# =========================
def reference_trajectory(t):
    """
    Piecewise-constant waypoint mission.
    Returns desired position, velocity, acceleration, yaw, and yaw rate.
    """
    if t < 3.0:
        pos = np.array([0.0, 0.0, 1.0])
    elif t < 7.0:
        pos = np.array([1.0, 0.0, 1.2])
    elif t < 11.0:
        pos = np.array([1.0, 1.0, 1.2])
    elif t < 15.0:
        pos = np.array([0.0, 1.0, 1.0])
    else:
        pos = np.array([0.0, 0.0, 1.0])

    return {
        "pos": pos,
        "vel": np.zeros(3),
        "acc": np.zeros(3),
        "yaw": 0.0,
        "yaw_rate": 0.0,
    }


# =========================
# Dynamics
# =========================

'''
### REMOVE LATER ###
def quad_dynamics(state, control, params):
    """
    State:
        [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

    Control:
        [T, tau_x, tau_y, tau_z]
    """
    m = params.mass
    g = params.g
    J = params.J

    pos = state[0:3]
    vel = state[3:6]
    phi, theta, psi = state[6:9]
    omega = state[9:12]   # [p, q, r]

    T = control[0]
    tau = control[1:4]

    R = euler_to_rotation(phi, theta, psi)

    # Translational dynamics
    thrust_world = R @ np.array([0.0, 0.0, T])
    pos_dot = vel
    vel_dot = thrust_world / m - np.array([0.0, 0.0, g])

    # Rotational dynamics
    E = euler_rate_matrix(phi, theta)
    euler_dot = E @ omega

    omega_dot = np.linalg.solve(J, tau - np.cross(omega, J @ omega))

    return np.concatenate([pos_dot, vel_dot, euler_dot, omega_dot])
'''


# =========================
# Controller
# =========================
class CascadedController:
    """
    Simple cascaded controller:
    - position loop -> desired acceleration
    - desired acceleration -> desired roll/pitch + thrust
    - attitude loop -> torques
    """

    def __init__(self, gains, params):
        self.gains = gains
        self.params = params

    def compute(self, state, ref):
        pos = state[0:3]
        vel = state[3:6]
        angles = state[6:9]
        omega = state[9:12]

        phi, theta, psi = angles
        psi_des = ref["yaw"]

        # ---- Position control ----
        e_pos = ref["pos"] - pos
        e_vel = ref["vel"] - vel

        a_cmd = (
            self.gains.kp_pos * e_pos
            + self.gains.kd_pos * e_vel
            + ref["acc"]
        )

        # Desired roll/pitch from horizontal acceleration command
        # Using common small-angle approximation with yaw reference.
        phi_des = (a_cmd[0] * np.sin(psi_des) - a_cmd[1] * np.cos(psi_des)) / self.params.g
        theta_des = (a_cmd[0] * np.cos(psi_des) + a_cmd[1] * np.sin(psi_des)) / self.params.g

        phi_des = clamp(phi_des, -self.params.max_tilt_rad, self.params.max_tilt_rad)
        theta_des = clamp(theta_des, -self.params.max_tilt_rad, self.params.max_tilt_rad)

        desired_angles = np.array([phi_des, theta_des, psi_des])

        # Collective thrust
        # Use vertical channel with compensation for current tilt.
        denom = max(np.cos(phi) * np.cos(theta), 0.3)
        thrust = self.params.mass * (self.params.g + a_cmd[2]) / denom
        thrust = float(clamp(thrust, self.params.thrust_min, self.params.thrust_max))

        # ---- Attitude control ----
        angle_error = desired_angles - angles
        angle_error[2] = wrap_to_pi(angle_error[2])

        desired_omega = np.array([0.0, 0.0, ref["yaw_rate"]])
        omega_error = desired_omega - omega

        tau = self.gains.kp_att * angle_error + self.gains.kd_att * omega_error
        tau = clamp(tau, -self.params.tau_max, self.params.tau_max)

        control = np.concatenate([[thrust], tau])

        debug = {
            "e_pos": e_pos,
            "e_vel": e_vel,
            "desired_angles": desired_angles,
            "a_cmd": a_cmd,
        }
        return control, debug


# =========================
# Simulation
# =========================
def simulate(params, controller, t_final=20.0, dt=0.01):
    n_steps = int(t_final / dt) + 1
    time = np.linspace(0.0, t_final, n_steps)

    # Initial state
    # Start on the ground, zero velocity, level attitude
    state = np.zeros(12)
    state[2] = 0.0  # z

    states = np.zeros((n_steps, 12))
    controls = np.zeros((n_steps, 4))
    refs = np.zeros((n_steps, 3))
    desired_angles_hist = np.zeros((n_steps, 3))

    for k, t in enumerate(time):
        ref = reference_trajectory(t)
        control, debug = controller.compute(state, ref)

        states[k] = state
        controls[k] = control
        refs[k] = ref["pos"]
        desired_angles_hist[k] = debug["desired_angles"]

        if k < n_steps - 1:
            state = rk4_step(quad_dynamics, state, control, dt, params)

            # Simple ground contact: do not allow z < 0
            if state[2] < 0.0:
                state[2] = 0.0
                if state[5] < 0.0:
                    state[5] = 0.0

            # Keep yaw wrapped for nicer plots
            state[8] = wrap_to_pi(state[8])

    return time, states, controls, refs, desired_angles_hist


# =========================
# Plotting
# =========================
def plot_results(time, states, controls, refs, desired_angles_hist):
    pos = states[:, 0:3]
    vel = states[:, 3:6]
    angles = states[:, 6:9]
    omega = states[:, 9:12]

    fig1, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels = ["x", "y", "z"]
    for i in range(3):
        ax[i].plot(time, pos[:, i], label=f"{labels[i]}")
        ax[i].plot(time, refs[:, i], "--", label=f"{labels[i]}_ref")
        ax[i].set_ylabel(f"{labels[i]} [m]")
        ax[i].grid(True)
        ax[i].legend(loc="best")
    ax[-1].set_xlabel("Time [s]")
    fig1.suptitle("Position Tracking")

    fig2, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    angle_labels = ["roll φ", "pitch θ", "yaw ψ"]
    for i in range(3):
        ax[i].plot(time, angles[:, i], label="actual")
        ax[i].plot(time, desired_angles_hist[:, i], "--", label="desired")
        ax[i].set_ylabel(f"{angle_labels[i]} [rad]")
        ax[i].grid(True)
        ax[i].legend(loc="best")
    ax[-1].set_xlabel("Time [s]")
    fig2.suptitle("Attitude")

    fig3, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ctrl_labels = ["Thrust [N]", "τx [N·m]", "τy [N·m]", "τz [N·m]"]
    for i in range(4):
        ax[i].plot(time, controls[:, i])
        ax[i].set_ylabel(ctrl_labels[i])
        ax[i].grid(True)
    ax[-1].set_xlabel("Time [s]")
    fig3.suptitle("Control Inputs")

    fig4, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(time, np.linalg.norm(refs - pos, axis=1))
    ax[0].set_ylabel("Position error [m]")
    ax[0].grid(True)

    ax[1].plot(time, np.linalg.norm(vel, axis=1))
    ax[1].set_ylabel("Speed [m/s]")
    ax[1].set_xlabel("Time [s]")
    ax[1].grid(True)
    fig4.suptitle("Basic Performance Signals")

    plt.tight_layout()


# =========================
# Animation
# =========================
def set_axes_equal(ax, xlim, ylim, zlim):
    """
    Make 3D axes look roughly equal.
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]

    max_range = max(x_range, y_range, z_range)

    x_mid = 0.5 * (xlim[0] + xlim[1])
    y_mid = 0.5 * (ylim[0] + ylim[1])
    z_mid = 0.5 * (zlim[0] + zlim[1])

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(max(0, z_mid - max_range / 2), z_mid + max_range / 2)


def animate_quadcopter(time, states, refs, arm_length):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    pos = states[:, 0:3]

    xlim = (-1.5, 1.5)
    ylim = (-1.5, 1.5)
    zlim = (0.0, 2.0)

    ax.set_title("Quadcopter 3D Animation")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    set_axes_equal(ax, xlim, ylim, zlim)

    # Draw reference path
    ax.plot(refs[:, 0], refs[:, 1], refs[:, 2], "--", linewidth=1.0, label="Reference path")

    # Ground plane corners
    ground_x = [-1.5, 1.5, 1.5, -1.5, -1.5]
    ground_y = [-1.5, -1.5, 1.5, 1.5, -1.5]
    ground_z = [0, 0, 0, 0, 0]
    ax.plot(ground_x, ground_y, ground_z, linewidth=1.0)

    arm1_line, = ax.plot([], [], [], linewidth=3)
    arm2_line, = ax.plot([], [], [], linewidth=3)
    trail_line, = ax.plot([], [], [], linewidth=1.5)
    body_point, = ax.plot([], [], [], "o")
    ref_point, = ax.plot([], [], [], "x", markersize=8)

    skip = 5  # show every 5th sample for lighter animation
    frame_ids = np.arange(0, len(time), skip)

    def update(frame_index):
        k = frame_ids[frame_index]
        x, y, z = states[k, 0:3]
        phi, theta, psi = states[k, 6:9]

        R = euler_to_rotation(phi, theta, psi)

        # Body axes projected as quad arms
        arm_x_body = np.array([arm_length, 0.0, 0.0])
        arm_y_body = np.array([0.0, arm_length, 0.0])

        p_center = np.array([x, y, z])

        p1 = p_center + R @ arm_x_body
        p2 = p_center - R @ arm_x_body
        p3 = p_center + R @ arm_y_body
        p4 = p_center - R @ arm_y_body

        arm1_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        arm1_line.set_3d_properties([p1[2], p2[2]])

        arm2_line.set_data([p3[0], p4[0]], [p3[1], p4[1]])
        arm2_line.set_3d_properties([p3[2], p4[2]])

        trail_line.set_data(pos[:k+1, 0], pos[:k+1, 1])
        trail_line.set_3d_properties(pos[:k+1, 2])

        body_point.set_data([x], [y])
        body_point.set_3d_properties([z])

        ref_point.set_data([refs[k, 0]], [refs[k, 1]])
        ref_point.set_3d_properties([refs[k, 2]])

        return arm1_line, arm2_line, trail_line, body_point, ref_point

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=20,
        blit=False,
        repeat=True
    )

    ax.legend(loc="upper left")
    return anim


# =========================
# Main
# =========================
def main():
    params = get_default_drone_params()
    sim_params = get_default_sim_params()
    gains = get_default_controller_gains()

    #controller = CascadedController(gains, params)
    controller = CascadedPIDController(gains, params)

    #trajectory = get_default_waypoint_trajectory()
    #trajectory = get_default_hover_trajectory()
    trajectory = get_default_circular_trajectory()
    ref = trajectory.get_reference(t)

    #environment = get_default_indoor_environment()
    environment = get_windy_outdoor_environment()
    disturbance_model = environment.get_disturbance_model()

    time, states, controls, refs, desired_angles_hist = simulate(
        params=params,
        controller=controller,
        t_final=sim_params.t_final,
        dt=sim_params.dt
    )

    final_error = np.linalg.norm(refs[-1] - states[-1, 0:3])
    mean_error = np.mean(np.linalg.norm(refs - states[:, 0:3], axis=1))

    print("Simulation finished.")
    print(f"Final position: {states[-1, 0:3]}")
    print(f"Final reference: {refs[-1]}")
    print(f"Final position error: {final_error:.4f} m")
    print(f"Mean position tracking error: {mean_error:.4f} m")

    plot_results(time, states, controls, refs, desired_angles_hist)
    anim = animate_quadcopter(time, states, refs, params.arm_length)

    # Keep a reference to animation alive
    _ = anim
    plt.show()


if __name__ == "__main__":
    main()