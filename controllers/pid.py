import numpy as np

from controllers.base_controller import BaseController
from dynamics.rigid_body import clamp, wrap_to_pi


class CascadedPIDController(BaseController):
    """
    First working controller for the project.

    Outer loop:
        position -> desired acceleration

    Inner loop:
        desired acceleration -> desired roll/pitch + thrust
        desired attitude -> body torques

    Notes:
    - With your current config, this behaves like a cascaded PD controller.
    - If later you add ki_pos / ki_att to the gains object, this class can use them.
    """

    def __init__(self, gains, params):
        super().__init__(params)
        self.gains = gains

        self.ki_pos = getattr(gains, "ki_pos", np.zeros(3))
        self.ki_att = getattr(gains, "ki_att", np.zeros(3))

        self.pos_integral = np.zeros(3)
        self.att_integral = np.zeros(3)

    def reset(self):
        self.pos_integral[:] = 0.0
        self.att_integral[:] = 0.0

    def compute(self, state, ref, dt=None):
        pos = state[0:3]
        vel = state[3:6]
        angles = state[6:9]
        omega = state[9:12]

        phi, theta, psi = angles
        psi_des = ref.get("yaw", 0.0)

        # Reference defaults
        pos_ref = ref.get("pos", np.zeros(3))
        vel_ref = ref.get("vel", np.zeros(3))
        acc_ref = ref.get("acc", np.zeros(3))
        yaw_rate_ref = ref.get("yaw_rate", 0.0)

        # ---- Position loop ----
        e_pos = pos_ref - pos
        e_vel = vel_ref - vel

        if dt is not None:
            self.pos_integral += e_pos * dt

        a_cmd = (
            self.gains.kp_pos * e_pos
            + self.gains.kd_pos * e_vel
            + self.ki_pos * self.pos_integral
            + acc_ref
        )

        # Small-angle mapping from desired horizontal acceleration to desired roll/pitch
        phi_des = (a_cmd[0] * np.sin(psi_des) - a_cmd[1] * np.cos(psi_des)) / self.params.g
        theta_des = (a_cmd[0] * np.cos(psi_des) + a_cmd[1] * np.sin(psi_des)) / self.params.g

        phi_des = clamp(phi_des, -self.params.max_tilt_rad, self.params.max_tilt_rad)
        theta_des = clamp(theta_des, -self.params.max_tilt_rad, self.params.max_tilt_rad)

        desired_angles = np.array([phi_des, theta_des, psi_des], dtype=float)

        # Collective thrust
        denom = max(np.cos(phi) * np.cos(theta), 0.3)
        thrust = self.params.mass * (self.params.g + a_cmd[2]) / denom
        thrust = float(clamp(thrust, self.params.thrust_min, self.params.thrust_max))

        # ---- Attitude loop ----
        angle_error = desired_angles - angles
        angle_error[2] = wrap_to_pi(angle_error[2])

        if dt is not None:
            self.att_integral += angle_error * dt

        desired_omega = np.array([0.0, 0.0, yaw_rate_ref], dtype=float)
        omega_error = desired_omega - omega

        tau = (
            self.gains.kp_att * angle_error
            + self.gains.kd_att * omega_error
            + self.ki_att * self.att_integral
        )

        tau = clamp(tau, -self.params.tau_max, self.params.tau_max)

        control = np.concatenate([[thrust], tau])

        debug = {
            "e_pos": e_pos,
            "e_vel": e_vel,
            "a_cmd": a_cmd,
            "desired_angles": desired_angles,
            "angle_error": angle_error,
            "omega_error": omega_error,
        }

        return control, debug