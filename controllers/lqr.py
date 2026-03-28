import numpy as np

from controllers.base_controller import BaseController
from dynamics.rigid_body import clamp, wrap_to_pi
from controllers.pid import CascadedPIDController
from config.controller_params import get_default_controller_gains


class HoverLQRController(BaseController):
    """
    First compatible LQR scaffold.

    Two modes:
    1. If K is provided:
       uses a hover-linearized regulation law
           u = u_eq - K @ x_error
    2. If K is None:
       falls back to the cascaded PID controller so the project still runs.

    This is intentionally a first version.
    Later you can replace it with a proper linearization + Riccati-based design.
    """

    def __init__(self, params, K=None, pid_fallback=None):
        super().__init__(params)
        self.K = K

        if pid_fallback is None:
            pid_fallback = CascadedPIDController(
                gains=get_default_controller_gains(),
                params=params,
            )

        self.pid_fallback = pid_fallback

    def reset(self):
        self.pid_fallback.reset()

    def compute(self, state, ref, dt=None):
        # If no LQR gain matrix is provided, use PID fallback
        if self.K is None:
            control, debug = self.pid_fallback.compute(state, ref, dt=dt)
            debug["mode"] = "pid_fallback"
            return control, debug

        pos = state[0:3]
        vel = state[3:6]
        angles = state[6:9]
        omega = state[9:12]

        pos_ref = ref.get("pos", np.zeros(3))
        vel_ref = ref.get("vel", np.zeros(3))
        yaw_ref = ref.get("yaw", 0.0)

        # Hover equilibrium around desired position and yaw
        x_err = np.concatenate([
            pos - pos_ref,
            vel - vel_ref,
            np.array([
                wrap_to_pi(angles[0] - 0.0),
                wrap_to_pi(angles[1] - 0.0),
                wrap_to_pi(angles[2] - yaw_ref),
            ]),
            omega,
        ])

        u_eq = np.array([self.params.mass * self.params.g, 0.0, 0.0, 0.0], dtype=float)
        u = u_eq - self.K @ x_err

        thrust = float(clamp(u[0], self.params.thrust_min, self.params.thrust_max))
        tau = clamp(u[1:4], -self.params.tau_max, self.params.tau_max)

        control = np.concatenate([[thrust], tau])

        debug = {
            "mode": "lqr",
            "x_err": x_err,
            "u_eq": u_eq,
        }
        return control, debug