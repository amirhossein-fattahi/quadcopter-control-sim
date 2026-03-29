import numpy as np

from controllers.base_controller import BaseController
from controllers.pid import CascadedPIDController
from config.controller_params import get_default_controller_gains


class MPCController(BaseController):
    """
    First compatible MPC scaffold.

    This class keeps the project structure ready for a future real MPC.
    For now it uses the PID controller internally so the simulator stays runnable.

    Later you can replace compute() with:
    - linear MPC
    - constrained MPC
    - trajectory-tracking MPC
    """

    def __init__(self, params, horizon_steps=20, dt_prediction=0.05, pid_fallback=None):
        super().__init__(params)
        self.horizon_steps = int(horizon_steps)
        self.dt_prediction = float(dt_prediction)

        if pid_fallback is None:
            pid_fallback = CascadedPIDController(
                gains=get_default_controller_gains(),
                params=params,
            )

        self.pid_fallback = pid_fallback

    def reset(self):
        self.pid_fallback.reset()

    def compute(self, state, ref, dt=None):
        control, debug = self.pid_fallback.compute(state, ref, dt=dt)
        debug["mode"] = "mpc_placeholder"
        debug["horizon_steps"] = self.horizon_steps
        debug["dt_prediction"] = self.dt_prediction
        return control, debug