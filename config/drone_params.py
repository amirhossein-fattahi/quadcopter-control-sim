from dataclasses import dataclass
import numpy as np


@dataclass
class QuadcopterParams:
    mass: float = 1.0
    g: float = 9.81
    arm_length: float = 0.22

    # Inertia components [kg·m^2]
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
    def J(self) -> np.ndarray:
        return np.diag([self.Jx, self.Jy, self.Jz])

    @property
    def tau_max(self) -> np.ndarray:
        return np.array([self.tau_x_max, self.tau_y_max, self.tau_z_max])


def get_default_drone_params() -> QuadcopterParams:
    return QuadcopterParams()