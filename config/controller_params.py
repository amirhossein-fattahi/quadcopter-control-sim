from dataclasses import dataclass
import numpy as np


@dataclass
class ControllerGains:
    # Outer loop: position control
    kp_pos: np.ndarray
    kd_pos: np.ndarray

    # Inner loop: attitude control
    kp_att: np.ndarray
    kd_att: np.ndarray


def get_default_controller_gains() -> ControllerGains:
    return ControllerGains(
        kp_pos=np.array([1.8, 1.8, 3.5]),
        kd_pos=np.array([1.4, 1.4, 2.0]),
        kp_att=np.array([8.0, 8.0, 4.0]),
        kd_att=np.array([2.5, 2.5, 1.5]),
    )