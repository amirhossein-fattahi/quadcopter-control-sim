from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationParams:
    t_final: float = 20.0
    dt: float = 0.01

    # Initial state:
    # [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    initial_state: np.ndarray = None

    # Optional simple world limits for plotting / future use
    xlim: tuple = (-1.5, 1.5)
    ylim: tuple = (-1.5, 1.5)
    zlim: tuple = (0.0, 2.0)

    def __post_init__(self):
        if self.initial_state is None:
            self.initial_state = np.zeros(12)
            self.initial_state[2] = 0.0  # start on ground


def get_default_sim_params() -> SimulationParams:
    return SimulationParams()