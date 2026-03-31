from dataclasses import dataclass, field
import numpy as np

from dynamics.disturbances import NoDisturbance


@dataclass
class IndoorEnvironment:
    """
    Simple indoor box environment.

    This first version provides:
    - room limits
    - collision/bound checks
    - optional state clamping
    - no wind by default
    """
    xlim: tuple = (-1.5, 1.5)
    ylim: tuple = (-1.5, 1.5)
    zlim: tuple = (0.0, 2.0)

    disturbance_model: object = field(default_factory=NoDisturbance)

    def get_disturbance_model(self):
        return self.disturbance_model

    def is_inside(self, position: np.ndarray) -> bool:
        x, y, z = position
        return (
            self.xlim[0] <= x <= self.xlim[1]
            and self.ylim[0] <= y <= self.ylim[1]
            and self.zlim[0] <= z <= self.zlim[1]
        )

    def get_bounds(self):
        return {
            "xlim": self.xlim,
            "ylim": self.ylim,
            "zlim": self.zlim,
        }

    def check_collision(self, state: np.ndarray):
        """
        Returns a dictionary describing whether the drone is out of bounds.
        """
        x, y, z = state[0:3]

        collisions = {
            "x_min": x < self.xlim[0],
            "x_max": x > self.xlim[1],
            "y_min": y < self.ylim[0],
            "y_max": y > self.ylim[1],
            "z_min": z < self.zlim[0],
            "z_max": z > self.zlim[1],
        }
        collisions["any"] = any(collisions.values())
        return collisions

    def clamp_state_to_bounds(self, state: np.ndarray) -> np.ndarray:
        """
        Hard-clamps position to the room boundaries.
        If a boundary is hit, the corresponding velocity component is zeroed.
        """
        state = np.array(state, dtype=float, copy=True)

        # x
        if state[0] < self.xlim[0]:
            state[0] = self.xlim[0]
            if state[3] < 0.0:
                state[3] = 0.0
        elif state[0] > self.xlim[1]:
            state[0] = self.xlim[1]
            if state[3] > 0.0:
                state[3] = 0.0

        # y
        if state[1] < self.ylim[0]:
            state[1] = self.ylim[0]
            if state[4] < 0.0:
                state[4] = 0.0
        elif state[1] > self.ylim[1]:
            state[1] = self.ylim[1]
            if state[4] > 0.0:
                state[4] = 0.0

        # z
        if state[2] < self.zlim[0]:
            state[2] = self.zlim[0]
            if state[5] < 0.0:
                state[5] = 0.0
        elif state[2] > self.zlim[1]:
            state[2] = self.zlim[1]
            if state[5] > 0.0:
                state[5] = 0.0

        return state


def get_default_indoor_environment() -> IndoorEnvironment:
    return IndoorEnvironment()