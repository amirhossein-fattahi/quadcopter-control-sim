from dataclasses import dataclass, field
import numpy as np


@dataclass
class HoverTrajectory:
    """
    Constant hover reference.
    """
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    yaw: float = 0.0

    def get_reference(self, t: float):
        return {
            "pos": np.array(self.position, dtype=float),
            "vel": np.zeros(3),
            "acc": np.zeros(3),
            "yaw": float(self.yaw),
            "yaw_rate": 0.0,
        }


def get_default_hover_trajectory() -> HoverTrajectory:
    return HoverTrajectory()


def reference_trajectory(t: float):
    """
    Compatible helper if you want to keep calling reference_trajectory(t)
    from main.py.
    """
    return get_default_hover_trajectory().get_reference(t)