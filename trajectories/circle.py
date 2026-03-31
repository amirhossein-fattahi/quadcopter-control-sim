from dataclasses import dataclass, field
import numpy as np


@dataclass
class CircularTrajectory:
    """
    Horizontal circular trajectory at constant altitude.

    position:
        x = cx + r cos(w t)
        y = cy + r sin(w t)
        z = constant
    """
    center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    radius: float = 1.0
    altitude: float = 1.2
    period: float = 8.0

    # yaw_mode:
    #   "constant" -> fixed yaw
    #   "tangent"  -> yaw follows velocity direction
    yaw_mode: str = "tangent"
    constant_yaw: float = 0.0

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        if self.center.shape != (2,):
            raise ValueError("center must be a 2D vector [cx, cy].")
        if self.radius <= 0.0:
            raise ValueError("radius must be positive.")
        if self.period <= 0.0:
            raise ValueError("period must be positive.")

        self.omega = 2.0 * np.pi / self.period

    def get_reference(self, t: float):
        t = float(t)
        w = self.omega
        cx, cy = self.center

        x = cx + self.radius * np.cos(w * t)
        y = cy + self.radius * np.sin(w * t)
        z = self.altitude

        vx = -self.radius * w * np.sin(w * t)
        vy = self.radius * w * np.cos(w * t)
        vz = 0.0

        ax = -self.radius * (w ** 2) * np.cos(w * t)
        ay = -self.radius * (w ** 2) * np.sin(w * t)
        az = 0.0

        if self.yaw_mode == "tangent":
            yaw = np.arctan2(vy, vx)
            yaw_rate = w
        elif self.yaw_mode == "constant":
            yaw = float(self.constant_yaw)
            yaw_rate = 0.0
        else:
            raise ValueError("yaw_mode must be 'constant' or 'tangent'.")

        return {
            "pos": np.array([x, y, z], dtype=float),
            "vel": np.array([vx, vy, vz], dtype=float),
            "acc": np.array([ax, ay, az], dtype=float),
            "yaw": float(yaw),
            "yaw_rate": float(yaw_rate),
        }


def get_default_circular_trajectory() -> CircularTrajectory:
    return CircularTrajectory()


def reference_trajectory(t: float):
    """
    Compatible helper if you want to use this module directly
    in place of a local reference_trajectory(t).
    """
    return get_default_circular_trajectory().get_reference(t)