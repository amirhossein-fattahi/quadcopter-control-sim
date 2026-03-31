from dataclasses import dataclass, field
import numpy as np


@dataclass
class WaypointTrajectory:
    """
    Piecewise-linear waypoint trajectory.

    waypoints: list/array of shape (N, 3)
    segment_durations: list/array of length (N - 1)

    For each segment:
        pos(t) is linearly interpolated
        vel(t) is constant
        acc(t) = 0
    """
    waypoints: np.ndarray = field(
        default_factory=lambda: np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.2],
            [1.0, 1.0, 1.2],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
    )
    segment_durations: np.ndarray = field(
        default_factory=lambda: np.array([4.0, 4.0, 4.0, 4.0], dtype=float)
    )
    yaw: float = 0.0

    def __post_init__(self):
        self.waypoints = np.asarray(self.waypoints, dtype=float)
        self.segment_durations = np.asarray(self.segment_durations, dtype=float)

        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ValueError("waypoints must have shape (N, 3).")

        if len(self.waypoints) < 2:
            raise ValueError("At least two waypoints are required.")

        if len(self.segment_durations) != len(self.waypoints) - 1:
            raise ValueError("segment_durations must have length N-1.")

        if np.any(self.segment_durations <= 0.0):
            raise ValueError("All segment durations must be positive.")

        self.cumulative_times = np.concatenate([[0.0], np.cumsum(self.segment_durations)])
        self.total_time = float(self.cumulative_times[-1])

    def get_reference(self, t: float):
        t = float(t)

        if t <= 0.0:
            return {
                "pos": self.waypoints[0].copy(),
                "vel": np.zeros(3),
                "acc": np.zeros(3),
                "yaw": float(self.yaw),
                "yaw_rate": 0.0,
            }

        if t >= self.total_time:
            return {
                "pos": self.waypoints[-1].copy(),
                "vel": np.zeros(3),
                "acc": np.zeros(3),
                "yaw": float(self.yaw),
                "yaw_rate": 0.0,
            }

        seg_idx = np.searchsorted(self.cumulative_times, t, side="right") - 1
        seg_idx = min(seg_idx, len(self.segment_durations) - 1)

        t0 = self.cumulative_times[seg_idx]
        t1 = self.cumulative_times[seg_idx + 1]

        p0 = self.waypoints[seg_idx]
        p1 = self.waypoints[seg_idx + 1]

        alpha = (t - t0) / (t1 - t0)
        pos = (1.0 - alpha) * p0 + alpha * p1
        vel = (p1 - p0) / (t1 - t0)
        acc = np.zeros(3)

        return {
            "pos": pos,
            "vel": vel,
            "acc": acc,
            "yaw": float(self.yaw),
            "yaw_rate": 0.0,
        }


def get_default_waypoint_trajectory() -> WaypointTrajectory:
    return WaypointTrajectory()


def reference_trajectory(t: float):
    """
    Compatible helper if you want to replace your current local
    reference_trajectory(t) in main.py.
    """
    return get_default_waypoint_trajectory().get_reference(t)