import numpy as np


class BaseDisturbance:
    """
    Base class for disturbance models.
    Returns:
        force_world: np.ndarray shape (3,)
        torque_body: np.ndarray shape (3,)
    """
    def get_force_torque(self, t, state):
        return np.zeros(3), np.zeros(3)


class NoDisturbance(BaseDisturbance):
    pass


class ConstantDisturbance(BaseDisturbance):
    def __init__(self, force_world=None, torque_body=None):
        self.force_world = np.zeros(3) if force_world is None else np.asarray(force_world, dtype=float)
        self.torque_body = np.zeros(3) if torque_body is None else np.asarray(torque_body, dtype=float)

    def get_force_torque(self, t, state):
        return self.force_world.copy(), self.torque_body.copy()


class WindGustDisturbance(BaseDisturbance):
    """
    Applies a constant force/torque only during a time window.
    """
    def __init__(self, start_time, end_time, force_world=None, torque_body=None):
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.force_world = np.zeros(3) if force_world is None else np.asarray(force_world, dtype=float)
        self.torque_body = np.zeros(3) if torque_body is None else np.asarray(torque_body, dtype=float)

    def get_force_torque(self, t, state):
        if self.start_time <= t <= self.end_time:
            return self.force_world.copy(), self.torque_body.copy()
        return np.zeros(3), np.zeros(3)


class SinusoidalDisturbance(BaseDisturbance):
    """
    Smooth time-varying disturbance.
    """
    def __init__(self, force_amplitude=None, torque_amplitude=None, frequency_hz=0.5):
        self.force_amplitude = np.zeros(3) if force_amplitude is None else np.asarray(force_amplitude, dtype=float)
        self.torque_amplitude = np.zeros(3) if torque_amplitude is None else np.asarray(torque_amplitude, dtype=float)
        self.frequency_hz = float(frequency_hz)

    def get_force_torque(self, t, state):
        s = np.sin(2.0 * np.pi * self.frequency_hz * t)
        return self.force_amplitude * s, self.torque_amplitude * s


class CombinedDisturbance(BaseDisturbance):
    def __init__(self, disturbances=None):
        self.disturbances = [] if disturbances is None else list(disturbances)

    def add(self, disturbance):
        self.disturbances.append(disturbance)

    def get_force_torque(self, t, state):
        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        for disturbance in self.disturbances:
            f, tau = disturbance.get_force_torque(t, state)
            total_force += f
            total_torque += tau

        return total_force, total_torque