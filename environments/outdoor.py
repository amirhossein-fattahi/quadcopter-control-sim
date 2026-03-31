from dataclasses import dataclass, field
import numpy as np

from dynamics.disturbances import (
    CombinedDisturbance,
    ConstantDisturbance,
    WindGustDisturbance,
    SinusoidalDisturbance,
)


@dataclass
class OutdoorEnvironment:
    """
    Simple outdoor environment.

    This first version provides:
    - larger flight region
    - steady wind option
    - gust option
    - sinusoidal disturbance option
    """
    xlim: tuple = (-10.0, 10.0)
    ylim: tuple = (-10.0, 10.0)
    zlim: tuple = (0.0, 8.0)

    steady_force_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    steady_torque_body: np.ndarray = field(default_factory=lambda: np.zeros(3))

    gust_start_time: float = 6.0
    gust_end_time: float = 8.0
    gust_force_world: np.ndarray = field(default_factory=lambda: np.array([1.5, 0.0, 0.0]))
    gust_torque_body: np.ndarray = field(default_factory=lambda: np.zeros(3))

    sinusoidal_force_amplitude: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sinusoidal_torque_amplitude: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sinusoidal_frequency_hz: float = 0.3

    def get_disturbance_model(self):
        disturbances = CombinedDisturbance()

        if np.linalg.norm(self.steady_force_world) > 0.0 or np.linalg.norm(self.steady_torque_body) > 0.0:
            disturbances.add(
                ConstantDisturbance(
                    force_world=self.steady_force_world,
                    torque_body=self.steady_torque_body,
                )
            )

        if np.linalg.norm(self.gust_force_world) > 0.0 or np.linalg.norm(self.gust_torque_body) > 0.0:
            disturbances.add(
                WindGustDisturbance(
                    start_time=self.gust_start_time,
                    end_time=self.gust_end_time,
                    force_world=self.gust_force_world,
                    torque_body=self.gust_torque_body,
                )
            )

        if (
            np.linalg.norm(self.sinusoidal_force_amplitude) > 0.0
            or np.linalg.norm(self.sinusoidal_torque_amplitude) > 0.0
        ):
            disturbances.add(
                SinusoidalDisturbance(
                    force_amplitude=self.sinusoidal_force_amplitude,
                    torque_amplitude=self.sinusoidal_torque_amplitude,
                    frequency_hz=self.sinusoidal_frequency_hz,
                )
            )

        return disturbances

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


def get_default_outdoor_environment() -> OutdoorEnvironment:
    return OutdoorEnvironment()


def get_windy_outdoor_environment() -> OutdoorEnvironment:
    return OutdoorEnvironment(
        steady_force_world=np.array([0.4, 0.0, 0.0]),
        gust_start_time=6.0,
        gust_end_time=8.0,
        gust_force_world=np.array([2.0, 0.6, 0.0]),
        sinusoidal_force_amplitude=np.array([0.2, 0.1, 0.0]),
        sinusoidal_frequency_hz=0.25,
    )