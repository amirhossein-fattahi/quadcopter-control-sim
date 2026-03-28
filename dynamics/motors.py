from dataclasses import dataclass
import numpy as np


@dataclass
class MotorParams:
    """
    Simple first-order motor model and thrust/drag mapping.
    """
    n_motors: int = 4

    # Rotor model:
    # thrust_i = kf * omega_i^2
    # yaw_drag_i = km * omega_i^2
    kf: float = 1.0e-5
    km: float = 2.0e-7

    omega_min: float = 0.0
    omega_max: float = 2200.0

    # First-order motor lag: domega/dt = (omega_cmd - omega)/tau
    motor_time_constant: float = 0.03

    # Rotor spin directions for yaw torque
    # +1 and -1 signs only
    spin_directions: tuple = (1, -1, 1, -1)


def clamp_rotor_speeds(omegas, motor_params: MotorParams):
    omegas = np.asarray(omegas, dtype=float)
    return np.clip(omegas, motor_params.omega_min, motor_params.omega_max)


def thrusts_from_omegas(omegas, motor_params: MotorParams):
    omegas = np.asarray(omegas, dtype=float)
    return motor_params.kf * omegas**2


def yaw_drag_torques_from_omegas(omegas, motor_params: MotorParams):
    omegas = np.asarray(omegas, dtype=float)
    spin = np.asarray(motor_params.spin_directions, dtype=float)
    return spin * motor_params.km * omegas**2


def total_thrust_from_omegas(omegas, motor_params: MotorParams):
    return float(np.sum(thrusts_from_omegas(omegas, motor_params)))


def rotor_speeds_to_wrench(omegas, quad_params, motor_params: MotorParams):
    """
    Converts rotor speeds to total wrench in a plus configuration.

    Rotor indexing:
        1: front  (+x)
        2: right  (+y)
        3: rear   (-x)
        4: left   (-y)

    Returns:
        np.array([T, tau_x, tau_y, tau_z])
    """
    omegas = np.asarray(omegas, dtype=float)
    thrusts = thrusts_from_omegas(omegas, motor_params)
    yaw_torques = yaw_drag_torques_from_omegas(omegas, motor_params)
    L = quad_params.arm_length

    f1, f2, f3, f4 = thrusts

    T = np.sum(thrusts)
    tau_x = L * (f2 - f4)
    tau_y = L * (f3 - f1)
    tau_z = np.sum(yaw_torques)

    return np.array([T, tau_x, tau_y, tau_z], dtype=float)


def first_order_motor_step(current_omegas, commanded_omegas, dt, motor_params: MotorParams):
    """
    Discrete-time first-order motor dynamics.
    """
    current_omegas = np.asarray(current_omegas, dtype=float)
    commanded_omegas = clamp_rotor_speeds(commanded_omegas, motor_params)

    tau = max(motor_params.motor_time_constant, 1e-6)
    domega = (commanded_omegas - current_omegas) / tau
    next_omegas = current_omegas + dt * domega

    return clamp_rotor_speeds(next_omegas, motor_params)