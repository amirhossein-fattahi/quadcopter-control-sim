import numpy as np
from dynamics.motors import clamp_rotor_speeds


def control_to_motor_thrusts(control, quad_params, motor_params):
    """
    Maps desired total thrust and body torques to individual rotor thrusts
    for a plus configuration.

    Control:
        [T, tau_x, tau_y, tau_z]

    Rotor indexing:
        1: front  (+x)
        2: right  (+y)
        3: rear   (-x)
        4: left   (-y)
    """
    T, tau_x, tau_y, tau_z = np.asarray(control, dtype=float)
    L = quad_params.arm_length

    if motor_params.kf <= 0.0:
        raise ValueError("motor_params.kf must be positive.")
    if motor_params.km <= 0.0:
        raise ValueError("motor_params.km must be positive.")

    # tau_z = (km/kf) * signed_sum_of_thrusts
    c = motor_params.km / motor_params.kf

    B = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [0.0, L, 0.0, -L],
        [-L, 0.0, L, 0.0],
        [c, -c, c, -c],
    ], dtype=float)

    wrench = np.array([T, tau_x, tau_y, tau_z], dtype=float)

    rotor_thrusts = np.linalg.solve(B, wrench)
    rotor_thrusts = np.clip(rotor_thrusts, 0.0, None)

    return rotor_thrusts


def motor_thrusts_to_omegas(rotor_thrusts, motor_params):
    rotor_thrusts = np.asarray(rotor_thrusts, dtype=float)
    omegas = np.sqrt(np.clip(rotor_thrusts / motor_params.kf, 0.0, None))
    return clamp_rotor_speeds(omegas, motor_params)


def control_to_rotor_speeds(control, quad_params, motor_params):
    rotor_thrusts = control_to_motor_thrusts(control, quad_params, motor_params)
    return motor_thrusts_to_omegas(rotor_thrusts, motor_params)