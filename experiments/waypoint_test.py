import numpy as np
from pathlib import Path

from config.drone_params import get_default_drone_params
from config.sim_params import get_default_sim_params
from config.controller_params import get_default_controller_gains

from controllers.pid import CascadedPIDController
from dynamics.rigid_body import simulate_step

from environments.indoor import get_default_indoor_environment
from trajectories.waypoint import get_default_waypoint_trajectory

from visualization.render import render_simulation_results, show_rendered_results
from utils.logger import SimpleLogger


def ensure_results_dirs():
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/videos").mkdir(parents=True, exist_ok=True)
    Path("results/logs").mkdir(parents=True, exist_ok=True)


def simulate_waypoint_experiment():
    params = get_default_drone_params()
    sim_params = get_default_sim_params()
    gains = get_default_controller_gains()

    environment = get_default_indoor_environment()
    trajectory = get_default_waypoint_trajectory()
    controller = CascadedPIDController(gains, params)

    disturbance_model = environment.get_disturbance_model()

    dt = sim_params.dt
    t_final = max(sim_params.t_final, trajectory.total_time + 2.0)
    n_steps = int(t_final / dt) + 1
    time = np.linspace(0.0, t_final, n_steps)

    state = np.array(sim_params.initial_state, dtype=float)

    states = np.zeros((n_steps, 12))
    controls = np.zeros((n_steps, 4))
    refs = np.zeros((n_steps, 3))
    desired_angles_hist = np.zeros((n_steps, 3))

    controller.reset()

    for k, t in enumerate(time):
        ref = trajectory.get_reference(t)
        control, debug = controller.compute(state, ref, dt=dt)

        states[k] = state
        controls[k] = control
        refs[k] = ref["pos"]
        desired_angles_hist[k] = debug["desired_angles"]

        if k < n_steps - 1:
            state = simulate_step(
                state=state,
                control=control,
                dt=dt,
                params=params,
                disturbance_model=disturbance_model,
                t=t,
                enforce_ground=True,
            )

            if hasattr(environment, "clamp_state_to_bounds"):
                state = environment.clamp_state_to_bounds(state)

    return {
        "time": time,
        "states": states,
        "controls": controls,
        "refs": refs,
        "desired_angles_hist": desired_angles_hist,
        "params": params,
        "environment": environment,
    }


def main():
    ensure_results_dirs()

    result = simulate_waypoint_experiment()

    tracking_error = np.linalg.norm(result["refs"] - result["states"][:, 0:3], axis=1)
    final_error = tracking_error[-1]
    mean_error = np.mean(tracking_error)
    max_error = np.max(tracking_error)

    print("Waypoint test finished.")
    print(f"Final position: {result['states'][-1, 0:3]}")
    print(f"Final reference: {result['refs'][-1]}")
    print(f"Final error: {final_error:.4f} m")
    print(f"Mean error: {mean_error:.4f} m")
    print(f"Max error: {max_error:.4f} m")

    rendered = render_simulation_results(
        time=result["time"],
        states=result["states"],
        controls=result["controls"],
        refs=result["refs"],
        desired_angles_hist=result["desired_angles_hist"],
        arm_length=result["params"].arm_length,
        bounds=result["environment"].get_bounds(),
        show_plots=True,
        show_animation=True,
    )

    logs_logger = SimpleLogger(base_dir="results/logs", run_name="waypoint_test")
    figs_logger = SimpleLogger(base_dir="results/figures", run_name="waypoint_test")

    logs_logger.save_metadata({
        "experiment": "waypoint_test",
        "controller": "CascadedPIDController",
        "environment": "IndoorEnvironment",
        "trajectory": "WaypointTrajectory",
        "final_error_m": final_error,
        "mean_error_m": mean_error,
        "max_error_m": max_error,
    })

    logs_logger.save_arrays_npz(
        time=result["time"],
        states=result["states"],
        controls=result["controls"],
        refs=result["refs"],
        desired_angles_hist=result["desired_angles_hist"],
    )

    logs_logger.save_time_series_csv(
        time=result["time"],
        states=result["states"],
        controls=result["controls"],
        refs=result["refs"],
    )

    figs_logger.save_figures(rendered["figures"], prefix="waypoint")

    _ = rendered["animation"]
    show_rendered_results()


if __name__ == "__main__":
    main()