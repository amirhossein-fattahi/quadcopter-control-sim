import numpy as np
import matplotlib.pyplot as plt


def plot_position_tracking(time, states, refs):
    pos = states[:, 0:3]

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels = ["x", "y", "z"]

    for i in range(3):
        ax[i].plot(time, pos[:, i], label=labels[i])
        ax[i].plot(time, refs[:, i], "--", label=f"{labels[i]}_ref")
        ax[i].set_ylabel(f"{labels[i]} [m]")
        ax[i].grid(True)
        ax[i].legend(loc="best")

    ax[-1].set_xlabel("Time [s]")
    fig.suptitle("Position Tracking")
    fig.tight_layout()
    return fig


def plot_attitude(time, states, desired_angles_hist=None):
    angles = states[:, 6:9]
    angle_labels = ["roll φ", "pitch θ", "yaw ψ"]

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for i in range(3):
        ax[i].plot(time, angles[:, i], label="actual")
        if desired_angles_hist is not None:
            ax[i].plot(time, desired_angles_hist[:, i], "--", label="desired")
        ax[i].set_ylabel(f"{angle_labels[i]} [rad]")
        ax[i].grid(True)
        ax[i].legend(loc="best")

    ax[-1].set_xlabel("Time [s]")
    fig.suptitle("Attitude")
    fig.tight_layout()
    return fig


def plot_controls(time, controls):
    ctrl_labels = ["Thrust [N]", "τx [N·m]", "τy [N·m]", "τz [N·m]"]

    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    for i in range(4):
        ax[i].plot(time, controls[:, i])
        ax[i].set_ylabel(ctrl_labels[i])
        ax[i].grid(True)

    ax[-1].set_xlabel("Time [s]")
    fig.suptitle("Control Inputs")
    fig.tight_layout()
    return fig


def plot_basic_performance(time, states, refs):
    pos = states[:, 0:3]
    vel = states[:, 3:6]

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(time, np.linalg.norm(refs - pos, axis=1))
    ax[0].set_ylabel("Position error [m]")
    ax[0].grid(True)

    ax[1].plot(time, np.linalg.norm(vel, axis=1))
    ax[1].set_ylabel("Speed [m/s]")
    ax[1].set_xlabel("Time [s]")
    ax[1].grid(True)

    fig.suptitle("Basic Performance Signals")
    fig.tight_layout()
    return fig


def plot_all_results(time, states, controls, refs, desired_angles_hist=None):
    figs = []
    figs.append(plot_position_tracking(time, states, refs))
    figs.append(plot_attitude(time, states, desired_angles_hist))
    figs.append(plot_controls(time, controls))
    figs.append(plot_basic_performance(time, states, refs))
    return figs