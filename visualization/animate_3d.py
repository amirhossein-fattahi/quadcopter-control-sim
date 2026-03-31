import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils.rotations import euler_to_rotation


def set_axes_equal(ax, xlim, ylim, zlim):
    """
    Make the 3D axes look roughly equally scaled.
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]

    max_range = max(x_range, y_range, z_range)

    x_mid = 0.5 * (xlim[0] + xlim[1])
    y_mid = 0.5 * (ylim[0] + ylim[1])
    z_mid = 0.5 * (zlim[0] + zlim[1])

    ax.set_xlim(x_mid - max_range / 2.0, x_mid + max_range / 2.0)
    ax.set_ylim(y_mid - max_range / 2.0, y_mid + max_range / 2.0)
    ax.set_zlim(max(0.0, z_mid - max_range / 2.0), z_mid + max_range / 2.0)


def animate_quadcopter(
    time,
    states,
    refs=None,
    arm_length=0.22,
    bounds=None,
    skip=5,
    title="Quadcopter 3D Animation",
):
    """
    Animate the quadcopter in 3D.

    Args:
        time: shape (N,)
        states: shape (N, 12)
        refs: optional shape (N, 3)
        arm_length: drone arm length
        bounds: dict like {"xlim": ..., "ylim": ..., "zlim": ...}
        skip: render every `skip` samples
    """
    pos = states[:, 0:3]

    if bounds is None:
        margin = 0.5
        xlim = (min(np.min(pos[:, 0]), -1.0) - margin, max(np.max(pos[:, 0]), 1.0) + margin)
        ylim = (min(np.min(pos[:, 1]), -1.0) - margin, max(np.max(pos[:, 1]), 1.0) + margin)
        zlim = (0.0, max(np.max(pos[:, 2]), 1.5) + margin)
    else:
        xlim = bounds["xlim"]
        ylim = bounds["ylim"]
        zlim = bounds["zlim"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    set_axes_equal(ax, xlim, ylim, zlim)

    # Ground boundary
    ground_x = [xlim[0], xlim[1], xlim[1], xlim[0], xlim[0]]
    ground_y = [ylim[0], ylim[0], ylim[1], ylim[1], ylim[0]]
    ground_z = [0.0, 0.0, 0.0, 0.0, 0.0]
    ax.plot(ground_x, ground_y, ground_z, linewidth=1.0)

    if refs is not None:
        ax.plot(refs[:, 0], refs[:, 1], refs[:, 2], "--", linewidth=1.0, label="Reference")

    arm1_line, = ax.plot([], [], [], linewidth=3)
    arm2_line, = ax.plot([], [], [], linewidth=3)
    trail_line, = ax.plot([], [], [], linewidth=1.5, label="Trajectory")
    body_point, = ax.plot([], [], [], "o")
    ref_point, = ax.plot([], [], [], "x", markersize=8)

    frame_ids = np.arange(0, len(time), max(1, int(skip)))

    def update(frame_index):
        k = frame_ids[frame_index]

        x, y, z = states[k, 0:3]
        phi, theta, psi = states[k, 6:9]

        R = euler_to_rotation(phi, theta, psi)
        p_center = np.array([x, y, z], dtype=float)

        arm_x_body = np.array([arm_length, 0.0, 0.0], dtype=float)
        arm_y_body = np.array([0.0, arm_length, 0.0], dtype=float)

        p1 = p_center + R @ arm_x_body
        p2 = p_center - R @ arm_x_body
        p3 = p_center + R @ arm_y_body
        p4 = p_center - R @ arm_y_body

        arm1_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        arm1_line.set_3d_properties([p1[2], p2[2]])

        arm2_line.set_data([p3[0], p4[0]], [p3[1], p4[1]])
        arm2_line.set_3d_properties([p3[2], p4[2]])

        trail_line.set_data(pos[:k + 1, 0], pos[:k + 1, 1])
        trail_line.set_3d_properties(pos[:k + 1, 2])

        body_point.set_data([x], [y])
        body_point.set_3d_properties([z])

        if refs is not None:
            ref_point.set_data([refs[k, 0]], [refs[k, 1]])
            ref_point.set_3d_properties([refs[k, 2]])
        else:
            ref_point.set_data([], [])
            ref_point.set_3d_properties([])

        return arm1_line, arm2_line, trail_line, body_point, ref_point

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=20,
        blit=False,
        repeat=True,
    )

    ax.legend(loc="upper left")
    return fig, anim