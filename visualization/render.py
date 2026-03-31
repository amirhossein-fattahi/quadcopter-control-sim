import matplotlib.pyplot as plt

from visualization.plots import plot_all_results
from visualization.animate_3d import animate_quadcopter


def render_simulation_results(
    time,
    states,
    controls,
    refs,
    desired_angles_hist=None,
    arm_length=0.22,
    bounds=None,
    show_plots=True,
    show_animation=True,
):
    """
    High-level rendering helper.

    Returns:
        dict with keys:
            "figures": list of figure objects
            "animation": animation object or None
            "animation_figure": figure or None
    """
    figures = []
    anim = None
    anim_fig = None

    if show_plots:
        figures = plot_all_results(
            time=time,
            states=states,
            controls=controls,
            refs=refs,
            desired_angles_hist=desired_angles_hist,
        )

    if show_animation:
        anim_fig, anim = animate_quadcopter(
            time=time,
            states=states,
            refs=refs,
            arm_length=arm_length,
            bounds=bounds,
        )

    return {
        "figures": figures,
        "animation": anim,
        "animation_figure": anim_fig,
    }


def show_rendered_results():
    plt.show()