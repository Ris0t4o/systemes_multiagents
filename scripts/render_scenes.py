"""
Headless scene captures from standalone_sim.

We import the simulator (which builds the world, fig, axes, and FuncAnimation)
once, immediately stop the animation, and then drive standalone_sim.update()
manually for each scene. Between scenes we reset every piece of mutable
state we know about.

For roam / hide scenes we relocate the seeker to (-1.5, 0, 0.8) — well clear
of every hider's spawn corner — so an early opportunistic catch does not
contaminate the demonstration. The catch scene puts the seeker deliberately
adjacent to a hider with clean line-of-sight.

Output: PNGs in ../report/figures/.
"""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")

# Hush the "plt.show() in Agg" warning from standalone_sim's module-level call.
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
warnings.filterwarnings("ignore", message="Animation was deleted without rendering")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, _ROOT)

import standalone_sim as sim  # noqa: E402
import tp_algos  # noqa: E402
import cf2_milestone1  # noqa: E402
import cf2_hider  # noqa: E402
import game_referee  # noqa: E402
import sim_viz  # noqa: E402
import rmtt_seeker  # noqa: E402

try:
    sim.ani.event_source.stop()
except Exception:
    pass


# Catch-detection toggle: certain scenes want a settled roam/hide geometry
# without an opportunistic seeker drive-by removing a hider mid-shot. We swap
# game_referee.update_catches with a no-op while the scene is being set up.
_real_update_catches = game_referee.update_catches


def _noop_update_catches(*args, **kwargs):
    return None


def disable_catches():
    game_referee.update_catches = _noop_update_catches


def enable_catches():
    game_referee.update_catches = _real_update_catches


# Cone-rendering toggle. Four cones at 0.10 alpha overlap into solid green
# on the wide 3D shots and hide the drone markers behind them. We strip
# cones from the overview/roam shots and keep them for hide/catch where
# they're informative.
_real_draw_fov_cone = sim_viz.draw_fov_cone


def _noop_draw_fov_cone(*args, **kwargs):
    return None


def disable_cones():
    sim_viz.draw_fov_cone = _noop_draw_fov_cone


def enable_cones():
    sim_viz.draw_fov_cone = _real_draw_fov_cone

FIG_DIR = os.path.join(_ROOT, "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
DPI = 180

# Default sim spawn (kept for the arena-overview shot).
SPAWN_RMTT_DEFAULT = np.array([[1.5], [1.0], [0.0]])
# Isolated seeker pose used for the roam / hide demos. Already at hover-z so
# we can skip the takeoff transient.
SAFE_RMTT_FLYING = np.array([[-1.5], [0.0], [0.8]])

SPAWN_CF2 = np.array([
    [-1.8, +1.8, -1.8, +1.8],
    [-3.8, -3.8, +3.8, +3.8],
    [ 0.0,  0.0,  0.0,  0.0],
])

# Make the saved figures roomier than the live-window default (10×8).
sim.fig.set_size_inches(12, 9)


def reset_world(*, seeker_xyz=None, seeker_state=1):
    """Put every piece of mutable state back to t = 0.

    seeker_xyz: optional (x, y, z) to override the seeker spawn.
    seeker_state: 0 (idle, will not move and emits no FoV cone),
                  1 (takeoff, the simulator's default),
                  2 (flying, controller takes over immediately).
    """
    if seeker_xyz is None:
        sim.rmtt_poses[:] = SPAWN_RMTT_DEFAULT
    else:
        sim.rmtt_poses[0, 0] = seeker_xyz[0]
        sim.rmtt_poses[1, 0] = seeker_xyz[1]
        sim.rmtt_poses[2, 0] = seeker_xyz[2]
    sim.cf2_poses[:] = SPAWN_CF2

    for i in range(sim.nbRMTT):
        sim.rmtt_states[i] = seeker_state
        sim.rmtt_timers[i] = sim.TAKEOFF_TIME if seeker_state == 1 else 0.0
    for i in range(sim.nbCF2):
        sim.cf2_states[i] = 0
        sim.cf2_timers[i] = 0.0

    sim.clock_time = 0.0
    sim.task_futures.clear()
    sim.last_cmds.clear()

    if hasattr(tp_algos.cf2_control_fn, "_takeoff_done_by_robot"):
        tp_algos.cf2_control_fn._takeoff_done_by_robot.clear()

    cf2_milestone1._t_offsets.clear()
    cf2_hider._last_seen["position"] = None
    cf2_hider._last_seen["timestamp"] = None
    cf2_hider.reset_cover_cache()

    game_referee._caught.clear()
    game_referee._catch_times.clear()

    for i, cone in enumerate(sim_viz._state.get("cf2_fov_cones", [])):
        if cone is not None:
            try:
                cone.remove()
            except Exception:
                pass
            sim_viz._state["cf2_fov_cones"][i] = None
    for i, cone in enumerate(sim_viz._state.get("rmtt_fov_cones", [])):
        if cone is not None:
            try:
                cone.remove()
            except Exception:
                pass
            sim_viz._state["rmtt_fov_cones"][i] = None
    for trail in sim_viz._state.get("cf2_trails", []):
        trail.clear()
    for line in sim_viz._state.get("cf2_trail_lines", []):
        line.set_data([], [])
        line.set_3d_properties([])


def step(n_frames):
    for _ in range(n_frames):
        sim.update(0)


def configure_axes(elev, azim, title=None):
    sim.ax.view_init(elev=elev, azim=azim)
    if title is not None:
        sim.ax.set_title(title)


def save(name):
    out = os.path.join(FIG_DIR, name)
    sim.fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"[render_scenes] wrote {out}")


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------
def scene_arena_overview():
    """Spawn configuration, default seeker pose, t = 0."""
    enable_catches()
    disable_cones()
    reset_world()
    step(1)
    configure_axes(
        elev=24, azim=-58,
        title="Arena, obstacles, and spawn configuration (t = 0)",
    )
    save("fig_arena_overview.png")


def scene_roam_dispersion():
    """Roam mode: settled Lissajous + Coulomb dispersion. Seeker parked
    idle and out of vision range so the swarm geometry is uncontaminated."""
    disable_catches()
    disable_cones()
    # Seeker parked idle 10 m below the floor: too far to be seen, no cone drawn.
    reset_world(seeker_xyz=(0.0, 0.0, -10.0), seeker_state=0)
    step(500)  # 25 s
    configure_axes(
        elev=26, azim=-66,
        title=f"Roam mode — Lissajous + Coulomb dispersion (t = {sim.clock_time:.1f} s)",
    )
    save("fig_roam_dispersion.png")


def _settle_then_force_hide(steps_after_inject):
    """Roam-settle with the seeker parked idle, then teleport the seeker into
    flying state at a sightline-relevant pose and keep re-broadcasting the
    sighting so the TTL never expires before the snapshot."""
    disable_catches()
    reset_world(seeker_xyz=(0.0, 0.0, -10.0), seeker_state=0)
    step(500)
    # Teleport seeker into a flying pose south of the arena.
    sim.rmtt_states[0] = 2
    sim.rmtt_timers[0] = 0.0
    sim.rmtt_poses[:] = np.array([[0.0], [-4.0], [0.8]])
    seeker_xyz = (0.0, -4.0, 0.8)
    # Force-inject across every settle tick so alert stays on at snapshot time.
    for _ in range(steps_after_inject):
        cf2_hider.report_seeker_sighting(seeker_xyz, sim.clock_time)
        step(1)


def scene_hide_assignment_3d():
    """Hide mode: sighting injected, swarm converged on distinct cover faces."""
    enable_cones()
    _settle_then_force_hide(steps_after_inject=70)  # 3.5 s — well inside TTL.
    configure_axes(
        elev=28, azim=-72,
        title=f"Hide mode — 4 hiders on distinct cover faces (t = {sim.clock_time:.1f} s)",
    )
    save("fig_hide_assignment.png")


def scene_hide_assignment_topdown():
    """Same instant as hide_assignment_3d, viewed from straight above so the
    cover-face geometry is unambiguous."""
    enable_cones()
    _settle_then_force_hide(steps_after_inject=70)
    configure_axes(
        elev=89, azim=-90,
        title=f"Hide mode (top-down) — seeker (orange) at y=-4 (t = {sim.clock_time:.1f} s)",
    )
    save("fig_hide_assignment_topdown.png")


def scene_catch_moment():
    """Deterministic catch: teleport the seeker 1.4 m from CF2_1 with clean
    LOS, force its heading toward the hider, capture the frame on which the
    referee marks the hider."""
    enable_cones()
    disable_catches()
    reset_world(seeker_xyz=(0.0, 0.0, -10.0), seeker_state=0)
    step(500)
    sim.rmtt_states[0] = 2
    sim.rmtt_timers[0] = 0.0

    target_idx = 0
    h_pose = sim.cf2_poses[:, target_idx].copy()
    direction = np.array([1.0, -1.0]) / np.sqrt(2.0)
    seeker_xy = h_pose[:2] + direction * 1.4
    sim.rmtt_poses[0, 0] = seeker_xy[0]
    sim.rmtt_poses[1, 0] = seeker_xy[1]
    sim.rmtt_poses[2, 0] = 1.0

    # Override the seeker heading so its FoV axis points at CF2_1 instead of
    # at its Lissajous waypoint. Used by both the visualization (sim_viz) and
    # the referee (game_referee.update_catches).
    seeker_pos = (float(sim.rmtt_poses[0, 0]),
                  float(sim.rmtt_poses[1, 0]),
                  float(sim.rmtt_poses[2, 0]))
    dxh = float(h_pose[0]) - seeker_pos[0]
    dyh = float(h_pose[1]) - seeker_pos[1]
    norm = (dxh * dxh + dyh * dyh) ** 0.5
    forced_heading = (dxh / norm, dyh / norm, 0.0)
    original_heading = rmtt_seeker.seeker_heading
    game_referee.seeker_heading = lambda pose, t: forced_heading
    rmtt_seeker.seeker_heading = lambda pose, t: forced_heading
    sim_viz.seeker_heading = lambda pose, t: forced_heading

    enable_catches()
    step(1)  # Referee runs update_catches with the forced heading, marks CF2_1.

    # Restore the real heading function so subsequent scenes are unaffected.
    game_referee.seeker_heading = original_heading
    rmtt_seeker.seeker_heading = original_heading
    sim_viz.seeker_heading = original_heading

    configure_axes(
        elev=22, azim=-50,
        title=f"Catch — seeker cone on CF2_{target_idx+1} with clean LOS (t = {sim.clock_time:.1f} s)",
    )
    save("fig_catch_moment.png")


def scene_end_game():
    """All hiders caught → seeker enters its trigger_land branch and descends."""
    enable_cones()
    disable_catches()
    reset_world(seeker_xyz=(0.0, 0.0, -10.0), seeker_state=0)
    step(500)
    sim.rmtt_states[0] = 2
    sim.rmtt_timers[0] = 0.0
    sim.rmtt_poses[:] = np.array([[0.0], [-3.0], [0.8]])
    enable_catches()
    for robot_no in range(1, sim.nbCF2 + 1):
        game_referee._caught.add(robot_no)
        game_referee._catch_times[robot_no] = float(sim.clock_time)
    step(40)  # 2 s into landing.
    configure_axes(
        elev=22, azim=-58,
        title=f"All hiders caught; seeker landing (t = {sim.clock_time:.1f} s)",
    )
    save("fig_end_game.png")


if __name__ == "__main__":
    scene_arena_overview()
    scene_roam_dispersion()
    scene_hide_assignment_3d()
    scene_hide_assignment_topdown()
    scene_catch_moment()
    scene_end_game()
    print(f"[render_scenes] done. Figures in {FIG_DIR}")
