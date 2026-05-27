"""
CF2 hider controller. Stacks goal + obstacle + dispersion + boundary forces,
then saturates. Goal switches between Lissajous roam and assigned cover when
the shared seeker sighting is fresh.
"""

import math

import numpy as np

try:
    from .cf2_sensing import intent_heading, sense_obstacles
    from .cf2_consensus import obstacle_consensus_cmd, inter_agent_dispersion_cmd
    from .cf2_hider import (
        report_seeker_sighting, get_last_seen,
        seek_cover_target, assign_cover_targets,
    )
    from .rmtt_seeker import can_see
except ImportError:
    from cf2_sensing import intent_heading, sense_obstacles
    from cf2_consensus import obstacle_consensus_cmd, inter_agent_dispersion_cmd
    from cf2_hider import (
        report_seeker_sighting, get_last_seen,
        seek_cover_target, assign_cover_targets,
    )
    from rmtt_seeker import can_see

# Arena bounds — must match standalone_sim.py.
_X_MIN, _X_MAX = -2.5, 2.5
_Y_MIN, _Y_MAX = -4.5, 4.5

# Per-drone Lissajous: amplitudes stay inside boundary repulsion zone (0.6)
# and below obstacle tops (z = 2.5).
_PATH_CENTER_X = 0.0
_PATH_CENTER_Y = 0.0
_PATH_CENTER_Z = 1.0
_PATH_AMP_X = 1.6
_PATH_AMP_Y = 3.0
_PATH_AMP_Z = 0.5
_Z_RATE_LIMIT = 0.3                 # m/call cap on z change, keeps xy saturation honest

# Forward cone used for seeker detection and visualization only.
# Obstacle sensing is omnidirectional (math.pi).
FOV_HALF_ANGLE = math.radians(60)
SENSING_RADIUS = 2.0
R_SAFE = 0.6
K_OBS = 2.0
K_BARRIER = 0.5

# Coulomb-style dispersion gain, comparable to goal_gain.
K_DISP = 0.4

HIDER_SEEKER_DETECTION_RANGE = 3.0


def _lissajous_target(robot_no, t):
    """Per-drone 3D Lissajous waypoint. Frequencies mirror intent_heading."""
    omega_x = 0.16 + 0.018 * (robot_no - 1)
    omega_y = 0.11 + 0.015 * (robot_no - 1)
    omega_z = 0.09 + 0.013 * (robot_no - 1)
    phase_x = (robot_no - 1) * (2.0 * math.pi / 3.0)
    phase_y = (robot_no - 1) * (2.0 * math.pi / 5.0)
    phase_z = (robot_no - 1) * (2.0 * math.pi / 7.0)
    tx = _PATH_CENTER_X + _PATH_AMP_X * math.sin(omega_x * t + phase_x)
    ty = _PATH_CENTER_Y + _PATH_AMP_Y * math.sin(omega_y * t + phase_y)
    tz = _PATH_CENTER_Z  # + _PATH_AMP_Z * math.sin(omega_z * t + phase_z)
    return tx, ty, tz


def _closest_point_to_obstacle(px, py, pz, ox, oy, sx, sy, sz):
    """Closest AABB-surface point. Local copy to avoid a circular import."""
    half_x = sx * 0.5
    half_y = sy * 0.5
    min_x = ox - half_x
    max_x = ox + half_x
    min_y = oy - half_y
    max_y = oy + half_y
    min_z = 0.0
    max_z = sz

    cx = min(max(px, min_x), max_x)
    cy = min(max(py, min_y), max_y)
    cz = min(max(pz, min_z), max_z)
    return cx, cy, cz


def _saturate_xy(vx, vy, vmax):
    """Cap 2D velocity by magnitude, preserving direction."""
    speed = math.hypot(vx, vy)
    if speed > vmax and speed > 1e-9:
        scale = vmax / speed
        return vx * scale, vy * scale
    return vx, vy


def compute_roaming_cmd(robot_no, robot_pose, cf2_poses, rmtt_poses, obstacle_pose, obstacle_size, clock):
    """Roam + obstacle consensus + hide-and-seek. Returns (vx, vy, z_dist, led)."""
    px, py, pz = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    t = float(clock)

    # tz is reused in hide mode so altitude stays consistent.
    tx_roam, ty_roam, tz = _lissajous_target(robot_no, t)

    heading = intent_heading(robot_no, robot_pose, t)

    # Seeker detection through this hider's forward cone; can_see also occludes.
    if rmtt_poses.size > 0:
        seeker_pos = (float(rmtt_poses[0, 0]), float(rmtt_poses[1, 0]), float(rmtt_poses[2, 0]))
        if can_see(
            (px, py, pz), seeker_pos, heading,
            FOV_HALF_ANGLE, HIDER_SEEKER_DETECTION_RANGE,
            obstacle_pose, obstacle_size,
        ):
            report_seeker_sighting(seeker_pos, t)

    # Mode switch: all hiders read the same shared sighting.
    last_seen = get_last_seen(t)
    if last_seen is not None:
        # Same deterministic matching on every hider → each gets a distinct face.
        cf2_xy = [(float(cf2_poses[0, j]), float(cf2_poses[1, j]))
                  for j in range(cf2_poses.shape[1] if cf2_poses.size else 0)]
        covers = assign_cover_targets(cf2_xy, last_seen, obstacle_pose, obstacle_size)
        tx, ty = covers[robot_no - 1]
    else:
        tx, ty = tx_roam, ty_roam

    # Goal attraction. Kept weak in roam mode so dispersion dominates and the
    # swarm doesn't settle into a static equilibrium.
    goal_gain = 0.15
    fx = goal_gain * (tx - px)
    fy = goal_gain * (ty - py)

    # Obstacle consensus (omnidirectional).
    emergency_dist = 0.25
    sensed = sense_obstacles(robot_pose, heading, math.pi, SENSING_RADIUS,
                             obstacle_pose, obstacle_size)
    fx_obs, fy_obs = obstacle_consensus_cmd(sensed, R_SAFE, K_OBS, K_BARRIER)
    fx += fx_obs
    fy += fy_obs

    is_emergency = any(s['distance'] < emergency_dist for s in sensed)

    # Long-range Coulomb dispersion (every pair contributes at every distance).
    n_cf2 = cf2_poses.shape[1] if cf2_poses.size else 0
    self_idx = robot_no - 1
    neighbor_xy = [
        (float(cf2_poses[0, j]), float(cf2_poses[1, j]))
        for j in range(n_cf2) if j != self_idx
    ]
    fx_disp, fy_disp = inter_agent_dispersion_cmd((px, py), neighbor_xy, K_DISP)
    fx += fx_disp
    fy += fy_disp

    # Soft 1/d boundary repulsion — gentle nudge, not a hard bounce.
    boundary_margin = 0.6
    boundary_gain = 0.09

    left = px - _X_MIN
    right = _X_MAX - px
    down = py - _Y_MIN
    up = _Y_MAX - py

    if left < boundary_margin:
        fx += boundary_gain * (1.0 / max(left, 1e-4) - 1.0 / boundary_margin)
    if right < boundary_margin:
        fx -= boundary_gain * (1.0 / max(right, 1e-4) - 1.0 / boundary_margin)
    if down < boundary_margin:
        fy += boundary_gain * (1.0 / max(down, 1e-4) - 1.0 / boundary_margin)
    if up < boundary_margin:
        fy -= boundary_gain * (1.0 / max(up, 1e-4) - 1.0 / boundary_margin)

    # 0.55 m/s sits above the seeker's 0.35 cap so hiders can outrun once spotted.
    vx, vy = _saturate_xy(fx, fy, vmax=0.55)

    # Rate-limit z so the sim's 3D speed cap doesn't shrink the xy avoidance.
    z_dist = max(pz - _Z_RATE_LIMIT, min(pz + _Z_RATE_LIMIT, tz))

    if is_emergency:
        led = (255, 0, 0)
    elif abs(vx) + abs(vy) > 0.02:
        led = (0, 255, 0)
    else:
        led = (255, 180, 0)

    return float(vx), float(vy), z_dist, led
