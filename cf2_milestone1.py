"""
Main hider controller (Crazyflie 2 / CF2).

Each CF2 drone is a hider. At every simulation tick the simulator calls
`compute_roaming_cmd` for each drone, which returns (vx, vy, z_dist, led).

The controller stacks four forces and saturates the sum:

   F_total = F_goal + F_obstacle + F_separation + F_boundary

The "goal" is either:
  - the drone's own per-agent Lissajous waypoint (default roaming mode), or
  - a cover point behind an obstacle relative to the last-known seeker position
    (hide mode, triggered when any hider's shared sighting is still fresh).

Obstacle avoidance and inter-drone separation are independent of the mode,
so the swarm stays collision-free regardless of what it is chasing.
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
    # Allow importing this file standalone for testing/debugging.
    from cf2_sensing import intent_heading, sense_obstacles
    from cf2_consensus import obstacle_consensus_cmd, inter_agent_dispersion_cmd
    from cf2_hider import (
        report_seeker_sighting, get_last_seen,
        seek_cover_target, assign_cover_targets,
    )
    from rmtt_seeker import can_see

# Milestone-1 constants tuned for stable roaming in standalone_sim bounds.
# These match X_MIN/X_MAX/Y_MIN/Y_MAX in standalone_sim.py and are duplicated
# here so the controller is self-contained (no import from the simulator).
_X_MIN, _X_MAX = -2.5, 2.5
_Y_MIN, _Y_MAX = -4.5, 4.5

# Each drone traces a smooth 3D Lissajous figure. Per-drone variation in
# (omega_x, omega_y, omega_z) keeps them from synchronizing onto the same orbit.
# Amplitudes are chosen to stay inside the soft boundary repulsion zone
# (boundary_margin = 0.6 below) and below the obstacle tops (z = 2.5),
# so xy avoidance handles obstacles and the path never fights wall/ceiling pushes.
_PATH_CENTER_X = 0.0
_PATH_CENTER_Y = 0.0
_PATH_CENTER_Z = 1.0
_PATH_AMP_X = 1.6
_PATH_AMP_Y = 3.0
_PATH_AMP_Z = 0.5
_Z_RATE_LIMIT = 0.3                 # m, max |z_dist - z| returned per call (keeps xy cap honest)

# Hider FoV / sensing tunables.
#   FOV_HALF_ANGLE  ... forward cone used ONLY for SEEKER detection and the
#                       on-screen cone visualization.
#   Obstacle sensing is OMNIDIRECTIONAL (math.pi) so a hider can never enter
#   an obstacle from a blind side (same proximity-sense pattern the seeker uses).
FOV_HALF_ANGLE = math.radians(60)
SENSING_RADIUS = 2.0
R_SAFE = 0.6                        # m, desired clearance to obstacle surface
K_OBS = 2.0                         # 1/s, linear consensus gain
K_BARRIER = 0.5                     # m/s, barrier strength near contact (dominates any bounded attraction as d -> 0)

# Swarm dispersion gain. Sets the scale of the inter-agent Coulomb law in
# cf2_consensus.inter_agent_dispersion_cmd: at d = 1 m the per-pair push has
# magnitude K_DISP. Comparable to the (now-weakened) goal_gain so dispersion
# can outweigh the Lissajous when the swarm is bunched up.
K_DISP = 0.4

# Seeker-detection (hider looks for the seeker through its own forward cone).
HIDER_SEEKER_DETECTION_RANGE = 3.0  # m, deliberately larger than SENSING_RADIUS


def _lissajous_target(robot_no, t):
    """
    Per-drone 3D Lissajous waypoint at time t. Frequencies and phases mirror
    those in cf2_sensing.intent_heading so the FoV always points at where the
    drone is trying to be.

    The z component is currently held at the center altitude (the AMP_Z * sin
    term is commented out): with only 1-3 hiders it's nicer to keep them at a
    consistent height for the cone visualization. Re-enable to add a slow
    vertical wobble if needed.
    """
    omega_x = 0.16 + 0.018 * (robot_no - 1)
    omega_y = 0.11 + 0.015 * (robot_no - 1)
    omega_z = 0.09 + 0.013 * (robot_no - 1)
    phase_x = (robot_no - 1) * (2.0 * math.pi / 3.0)
    phase_y = (robot_no - 1) * (2.0 * math.pi / 5.0)
    phase_z = (robot_no - 1) * (2.0 * math.pi / 7.0)
    tx = _PATH_CENTER_X + _PATH_AMP_X * math.sin(omega_x * t + phase_x)
    ty = _PATH_CENTER_Y + _PATH_AMP_Y * math.sin(omega_y * t + phase_y)
    tz = _PATH_CENTER_Z #+ _PATH_AMP_Z * math.sin(omega_z * t + phase_z)
    return tx, ty, tz


def _closest_point_to_obstacle(px, py, pz, ox, oy, sx, sy, sz):
    """Closest AABB-surface point. Duplicated locally for the same reasons as
    in cf2_sensing -- avoids a circular import."""
    half_x = sx * 0.5
    half_y = sy * 0.5
    min_x = ox - half_x
    max_x = ox + half_x
    min_y = oy - half_y
    max_y = oy + half_y

    # Match standalone_sim convention: obstacles stand on floor z=0 up to z=size_z.
    min_z = 0.0
    max_z = sz

    cx = min(max(px, min_x), max_x)
    cy = min(max(py, min_y), max_y)
    cz = min(max(pz, min_z), max_z)
    return cx, cy, cz


def _saturate_xy(vx, vy, vmax):
    """Cap the 2D velocity vector by magnitude, preserving its direction."""
    speed = math.hypot(vx, vy)
    if speed > vmax and speed > 1e-9:
        scale = vmax / speed
        return vx * scale, vy * scale
    return vx, vy


def compute_roaming_cmd(robot_no, robot_pose, cf2_poses, rmtt_poses, obstacle_pose, obstacle_size, clock):
    """
    Roaming + obstacle consensus + hide-and-seek behavior.

    - Default: Lissajous roaming.
    - If this hider sees the seeker through its forward cone, broadcast the
      seeker's position to the other hiders via cf2_hider's shared memory.
    - If the shared memory holds a fresh sighting (< SEEKER_MEMORY_TTL),
      every hider switches to cover-seeking toward a position behind an
      obstacle relative to the last-known seeker position.

    Returns (vx, vy, z_dist, led).
    """
    px, py, pz = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    t = float(clock)

    # Always compute the Lissajous target -- we need its tz even in hide mode,
    # so the drone keeps oscillating gently in altitude rather than freezing at z.
    tx_roam, ty_roam, tz = _lissajous_target(robot_no, t)

    # Heading + sensing (used for both obstacle avoidance and seeker detection).
    heading = intent_heading(robot_no, robot_pose, t)

    # Seeker detection through this hider's own forward cone.
    # Convention: rmtt_poses[:, 0] is the (only) seeker. can_see does both the
    # cone test and an obstacle-occlusion test, so a seeker behind a wall won't
    # falsely trigger the broadcast.
    if rmtt_poses.size > 0:
        seeker_pos = (float(rmtt_poses[0, 0]), float(rmtt_poses[1, 0]), float(rmtt_poses[2, 0]))
        if can_see(
            (px, py, pz), seeker_pos, heading,
            FOV_HALF_ANGLE, HIDER_SEEKER_DETECTION_RANGE,
            obstacle_pose, obstacle_size,
        ):
            report_seeker_sighting(seeker_pos, t)

    # Mode switch: hide if there's a fresh sighting in shared memory, else roam.
    # All hiders read the same shared memory, so the swarm switches modes
    # collectively as soon as ANY hider has reported a sighting in the last TTL.
    last_seen = get_last_seen(t)
    if last_seen is not None:
        # Multi-agent cover assignment: every hider runs the same deterministic
        # min-cost matching, so each gets a DISTINCT cover face. Stops the old
        # behavior where all hiders chased the same nearest cover and stacked
        # up where the seeker could sweep them in one cone.
        cf2_xy = [(float(cf2_poses[0, j]), float(cf2_poses[1, j]))
                  for j in range(cf2_poses.shape[1] if cf2_poses.size else 0)]
        covers = assign_cover_targets(cf2_xy, last_seen, obstacle_pose, obstacle_size)
        tx, ty = covers[robot_no - 1]
    else:
        tx, ty = tx_roam, ty_roam

    # 1) Goal attraction toward (tx, ty) -- Lissajous target or cover target.
    # In roam mode the Lissajous is a WEAK exploration drift (small gain) so the
    # dispersion consensus below is the dominant driver and the swarm doesn't
    # freeze at a static equilibrium the seeker can memorize. In hide mode the
    # same small gain is fine because (tx, ty) is the assigned cover and we
    # don't want a giant force overwhelming the obstacle barrier near it.
    goal_gain = 0.15
    fx = goal_gain * (tx - px)
    fy = goal_gain * (ty - py)

    # 2) Obstacle consensus (linear + barrier). Sensing is OMNIDIRECTIONAL
    #    (math.pi) so there are no blind spots -- the forward cone is reserved
    #    for seeker detection above.
    emergency_dist = 0.25
    sensed = sense_obstacles(robot_pose, heading, math.pi, SENSING_RADIUS,
                             obstacle_pose, obstacle_size)
    fx_obs, fy_obs = obstacle_consensus_cmd(sensed, R_SAFE, K_OBS, K_BARRIER)
    fx += fx_obs
    fy += fy_obs

    # Used only to colour the LED red when we're dangerously close, not to
    # change the control law -- the barrier already handles the dynamics.
    is_emergency = any(s['distance'] < emergency_dist for s in sensed)

    # 3) Inter-agent dispersion consensus (long-range, gradient flow).
    # Per-agent gradient of the swarm potential V = (k/2)*sum 1/d_ij. Acts at
    # all ranges, so every hider always pulls every other hider apart -- unlike
    # the old 0.9 m repulsion, which was silent past 0.9 m. See
    # cf2_consensus.inter_agent_dispersion_cmd for the Lyapunov story.
    n_cf2 = cf2_poses.shape[1] if cf2_poses.size else 0
    self_idx = robot_no - 1
    neighbor_xy = [
        (float(cf2_poses[0, j]), float(cf2_poses[1, j]))
        for j in range(n_cf2) if j != self_idx
    ]
    fx_disp, fy_disp = inter_agent_dispersion_cmd((px, py), neighbor_xy, K_DISP)
    fx += fx_disp
    fy += fy_disp

    # 4) Soft boundary repulsion
    # Each wall gets the same 1/d - 1/d0 push as the inter-drone separation.
    # boundary_gain is intentionally small (0.09) -- we want a gentle nudge
    # that keeps the path inside, not a hard bounce.
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

    # Saturate the final (vx, vy) by magnitude. 0.45 m/s leaves headroom under
    # MAX_V_CF2 = 0.6 so the z component can move without the 3D clamp in the
    # simulator scaling down the xy avoidance.
    vx, vy = _saturate_xy(fx, fy, vmax=0.45)

    # Rate-limit z command so |vz| stays small enough that the sim's 3D speed cap
    # (clamp_vel3d at MAX_V_CF2=0.6) doesn't scale down the xy avoidance.
    z_dist = max(pz - _Z_RATE_LIMIT, min(pz + _Z_RATE_LIMIT, tz))

    # LED is purely cosmetic: red = emergency clearance, green = moving normally,
    # amber = essentially stationary (saturated boundaries / dead zone).
    if is_emergency:
        led = (255, 0, 0)
    elif abs(vx) + abs(vy) > 0.02:
        led = (0, 255, 0)
    else:
        led = (255, 180, 0)

    return float(vx), float(vy), z_dist, led
