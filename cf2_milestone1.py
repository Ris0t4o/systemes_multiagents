import math

import numpy as np

# Milestone-1 constants tuned for stable roaming in standalone_sim bounds.
_X_MIN, _X_MAX = -2.5, 2.5
_Y_MIN, _Y_MAX = -4.5, 4.5
_Z_TARGET = 1.0

_WAYPOINTS = np.array([
    [-2.0, -3.0],
    [1.8, -3.0],
    [2.1, -0.3],
    [1.7, 2.8],
    [-1.4, 3.1],
    [-2.1, 0.3],
], dtype=float)

_SEGMENT_TIME = 4.0


def _closest_point_to_obstacle(px, py, pz, ox, oy, sx, sy, sz):
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
    speed = math.hypot(vx, vy)
    if speed > vmax and speed > 1e-9:
        scale = vmax / speed
        return vx * scale, vy * scale
    return vx, vy


def compute_roaming_cmd(robot_no, robot_pose, cf2_poses, obstacle_pose, obstacle_size, clock):
    """
    Deterministic roaming + reactive avoidance for Milestone 1.

    Returns:
        (vx, vy, z_dist, led)
    """
    px, py, pz = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])

    phase = (robot_no - 1) * 2
    waypoint_idx = (int(clock / _SEGMENT_TIME) + phase) % len(_WAYPOINTS)
    tx, ty = _WAYPOINTS[waypoint_idx]

    # 1) Waypoint attraction
    goal_gain = 0.65
    fx = goal_gain * (tx - px)
    fy = goal_gain * (ty - py)

    # 2) Obstacle repulsion (reactive detour)
    obs_influence = 1.0
    obs_gain = 0.11
    emergency_dist = 0.25
    is_emergency = False

    n_obstacles = obstacle_pose.shape[1] if obstacle_pose.size else 0
    for i in range(n_obstacles):
        ox = float(obstacle_pose[0, i])
        oy = float(obstacle_pose[1, i])
        sx = float(obstacle_size[0, i])
        sy = float(obstacle_size[1, i])
        sz = float(obstacle_size[2, i])

        cx, cy, cz = _closest_point_to_obstacle(px, py, pz, ox, oy, sx, sy, sz)
        dx = px - cx
        dy = py - cy
        dz = pz - cz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < emergency_dist:
            is_emergency = True

        if dist < obs_influence:
            planar_norm = math.hypot(dx, dy)
            if planar_norm < 1e-6:
                # Degenerate case: pick deterministic push direction per robot.
                ang = 1.7 * robot_no
                ux = math.cos(ang)
                uy = math.sin(ang)
            else:
                ux = dx / planar_norm
                uy = dy / planar_norm

            safe_dist = max(dist, 1e-4)
            mag = obs_gain * (1.0 / safe_dist - 1.0 / obs_influence) / (safe_dist * safe_dist)
            fx += mag * ux
            fy += mag * uy

    # 3) CF2-to-CF2 separation
    sep_influence = 0.9
    sep_gain = 0.12
    n_cf2 = cf2_poses.shape[1] if cf2_poses.size else 0
    self_idx = robot_no - 1
    for j in range(n_cf2):
        if j == self_idx:
            continue
        ox = float(cf2_poses[0, j])
        oy = float(cf2_poses[1, j])
        dx = px - ox
        dy = py - oy
        dist = math.hypot(dx, dy)
        if dist < sep_influence and dist > 1e-6:
            ux = dx / dist
            uy = dy / dist
            mag = sep_gain * (1.0 / dist - 1.0 / sep_influence) / (dist * dist)
            fx += mag * ux
            fy += mag * uy

    # 4) Soft boundary repulsion
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

    vx, vy = _saturate_xy(fx, fy, vmax=0.45)

    if is_emergency:
        led = (255, 0, 0)
    elif abs(vx) + abs(vy) > 0.02:
        led = (0, 255, 0)
    else:
        led = (255, 180, 0)

    return float(vx), float(vy), _Z_TARGET, led
