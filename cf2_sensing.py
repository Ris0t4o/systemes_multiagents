"""
Hider sensing: intent heading (FoV axis) and obstacle FoV detection.

Lissajous constants are duplicated from cf2_milestone1 to avoid a circular
import — keep them in sync.
"""

import math


_PATH_AMP_X = 1.6
_PATH_AMP_Y = 3.0
_PATH_CENTER_Z = 1.0
_PATH_AMP_Z = 0.5


def intent_heading(robot_no, current_pose, t):
    """
    Unit 3D vector from drone toward its current Lissajous target.

    Pointing at the goal (not the path tangent) means the FoV catches
    obstacles between drone and target before collision.
    """
    px = float(current_pose[0])
    py = float(current_pose[1])
    pz = float(current_pose[2])
    omega_x = 0.16 + 0.018 * (robot_no - 1)
    omega_y = 0.11 + 0.015 * (robot_no - 1)
    omega_z = 0.09 + 0.013 * (robot_no - 1)
    phase_x = (robot_no - 1) * (2.0 * math.pi / 3.0)
    phase_y = (robot_no - 1) * (2.0 * math.pi / 5.0)
    phase_z = (robot_no - 1) * (2.0 * math.pi / 7.0)
    tx = _PATH_AMP_X * math.sin(omega_x * t + phase_x)
    ty = _PATH_AMP_Y * math.sin(omega_y * t + phase_y)
    tz = _PATH_CENTER_Z + _PATH_AMP_Z * math.sin(omega_z * t + phase_z)
    dx = tx - px
    dy = ty - py
    dz = tz - pz
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-6:
        return 1.0, 0.0, 0.0
    return dx / norm, dy / norm, dz / norm


def _closest_point_to_obstacle(px, py, pz, ox, oy, sx, sy, sz):
    """Closest point on an AABB to (px, py, pz). Obstacles stand on z = 0."""
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


def sense_obstacles(robot_pose, heading, fov_half_angle, sensing_radius,
                    obstacle_pose, obstacle_size):
    """
    Return obstacles whose closest surface point falls in the 3D forward cone.

    Each entry: {'closest_point', 'normal', 'distance', 'obstacle_idx'}.
    Normal points from surface toward the drone.
    """
    px, py, pz = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    hx, hy, hz = heading
    cos_fov = math.cos(fov_half_angle)

    n_obstacles = obstacle_pose.shape[1] if obstacle_pose.size else 0
    sensed = []
    for i in range(n_obstacles):
        ox = float(obstacle_pose[0, i])
        oy = float(obstacle_pose[1, i])
        sx = float(obstacle_size[0, i])
        sy = float(obstacle_size[1, i])
        sz = float(obstacle_size[2, i])

        # Using the closest surface point (not center) keeps geometry honest
        # for long/thin boxes.
        cx, cy, cz = _closest_point_to_obstacle(px, py, pz, ox, oy, sx, sy, sz)
        dx = cx - px
        dy = cy - py
        dz = cz - pz
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance > sensing_radius:
            continue

        if distance < 1e-6:
            # Drone is on / inside the box — push opposite to heading.
            in_cone = True
            nx, ny, nz = -hx, -hy, -hz
        else:
            cos_angle = (dx * hx + dy * hy + dz * hz) / distance
            in_cone = cos_angle >= cos_fov
            nx = -dx / distance
            ny = -dy / distance
            nz = -dz / distance

        if not in_cone:
            continue

        sensed.append({
            'closest_point': (cx, cy, cz),
            'normal':        (nx, ny, nz),
            'distance':      distance,
            'obstacle_idx':  i,
        })
    return sensed
