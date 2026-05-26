"""
Sensing primitives for the hider drones.

Two pieces:
  - intent_heading(): the unit 3D vector that points from the drone toward the
    place it currently wants to be (its Lissajous target). This is the axis of
    the FoV cone used both for visualization and for seeker detection.
  - sense_obstacles(): which obstacles fall inside the drone's 3D FoV cone,
    along with their closest-surface point and the outward normal. The output
    is consumed by cf2_consensus.obstacle_consensus_cmd.

Design note: the Lissajous coefficients are duplicated from cf2_milestone1.py
(rather than imported) so the two files have no circular dependency. They MUST
stay in sync -- if you change the path in one, mirror it here.
"""

import math


# Mirror the Lissajous amplitudes from cf2_milestone1.py so the target tracking
# stays consistent with the actual roaming path.
_PATH_AMP_X = 1.6
_PATH_AMP_Y = 3.0
_PATH_CENTER_Z = 1.0
_PATH_AMP_Z = 0.5


def intent_heading(robot_no, current_pose, t):
    """
    Unit 3D vector from the drone toward its current Lissajous target.

    Compared to a pure path-tangent heading, this points at where the drone
    actually wants to be -- so when an obstacle sits between the drone and its
    target, the heading (and hence the FoV cone) point at the obstacle, and
    the drone can see it before colliding.
    """
    px = float(current_pose[0])
    py = float(current_pose[1])
    pz = float(current_pose[2])
    # Per-drone frequencies (mirror cf2_milestone1._lissajous_target): each
    # drone gets a slightly different (omega_x, omega_y, omega_z) so the swarm
    # never collapses onto a single shared orbit.
    omega_x = 0.16 + 0.018 * (robot_no - 1)
    omega_y = 0.11 + 0.015 * (robot_no - 1)
    omega_z = 0.09 + 0.013 * (robot_no - 1)
    # Coprime phase offsets (2pi/3, 2pi/5, 2pi/7) further desynchronize the
    # axes between drones so their targets are spread across the volume.
    phase_x = (robot_no - 1) * (2.0 * math.pi / 3.0)
    phase_y = (robot_no - 1) * (2.0 * math.pi / 5.0)
    phase_z = (robot_no - 1) * (2.0 * math.pi / 7.0)
    tx = _PATH_AMP_X * math.sin(omega_x * t + phase_x)
    ty = _PATH_AMP_Y * math.sin(omega_y * t + phase_y)
    tz = _PATH_CENTER_Z + _PATH_AMP_Z * math.sin(omega_z * t + phase_z)
    # Heading = unit(target - position). Numerically guard the rare case where
    # we are exactly on the target by returning +x.
    dx = tx - px
    dy = ty - py
    dz = tz - pz
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-6:
        return 1.0, 0.0, 0.0
    return dx / norm, dy / norm, dz / norm


def _closest_point_to_obstacle(px, py, pz, ox, oy, sx, sy, sz):
    """
    Closest point on an axis-aligned obstacle box to the drone at (px, py, pz).
    Obstacles stand on the floor (z in [0, sz]), so the z-min is hardcoded to 0
    to match the convention used everywhere else in this project.

    For a point P and AABB [min, max], the closest box point is the per-axis
    clamp of P into [min, max] -- a textbook result, but worth recalling that
    it returns P itself when P is inside the box (distance becomes 0).
    """
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
    Return obstacles whose closest surface point falls inside the drone's
    3D forward cone (within sensing_radius AND within +/- fov_half_angle of
    the 3D heading direction).

    Each entry: {
        'closest_point': (cx, cy, cz),
        'normal':        (nx, ny, nz),   # unit 3D vector from surface toward drone
        'distance':      float,           # 3D distance to closest surface point
        'obstacle_idx':  int,
    }
    """
    px, py, pz = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    hx, hy, hz = heading
    # Precompute cos(fov_half_angle) once -- the in-cone test is a dot product
    # comparison, which is cheaper than calling acos per obstacle.
    cos_fov = math.cos(fov_half_angle)

    n_obstacles = obstacle_pose.shape[1] if obstacle_pose.size else 0
    sensed = []
    for i in range(n_obstacles):
        ox = float(obstacle_pose[0, i])
        oy = float(obstacle_pose[1, i])
        sx = float(obstacle_size[0, i])
        sy = float(obstacle_size[1, i])
        sz = float(obstacle_size[2, i])

        # Use the closest surface point (not the obstacle center) so the
        # distance and normal reflect the real geometry -- a long thin box
        # would otherwise look "far" by its center while one face is right
        # next to the drone.
        cx, cy, cz = _closest_point_to_obstacle(px, py, pz, ox, oy, sx, sy, sz)
        dx = cx - px
        dy = cy - py
        dz = cz - pz
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Range gate: ignore obstacles outside the FoV radius.
        if distance > sensing_radius:
            continue

        if distance < 1e-6:
            # Degenerate case: drone is on / inside the box. We can't form a
            # surface normal, so we treat it as an emergency and push back
            # along (the opposite of) the heading. The downstream consensus
            # law will see a large violation and react strongly.
            in_cone = True
            nx, ny, nz = -hx, -hy, -hz
        else:
            # Standard angular gate: in-cone iff cos(angle) >= cos(fov_half).
            cos_angle = (dx * hx + dy * hy + dz * hz) / distance
            in_cone = cos_angle >= cos_fov
            # Outward normal = unit vector pointing surface -> drone, i.e.
            # the direction the consensus law will push us along.
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
