"""
Seeker controller (RMTT). Roams a slow Lissajous, avoids obstacles, and
exposes `can_see` (used by both the referee and the hiders).

The seeker has no access to hider positions — that asymmetry is the game.
"""

import math
try:
    from .cf2_sensing import sense_obstacles
    from .cf2_consensus import obstacle_consensus_cmd
except ImportError:
    from cf2_sensing import sense_obstacles
    from cf2_consensus import obstacle_consensus_cmd


# Slow Lissajous sweep — quadrature in y produces a figure-of-eight.
_CENTER_X = 0.0
_CENTER_Y = 0.0
_CENTER_Z = 1.0
_AMP_X = 1.8
_AMP_Y = 3.5
_AMP_Z = 0.4
_OMEGA_X = 0.18                 # ~35 s / cycle
_OMEGA_Y = 0.13                 # ~48 s
_OMEGA_Z = 0.08                 # ~79 s
_PHASE_Y = math.pi / 2.0

# Narrow vision cone. Exposed for the sim/referee.
SEEKER_FOV_HALF_ANGLE = math.radians(30)
SEEKER_VISION_RANGE = 2.0

# Omnidirectional proximity sensing — separate from vision.
_OBS_SENSING_RADIUS = 2.0
_OBS_FOV_HALF_ANGLE = math.pi
_R_SAFE = 0.6
_K_OBS = 2.0
_K_BARRIER = 0.5

# Slower than hider cap so hiders can outrun once spotted.
_VMAX_XY = 0.35
_VMAX_Z = 0.25
_GOAL_GAIN = 0.6


def _seeker_target(t):
    """Instantaneous Lissajous target."""
    tx = _CENTER_X + _AMP_X * math.sin(_OMEGA_X * t)
    ty = _CENTER_Y + _AMP_Y * math.sin(_OMEGA_Y * t + _PHASE_Y)
    tz = _CENTER_Z + _AMP_Z * math.sin(_OMEGA_Z * t)
    return tx, ty, tz


def seeker_heading(current_pose, t):
    """Unit 3D vector from the seeker toward its current sweep target."""
    px = float(current_pose[0])
    py = float(current_pose[1])
    pz = float(current_pose[2])
    tx, ty, tz = _seeker_target(t)
    dx = tx - px
    dy = ty - py
    dz = tz - pz
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-6:
        return 1.0, 0.0, 0.
    return dx / norm, dy / norm, dz / norm


def compute_seeker_cmd(robot_no, robot_pose, obstacle_pose, obstacle_size, clock):
    """Lissajous roam + omnidirectional obstacle consensus. Returns (vx, vy, vz, led)."""
    px = float(robot_pose[0])
    py = float(robot_pose[1])
    pz = float(robot_pose[2])
    t = float(clock)

    tx, ty, tz = _seeker_target(t)

    fx = _GOAL_GAIN * (tx - px)
    fy = _GOAL_GAIN * (ty - py)
    fz = _GOAL_GAIN * (tz - pz)

    # Omnidirectional sensing — pass any heading; the math.pi half-angle
    # collapses the cone test to a pure range test.
    heading = seeker_heading(robot_pose, t)
    sensed = sense_obstacles(
        robot_pose, heading,
        _OBS_FOV_HALF_ANGLE,
        _OBS_SENSING_RADIUS,
        obstacle_pose, obstacle_size,
    )
    fx_obs, fy_obs = obstacle_consensus_cmd(
        sensed, _R_SAFE, _K_OBS, _K_BARRIER,
    )
    fx += fx_obs
    fy += fy_obs

    speed_xy = math.hypot(fx, fy)
    if speed_xy > _VMAX_XY and speed_xy > 1e-9:
        scale = _VMAX_XY / speed_xy
        fx *= scale
        fy *= scale
    fz = max(-_VMAX_Z, min(_VMAX_Z, fz))

    led = (255, 120, 0)
    return fx, fy, fz, led


def _segment_intersects_aabb(p0, p1, box_min, box_max):
    """Slab test (Kay & Kajiya): True if segment p0->p1 hits the AABB."""
    eps = 1e-9
    t_enter = 0.0
    t_exit = 1.0
    for axis in range(3):
        d = p1[axis] - p0[axis]
        if abs(d) < eps:
            if p0[axis] < box_min[axis] or p0[axis] > box_max[axis]:
                return False
        else:
            t1 = (box_min[axis] - p0[axis]) / d
            t2 = (box_max[axis] - p0[axis]) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_enter = max(t_enter, t1)
            t_exit = min(t_exit, t2)
            if t_enter > t_exit:
                return False
    return True


def can_see(seeker_pos, target_pos, heading_xyz, fov_half_angle, vision_range,
            obstacle_pose, obstacle_size):
    """True iff target is in the cone AND not occluded by any obstacle."""
    sx, sy, sz = seeker_pos
    tx, ty, tz = target_pos
    dx, dy, dz = tx - sx, ty - sy, tz - sz
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance > vision_range:
        return False
    if distance < 1e-6:
        return True
    hx, hy, hz = heading_xyz
    cos_angle = (dx * hx + dy * hy + dz * hz) / distance
    if cos_angle < math.cos(fov_half_angle):
        return False
    n_obstacles = obstacle_pose.shape[1] if obstacle_pose.size else 0
    for i in range(n_obstacles):
        ox = float(obstacle_pose[0, i])
        oy = float(obstacle_pose[1, i])
        bx = float(obstacle_size[0, i])
        by = float(obstacle_size[1, i])
        bz = float(obstacle_size[2, i])
        box_min = (ox - bx / 2.0, oy - by / 2.0, 0.0)
        box_max = (ox + bx / 2.0, oy + by / 2.0, bz)
        if _segment_intersects_aabb((sx, sy, sz), (tx, ty, tz), box_min, box_max):
            return False
    return True
