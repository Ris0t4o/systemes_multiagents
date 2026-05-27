"""
Seeker controller (DJI RoboMaster TT / RMTT).

The seeker has two jobs:
  1. Roam the arena on a slow 3D sweep, never colliding with obstacles.
  2. Catch hiders by having them fall inside its forward vision cone with
     unobstructed line of sight.

This module also exposes `can_see`, the LOS predicate used both by the seeker
itself (for catches, via game_referee) and by the hiders (to know when the
seeker has been spotted, via cf2_milestone1).

The seeker's control inputs do NOT include hider positions. That asymmetry is
the whole game -- it has to find them with vision, not by reading their state.
"""

import math
try:
    from .cf2_sensing import sense_obstacles
    from .cf2_consensus import obstacle_consensus_cmd
except ImportError:
    # For standalone testing of this module, allow importing from the same directory.
    from cf2_sensing import sense_obstacles
    from cf2_consensus import obstacle_consensus_cmd


# --- Sweep path ----------------------------------------------------------
# Slow 3D Lissajous, deliberately decoupled from the hiders' parameters.
# Stays inside arena bounds with comfortable margins.
_CENTER_X = 0.0
_CENTER_Y = 0.0
_CENTER_Z = 1.0
_AMP_X = 1.8
_AMP_Y = 3.5
_AMP_Z = 0.4
_OMEGA_X = 0.18                 # 2 * pi / 0.18 ~ 35 s per cycle
_OMEGA_Y = 0.13                 #                ~ 48 s
_OMEGA_Z = 0.08                 #                ~ 79 s
_PHASE_Y = math.pi / 2.0        # quadrature with x -> figure-of-eight sweep

# --- Search cone (for step 2: hider detection). Exposed for the sim. -----
# Narrower than the hiders' cone so the seeker has to actively aim, but a
# longer range so distant hiders can still be caught from across the room.
SEEKER_FOV_HALF_ANGLE = math.radians(30)    # narrower than hiders (which use 60 deg)
SEEKER_VISION_RANGE = 3.5                   # m

# --- Obstacle avoidance --------------------------------------------------
# Omnidirectional sensing (math.pi half-angle): the seeker's "proximity" sense
# is separate from its narrow forward "vision", so it doesn't blind-spot into
# walls while sweeping. Consensus law shared with the hiders.
_OBS_SENSING_RADIUS = 2.0
_OBS_FOV_HALF_ANGLE = math.pi
_R_SAFE = 0.6
_K_OBS = 2.0
_K_BARRIER = 0.5

# --- Cruise speeds -------------------------------------------------------
# Deliberately slower than the hider cap (0.55 in cf2_milestone1) so the
# hiders can outrun the seeker once spotted -- the game was unwinnable for
# the hiders when the seeker was the faster agent.
_VMAX_XY = 0.35
_VMAX_Z = 0.25
_GOAL_GAIN = 0.6


def _seeker_target(t):
    """The seeker's instantaneous Lissajous target. Quadrature in y produces a
    smooth figure-of-eight sweep over the floor while z wobbles gently."""
    tx = _CENTER_X + _AMP_X * math.sin(_OMEGA_X * t)
    ty = _CENTER_Y + _AMP_Y * math.sin(_OMEGA_Y * t + _PHASE_Y)
    tz = _CENTER_Z + _AMP_Z * math.sin(_OMEGA_Z * t)
    return tx, ty, tz


def seeker_heading(current_pose, t):
    """Unit 3D vector from the seeker toward its current sweep target.

    The same vector is reused as the search-cone axis (step 2) and as the
    heading argument to the omnidirectional obstacle sensor (here).
    """
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
    """
    Step-1 seeker controller.

    Roams a slow 3D Lissajous, avoids obstacles with the same consensus law as
    the hiders (using its omnidirectional proximity sense), and NEVER references
    hider positions -- enforced structurally: this function doesn't accept them.

    Returns (vx, vy, vz, led).
    """
    px = float(robot_pose[0])
    py = float(robot_pose[1])
    pz = float(robot_pose[2])
    t = float(clock)

    tx, ty, tz = _seeker_target(t)

    # 1) Smooth attraction toward sweep target.
    # Same proportional law as the hiders, with a slightly lower gain (0.6)
    # since the seeker doesn't need to chase anything aggressively.
    fx = _GOAL_GAIN * (tx - px)
    fy = _GOAL_GAIN * (ty - py)
    fz = _GOAL_GAIN * (tz - pz)

    # 2) Obstacle consensus (omnidirectional sensing).
    # We feed the same Lissajous heading as the "look direction" but with a
    # 180-degree (math.pi) half-angle, which effectively turns the cone test
    # in sense_obstacles into a pure range test.
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

    # 3) Saturation. xy via magnitude cap, z via per-axis clamp.
    speed_xy = math.hypot(fx, fy)
    if speed_xy > _VMAX_XY and speed_xy > 1e-9:
        scale = _VMAX_XY / speed_xy
        fx *= scale
        fy *= scale
    fz = max(-_VMAX_Z, min(_VMAX_Z, fz))

    led = (255, 120, 0)   # orange -- matches the sim's RMTT marker color
    return fx, fy, fz, led


# --- Line-of-sight ------------------------------------------------------

def _segment_intersects_aabb(p0, p1, box_min, box_max):
    """Slab test: True if the line segment p0->p1 enters the axis-aligned box.

    Standard ray/AABB intersection (Kay & Kajiya): for each axis, compute the
    interval of parameter t in which the ray lies inside the slab between the
    two parallel faces. If the intersection of the three intervals overlaps
    [0, 1], the segment hits the box.
    """
    eps = 1e-9
    t_enter = 0.0
    t_exit = 1.0
    for axis in range(3):
        d = p1[axis] - p0[axis]
        if abs(d) < eps:
            # Ray is parallel to this pair of faces. Miss unless the starting
            # point already lies between them.
            if p0[axis] < box_min[axis] or p0[axis] > box_max[axis]:
                return False
        else:
            # Parameters at which the ray crosses the two slab planes.
            t1 = (box_min[axis] - p0[axis]) / d
            t2 = (box_max[axis] - p0[axis]) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_enter = max(t_enter, t1)
            t_exit = min(t_exit, t2)
            # If the running intersection becomes empty, we miss the box.
            if t_enter > t_exit:
                return False
    return True


def can_see(seeker_pos, target_pos, heading_xyz, fov_half_angle, vision_range,
            obstacle_pose, obstacle_size):
    """
    True iff target_pos is (a) inside the cone with apex seeker_pos, axis
    heading_xyz, half-angle fov_half_angle, slant length vision_range; AND
    (b) not occluded by any obstacle box.
    """
    sx, sy, sz = seeker_pos
    tx, ty, tz = target_pos
    dx, dy, dz = tx - sx, ty - sy, tz - sz
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    # Range gate first -- cheap and rejects most targets immediately.
    if distance > vision_range:
        return False
    if distance < 1e-6:
        # Same point: trivially visible.
        return True
    hx, hy, hz = heading_xyz
    # Cone test via dot product. cos_angle = (target - apex) . heading / |.|
    cos_angle = (dx * hx + dy * hy + dz * hz) / distance
    if cos_angle < math.cos(fov_half_angle):
        return False
    # Occlusion test: any obstacle box that intersects the LOS segment blocks vision.
    n_obstacles = obstacle_pose.shape[1] if obstacle_pose.size else 0
    for i in range(n_obstacles):
        ox = float(obstacle_pose[0, i])
        oy = float(obstacle_pose[1, i])
        bx = float(obstacle_size[0, i])
        by = float(obstacle_size[1, i])
        bz = float(obstacle_size[2, i])
        # Obstacles stand on the floor (z in [0, bz]) -- same convention as
        # the rest of the project.
        box_min = (ox - bx / 2.0, oy - by / 2.0, 0.0)
        box_max = (ox + bx / 2.0, oy + by / 2.0, bz)
        if _segment_intersects_aabb((sx, sy, sz), (tx, ty, tz), box_min, box_max):
            return False
    return True
