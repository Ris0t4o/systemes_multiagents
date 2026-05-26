"""
Game referee: tracks which hiders have been caught.

The simulator calls `update_catches` once per frame after all kinematics have
been integrated. For each flying seeker we ask `rmtt_seeker.can_see` against
every flying hider; the first time a hider is visible, we add it to a "caught"
set and record the catch time. The hider's controller (cf2_milestone1) then
sees the caught flag via `is_caught` and triggers a landing.

The module exposes three things:
  - update_catches(...) : per-frame catch detection
  - is_caught(robot_no) : queried by the hider controller
  - catch_count()       : queried by the HUD in sim_viz
"""

try:
    from .rmtt_seeker import seeker_heading, can_see, SEEKER_FOV_HALF_ANGLE, SEEKER_VISION_RANGE
except ImportError:
    from rmtt_seeker import seeker_heading, can_see, SEEKER_FOV_HALF_ANGLE, SEEKER_VISION_RANGE


# Module-level state. A hider is "caught" forever once caught -- this matches
# the hide-and-seek game where a tagged player is out for the round.
_caught = set()
_catch_times = {}


def update_catches(rmtt_poses, rmtt_states, cf2_poses, cf2_states,
                   obstacle_pose, obstacle_size, t):
    """Per-frame: mark any flying hider currently in a flying seeker's cone."""
    nb_rmtt = rmtt_poses.shape[1] if rmtt_poses.size else 0
    nb_cf2 = cf2_poses.shape[1] if cf2_poses.size else 0
    for s_idx in range(nb_rmtt):
        # State == 2 is "Flying" in the simulator's state machine. We skip
        # taking-off/landing/idle seekers so a half-deployed seeker can't tag
        # anyone.
        if rmtt_states[s_idx] != 2:
            continue
        s_pose = rmtt_poses[:, s_idx]
        # Same heading the seeker controller uses, so the catch cone is exactly
        # the on-screen vision cone -- no surprise tags from a divergent axis.
        s_heading = seeker_heading(s_pose, t)
        s_pos = (float(s_pose[0]), float(s_pose[1]), float(s_pose[2]))
        for h_idx in range(nb_cf2):
            robot_no = h_idx + 1
            # "Out for the round": already-caught hiders stay caught.
            if robot_no in _caught:
                continue
            if cf2_states[h_idx] != 2:
                continue
            h_pos = (float(cf2_poses[0, h_idx]),
                     float(cf2_poses[1, h_idx]),
                     float(cf2_poses[2, h_idx]))
            # can_see handles both the FoV cone test and the obstacle LOS
            # occlusion, so a hider behind a box can't be caught even if it's
            # within the cone.
            if can_see(s_pos, h_pos, s_heading,
                       SEEKER_FOV_HALF_ANGLE, SEEKER_VISION_RANGE,
                       obstacle_pose, obstacle_size):
                _caught.add(robot_no)
                _catch_times[robot_no] = float(t)
                print(f"[SEEKER] CF2_{robot_no} spotted at t={float(t):.1f}s -- landing.")


def is_caught(robot_no):
    """Used by the hider controller to know it should trigger a landing."""
    return int(robot_no) in _caught


def catch_count():
    """Used by the HUD to display the running tally."""
    return len(_caught)
