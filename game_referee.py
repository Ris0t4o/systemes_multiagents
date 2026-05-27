"""
Per-frame catch detection. A hider is caught (forever) the first time any
flying seeker sees it through its vision cone with unobstructed LOS.
"""

try:
    from .rmtt_seeker import seeker_heading, can_see, SEEKER_FOV_HALF_ANGLE, SEEKER_VISION_RANGE
except ImportError:
    from rmtt_seeker import seeker_heading, can_see, SEEKER_FOV_HALF_ANGLE, SEEKER_VISION_RANGE


_caught = set()
_catch_times = {}


def update_catches(rmtt_poses, rmtt_states, cf2_poses, cf2_states,
                   obstacle_pose, obstacle_size, t):
    """Mark any flying hider currently in a flying seeker's cone."""
    nb_rmtt = rmtt_poses.shape[1] if rmtt_poses.size else 0
    nb_cf2 = cf2_poses.shape[1] if cf2_poses.size else 0
    for s_idx in range(nb_rmtt):
        # State 2 == Flying. Half-deployed seekers can't tag.
        if rmtt_states[s_idx] != 2:
            continue
        s_pose = rmtt_poses[:, s_idx]
        s_heading = seeker_heading(s_pose, t)
        s_pos = (float(s_pose[0]), float(s_pose[1]), float(s_pose[2]))
        for h_idx in range(nb_cf2):
            robot_no = h_idx + 1
            if robot_no in _caught:
                continue
            if cf2_states[h_idx] != 2:
                continue
            h_pos = (float(cf2_poses[0, h_idx]),
                     float(cf2_poses[1, h_idx]),
                     float(cf2_poses[2, h_idx]))
            if can_see(s_pos, h_pos, s_heading,
                       SEEKER_FOV_HALF_ANGLE, SEEKER_VISION_RANGE,
                       obstacle_pose, obstacle_size):
                _caught.add(robot_no)
                _catch_times[robot_no] = float(t)
                print(f"[SEEKER] CF2_{robot_no} spotted at t={float(t):.1f}s -- landing.")


def is_caught(robot_no):
    return int(robot_no) in _caught


def catch_count():
    return len(_caught)
