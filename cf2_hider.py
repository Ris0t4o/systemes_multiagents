"""
Hider-side coordination: shared seeker-sighting memory and cover geometry.
The seeker never touches _last_seen. communication is hider-only.
"""

import math


# Shared broadcast channel. Last write wins; no locking needed (atomic dict writes).
_last_seen = {
    'position': None,
    'timestamp': None,
}

# TTL chosen larger than a seeker half-cycle (~30s) so the swarm stays in hide
# mode while the threat is plausibly nearby, but short enough to return to roam.
SEEKER_MEMORY_TTL = 5.0


def report_seeker_sighting(seeker_pos, timestamp):
    """A hider that just saw the seeker writes here."""
    prev_ts = _last_seen['timestamp']
    ts = float(timestamp)
    # Entering alert from roam (no sighting, or last one expired) → fresh cover solve.
    if prev_ts is None or ts - prev_ts > SEEKER_MEMORY_TTL:
        reset_cover_cache()
    _last_seen['position'] = (float(seeker_pos[0]), float(seeker_pos[1]), float(seeker_pos[2]))
    _last_seen['timestamp'] = ts


def get_last_seen(current_time):
    """Return (x, y, z) if a sighting is fresh, else None."""
    pos = _last_seen['position']
    ts = _last_seen['timestamp']
    if pos is None or ts is None:
        return None
    if float(current_time) - ts > SEEKER_MEMORY_TTL:
        return None
    return pos


# COVER_OFFSET > R_SAFE (=0.6) so the cover point sits outside the obstacle
# consensus push zone — the hider can rest there without fighting avoidance.
COVER_OFFSET = 0.8


def seek_cover_target(robot_pose, seeker_pos, obstacle_pose, obstacle_size):
    """
    Pick an xy point behind the nearest obstacle relative to the seeker.

    Cover candidate per obstacle: O + (t_surface + COVER_OFFSET) * unit(O - S),
    where t_surface is the AABB exit distance along (O - S). This guarantees
    the point sits outside the box even for long/asymmetric obstacles.

    Falls back to a pure-flee step if no usable obstacle exists.
    """
    px = float(robot_pose[0])
    py = float(robot_pose[1])
    pz = float(robot_pose[2])
    sx = float(seeker_pos[0])
    sy = float(seeker_pos[1])

    n_obstacles = obstacle_pose.shape[1] if obstacle_pose.size else 0
    best = None
    best_d2 = float('inf')
    for i in range(n_obstacles):
        ox = float(obstacle_pose[0, i])
        oy = float(obstacle_pose[1, i])
        half_x = float(obstacle_size[0, i]) * 0.5
        half_y = float(obstacle_size[1, i]) * 0.5
        dox = ox - sx
        doy = oy - sy
        norm = math.hypot(dox, doy)
        if norm < 1e-6:
            continue
        ux = dox / norm
        uy = doy / norm
        ax = abs(ux)
        ay = abs(uy)
        if ax < 1e-9:
            t_surface = half_y / max(ay, 1e-9)
        elif ay < 1e-9:
            t_surface = half_x / max(ax, 1e-9)
        else:
            t_surface = min(half_x / ax, half_y / ay)
        cover_d = t_surface + COVER_OFFSET
        cx = ox + cover_d * ux
        cy = oy + cover_d * uy
        d2 = (cx - px) ** 2 + (cy - py) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (cx, cy, pz)

    if best is not None:
        return best

    # Fallback: 1 m further from the seeker along hider->seeker direction.
    dx = px - sx
    dy = py - sy
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return (px + 1.0, py, pz)
    return (px + dx / norm, py + dy / norm, pz)


# AABB face normals in 2D — one cover candidate per face on the far side.
_AABB_FACE_NORMALS = ((+1.0, 0.0), (-1.0, 0.0), (0.0, +1.0), (0.0, -1.0))


# Lock the (hider -> face) mapping for the duration of an alert period so the
# brute-force matching can't oscillate frame-to-frame on near-ties.
_cover_assignment_cache = {
    'assignment': None,            # list of (cx, cy) per hider
    'computed_at': None,           # sim time when computed
    'seeker_pos_at_compute': None, # seeker xyz when computed
}

# Refresh interval (s) for the cached assignment, so a slowly drifting seeker
# eventually triggers a re-solve even if it never crosses the position gate.
_COVER_CACHE_MAX_AGE = 1.0
# Position gate: seeker moves more than this since the cache was built -> resolve.
_COVER_CACHE_SEEKER_DELTA = 0.5


def reset_cover_cache():
    """Invalidate the cached assignment. Called when the swarm re-enters alert."""
    _cover_assignment_cache['assignment'] = None
    _cover_assignment_cache['computed_at'] = None
    _cover_assignment_cache['seeker_pos_at_compute'] = None


def _enumerate_cover_candidates(seeker_pos, obstacle_pose, obstacle_size):
    """Per-face cover points on the far side of each obstacle."""
    sx = float(seeker_pos[0])
    sy = float(seeker_pos[1])

    n_obstacles = obstacle_pose.shape[1] if obstacle_pose.size else 0
    candidates = []
    for i in range(n_obstacles):
        ox = float(obstacle_pose[0, i])
        oy = float(obstacle_pose[1, i])
        half_x = float(obstacle_size[0, i]) * 0.5
        half_y = float(obstacle_size[1, i]) * 0.5
        dox = ox - sx
        doy = oy - sy
        for nx, ny in _AABB_FACE_NORMALS:
            # Keep only faces on the far side (normal . (O - S) > 0).
            if nx * dox + ny * doy <= 0.0:
                continue
            face_cx = ox + nx * half_x
            face_cy = oy + ny * half_y
            cx = face_cx + nx * COVER_OFFSET
            cy = face_cy + ny * COVER_OFFSET
            candidates.append((cx, cy))
    return candidates


def _best_assignment(cost_matrix):
    """Brute-force optimal injective assignment (N rows to M cols, N <= M).

    Search space stays tiny (<= 1680 perms for N=4, M=8). avoids a SciPy dep.
    """
    n = len(cost_matrix)
    if n == 0:
        return []
    m = len(cost_matrix[0])
    best_total = float('inf')
    best_assign = None

    chosen = [-1] * n
    used = [False] * m

    def recurse(row, total):
        nonlocal best_total, best_assign
        if total >= best_total:
            return
        if row == n:
            best_total = total
            best_assign = chosen.copy()
            return
        for col in range(m):
            if used[col]:
                continue
            used[col] = True
            chosen[row] = col
            recurse(row + 1, total + cost_matrix[row][col])
            used[col] = False

    recurse(0, 0.0)
    return best_assign if best_assign is not None else list(range(n))


def assign_cover_targets(cf2_xy, seeker_pos, obstacle_pose, obstacle_size, current_time):
    """
    Decentralized cover assignment: every hider runs this deterministic
    min-cost matching on identical inputs and reaches the same global
    assignment without exchanging messages.

    Caches the result for ~1 s (and until the seeker moves more than ~0.5 m)
    so the assignment can't flicker between near-tie candidates frame to frame.

    Returns one (cx, cy) per hider (indexed by robot_no - 1). Hiders left
    without a real cover candidate get a fanned-out fallback around the seeker.
    """
    # Cache hit: same seeker position (approximately), still fresh.
    cached = _cover_assignment_cache['assignment']
    if cached is not None and len(cached) == len(cf2_xy):
        t_prev = _cover_assignment_cache['computed_at']
        sp = _cover_assignment_cache['seeker_pos_at_compute']
        if t_prev is not None and sp is not None:
            age = float(current_time) - float(t_prev)
            ds2 = ((float(seeker_pos[0]) - sp[0]) ** 2
                   + (float(seeker_pos[1]) - sp[1]) ** 2
                   + (float(seeker_pos[2]) - sp[2]) ** 2)
            if age < _COVER_CACHE_MAX_AGE and ds2 < _COVER_CACHE_SEEKER_DELTA ** 2:
                return cached

    n_hiders = len(cf2_xy)
    candidates = _enumerate_cover_candidates(seeker_pos, obstacle_pose, obstacle_size)
    sx = float(seeker_pos[0])
    sy = float(seeker_pos[1])

    targets = [None] * n_hiders

    if candidates and n_hiders > 0:
        m = len(candidates)
        n_assigned = min(n_hiders, m)
        cost = [
            [
                (float(cf2_xy[i][0]) - candidates[j][0]) ** 2
                + (float(cf2_xy[i][1]) - candidates[j][1]) ** 2
                for j in range(m)
            ]
            for i in range(n_assigned)
        ]
        assignment = _best_assignment(cost)
        for i in range(n_assigned):
            targets[i] = candidates[assignment[i]]

    # Fan unassigned hiders radially around the seeker so they don't stack up.
    fallback_radius = 1.5
    fallback_count = sum(1 for t in targets if t is None)
    fallback_idx = 0
    for i in range(n_hiders):
        if targets[i] is not None:
            continue
        angle = (2.0 * math.pi * fallback_idx) / max(fallback_count, 1)
        cx = sx + fallback_radius * math.cos(angle)
        cy = sy + fallback_radius * math.sin(angle)
        targets[i] = (cx, cy)
        fallback_idx += 1

    _cover_assignment_cache['assignment'] = targets
    _cover_assignment_cache['computed_at'] = float(current_time)
    _cover_assignment_cache['seeker_pos_at_compute'] = (
        float(seeker_pos[0]), float(seeker_pos[1]), float(seeker_pos[2])
    )
    return targets
