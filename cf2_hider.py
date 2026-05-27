"""
Inter-hider communication and cover-seeking geometry.

Two responsibilities, one module:

1. A *shared-memory broadcast channel* for the swarm. Whichever hider sees the
   seeker writes its last-known position to `_last_seen`; every other hider
   reads it. A TTL is applied so the swarm forgets stale sightings and
   eventually returns to roaming. The seeker NEVER touches this dict --
   communication is hider-only by construction.

2. A `seek_cover_target` function that, given the last-known seeker position,
   picks a 2D point that sits BEHIND an obstacle relative to the seeker, with
   enough offset to avoid being repelled by the consensus law.
"""

import math


# --- Inter-hider communication (shared last-known seeker position) -----
# Module-level state is the broadcast channel: whichever hider sees the seeker
# writes here, and every other hider reads from here. The seeker never touches it.
_last_seen = {
    'position': None,    # tuple (x, y, z) or None
    'timestamp': None,   # sim time when reported
}

# How long a sighting remains "fresh". Picked larger than the typical time a
# seeker takes to cross the arena (~30s for a half cycle of its Lissajous), so
# the swarm stays in hide mode while the threat is plausibly nearby, but short
# enough that they return to free roaming once it has clearly moved on.
SEEKER_MEMORY_TTL = 5.0   # seconds


def report_seeker_sighting(seeker_pos, timestamp):
    """A hider that just saw the seeker calls this. Last write wins."""
    # No locking: the simulator runs controllers in a thread pool, but each
    # controller writes the whole record atomically (3-tuple + timestamp), and
    # readers tolerate an out-of-order write since they just use the latest.
    _last_seen['position'] = (float(seeker_pos[0]), float(seeker_pos[1]), float(seeker_pos[2]))
    _last_seen['timestamp'] = float(timestamp)


def get_last_seen(current_time):
    """Returns (x, y, z) if a sighting is fresh (within TTL), else None."""
    pos = _last_seen['position']
    ts = _last_seen['timestamp']
    if pos is None or ts is None:
        return None
    # TTL gate: forget sightings older than SEEKER_MEMORY_TTL so the swarm
    # eventually returns to its default (roaming) behaviour.
    if float(current_time) - ts > SEEKER_MEMORY_TTL:
        return None
    return pos


# --- Cover-seeking ------------------------------------------------------
# COVER_OFFSET sits PAST THE OBSTACLE SURFACE (not the center) and exceeds
# R_SAFE = 0.6 so the cover point itself is outside the obstacle consensus
# push zone -- the hider can come to rest there without fighting avoidance.
COVER_OFFSET = 0.8


def seek_cover_target(robot_pose, seeker_pos, obstacle_pose, obstacle_size):
    """
    Pick an xy point behind an obstacle relative to the (last-known) seeker
    position. For each obstacle, the cover candidate is

        O + (t_surface + COVER_OFFSET) * unit(O - S)

    where t_surface is the distance from the obstacle center to its AABB
    surface along the direction (O - S). This guarantees the cover point is
    outside the obstacle box -- critical for long/asymmetric obstacles.

    Falls back to a pure-flee point 1 m further away from the seeker along the
    hider->seeker vector if no usable obstacle exists. Returns (cx, cy, pz) --
    z stays at the hider's current altitude.
    """
    px = float(robot_pose[0])
    py = float(robot_pose[1])
    pz = float(robot_pose[2])
    sx = float(seeker_pos[0])
    sy = float(seeker_pos[1])

    n_obstacles = obstacle_pose.shape[1] if obstacle_pose.size else 0
    # We rank candidate cover points by distance from the hider -- nearest
    # cover wins, which keeps the manoeuvre short and predictable.
    best = None
    best_d2 = float('inf')
    for i in range(n_obstacles):
        ox = float(obstacle_pose[0, i])
        oy = float(obstacle_pose[1, i])
        half_x = float(obstacle_size[0, i]) * 0.5
        half_y = float(obstacle_size[1, i]) * 0.5
        # Direction from seeker to obstacle center: cover lies *past* the
        # obstacle along this same direction.
        dox = ox - sx
        doy = oy - sy
        norm = math.hypot(dox, doy)
        if norm < 1e-6:
            # Seeker is at the obstacle center -- no meaningful "behind".
            continue
        ux = dox / norm
        uy = doy / norm
        # Parametric exit distance of the ray (O + t*u) from the AABB.
        # For each axis with non-zero direction component, t_axis = half_axis / |u_axis|
        # is the parameter at which the ray crosses that face; the actual exit
        # is the minimum over axes -- that's where the ray first leaves the box.
        ax = abs(ux)
        ay = abs(uy)
        if ax < 1e-9:
            # Ray is purely along +/- y, exit through a y-face.
            t_surface = half_y / max(ay, 1e-9)
        elif ay < 1e-9:
            # Ray is purely along +/- x, exit through an x-face.
            t_surface = half_x / max(ax, 1e-9)
        else:
            t_surface = min(half_x / ax, half_y / ay)
        # Stand COVER_OFFSET behind the far face -- and since COVER_OFFSET > R_SAFE,
        # we sit outside the obstacle consensus push zone.
        cover_d = t_surface + COVER_OFFSET
        cx = ox + cover_d * ux
        cy = oy + cover_d * uy
        # Use squared distance: same ranking, no sqrt.
        d2 = (cx - px) ** 2 + (cy - py) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (cx, cy, pz)

    if best is not None:
        return best

    # Pure-flee fallback: move 1 m further from the seeker along hider->seeker direction.
    # Used only when no obstacle is usable (e.g. seeker stands at an obstacle
    # center). Better than freezing in place even though it offers no occlusion.
    dx = px - sx
    dy = py - sy
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        # Hider exactly at seeker -- arbitrary +x flee.
        return (px + 1.0, py, pz)
    return (px + dx / norm, py + dy / norm, pz)


# --- Multi-agent cover assignment ---------------------------------------
# Generate cover candidates per AABB face (4 faces per obstacle in 2D), keep
# only the faces that are on the FAR side of the obstacle relative to the
# seeker -- those are the ones that actually occlude line of sight.
_AABB_FACE_NORMALS = ((+1.0, 0.0), (-1.0, 0.0), (0.0, +1.0), (0.0, -1.0))


def _enumerate_cover_candidates(seeker_pos, obstacle_pose, obstacle_size):
    """Build the list of xy cover points that sit COVER_OFFSET behind one
    AABB face of one obstacle. Same geometry as `seek_cover_target`, but
    one point per face rather than one per obstacle, so multiple hiders can
    occupy distinct cover points around the same obstacle."""
    sx = float(seeker_pos[0])
    sy = float(seeker_pos[1])

    n_obstacles = obstacle_pose.shape[1] if obstacle_pose.size else 0
    candidates = []
    for i in range(n_obstacles):
        ox = float(obstacle_pose[0, i])
        oy = float(obstacle_pose[1, i])
        half_x = float(obstacle_size[0, i]) * 0.5
        half_y = float(obstacle_size[1, i]) * 0.5
        # Seeker -> obstacle direction. Faces with normal . this_direction > 0
        # are on the far side and provide real occlusion.
        dox = ox - sx
        doy = oy - sy
        for nx, ny in _AABB_FACE_NORMALS:
            if nx * dox + ny * doy <= 0.0:
                continue
            # Cover point: face center pushed COVER_OFFSET along the outward
            # normal. Face center is the obstacle center shifted by half the
            # box side along the face's axis.
            face_cx = ox + nx * half_x
            face_cy = oy + ny * half_y
            cx = face_cx + nx * COVER_OFFSET
            cy = face_cy + ny * COVER_OFFSET
            candidates.append((cx, cy))
    return candidates


def _best_assignment(cost_matrix):
    """Brute-force optimal injective assignment of N rows (hiders) to M
    columns (candidates), with N <= M. Returns a list `assign` of length N
    where `assign[i]` is the column index chosen for row i, minimizing the
    sum of cost_matrix[i, assign[i]].

    With N <= 4 and M <= 8 the search space is at most 8*7*6*5 = 1680
    permutations -- negligible. Avoids a SciPy dependency."""
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
            # Lower-bound prune: cannot improve.
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


def assign_cover_targets(cf2_xy, seeker_pos, obstacle_pose, obstacle_size):
    """
    Decentralized multi-agent cover assignment.

    Every hider runs this same deterministic function on identical inputs
    (the shared seeker sighting plus the simulator-provided pose matrix),
    so they reach the same global assignment without exchanging messages.
    This is the "every node solves the same optimization" flavor of
    consensus -- a one-shot agreement on which cover belongs to whom.

    Algorithm:
      1. Enumerate candidate cover points: one per AABB face that lies on
         the far side of an obstacle relative to the seeker.
      2. Build a cost matrix C[i, j] = squared xy distance from hider i to
         candidate j.
      3. Solve min-cost injective assignment via brute force.
      4. If there are fewer candidates than hiders, fill the rest with a
         pure-flee fallback fanned out by 2*pi/N around the seeker so they
         don't pile up on the same line.

    Returns a list of (cx, cy) tuples, one per hider, indexed by robot_no-1.
    """
    n_hiders = len(cf2_xy)
    candidates = _enumerate_cover_candidates(seeker_pos, obstacle_pose, obstacle_size)
    sx = float(seeker_pos[0])
    sy = float(seeker_pos[1])

    targets = [None] * n_hiders

    if candidates and n_hiders > 0:
        m = len(candidates)
        # If we have more hiders than far-side cover faces, only the first
        # min(n, m) hiders get a real cover point; the rest fall back below.
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

    # Fallback for any hider that didn't get a real cover candidate: stand
    # 1.5 m from the seeker on an angular offset, so they spread radially
    # around the seeker instead of stacking up on one ray.
    fallback_radius = 1.5
    fallback_count = sum(1 for t in targets if t is None)
    fallback_idx = 0
    for i in range(n_hiders):
        if targets[i] is not None:
            continue
        # Angle is per-hider so the fan is stable across frames.
        angle = (2.0 * math.pi * fallback_idx) / max(fallback_count, 1)
        cx = sx + fallback_radius * math.cos(angle)
        cy = sy + fallback_radius * math.sin(angle)
        targets[i] = (cx, cy)
        fallback_idx += 1

    return targets
