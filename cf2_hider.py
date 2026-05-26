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
SEEKER_MEMORY_TTL = 8.0   # seconds


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
