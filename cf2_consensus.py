"""
Consensus laws: per-agent forces from obstacles and from other hiders.
Each returns a 2D (fx, fy); the caller composes them and saturates.
"""

import math


def obstacle_consensus_cmd(sensed_obstacles, r_safe, k_obs, k_barrier=0.0):
    """
    Linear + barrier push away from sensed obstacles.

        u = sum_j (k_obs * e_j + k_barrier * (r_safe/d_j - 1)) * n_j
        with e_j = max(0, r_safe - d_j), n_j = outward normal.

    The barrier diverges as d -> 0, so it dominates any bounded attraction
    near contact. V = (1/2) sum e_j^2 is a Lyapunov function on the safe set.
    Saturation is the caller's job.
    """
    fx = 0.0
    fy = 0.0
    for s in sensed_obstacles:
        d = s['distance']
        if d >= r_safe:
            continue
        e = r_safe - d
        nx, ny = s['normal'][0], s['normal'][1]
        mag = k_obs * e
        if k_barrier > 0.0:
            # Floor d at 2% of r_safe to avoid divide-by-zero on penetration.
            d_safe = max(d, 0.02 * r_safe)
            mag += k_barrier * (r_safe / d_safe - 1.0)
        fx += mag * nx
        fy += mag * ny
    return fx, fy


def inter_agent_dispersion_cmd(self_xy, neighbor_xy, k_disp, eps=0.05):
    """
    Gradient of the Coulomb swarm potential V = (k/2) * sum 1/||p_i - p_j||.

        u_i = k * sum_{j != i} (p_i - p_j) / ||p_i - p_j||^3

    Long-range and symmetric: every hider repels every other, so the swarm
    maximizes pairwise distances while obstacle/boundary terms steer the
    centroid. `eps` floors the pairwise distance against numerical blow-up.
    """
    sx = float(self_xy[0])
    sy = float(self_xy[1])

    fx = 0.0
    fy = 0.0
    eps2 = eps * eps
    for nx, ny in neighbor_xy:
        dx = sx - float(nx)
        dy = sy - float(ny)
        d2 = dx * dx + dy * dy
        if d2 < eps2:
            d2 = eps2
        d = math.sqrt(d2)
        inv_d3 = 1.0 / (d * d2)
        fx += k_disp * dx * inv_d3
        fy += k_disp * dy * inv_d3
    return fx, fy
