"""
Consensus laws used by the CF2 hider swarm.

Two primitives live here:

  - `obstacle_consensus_cmd`: per-agent, Lyapunov-decreasing + barrier push away
    from sensed obstacles. Used by both hiders and the seeker.
  - `inter_agent_dispersion_cmd`: per-agent gradient of the swarm-level
    dispersion potential V = (k/2) * sum_{i != j} 1 / ||p_i - p_j|| (Coulomb
    form). Drives the hider swarm toward a maximally-dispersed configuration
    so the seeker cannot catch everyone in one FoV sweep.

Both functions return a 2D (fx, fy) force; the caller composes them with the
goal attraction and the actuator saturation.
"""

import math


def obstacle_consensus_cmd(sensed_obstacles, r_safe, k_obs, k_barrier=0.0):
    """
    Obstacle consensus law (linear + barrier).

    For each sensed obstacle j with closest-surface distance d_j and outward
    normal n_j (unit, surface -> drone), define the violation
        e_j = max(0, r_safe - d_j).
    The control law is
        u = sum_j ( k_obs * e_j  +  k_barrier * (r_safe / max(d_j, eps) - 1) ) * n_j
    where the second term is a barrier that grows unbounded as d_j -> 0, so it
    dominates any bounded goal-attraction force near contact. eps = 0.02*r_safe.

    Properties:
      - Asymptotic: V(p) = 1/2 * sum_j e_j^2 (plus a strictly positive barrier
        term inside r_safe) is a Lyapunov function for static obstacles. Its
        derivative along the closed-loop trajectory is non-positive and zero
        only when e_j = 0 for all j, so the drone converges to the safe set
        { p : d_j(p) >= r_safe }.
      - Bounded actuation: this function does NOT saturate on its own. The
        caller (cf2_milestone1.compute_roaming_cmd) applies the final
        vmax=0.45 m/s saturation on the summed command. The barrier therefore
        always dominates the bounded attraction once d -> 0, while the
        actuator stays within its limit.

    Returns (fx, fy).
    """
    # Accumulators for the 2D push vector. z is left to the caller -- altitude
    # is governed by the path target, not the consensus law.
    fx = 0.0
    fy = 0.0
    for s in sensed_obstacles:
        d = s['distance']
        # Outside the safety bubble: this obstacle contributes nothing. This is
        # what makes the law "consensus-like" -- it is identically zero on the
        # safe set and only switches on when we start violating it.
        if d >= r_safe:
            continue
        # Linear violation: vanishes smoothly at d = r_safe, grows linearly inside.
        e = r_safe - d
        nx, ny = s['normal'][0], s['normal'][1]
        # Asymptotic part: proportional to the violation (Lyapunov-decreasing).
        mag = k_obs * e
        if k_barrier > 0.0:
            # Soft eps clamp prevents a divide-by-zero when the drone briefly
            # penetrates the obstacle in simulation. 2% of r_safe is small
            # enough that the barrier is already very large there.
            d_safe = max(d, 0.02 * r_safe)
            # Barrier part: r_safe/d_safe - 1 -> +inf as d -> 0, so the total
            # magnitude beats any bounded goal-attraction near contact.
            mag += k_barrier * (r_safe / d_safe - 1.0)
        # Accumulate along the outward normal (surface -> drone), which is the
        # direction we want to be pushed.
        fx += mag * nx
        fy += mag * ny
    return fx, fy


def inter_agent_dispersion_cmd(self_xy, neighbor_xy, k_disp, eps=0.05):
    """
    Per-agent dispersion law: gradient of the swarm potential
        V_disp(p_1, ..., p_N) = (k/2) * sum_{i != j} 1 / ||p_i - p_j||
    evaluated at this agent's position. Returns u_i = -grad_i V_disp =
        k * sum_{j != i} (p_i - p_j) / ||p_i - p_j||^3

    Properties (this is exactly what makes it "real" consensus, not the
    short-range repulsion it replaces):
      - Aggregate Lyapunov: V_disp is the same scalar function that every
        agent's law decreases along the joint gradient flow. The swarm
        therefore minimizes V_disp -- equivalently maximizes pairwise
        distances -- combined with whatever boundary/obstacle constraints
        the caller adds.
      - Long-range: no cutoff. Every other hider contributes at every
        distance, so the swarm always knows about its full geometry, not
        just the neighbors within 0.9 m.
      - Symmetric and Newton's-third-law-like: agent i pushes j by the same
        magnitude that j pushes i, so the centroid of the swarm is invariant
        under this law alone (the boundary/goal terms then steer the centroid).

    `eps` floors the pairwise distance to avoid a numerical blow-up if two
    drones briefly co-locate. 0.05 m is half the CF2 glob radius.

    Caller responsibility: saturate the final (fx, fy) at an actuator limit.
    """
    sx = float(self_xy[0])
    sy = float(self_xy[1])

    fx = 0.0
    fy = 0.0
    eps2 = eps * eps
    for nx, ny in neighbor_xy:
        dx = sx - float(nx)
        dy = sy - float(ny)
        # Squared distance with eps floor: keeps d^3 well above zero so the
        # gradient stays bounded even if two hiders briefly overlap.
        d2 = dx * dx + dy * dy
        if d2 < eps2:
            d2 = eps2
        d = math.sqrt(d2)
        # u_i contribution from this neighbor: k * (p_i - p_j) / d^3.
        inv_d3 = 1.0 / (d * d2)
        fx += k_disp * dx * inv_d3
        fy += k_disp * dy * inv_d3
    return fx, fy
