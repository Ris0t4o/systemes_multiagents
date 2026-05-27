"""
Obstacle consensus law shared by hiders (CF2) and the seeker (RMTT).

This module implements a single control primitive --
"how strongly should I be pushed away from the obstacles I currently sense" --
in a way that is both *asymptotically* safe (a Lyapunov-decreasing linear term)
and *bounded* near contact (an unbounded barrier term that overpowers any
bounded goal-attraction once we are too close).

The caller is responsible for:
  - producing the `sensed_obstacles` list (see cf2_sensing.sense_obstacles),
  - choosing R_SAFE, K_OBS, K_BARRIER,
  - saturating the resulting (fx, fy) at an actuator limit.
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
