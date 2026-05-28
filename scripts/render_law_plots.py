"""
Analytical figures for the technical report.

No simulator dependency, no live state. Each figure is a clean illustration
of a single law or geometry. Writes PNGs to ../report/figures/.

Figures:
  - fig_barrier_curves.png       : obstacle consensus + inter-drone barrier
  - fig_dispersion_field.png     : Coulomb gradient flow around 4 hiders
  - fig_lissajous_footprint.png  : 4 hider Lissajous + seeker figure-eight
  - fig_fov_geometry.png         : cone + AABB occluder schematic
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow, Polygon, Rectangle, Wedge
import numpy as np

# Make the project root importable so we can reuse the live control laws
# instead of duplicating them.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, _ROOT)

import cf2_consensus  # noqa: E402  (sys.path tweak above)
import cf2_milestone1 as cf2m  # noqa: E402
import rmtt_seeker as rmtt  # noqa: E402

FIG_DIR = os.path.join(_ROOT, "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 180


# ---------------------------------------------------------------------------
# 1. Obstacle consensus + inter-drone barrier curves
# ---------------------------------------------------------------------------
def fig_barrier_curves():
    r_safe = cf2m.R_SAFE
    k_obs = cf2m.K_OBS
    k_bar = cf2m.K_BARRIER
    eps = 0.02 * r_safe

    d_obs = np.linspace(eps, r_safe * 1.05, 400)
    linear = np.where(d_obs < r_safe, k_obs * (r_safe - d_obs), 0.0)
    d_clamped = np.maximum(d_obs, eps)
    barrier = np.where(d_obs < r_safe, k_bar * (r_safe / d_clamped - 1.0), 0.0)
    total = linear + barrier

    # Inter-drone repulsion (from tp_algos._compute_drone_repulsion_3d).
    d_min = 1.0
    d_act = 1.5
    g_rep = 30.0
    d_inter = np.linspace(d_min * 1.001, d_act * 1.05, 400)
    inv_d_band = 1.0 / (d_act - d_min)
    inter_mag = np.where(
        d_inter < d_act,
        g_rep * (1.0 / (d_inter - d_min) - inv_d_band),
        0.0,
    )
    inter_mag = np.maximum(inter_mag, 0.0)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Left: obstacle consensus.
    ax_l.plot(d_obs, linear, color="#2c7fb8", lw=1.5,
              label=r"linear $k_\mathrm{obs}\,(r_\mathrm{safe}-d)$")
    ax_l.plot(d_obs, barrier, color="#d95f0e", lw=1.5,
              label=r"barrier $k_\mathrm{bar}\,(r_\mathrm{safe}/d-1)$")
    ax_l.plot(d_obs, total, color="black", lw=2.0,
              label="sum (commanded magnitude)")
    ax_l.axvline(r_safe, color="0.5", lw=0.8, ls="--")
    ax_l.axvline(eps, color="0.5", lw=0.8, ls=":")
    ax_l.text(r_safe, ax_l.get_ylim()[1] * 0.9 if ax_l.get_ylim()[1] else 1.0,
              r"  $r_\mathrm{safe}$", color="0.4", ha="left", va="top")
    ax_l.text(eps, 0.6 * total.max(), r"$\varepsilon$",
              color="0.4", ha="left", va="bottom")
    ax_l.set_xlim(0, r_safe * 1.05)
    ax_l.set_ylim(0, total.max() * 1.1)
    ax_l.set_xlabel("distance to obstacle surface $d$ (m)")
    ax_l.set_ylabel("force magnitude (m/s)")
    ax_l.set_title("Obstacle consensus: linear + barrier")
    ax_l.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax_l.grid(alpha=0.25)

    # Right: inter-drone barrier.
    ax_r.plot(d_inter, inter_mag, color="black", lw=2.0)
    ax_r.axvline(d_min, color="#d95f0e", lw=1.0, ls="--", label=r"$d_\mathrm{min}=1.0$ m")
    ax_r.axvline(d_act, color="#2c7fb8", lw=1.0, ls="--", label=r"$d_\mathrm{act}=1.5$ m")
    # Annotate the per-tick closure budget 2 * v_max * dt = 0.06 m.
    closure = 0.06
    ax_r.axvspan(d_min, d_min + closure, color="#fdae61", alpha=0.25,
                 label="per-tick closure budget $2v_\\mathrm{max}\\Delta t = 0.06$ m")
    ax_r.set_xlim(d_min - 0.05, d_act * 1.05)
    ax_r.set_ylim(0, min(inter_mag.max(), 400.0) * 1.05)
    ax_r.set_xlabel("inter-drone distance $d$ (m)")
    ax_r.set_ylabel("repulsion magnitude (m/s, pre-clamp)")
    ax_r.set_title("Inter-drone 3D barrier: divergent at $d=d_\\mathrm{min}^+$")
    ax_r.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax_r.grid(alpha=0.25)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_barrier_curves.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_law_plots] wrote {out}")


# ---------------------------------------------------------------------------
# 2. Coulomb dispersion vector field
# ---------------------------------------------------------------------------
def fig_dispersion_field():
    # Three "other" hiders fixed in a contracted equilateral triangle. We
    # sweep the position of a probe hider through xy and evaluate the
    # gradient flow it would experience.
    r = 0.9
    others = [(r * math.cos(t), r * math.sin(t))
              for t in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3,
                        math.pi / 2 - 2 * math.pi / 3)]
    k_disp = cf2m.K_DISP

    # Grid restricted to the arena box.
    x = np.linspace(-2.4, 2.4, 25)
    y = np.linspace(-4.4, 4.4, 35)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            fx, fy = cf2_consensus.inter_agent_dispersion_cmd(
                (X[i, j], Y[i, j]), others, k_disp,
            )
            U[i, j] = fx
            V[i, j] = fy
    mag = np.hypot(U, V)
    # Normalize for direction-only quiver, color by magnitude.
    safe = np.where(mag > 1e-9, mag, 1.0)
    U_n = U / safe
    V_n = V / safe

    fig, ax = plt.subplots(figsize=(6.0, 8.0))
    q = ax.quiver(X, Y, U_n, V_n, np.log10(mag + 1e-3),
                  cmap="viridis", scale=30, width=0.0035,
                  pivot="tail")
    cbar = fig.colorbar(q, ax=ax, shrink=0.85)
    cbar.set_label(r"$\log_{10}\|\mathbf{F}_\mathrm{disp}\|$  (m/s)")
    # Plot the three fixed hiders.
    ox, oy = zip(*others)
    ax.scatter(ox, oy, marker="*", s=180, color="limegreen",
               edgecolor="black", linewidth=0.6, zorder=5, label="fixed hiders")
    # Arena box.
    ax.add_patch(Rectangle((-2.5, -4.5), 5.0, 9.0, fill=False,
                           edgecolor="black", linewidth=1.2))
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-4.7, 4.7)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_title(r"Dispersion field $-\nabla_{\mathbf{p}_i} V_\mathrm{disp}$"
                 r"  (3 hiders fixed, probe hider position varies)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_dispersion_field.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_law_plots] wrote {out}")


# ---------------------------------------------------------------------------
# 3. Lissajous footprint (4 hiders + seeker)
# ---------------------------------------------------------------------------
def fig_lissajous_footprint():
    # Honour the per-drone offsets: pick each drone's "spawn pose" identical
    # to the corner-rectangle init in standalone_sim, then resolve the same
    # nearest-phase logic that cf2_milestone1._ensure_t_offset uses, but
    # offline so we don't have to import the cache.
    corners = [(-1.8, -3.8), (1.8, -3.8), (-1.8, 3.8), (1.8, 3.8)]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    def nearest_phase_offset(robot_no, spawn_xy):
        best_t = 0.0
        best_d2 = float("inf")
        t = 0.0
        while t <= 60.0:
            tx, ty, _ = cf2m._lissajous_target(robot_no, t)
            d2 = (tx - spawn_xy[0]) ** 2 + (ty - spawn_xy[1]) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_t = t
            t += 0.25
        return best_t

    duration = 120.0
    dt = 0.1
    ts = np.arange(0.0, duration, dt)

    fig, ax = plt.subplots(figsize=(7.5, 9.0))
    # Obstacles (xy footprint of the three current boxes).
    obstacles = [
        ((-0.5, 1.5), (1.0, 0.5)),
        ((0.8, -2.0), (0.5, 1.1)),
        ((0.0, 0.0), (3.0, 1.0)),
    ]
    for (ox, oy), (sx, sy) in obstacles:
        ax.add_patch(Rectangle(
            (ox - sx / 2, oy - sy / 2), sx, sy,
            facecolor="black", alpha=0.18, edgecolor="black", linewidth=0.8,
        ))

    for idx, (spawn, col) in enumerate(zip(corners, colors)):
        robot_no = idx + 1
        offset = nearest_phase_offset(robot_no, spawn)
        xs = []
        ys = []
        for t in ts:
            tx, ty, _ = cf2m._lissajous_target(robot_no, t + offset)
            xs.append(tx)
            ys.append(ty)
        ax.plot(xs, ys, color=col, lw=1.0, alpha=0.85,
                label=f"hider {robot_no}")
        ax.scatter([spawn[0]], [spawn[1]], marker="*", s=120,
                   color=col, edgecolor="black", linewidth=0.6, zorder=5)

    # Seeker figure-of-eight (same time horizon).
    sx_path = [rmtt._seeker_target(t)[0] for t in ts]
    sy_path = [rmtt._seeker_target(t)[1] for t in ts]
    ax.plot(sx_path, sy_path, color="#e6550d", lw=1.4, alpha=0.85,
            label="seeker")

    ax.add_patch(Rectangle((-2.5, -4.5), 5.0, 9.0, fill=False,
                           edgecolor="black", linewidth=1.2))
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-4.8, 4.8)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_title("Lissajous footprints over 120 s (xy projection)\n"
                 "spawns marked by stars; obstacles in grey")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_lissajous_footprint.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_law_plots] wrote {out}")


# ---------------------------------------------------------------------------
# 4. FoV cone + AABB occluder schematic
# ---------------------------------------------------------------------------
def fig_fov_geometry():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    apex = np.array([0.0, 0.0])
    heading = np.array([1.0, 0.0])
    half_angle_deg = 30.0
    rng = 3.0

    # Cone wedge.
    ax.add_patch(Wedge(
        center=apex, r=rng,
        theta1=-half_angle_deg, theta2=+half_angle_deg,
        facecolor="#fde0a0", edgecolor="#d99410", alpha=0.55, linewidth=1.2,
    ))

    # Heading axis.
    ax.annotate(
        "", xy=apex + heading * rng * 0.95, xytext=apex,
        arrowprops=dict(arrowstyle="->", color="#d99410", lw=1.5),
    )
    ax.text(rng * 0.5, 0.12, r"$\hat{\mathbf{h}}$", color="#9b6a08", fontsize=12)

    # Range arc label.
    ax.text(rng * math.cos(math.radians(half_angle_deg)) * 0.97,
            rng * math.sin(math.radians(half_angle_deg)) * 1.0 + 0.05,
            r"range $R$", color="#9b6a08", fontsize=10, ha="left", va="bottom")
    ax.text(0.05, 0.05, r"apex $\mathbf{a}$", color="black", fontsize=10)

    # Half-angle annotation.
    arc = mpatches.Arc(apex, 1.2, 1.2, angle=0,
                       theta1=0, theta2=half_angle_deg, color="black", lw=1.0)
    ax.add_patch(arc)
    ax.text(0.6, 0.18, r"$\theta_\mathrm{fov}$", fontsize=11, color="black")

    # AABB occluder intersecting a target ray.
    obstacle_xy = (1.3, -0.4)
    obstacle_w = 0.7
    obstacle_h = 0.9
    ax.add_patch(Rectangle(obstacle_xy, obstacle_w, obstacle_h,
                           facecolor="black", alpha=0.25,
                           edgecolor="black", linewidth=1.0))
    ax.text(obstacle_xy[0] + obstacle_w / 2,
            obstacle_xy[1] + obstacle_h / 2,
            "obstacle\n(AABB)",
            ha="center", va="center", fontsize=9)

    # Two targets: one visible (above the obstacle), one occluded (behind).
    visible = np.array([2.6, 0.7])
    occluded = np.array([2.6, 0.0])

    ax.scatter(*visible, marker="*", s=190, color="limegreen",
               edgecolor="black", linewidth=0.7, zorder=5)
    ax.text(visible[0] + 0.08, visible[1] + 0.05, "visible target",
            fontsize=9, color="darkgreen")
    ax.plot([apex[0], visible[0]], [apex[1], visible[1]],
            color="green", lw=1.2, ls="--", alpha=0.85)

    ax.scatter(*occluded, marker="*", s=190, color="crimson",
               edgecolor="black", linewidth=0.7, zorder=5)
    ax.text(occluded[0] + 0.08, occluded[1] - 0.15, "occluded target",
            fontsize=9, color="darkred")
    ax.plot([apex[0], occluded[0]], [apex[1], occluded[1]],
            color="red", lw=1.2, ls=":", alpha=0.85)

    ax.scatter([apex[0]], [apex[1]], marker="^", s=120,
               color="orange", edgecolor="black", linewidth=0.7, zorder=6)

    ax.set_xlim(-0.5, 3.4)
    ax.set_ylim(-1.6, 2.0)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (m, local frame at apex)")
    ax.set_ylabel("$y$ (m)")
    ax.set_title(
        "Field-of-view geometry. "
        "can_see passes iff (i) target in the cone "
        "AND (ii) segment apex$\\to$target hits no AABB."
    )
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_fov_geometry.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_law_plots] wrote {out}")


if __name__ == "__main__":
    fig_barrier_curves()
    fig_dispersion_field()
    fig_lissajous_footprint()
    fig_fov_geometry()
    print(f"[render_law_plots] done. Figures in {FIG_DIR}")
