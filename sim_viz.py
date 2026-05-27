"""
Sim overlays: FoV cones, hider trails, and a small HUD.
init() once, update() per frame.
"""

import numpy as np
from collections import deque
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rmtt_seeker import seeker_heading, SEEKER_FOV_HALF_ANGLE, SEEKER_VISION_RANGE
from cf2_milestone1 import FOV_HALF_ANGLE, SENSING_RADIUS
from cf2_sensing import intent_heading
from cf2_hider import get_last_seen
import game_referee


TRAIL_LEN = 80

_state = {
    'ax': None,
    'nbCF2': 0,
    'nbRMTT': 0,
    'cf2_fov_cones': [],
    'cf2_trails': [],
    'cf2_trail_lines': [],
    'rmtt_fov_cones': [],
    'status_text': None,
}


def init(ax, nbCF2, nbRMTT):
    """Wire the axes and allocate per-drone artist slots."""
    _state['ax'] = ax
    _state['nbCF2'] = nbCF2
    _state['nbRMTT'] = nbRMTT
    _state['cf2_fov_cones'] = [None] * nbCF2
    _state['cf2_trails'] = [deque(maxlen=TRAIL_LEN) for _ in range(nbCF2)]
    _state['cf2_trail_lines'] = [
        ax.plot([], [], [], '-', color='limegreen', alpha=0.45, linewidth=1.1)[0]
        for _ in range(nbCF2)
    ]
    _state['rmtt_fov_cones'] = [None] * nbRMTT
    _state['status_text'] = ax.text2D(
        0.02, 0.97, '', transform=ax.transAxes,
        fontsize=9, color=(0.25, 0.25, 0.25),
        verticalalignment='top', family='monospace',
    )


def draw_fov_cone(ax, apex, heading_xyz, half_angle, radius,
                  color='limegreen', alpha=0.10):
    """3D cone (triangle fan) with apex at the drone, axis along heading_xyz."""
    px, py, pz = apex
    hx, hy, hz = heading_xyz

    # Orthonormal basis in the plane perpendicular to heading.
    # Reference is world-z, or world-x if heading is nearly vertical.
    if abs(hz) < 0.9:
        rx, ry, rz = 0.0, 0.0, 1.0
    else:
        rx, ry, rz = 1.0, 0.0, 0.0
    ux = hy * rz - hz * ry
    uy = hz * rx - hx * rz
    uz = hx * ry - hy * rx
    un = np.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / un, uy / un, uz / un
    vx = hy * uz - hz * uy
    vy = hz * ux - hx * uz
    vz = hx * uy - hy * ux

    bcx = px + radius * hx
    bcy = py + radius * hy
    bcz = pz + radius * hz
    br = radius * np.tan(half_angle)

    n_segments = 16
    base_pts = []
    for k in range(n_segments + 1):
        a = 2.0 * np.pi * k / n_segments
        c = np.cos(a) * br
        s = np.sin(a) * br
        base_pts.append((
            bcx + c * ux + s * vx,
            bcy + c * uy + s * vy,
            bcz + c * uz + s * vz,
        ))
    apex_t = (px, py, pz)
    tris = [[apex_t, base_pts[k], base_pts[k + 1]] for k in range(n_segments)]
    coll = Poly3DCollection(tris, facecolor=color, alpha=alpha,
                            edgecolor=color, linewidth=0.4)
    ax.add_collection3d(coll)
    return coll


def update(rmtt_poses, rmtt_states, cf2_poses, cf2_states, t):
    """Redraw cones, trails, and HUD."""
    ax = _state['ax']
    if ax is None:
        return

    # Hiders' cones turn gold when a sighting is fresh in shared memory.
    alerted = get_last_seen(t) is not None
    cone_color = 'gold' if alerted else 'limegreen'
    cone_alpha = 0.16 if alerted else 0.10

    for i in range(_state['nbCF2']):
        # Throw away and rebuild — cheap for this many drones.
        if _state['cf2_fov_cones'][i] is not None:
            _state['cf2_fov_cones'][i].remove()
            _state['cf2_fov_cones'][i] = None
        if cf2_states[i] == 2 and not game_referee.is_caught(i + 1):
            heading = intent_heading(i + 1, cf2_poses[:, i], t)
            _state['cf2_fov_cones'][i] = draw_fov_cone(
                ax,
                (cf2_poses[0, i], cf2_poses[1, i], cf2_poses[2, i]),
                heading, FOV_HALF_ANGLE, SENSING_RADIUS,
                color=cone_color, alpha=cone_alpha,
            )
            _state['cf2_trails'][i].append(
                (cf2_poses[0, i], cf2_poses[1, i], cf2_poses[2, i])
            )
        elif cf2_states[i] == 0:
            _state['cf2_trails'][i].clear()
        if len(_state['cf2_trails'][i]) >= 2:
            xs, ys, zs = zip(*_state['cf2_trails'][i])
            _state['cf2_trail_lines'][i].set_data(xs, ys)
            _state['cf2_trail_lines'][i].set_3d_properties(zs)
        else:
            _state['cf2_trail_lines'][i].set_data([], [])
            _state['cf2_trail_lines'][i].set_3d_properties([])

    for i in range(_state['nbRMTT']):
        if _state['rmtt_fov_cones'][i] is not None:
            _state['rmtt_fov_cones'][i].remove()
            _state['rmtt_fov_cones'][i] = None
        if rmtt_states[i] == 2:
            s_heading = seeker_heading(rmtt_poses[:, i], t)
            _state['rmtt_fov_cones'][i] = draw_fov_cone(
                ax,
                (rmtt_poses[0, i], rmtt_poses[1, i], rmtt_poses[2, i]),
                s_heading, SEEKER_FOV_HALF_ANGLE, SEEKER_VISION_RANGE,
                color='orangered', alpha=0.16,
            )

    alert_str = "on " if alerted else "off"
    _state['status_text'].set_text(
        f"t = {t:6.1f} s   caught = {game_referee.catch_count()}/{_state['nbCF2']}   alert = {alert_str}"
    )
