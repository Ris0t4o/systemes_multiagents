import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from concurrent.futures import ThreadPoolExecutor
# Import the student's algorithms
from tp_algos import tb3B_controller, tb3W_controller, rmtt_controller, cf2_controller, rmep_controller
from cf2_sensing import intent_heading
from cf2_milestone1 import FOV_HALF_ANGLE, SENSING_RADIUS
from rmtt_seeker import (
    seeker_heading, can_see,
    SEEKER_FOV_HALF_ANGLE, SEEKER_VISION_RANGE,
)
from cf2_hider import get_last_seen as hider_get_last_seen

# ==========================================
# 1. SIMULATION INPUTS & CONFIGURATION
# ==========================================
dt = 0.05  # Simulation time step (seconds)
SPEEDUP = 5  # Render frames faster than real-time (1 = real-time)

nbTb3B = 0
Tb3B_pose = []

nbTb3W = 0
Tb3W_pose = []

nbRMTT = 1
RMTT_pose = [[2.0, 0.0, 0.0]]   # Seeker drone -- orange triangle marker

# On place 3 drones CF2 a gauche du terrain
nbCF2 = 1
CF2_pose = [[-2.1, -1.2, 0.0], [-2.1, 0.0, 0.0], [-2.1, 1.2, 0.0]]  # x, y, z

nbRMEP = 0
RMEP_pose = []

nbObstacle = 2
obstacle_size = [[0.8, 3.0, 2.5], [0.8, 1.2, 2.5]]
obstacle_pose = [[0.0, 0.0, 0.0], [1.4, 2.2, 0.0]]

# ==========================================
# 2. HARDWARE SPECS (SPEEDS, RADII, TIMERS)
# ==========================================
# Max Speeds (m/s)
MAX_V_TB3B = 0.15
MAX_V_TB3W = 0.18
MAX_W_TB3  = 2.84  # Max angular velocity for tb3 (rad/s)
MAX_V_RMTT = 0.6
MAX_V_CF2  = 0.6
MAX_V_RMEP = 0.75

# Base Radii (m) -> Globs will be 2x these
RAD_TB3B = 0.08
RAD_TB3W = 0.18
RAD_RMTT = 0.08
RAD_CF2  = 0.05
RAD_RMEP = 0.15

# Drone Hover Info
RMTT_HOVER_Z = 0.8
CF2_HOVER_Z = 1.0
TAKEOFF_TIME = 3.0 # seconds
LANDING_TIME = 3.0 # seconds

# Virtual Noise Standard Deviation (in meters per dt step)
# 0.005 means ~5mm of random drift per simulation step
DRONE_POS_NOISE_STD = 0.005 

# ==========================================
# 3. INITIALIZE STATE ARRAYS FOR TP_ALGOS
# ==========================================
tb3B_poses = np.array(Tb3B_pose).T if nbTb3B > 0 else np.zeros((3, 0))
tb3W_poses = np.array(Tb3W_pose).T if nbTb3W > 0 else np.zeros((3, 0))
rmtt_poses = np.array(RMTT_pose).T if nbRMTT > 0 else np.zeros((3, 0))
cf2_poses  = np.array(CF2_pose).T if nbCF2 > 0 else np.zeros((3, 0))
rmep_poses = np.array(RMEP_pose).T if nbRMEP > 0 else np.zeros((3, 0))

obs_poses = np.array(obstacle_pose).T if nbObstacle > 0 else np.zeros((3, 0))
obs_sizes = np.array(obstacle_size).T if nbObstacle > 0 else np.zeros((3, 0))

# Drone State Machines: 0 = Idle, 1 = Takeoff, 2 = Flying, 3 = Landing
# RMTT forced taking off immediately
rmtt_states = [1] * nbRMTT
rmtt_timers = [TAKEOFF_TIME] * nbRMTT

# CF2 waits for algorithm trigger_takeoff
cf2_states = [0] * nbCF2
cf2_timers = [0.0] * nbCF2

# ==========================================
# 4. SET UP MATPLOTLIB 3D FIGURE
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("CF2 Hider Swarm — FoV Cones & Bounded Consensus", pad=14)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.view_init(elev=26, azim=-58)

# Soften the default 3D panes / grid so the drones and obstacles stand out.
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_facecolor((0.97, 0.97, 0.99, 1.0))
    pane.set_edgecolor((0.85, 0.85, 0.88, 1.0))
ax.grid(True, linestyle=':', alpha=0.35)

# Lock standard viewing interaction using mouse
ax.set_navigate(False)

# Environment Limits
X_MIN, X_MAX = -2.5, 2.5
Y_MIN, Y_MAX = -4.5, 4.5
Z_MIN, Z_MAX = 0.0, 3.5

ax.set_xlim([X_MIN, X_MAX])
ax.set_ylim([Y_MIN, Y_MAX])
ax.set_zlim([Z_MIN, Z_MAX])
ax.set_box_aspect((5, 9, 3.5))

# Draw Obstacles as 3D bars
for i in range(nbObstacle):
    x, y, z = obs_poses[:, i]
    dx, dy, dz = obs_sizes[:, i]
    ax.bar3d(x - dx/2, y - dy/2, 0, dx, dy, dz,
             color=(0.35, 0.35, 0.40), alpha=0.55,
             edgecolor=(0.15, 0.15, 0.18), linewidth=0.8, shade=True)

# Core markers for robots. Legends are assigned only to index 0 of each group.
tb3B_plots = [ax.plot([], [], [], marker='s', color='blue', linestyle='', label='TB3 Burger' if _ == 0 else '')[0] for _ in range(nbTb3B)]
tb3W_plots = [ax.plot([], [], [], marker='D', color='cyan', linestyle='', label='TB3 Waffle' if _ == 0 else '')[0] for _ in range(nbTb3W)]
rmtt_plots = [ax.plot([], [], [], marker='^', color='orange', linestyle='', label='RMTT Drone' if _ == 0 else '')[0] for _ in range(nbRMTT)]
cf2_plots  = [ax.plot([], [], [], marker='*', color='green', linestyle='', label='CF2 Drone' if _ == 0 else '')[0] for _ in range(nbCF2)]
rmep_plots = [ax.plot([], [], [], marker='h', color='magenta', linestyle='', label='RMEP Robot' if _ == 0 else '')[0] for _ in range(nbRMEP)]

ax.legend(loc='upper right', framealpha=0.9, title="Robots")

# Storage for dynamic 3D glob collections
tb3B_globs = [None] * nbTb3B
tb3W_globs = [None] * nbTb3W
rmtt_globs = [None] * nbRMTT
cf2_globs  = [None] * nbCF2
rmep_globs = [None] * nbRMEP

# FoV cones and trajectory trails for CF2 hider drones (visual aids only)
cf2_fov_cones = [None] * nbCF2
TRAIL_LEN = 80   # ~4 s of history at dt=0.05
cf2_trails = [deque(maxlen=TRAIL_LEN) for _ in range(nbCF2)]
cf2_trail_lines = [
    ax.plot([], [], [], '-', color='limegreen', alpha=0.45, linewidth=1.1)[0]
    for _ in range(nbCF2)
]

# Hide-and-seek game state: a CF2 stays True once the seeker has spotted it.
cf2_deactivated = [False] * nbCF2

# Seeker FoV cone (one per RMTT drone, only the first is the seeker for now)
rmtt_fov_cones = [None] * nbRMTT

# On-screen clock / speedup indicator
status_text = ax.text2D(0.02, 0.97, '', transform=ax.transAxes,
                        fontsize=9, color=(0.25, 0.25, 0.25),
                        verticalalignment='top', family='monospace')

clock_time = 0.0

# Helpers
def clamp_vel2d(vx, vy, max_v):
    speed = np.hypot(vx, vy)
    return (vx * max_v / speed, vy * max_v / speed) if speed > max_v else (vx, vy)

def clamp_vel3d(vx, vy, vz, max_v):
    speed = np.linalg.norm([vx, vy, vz])
    return (vx * max_v / speed, vy * max_v / speed, vz * max_v / speed) if speed > max_v else (vx, vy, vz)

def unicycle_kinematics(vx, vy, theta, max_v, max_w):
    """Converts holonomic (vx, vy) commands into unicycle (v, w) commands allowing NO backward motion."""
    v_mag = np.hypot(vx, vy)
    if v_mag < 1e-3:
        return 0.0, 0.0
    
    theta_des = np.arctan2(vy, vx)
    e_theta = np.arctan2(np.sin(theta_des - theta), np.cos(theta_des - theta))
    
    # Proportional control for angular velocity
    wz = 2.5 * e_theta
    wz = np.clip(wz, -max_w, max_w)
    
    # Linear velocity: Scale by cosine of error to smoothly drive only when facing the target.
    # The max(0.0, ...) strictly forbids resolving the command by engaging reverse thrust.
    v = v_mag * max(0.0, np.cos(e_theta))
    v = min(v, max_v)
    
    return v, wz

def draw_glob(ax, x, y, z, radius, color):
    """Draws a semi-transparent wireframe sphere representing the safety glob"""
    u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:8j]
    X = x + radius * np.cos(u) * np.sin(v)
    Y = y + radius * np.sin(u) * np.sin(v)
    Z = z + radius * np.cos(v)
    return ax.plot_wireframe(X, Y, Z, color=color, alpha=0.15)

def draw_fov_cone(ax, apex, heading_xyz, half_angle, radius,
                  color='limegreen', alpha=0.10):
    """3D cone (triangle fan) with apex at the drone, axis along heading_xyz."""
    px, py, pz = apex
    hx, hy, hz = heading_xyz

    # Orthonormal basis (u, v) perpendicular to the cone axis.
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

    # Base disc center and radius (radius = R*tan(half_angle), where R is axial length).
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

def check_boundary_collision(cx, cy, cz, r, is_drone, drone_state=None):
    if cx - r < X_MIN or cx + r > X_MAX: return True
    if cy - r < Y_MIN or cy + r > Y_MAX: return True
    if is_drone:
        # Always check ceiling
        if cz + r > Z_MAX: return True
        
        # Check floor only if actively flying (state == 2)
        # States: 0=Idle, 1=Takeoff, 2=Flying, 3=Landing
        if cz - r < Z_MIN and drone_state == 2:
            return True
            
    return False

def check_obstacle_collision(cx, cy, cz, r):
    for i in range(nbObstacle):
        ox, oy, _ = obs_poses[:, i]
        dx, dy, dz = obs_sizes[:, i]
        closest_x = max(ox - dx/2, min(cx, ox + dx/2))
        closest_y = max(oy - dy/2, min(cy, oy + dy/2))
        closest_z = max(0, min(cz, dz))
        if (closest_x - cx)**2 + (closest_y - cy)**2 + (closest_z - cz)**2 < r**2:
            return True
    return False

def clamp_outside_obstacles(px, py, pz, drone_radius):
    """
    Hard last-line-of-defense: project the drone center out of any obstacle
    box expanded by drone_radius. Penetration is mathematically impossible
    after this call regardless of upstream controller bugs.
    """
    for k in range(nbObstacle):
        ox = float(obs_poses[0, k]); oy = float(obs_poses[1, k])
        hx = float(obs_sizes[0, k]) * 0.5
        hy = float(obs_sizes[1, k]) * 0.5
        sz_box = float(obs_sizes[2, k])
        cx = max(ox - hx, min(px, ox + hx))
        cy = max(oy - hy, min(py, oy + hy))
        cz = max(0.0, min(pz, sz_box))
        dx = px - cx; dy = py - cy; dz = pz - cz
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        if d >= drone_radius:
            continue
        if d < 1e-6:
            # Center inside the box -- escape via the closest face.
            face_pushes = [
                (px - (ox - hx), (ox - hx - drone_radius, py, pz)),
                ((ox + hx) - px, (ox + hx + drone_radius, py, pz)),
                (py - (oy - hy), (px, oy - hy - drone_radius, pz)),
                ((oy + hy) - py, (px, oy + hy + drone_radius, pz)),
                (sz_box - pz, (px, py, sz_box + drone_radius)),
            ]
            face_pushes.sort(key=lambda f: f[0])
            px, py, pz = face_pushes[0][1]
        else:
            scale = drone_radius / d
            px = cx + dx * scale
            py = cy + dy * scale
            pz = cz + dz * scale
    return px, py, pz

# Initialize dictionary to limit print frequency
last_log_time = {}

# ==========================================
# 5. MATPLOTLIB ANIMATION LOOP
# ==========================================
# Initialize thread pool for non-blocking student functions
executor = ThreadPoolExecutor(max_workers=20)
task_futures = {}
last_cmds = {}

def get_async_cmd(robot_type, idx, default_cmd, func, *args):
    """Runs a robot's controller in the background. If it blocks (sleeps), returns the last known command."""
    key = f"{robot_type}_{idx}"
    
    # If no task is running, start one
    if key not in task_futures or task_futures[key] is None:
        task_futures[key] = executor.submit(func, *args)
        
    # Check if the async task is done
    if task_futures[key].done():
        try:
            res = task_futures[key].result()
            last_cmds[key] = res
        except Exception as e:
            print(f"[ERROR] Exception in {key} controller: {e}")
        # Reset to allow next frame execution
        task_futures[key] = None 
        
    return last_cmds.get(key, default_cmd)

def update(frame):
    global clock_time
    
    # 1. First, Update all kinematics (calculate next step)
    
    # Snapshot of the world for the background threads
    tb3B_snap = tb3B_poses.copy() if nbTb3B > 0 else tb3B_poses
    tb3W_snap = tb3W_poses.copy() if nbTb3W > 0 else tb3W_poses
    rmtt_snap = rmtt_poses.copy() if nbRMTT > 0 else rmtt_poses
    cf2_snap  = cf2_poses.copy()  if nbCF2 > 0 else cf2_poses
    rmep_snap = rmep_poses.copy() if nbRMEP > 0 else rmep_poses

    # --- Update TB3 Burgers (Unicycle) ---
    for i in range(nbTb3B):
        pose = tb3B_poses[:, i]
        default = (0.0, 0.0)
        vx, vy = get_async_cmd('tb3B', i, default, tb3B_controller, 
                               i+1, pose.copy(), tb3B_snap, tb3W_snap, rmtt_snap, cf2_snap, rmep_snap, obs_poses, obs_sizes, [], clock_time)
        
        v, wz = unicycle_kinematics(vx, vy, pose[2], MAX_V_TB3B, MAX_W_TB3)
        tb3B_poses[0, i] += v * np.cos(pose[2]) * dt
        tb3B_poses[1, i] += v * np.sin(pose[2]) * dt
        tb3B_poses[2, i] += wz * dt

    # --- Update TB3 Waffles (Unicycle) ---
    for i in range(nbTb3W):
        pose = tb3W_poses[:, i]
        default = (0.0, 0.0)
        vx, vy = get_async_cmd('tb3W', i, default, tb3W_controller, 
                               i+1, pose.copy(), tb3B_snap, tb3W_snap, rmtt_snap, cf2_snap, rmep_snap, obs_poses, obs_sizes, [], clock_time)
        
        v, wz = unicycle_kinematics(vx, vy, pose[2], MAX_V_TB3W, MAX_W_TB3)
        tb3W_poses[0, i] += v * np.cos(pose[2]) * dt
        tb3W_poses[1, i] += v * np.sin(pose[2]) * dt
        tb3W_poses[2, i] += wz * dt

    # --- Update RMTT Drones ---
    for i in range(nbRMTT):
        pose = rmtt_poses[:, i]
        default = (0.0, 0.0, 0.0, False, (0,0,0))
        vx, vy, vz, trigger_land, led = get_async_cmd('rmtt', i, default, rmtt_controller, 
                                                      i+1, pose.copy(), tb3B_snap, tb3W_snap, rmtt_snap, cf2_snap, rmep_snap, obs_poses, obs_sizes, clock_time)
        
        # State Machine Logic
        if rmtt_states[i] == 1:   
             rmtt_poses[2, i] += (RMTT_HOVER_Z / TAKEOFF_TIME) * dt
             rmtt_timers[i] -= dt
             if rmtt_timers[i] <= 0:
                 rmtt_poses[2, i] = RMTT_HOVER_Z
                 rmtt_states[i] = 2   
                 
        elif rmtt_states[i] == 2:
             if trigger_land:
                 rmtt_states[i] = 3
                 rmtt_timers[i] = LANDING_TIME
             else:
                 vx, vy, vz = clamp_vel3d(vx, vy, vz, MAX_V_RMTT)
                 rmtt_poses[0, i] += vx * dt + np.random.normal(0, DRONE_POS_NOISE_STD)
                 rmtt_poses[1, i] += vy * dt + np.random.normal(0, DRONE_POS_NOISE_STD)
                 rmtt_poses[2, i] += vz * dt + np.random.normal(0, DRONE_POS_NOISE_STD)
                 rmtt_poses[0, i], rmtt_poses[1, i], rmtt_poses[2, i] = clamp_outside_obstacles(
                     rmtt_poses[0, i], rmtt_poses[1, i], rmtt_poses[2, i], RAD_RMTT * 2
                 )
                 
        elif rmtt_states[i] == 3: 
             rmtt_poses[2, i] -= (RMTT_HOVER_Z / LANDING_TIME) * dt
             rmtt_timers[i] -= dt
             if rmtt_timers[i] <= 0 or rmtt_poses[2, i] <= 0:
                 rmtt_poses[2, i] = 0.0
                 rmtt_states[i] = 0   

    # --- Update CF2 Drones ---
    for i in range(nbCF2):
        pose = cf2_poses[:, i]
        default = (0.0, 0.0, pose[2], False, False, (0,0,0))
        vx, vy, z_dist, trigger_takeoff, trigger_land, led = get_async_cmd('cf2', i, default, cf2_controller, 
                                                                           i+1, pose.copy(), tb3B_snap, tb3W_snap, rmtt_snap, cf2_snap, rmep_snap, obs_poses, obs_sizes, clock_time)
        
        # State Machine Logic
        if cf2_states[i] == 0:    
             if trigger_takeoff:
                 cf2_states[i] = 1
                 cf2_timers[i] = TAKEOFF_TIME
                 
        elif cf2_states[i] == 1:  
             cf2_poses[2, i] += (CF2_HOVER_Z / TAKEOFF_TIME) * dt
             cf2_timers[i] -= dt
             if cf2_timers[i] <= 0:
                 cf2_poses[2, i] = CF2_HOVER_Z
                 cf2_states[i] = 2   
                 
        elif cf2_states[i] == 2:
             if trigger_land:
                 cf2_states[i] = 3
                 cf2_timers[i] = LANDING_TIME
             else:
                 vz = (z_dist - cf2_poses[2, i])
                 vx, vy, vz = clamp_vel3d(vx, vy, vz, MAX_V_CF2)
                 cf2_poses[0, i] += vx * dt + np.random.normal(0, DRONE_POS_NOISE_STD)
                 cf2_poses[1, i] += vy * dt + np.random.normal(0, DRONE_POS_NOISE_STD)
                 cf2_poses[2, i] += vz * dt + np.random.normal(0, DRONE_POS_NOISE_STD)
                 cf2_poses[0, i], cf2_poses[1, i], cf2_poses[2, i] = clamp_outside_obstacles(
                     cf2_poses[0, i], cf2_poses[1, i], cf2_poses[2, i], RAD_CF2 * 2
                 )
                 
        elif cf2_states[i] == 3:
             cf2_poses[2, i] -= (CF2_HOVER_Z / LANDING_TIME) * dt
             cf2_timers[i] -= dt
             if cf2_timers[i] <= 0 or cf2_poses[2, i] <= 0:
                 cf2_poses[2, i] = 0.0
                 cf2_states[i] = 0

    # --- Hide-and-seek: seeker line-of-sight check ---
    # If a flying hider falls inside the seeker's cone AND is not occluded,
    # it gets deactivated -- triggering a landing it can never come back from
    # (the cf2 controller's _takeoff_done_by_robot flag stays True).
    for s_idx in range(nbRMTT):
        if rmtt_states[s_idx] != 2:
            continue
        s_pose = rmtt_poses[:, s_idx]
        s_heading = seeker_heading(s_pose, clock_time)
        s_pos = (float(s_pose[0]), float(s_pose[1]), float(s_pose[2]))
        for h_idx in range(nbCF2):
            if cf2_deactivated[h_idx]:
                continue
            if cf2_states[h_idx] != 2:
                continue
            h_pos = (float(cf2_poses[0, h_idx]), float(cf2_poses[1, h_idx]), float(cf2_poses[2, h_idx]))
            if can_see(s_pos, h_pos, s_heading,
                       SEEKER_FOV_HALF_ANGLE, SEEKER_VISION_RANGE,
                       obs_poses, obs_sizes):
                cf2_deactivated[h_idx] = True
                cf2_states[h_idx] = 3
                cf2_timers[h_idx] = LANDING_TIME
                print(f"[SEEKER] CF2_{h_idx+1} spotted at t={clock_time:.1f}s -- landing.")

    # --- Update RMEP Robots ---
    for i in range(nbRMEP):
        pose = rmep_poses[:, i]
        default = (0.0, 0.0, 0.0)
        vx, vy, wz = get_async_cmd('rmep', i, default, rmep_controller, 
                                   i+1, pose.copy(), tb3B_snap, tb3W_snap, rmtt_snap, cf2_snap, rmep_snap, obs_poses, obs_sizes, clock_time)
        
        vx, vy = clamp_vel2d(vx, vy, MAX_V_RMEP)
        rmep_poses[0, i] += vx * dt
        rmep_poses[1, i] += vy * dt
        rmep_poses[2, i] += wz * dt


    # 2. Gather state information for collision detection
    # Format: [name_id, x, y, z, glob_radius, default_color, draw_glob_ref, plot_ref, drone_state]
    all_robots = []
    for i in range(nbTb3B):
        all_robots.append(["TB3B_" + str(i+1), tb3B_poses[0,i], tb3B_poses[1,i], 0.1, RAD_TB3B*2, 'blue', tb3B_globs, i, tb3B_plots, None])
    for i in range(nbTb3W):
        all_robots.append(["TB3W_" + str(i+1), tb3W_poses[0,i], tb3W_poses[1,i], 0.15, RAD_TB3W*2, 'cyan', tb3W_globs, i, tb3W_plots, None])
    for i in range(nbRMTT):
        all_robots.append(["RMTT_" + str(i+1), rmtt_poses[0,i], rmtt_poses[1,i], rmtt_poses[2,i], RAD_RMTT*2, 'orange', rmtt_globs, i, rmtt_plots, rmtt_states[i]])
    for i in range(nbCF2):
        all_robots.append(["CF2_" + str(i+1), cf2_poses[0,i], cf2_poses[1,i], cf2_poses[2,i], RAD_CF2*2, 'green', cf2_globs, i, cf2_plots, cf2_states[i]])
    for i in range(nbRMEP):
        all_robots.append(["RMEP_" + str(i+1), rmep_poses[0,i], rmep_poses[1,i], 0.0, RAD_RMEP*2, 'magenta', rmep_globs, i, rmep_plots, None])

    # 3. Check Collisions & Assign Target Colors
    collision_states = {rob[0]: False for rob in all_robots}

    for rob in all_robots:
        name, cx, cy, cz, r, drone_state = rob[0], rob[1], rob[2], rob[3], rob[4], rob[9]
        is_drone = name.startswith("RMTT") or name.startswith("CF2")
        
        if check_boundary_collision(cx, cy, cz, r, is_drone, drone_state):
            log_key = f"{name}_boundary"
            if clock_time - last_log_time.get(log_key, -1.0) >= 1.0:
                print(f"[WARNING] {name} is colliding with the environment boundary!")
                last_log_time[log_key] = clock_time
            collision_states[name] = True
            
        elif check_obstacle_collision(cx, cy, cz, r):
            log_key = f"{name}_obstacle"
            if clock_time - last_log_time.get(log_key, -1.0) >= 1.0:
                print(f"[WARNING] {name} is colliding with an obstacle!")
                last_log_time[log_key] = clock_time
            collision_states[name] = True

    for i in range(len(all_robots)):
        for j in range(i + 1, len(all_robots)):
            name1, x1, y1, z1, r1 = all_robots[i][0:5]
            name2, x2, y2, z2, r2 = all_robots[j][0:5]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            if dist < (r1 + r2):
                log_key = f"{name1}_{name2}_collision"
                if clock_time - last_log_time.get(log_key, -1.0) >= 1.0:
                    print(f"[WARNING] {name1} and {name2} are intersecting/colliding!")
                    last_log_time[log_key] = clock_time
                collision_states[name1] = True
                collision_states[name2] = True


    # 4. Final Rendering
    for rob in all_robots:
        name, cx, cy, cz, glob_r, default_color, glob_list, idx, plot_list, drone_state = rob
        plot_list[idx].set_data([cx], [cy])
        plot_list[idx].set_3d_properties([cz])

        # Deactivated hiders force red regardless of collision/default color.
        deactivated = name.startswith("CF2_") and cf2_deactivated[idx]
        if deactivated or collision_states[name]:
            target_color = 'red'
        else:
            target_color = default_color

        if glob_list[idx]: glob_list[idx].remove()
        glob_list[idx] = draw_glob(ax, cx, cy, cz, glob_r, target_color)

    # 4b. FoV cones + trails for CF2 hider drones (suppressed once deactivated)
    alerted = hider_get_last_seen(clock_time) is not None
    cone_color = 'gold' if alerted else 'limegreen'
    cone_alpha = 0.16 if alerted else 0.10
    for i in range(nbCF2):
        if cf2_fov_cones[i] is not None:
            cf2_fov_cones[i].remove()
            cf2_fov_cones[i] = None
        if cf2_states[i] == 2 and not cf2_deactivated[i]:
            heading = intent_heading(i + 1, cf2_poses[:, i], clock_time)
            cf2_fov_cones[i] = draw_fov_cone(
                ax,
                (cf2_poses[0, i], cf2_poses[1, i], cf2_poses[2, i]),
                heading, FOV_HALF_ANGLE, SENSING_RADIUS,
                color=cone_color, alpha=cone_alpha,
            )
            cf2_trails[i].append((cf2_poses[0, i], cf2_poses[1, i], cf2_poses[2, i]))
        elif cf2_states[i] == 0:
            cf2_trails[i].clear()
        if len(cf2_trails[i]) >= 2:
            xs, ys, zs = zip(*cf2_trails[i])
            cf2_trail_lines[i].set_data(xs, ys)
            cf2_trail_lines[i].set_3d_properties(zs)
        else:
            cf2_trail_lines[i].set_data([], [])
            cf2_trail_lines[i].set_3d_properties([])

    # 4c. Seeker (RMTT) FoV cone -- narrower, orange-red.
    for i in range(nbRMTT):
        if rmtt_fov_cones[i] is not None:
            rmtt_fov_cones[i].remove()
            rmtt_fov_cones[i] = None
        if rmtt_states[i] == 2:
            s_heading = seeker_heading(rmtt_poses[:, i], clock_time)
            rmtt_fov_cones[i] = draw_fov_cone(
                ax,
                (rmtt_poses[0, i], rmtt_poses[1, i], rmtt_poses[2, i]),
                s_heading, SEEKER_FOV_HALF_ANGLE, SEEKER_VISION_RANGE,
                color='orangered', alpha=0.16,
            )

    caught = sum(cf2_deactivated)
    alert_str = "on " if alerted else "off"
    status_text.set_text(
        f"t = {clock_time:6.1f} s   speedup = {SPEEDUP}x   caught = {caught}/{nbCF2}   alert = {alert_str}"
    )

    clock_time += dt
    return tb3B_plots + tb3W_plots + rmtt_plots + cf2_plots + rmep_plots

ani = animation.FuncAnimation(fig, update, interval=max(1, int(dt*1000/SPEEDUP)), blit=False, cache_frame_data=False)

plt.show()
