import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from concurrent.futures import ThreadPoolExecutor
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# Import the student's algorithms
import tp_algos

# ==========================================
# 1. SIMULATION INPUTS & CONFIGURATION
# ==========================================
SIMULATION_STOPPED = False

dt = 0.05  # Simulation time step (seconds)

nbTb3B = 0
Tb3B_pose = [[0.0, 1.4, 0.0], [0.0, -1.4, 0.0], [-1.4, 0.0, 0.0]]  # x, y, theta

nbTb3W = 0
Tb3W_pose = [[1.0, 0.0, 0.0]]  # x, y, theta

nbRMTT = 1
# Started at z=0.0 so we can see the 3s straight line takeoff
RMTT_pose = [[0.0, 1.0, 0.0]]  # x, y, z 

nbCF2 = 1
CF2_pose = [[-1.0, 0.0, 0.0]]  # x, y, z

nbRMEP = 0
RMEP_pose = [[-1.0, -1.0, 0.0]]  # x, y, theta

nbObstacle = 2
obstacle_size = [[1.0, 0.5, 2.5], [0.5, 1.1, 2.5]]
obstacle_pose = [[0.0, 0.0, 0.0], [3.0, -3.0, 0.0]]

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

# ==========================================
# VISION CONE PARAMETERS
# ==========================================
FOV_ANGLE = np.radians(60)
FOV_RANGE = 2.5

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
ax.set_title("Swarm Simulation (Collision Detection & Fixed Boundaries)")
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

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
    ax.bar3d(x - dx/2, y - dy/2, 0, dx, dy, dz, color='k', alpha=0.3)

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
# Vision cone display
cf2_fov_lines = [None] * nbCF2
cf2_yaws = [0.0] * nbCF2
update_prev_cf2_positions = {}
rmep_globs = [None] * nbRMEP

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


def draw_fov(ax, x, y, z, yaw, fov_angle, fov_range, detected=False):

    n = 20

    cone_radius = fov_range * np.tan(fov_angle / 2)

    theta = np.linspace(0, 2*np.pi, n)

    dx = np.cos(yaw)
    dy = np.sin(yaw)

    cx = x + fov_range * dx
    cy = y + fov_range * dy
    cz = z

    ux = -dy
    uy = dx

    vx = 0
    vy = 0
    vz = 1

    circle_points = []

    for t in theta:

        px = cx + cone_radius * (np.cos(t)*ux + np.sin(t)*vx)
        py = cy + cone_radius * (np.cos(t)*uy + np.sin(t)*vy)
        pz = cz + cone_radius * (np.cos(t)*0  + np.sin(t)*vz)

        circle_points.append([px, py, pz])

    faces = []

    apex = [x, y, z]

    for i in range(n-1):
        faces.append([
            apex,
            circle_points[i],
            circle_points[i+1]
        ])

    faces.append([
        apex,
        circle_points[-1],
        circle_points[0]
    ])

    # COLOR CHANGES HERE
    face_color = 'lime' if detected else 'red'
    edge_color = 'green' if detected else 'darkred'

    cone = Poly3DCollection(
        faces,
        alpha=0.25,
        facecolor=face_color,
        edgecolor=edge_color
    )

    ax.add_collection3d(cone)

    return cone

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

def point_in_cone(
    tx, ty, tz,
    cx, cy, cz,
    yaw,
    fov_angle,
    fov_range
):

    dx = tx - cx
    dy = ty - cy
    dz = tz - cz

    horizontal_dist = np.hypot(dx, dy)

    # Too far away
    if horizontal_dist > fov_range:
        return False

    # Ignore almost-zero distance
    if horizontal_dist < 1e-6:
        return True

    angle_to_target = np.arctan2(dy, dx)

    angle_error = np.arctan2(
        np.sin(angle_to_target - yaw),
        np.cos(angle_to_target - yaw)
    )

    return abs(angle_error) <= (fov_angle / 2)

def update(frame):

    global SIMULATION_STOPPED
    global clock_time
    global update_prev_cf2_positions

    if SIMULATION_STOPPED:
        ani.event_source.stop()
        return []

    # ==========================================
    # SNAPSHOTS
    # ==========================================

    tb3B_snap = tb3B_poses.copy() if nbTb3B > 0 else tb3B_poses
    tb3W_snap = tb3W_poses.copy() if nbTb3W > 0 else tb3W_poses
    rmtt_snap = rmtt_poses.copy() if nbRMTT > 0 else rmtt_poses
    cf2_snap  = cf2_poses.copy() if nbCF2 > 0 else cf2_poses
    rmep_snap = rmep_poses.copy() if nbRMEP > 0 else rmep_poses

    # ==========================================
    # UPDATE TB3 BURGERS
    # ==========================================

    for i in range(nbTb3B):

        pose = tb3B_poses[:, i]

        default = (0.0, 0.0)

        vx, vy = get_async_cmd(
            'tb3B',
            i,
            default,
            tp_algos.tb3B_controller,
            i+1,
            pose.copy(),
            tb3B_snap,
            tb3W_snap,
            rmtt_snap,
            cf2_snap,
            rmep_snap,
            obs_poses,
            obs_sizes,
            [],
            clock_time
        )

        v, wz = unicycle_kinematics(
            vx,
            vy,
            pose[2],
            MAX_V_TB3B,
            MAX_W_TB3
        )

        tb3B_poses[0, i] += v * np.cos(pose[2]) * dt
        tb3B_poses[1, i] += v * np.sin(pose[2]) * dt
        tb3B_poses[2, i] += wz * dt

    # ==========================================
    # UPDATE TB3 WAFFLES
    # ==========================================

    for i in range(nbTb3W):

        pose = tb3W_poses[:, i]

        default = (0.0, 0.0)

        vx, vy = get_async_cmd(
            'tb3W',
            i,
            default,
            tp_algos.tb3W_controller,
            i+1,
            pose.copy(),
            tb3B_snap,
            tb3W_snap,
            rmtt_snap,
            cf2_snap,
            rmep_snap,
            obs_poses,
            obs_sizes,
            [],
            clock_time
        )

        v, wz = unicycle_kinematics(
            vx,
            vy,
            pose[2],
            MAX_V_TB3W,
            MAX_W_TB3
        )

        tb3W_poses[0, i] += v * np.cos(pose[2]) * dt
        tb3W_poses[1, i] += v * np.sin(pose[2]) * dt
        tb3W_poses[2, i] += wz * dt

    # ==========================================
    # UPDATE RMTT DRONES
    # ==========================================

    for i in range(nbRMTT):

        pose = rmtt_poses[:, i]

        default = (0.0, 0.0, 0.0, False, (0,0,0))

        vx, vy, vz, trigger_land, led = get_async_cmd(
            'rmtt',
            i,
            default,
            tp_algos.rmtt_controller,
            i+1,
            pose.copy(),
            tb3B_snap,
            tb3W_snap,
            rmtt_snap,
            cf2_snap,
            rmep_snap,
            obs_poses,
            obs_sizes,
            clock_time
        )

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

                vx, vy, vz = clamp_vel3d(
                    vx,
                    vy,
                    vz,
                    MAX_V_RMTT
                )

                rmtt_poses[0, i] += vx * dt + np.random.normal(0, DRONE_POS_NOISE_STD)
                rmtt_poses[1, i] += vy * dt + np.random.normal(0, DRONE_POS_NOISE_STD)
                rmtt_poses[2, i] += vz * dt + np.random.normal(0, DRONE_POS_NOISE_STD)

        elif rmtt_states[i] == 3:

            rmtt_poses[2, i] -= (RMTT_HOVER_Z / LANDING_TIME) * dt
            rmtt_timers[i] -= dt

            if rmtt_timers[i] <= 0 or rmtt_poses[2, i] <= 0:

                rmtt_poses[2, i] = 0.0
                rmtt_states[i] = 0

    # ==========================================
    # UPDATE CF2 DRONES
    # ==========================================

    for i in range(nbCF2):

        pose = cf2_poses[:, i]

        default = (0.0, 0.0, pose[2], False, False, (0,0,0))

        vx, vy, z_dist, trigger_takeoff, trigger_land, led = get_async_cmd(
            'cf2',
            i,
            default,
            tp_algos.cf2_controller,
            i+1,
            pose.copy(),
            tb3B_snap,
            tb3W_snap,
            rmtt_snap,
            cf2_snap,
            rmep_snap,
            obs_poses,
            obs_sizes,
            clock_time
        )

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

            # ------------------------------------------
            # UPDATE DRONE YAW FIRST
            # ------------------------------------------

            if i not in update_prev_cf2_positions:
                update_prev_cf2_positions[i] = (
                    cf2_poses[0, i],
                    cf2_poses[1, i]
                )

            prev_x, prev_y = update_prev_cf2_positions[i]

            dx_yaw = cf2_poses[0, i] - prev_x
            dy_yaw = cf2_poses[1, i] - prev_y

            speed_yaw = np.hypot(dx_yaw, dy_yaw)

            if speed_yaw > 1e-4:

                cf2_yaws[i] = np.arctan2(dy_yaw, dx_yaw)

            update_prev_cf2_positions[i] = (
                cf2_poses[0, i],
                cf2_poses[1, i]
            )

            # ------------------------------------------
            # TARGET DETECTION
            # ------------------------------------------

            target_detected = False

            for j in range(nbRMTT):

                tx = rmtt_poses[0, j]
                ty = rmtt_poses[1, j]
                tz = rmtt_poses[2, j]

                if point_in_cone(
                    tx,
                    ty,
                    tz,
                    cf2_poses[0, i],
                    cf2_poses[1, i],
                    cf2_poses[2, i],
                    cf2_yaws[i],
                    FOV_ANGLE,
                    FOV_RANGE
                ):

                    target_detected = True
                    break

            # ------------------------------------------
            # STOP SIMULATION IMMEDIATELY
            # ------------------------------------------

            if target_detected:

                print("TARGET FOUND — STOPPING SIMULATION")

                SIMULATION_STOPPED = True

                cf2_states[i] = 3
                cf2_timers[i] = LANDING_TIME

            else:

                vz = (z_dist - cf2_poses[2, i])

                vx, vy, vz = clamp_vel3d(
                    vx,
                    vy,
                    vz,
                    MAX_V_CF2
                )

                cf2_poses[0, i] += vx * dt + np.random.normal(0, DRONE_POS_NOISE_STD)
                cf2_poses[1, i] += vy * dt + np.random.normal(0, DRONE_POS_NOISE_STD)
                cf2_poses[2, i] += vz * dt + np.random.normal(0, DRONE_POS_NOISE_STD)

        elif cf2_states[i] == 3:

            cf2_poses[2, i] -= (CF2_HOVER_Z / LANDING_TIME) * dt
            cf2_timers[i] -= dt

            if cf2_timers[i] <= 0 or cf2_poses[2, i] <= 0:

                cf2_poses[2, i] = 0.0
                cf2_states[i] = 0

 # ==========================================
# DRAW ROBOTS + GLOBS
# ==========================================

    all_robots = []

    for i in range(nbTb3B):
        all_robots.append([
            tb3B_poses[0,i],
            tb3B_poses[1,i],
            0.1,
            RAD_TB3B*2,
            'blue',
            tb3B_globs,
            tb3B_plots,
            i
        ])

    for i in range(nbTb3W):
        all_robots.append([
            tb3W_poses[0,i],
            tb3W_poses[1,i],
            0.15,
            RAD_TB3W*2,
            'cyan',
            tb3W_globs,
            tb3W_plots,
            i
        ])

    for i in range(nbRMTT):
        all_robots.append([
            rmtt_poses[0,i],
            rmtt_poses[1,i],
            rmtt_poses[2,i],
            RAD_RMTT*2,
            'orange',
            rmtt_globs,
            rmtt_plots,
            i
        ])

    for i in range(nbCF2):
        all_robots.append([
            cf2_poses[0,i],
            cf2_poses[1,i],
            cf2_poses[2,i],
            RAD_CF2*2,
            'green',
            cf2_globs,
            cf2_plots,
            i
        ])

    for i in range(nbRMEP):
        all_robots.append([
            rmep_poses[0,i],
            rmep_poses[1,i],
            0.0,
            RAD_RMEP*2,
            'magenta',
            rmep_globs,
            rmep_plots,
            i
        ])

    # Draw markers + globs
    for rob in all_robots:

        x, y, z, radius, color, glob_list, plot_list, idx = rob

        plot_list[idx].set_data([x], [y])
        plot_list[idx].set_3d_properties([z])

        if glob_list[idx] is not None:
            glob_list[idx].remove()

        glob_list[idx] = draw_glob(
            ax,
            x,
            y,
            z,
            radius,
            color
        )

    # ==========================================
    # DRAW CF2 CONES
    # ==========================================

    for i in range(nbCF2):

        x = cf2_poses[0, i]
        y = cf2_poses[1, i]
        z = cf2_poses[2, i]

        if cf2_fov_lines[i] is not None:
            cf2_fov_lines[i].remove()

        if i not in update_prev_cf2_positions:
            update_prev_cf2_positions[i] = (x, y)

        prev_x, prev_y = update_prev_cf2_positions[i]

        dx = x - prev_x
        dy = y - prev_y

        speed = np.hypot(dx, dy)

        if speed > 1e-4:

            target_yaw = np.arctan2(dy, dx)

            yaw_error = np.arctan2(
                np.sin(target_yaw - cf2_yaws[i]),
                np.cos(target_yaw - cf2_yaws[i])
            )

            cf2_yaws[i] += 0.15 * yaw_error

        update_prev_cf2_positions[i] = (x, y)

        detected = False

        for j in range(nbRMTT):

            if point_in_cone(
                rmtt_poses[0, j],
                rmtt_poses[1, j],
                rmtt_poses[2, j],
                x,
                y,
                z,
                cf2_yaws[i],
                FOV_ANGLE,
                FOV_RANGE
            ):

                detected = True
                break

        cf2_fov_lines[i] = draw_fov(
            ax,
            x,
            y,
            z,
            cf2_yaws[i],
            FOV_ANGLE,
            FOV_RANGE,
            detected
        )

    clock_time += dt

    return (
        tb3B_plots
        + tb3W_plots
        + rmtt_plots
        + cf2_plots
        + rmep_plots
    )

ani = animation.FuncAnimation(fig, update, interval=int(dt*1000), blit=False, cache_frame_data=False)

plt.show()
