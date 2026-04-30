import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from concurrent.futures import ThreadPoolExecutor
# Import the student's algorithms
import tp_algos

# ==========================================
# 1. SIMULATION INPUTS & CONFIGURATION
# ==========================================
dt = 0.05  # Simulation time step (seconds)

nbTb3B = 0
Tb3B_pose = [[0.0, 1.4, 0.0], [0.0, -1.4, 0.0], [-1.4, 0.0, 0.0]]  # x, y, theta

nbTb3W = 0
Tb3W_pose = [[1.0, 0.0, 0.0]]  # x, y, theta

nbRMTT = 0
# Started at z=0.0 so we can see the 3s straight line takeoff
RMTT_pose = [[-1.0, 1.0, 0.0]]  # x, y, z 

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
        vx, vy = get_async_cmd('tb3B', i, default, tp_algos.tb3B_controller, 
                               i+1, pose.copy(), tb3B_snap, tb3W_snap, rmtt_snap, cf2_snap, rmep_snap, obs_poses, obs_sizes, [], clock_time)
        
        v, wz = unicycle_kinematics(vx, vy, pose[2], MAX_V_TB3B, MAX_W_TB3)
        tb3B_poses[0, i] += v * np.cos(pose[2]) * dt
        tb3B_poses[1, i] += v * np.sin(pose[2]) * dt
        tb3B_poses[2, i] += wz * dt

    # --- Update TB3 Waffles (Unicycle) ---
    for i in range(nbTb3W):
        pose = tb3W_poses[:, i]
        default = (0.0, 0.0)
        vx, vy = get_async_cmd('tb3W', i, default, tp_algos.tb3W_controller, 
                               i+1, pose.copy(), tb3B_snap, tb3W_snap, rmtt_snap, cf2_snap, rmep_snap, obs_poses, obs_sizes, [], clock_time)
        
        v, wz = unicycle_kinematics(vx, vy, pose[2], MAX_V_TB3W, MAX_W_TB3)
        tb3W_poses[0, i] += v * np.cos(pose[2]) * dt
        tb3W_poses[1, i] += v * np.sin(pose[2]) * dt
        tb3W_poses[2, i] += wz * dt

    # --- Update RMTT Drones ---
    for i in range(nbRMTT):
        pose = rmtt_poses[:, i]
        default = (0.0, 0.0, 0.0, False, (0,0,0))
        vx, vy, vz, trigger_land, led = get_async_cmd('rmtt', i, default, tp_algos.rmtt_controller, 
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
        vx, vy, z_dist, trigger_takeoff, trigger_land, led = get_async_cmd('cf2', i, default, tp_algos.cf2_controller, 
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
                 
        elif cf2_states[i] == 3:  
             cf2_poses[2, i] -= (CF2_HOVER_Z / LANDING_TIME) * dt
             cf2_timers[i] -= dt
             if cf2_timers[i] <= 0 or cf2_poses[2, i] <= 0:
                 cf2_poses[2, i] = 0.0
                 cf2_states[i] = 0   

    # --- Update RMEP Robots ---
    for i in range(nbRMEP):
        pose = rmep_poses[:, i]
        default = (0.0, 0.0, 0.0)
        vx, vy, wz = get_async_cmd('rmep', i, default, tp_algos.rmep_controller, 
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
        
        target_color = 'red' if collision_states[name] else default_color
        
        if glob_list[idx]: glob_list[idx].remove()
        glob_list[idx] = draw_glob(ax, cx, cy, cz, glob_r, target_color)

    clock_time += dt
    return tb3B_plots + tb3W_plots + rmtt_plots + cf2_plots + rmep_plots

ani = animation.FuncAnimation(fig, update, interval=int(dt*1000), blit=False, cache_frame_data=False)

plt.show()