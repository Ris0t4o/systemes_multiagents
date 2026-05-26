#!/usr/bin/python3
'''
    CentraleSupelec TP 2A/3A
    Aarsh THAKKER,2025
    (all variables in SI unit)

###########################################################################################

============================ READ THIS BEFORE STARTING TO CODE ============================

    You ONLY modify the part that is marked << TO BE MODIFIED >> in the functions
    YOU MUST NOT MODIFY THE NAME OF THE FILE OR THE NAME OF THE FUNCTIONS OR THE INPUT/OUTPUT OF THE FUNCTIONS
    FOR THE SUBMISSION, ONLY THE CODE INSIDE THE FUNCTION (MARKED << TO BE MODIFIED >>) WILL BE CONSIDERED FOR EVALUATION
    variables used by the functions of this script
        - robotNo: Current robot number in the fleet of same type of robots
        - robotPose: current position of the robot (x,y,z, ... depending on the robot)
        - nbTB3B: number of total tb3-Burger robots in the fleet (>=0)
        - nbTB3W: number of total tb3-Waffle robots in the fleet (>=0)
        - nbRMTT: number of total dji robomaster TT drones in the fleet (>=0)
        - nbCF2: number of total crazyflie 2 drones in the fleet (>=0)
        - nbRMEP: number of total dji robomaster EP in the fleet (>=0)  
        - nbOBSTACLE: number of total obstacle positions in the environment (>=0)
        
        - tb3B_poses:  size (3 x nbTB3B) 
            eg. of use: for robot number 'robotNo', position of the robot can be obtained by: 
            tb3B_poses[:,robotNo-1]   (indexes in Python start from 0 !)
            tb3B_poses[0,robotNo-1]: x-coordinate of robot position (in m)
            tb3B_poses[1,robotNo-1]: y-coordinate of robot position (in m)
            tb3B_poses[2,robotNo-1]: orientation angle of robot (in rad)
            
        - tb3W_poses:  size (3 x nbTB3W) 
            tb3W_poses[0,robotNo-1]: x-coordinate of robot position (in m)
            tb3W_poses[1,robotNo-1]: y-coordinate of robot position (in m)
            tb3W_poses[2,robotNo-1]: orientation angle of robot (in rad)

        - rmtt_poses:  size (3 x nbRMTT) 
            rmtt_poses[0,robotNo-1]: x-coordinate of robot position (in m)
            rmtt_poses[1,robotNo-1]: y-coordinate of robot position (in m)
            rmtt_poses[2,robotNo-1]: z-coordinate of robot position (in m)
            rmtt_poses[3,robotNo-1]: orientation angle of robot (in rad) (Ask Supervisor if needed)
            
        - cf2_poses:  size (3 x nbCF2) 
            cf2_poses[0,robotNo-1]: x-coordinate of robot position (in m)
            cf2_poses[1,robotNo-1]: y-coordinate of robot position (in m)
            cf2_poses[2,robotNo-1]: z-coordinate of robot position (in m)

        - rmep_poses:  size (3 x nbRMEP) 
            rmep_poses[0,robotNo-1]: x-coordinate of robot position (in m)
            rmep_poses[1,robotNo-1]: y-coordinate of robot position (in m)
            rmep_poses[2,robotNo-1]: orientation angle of robot (in rad)
            rmep_poses[2,robotNo-1]: orientation angle of robot (in rad)

        - obstacle_pose:  size (3 x nbOBSTACLE)  
            obstacle_pose[0,nbOBSTACLE-1]: x-coordinate of center position of obstacle (in m)
            obstacle_pose[1,nbOBSTACLE-1]: y-coordinate of center position of obstacle (in m)
            obstacle_pose[2,nbOBSTACLE-1]: z-coordinate of center position of obstacle (in m)
        
        - obstacle_size: size (3 x nbOBSTACLE)
            obstacle_size[0,nbOBSTACLE-1]: size of the obstacle in x (in m)
            obstacle_size[1,nbOBSTACLE-1]: size of the obstacle in y (in m)
            obstacle_size[2,nbOBSTACLE-1]: size of the obstacle in z (in m)

    In case of doubt related to the robots, this code or may be something else,
    open a discussion at https://tp-cs.talkyard.net/
    Use your own GitHub account or CS email to signup.
###########################################################################################

'''

import random

import numpy as np
import math, time

# ==============   "GLOBAL" VARIABLES KNOWN BY ALL THE FUNCTIONS ===================
# all variables declared here will be known by functions below
# use keyword "global" inside a function if the variable needs to be modified by the function



global TAKEOFF_DONE, Time2Takeoff
TAKEOFF_DONE = False
Time2Takeoff = 5 # time to wait before takeoff for the cf2 drone (in seconds)

# Controller-level drone separation (3D).
DRONE_MIN_DISTANCE = 1.0
DRONE_REPULSION_EPS = 1e-6
DRONE_REPULSION_GAIN = 1.0


def _compute_drone_repulsion_3d(self_pose, neighbor_positions, min_dist=DRONE_MIN_DISTANCE):
    """Return a 3D velocity offset that pushes the drone away from close neighbors."""
    sx = float(self_pose[0])
    sy = float(self_pose[1])
    sz = float(self_pose[2])

    dvx = 0.0
    dvy = 0.0
    dvz = 0.0

    for nx, ny, nz in neighbor_positions:
        dx = sx - float(nx)
        dy = sy - float(ny)
        dz = sz - float(nz)
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist >= min_dist:
            continue
        if dist < DRONE_REPULSION_EPS:
            ux, uy, uz = 1.0, 0.0, 0.0
        else:
            ux = dx / dist
            uy = dy / dist
            uz = dz / dist

        mag = DRONE_REPULSION_GAIN * (min_dist - dist) / max(min_dist, DRONE_REPULSION_EPS)
        dvx += mag * ux
        dvy += mag * uy
        dvz += mag * uz

    return dvx, dvy, dvz

# ===================================================================================
# Control function for turtlebot3 Burger ground vehicle Unicycle model
# should ONLY return (vx,vy) for the robot command
# max useable numbers of robots = 6 
# ====================================
def tb3B_control_fn(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, lidar_scan, clock):
# ====================================

    nbTB3= len(tb3B_poses[0]) # number of total tb3 robots in the use
    nbTB3W = len(tb3W_poses[0]) # number of total tb3W robots in the use
    nbRMTT = len(rmtt_poses[0]) # number of total dji rmtt drones in the use
    nbCF2 = len(cf2_poses[0]) # number of total cf2 drones in the use
    nbRMEP = len(rmep_poses[0]) # number of total dji rmep in the use
    nbOBSTACLE = len(obstacle_pose[0]) # number of total obstacle positions in the environment

    #  --- TO BE MODIFIED --- 
    if robotNo == 1:    
        goal = [2,-2]
    if robotNo == 2:    
        goal = [1,-2]
    if robotNo == 3:
        time.sleep(2)
        goal = [0,-2]  
    vx = 0.2 * (-robotPose[0] + goal[0])
    vy = 0.2 * (-robotPose[1] + goal[1])
    # -----------------------

    return vx,vy
# ====================================        


# ===================================================================================
# Control function for turtlebot3 Waffle ground vehicle Unicycle model
# should ONLY return (vx,vy) for the robot command
# max useable numbers of robots = 2
# ====================================
def tb3W_control_fn(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, lidar_scan, clock):
# ====================================

    nbTB3= len(tb3B_poses[0]) # number of total tb3 robots in the use
    nbTB3W = len(tb3W_poses[0]) # number of total tb3W robots in the use
    nbRMTT = len(rmtt_poses[0]) # number of total dji rmtt drones in the use
    nbCF2 = len(cf2_poses[0]) # number of total cf2 drones in the use
    nbRMEP = len(rmep_poses[0]) # number of total dji rmep in the use
    nbOBSTACLE = len(obstacle_pose[0]) # number of total obstacle positions in the environment

    #  --- TO BE MODIFIED --- 
    goal = [-1,1]
    vx = 0.2 * (-robotPose[0] + goal[0])
    vy = 0.2 * (-robotPose[1] + goal[1])
    # -----------------------

    return vx,vy
# ====================================   


# ====================================        
# Control function for dji rmtt drones
# should ONLY return (vx,vy,vz) for the robot command
# max useable numbers of drones = 4
# ====================================
def rmtt_control_fn(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, clock):
# ====================================
    nbTB3= len(tb3B_poses[0]) # number of total tb3 robots in the use
    nbTB3W = len(tb3W_poses[0]) # number of total tb3W robots in the use
    nbRMTT = len(rmtt_poses[0]) # number of total dji rmtt drones in the use
    nbCF2 = len(cf2_poses[0]) # number of total cf2 drones in the use
    nbRMEP = len(rmep_poses[0]) # number of total dji rmep in the use
    nbOBSTACLE = len(obstacle_pose[0]) # number of total obstacle positions in the environment
    led = (0,0,0) # led color (r,g,b) in range [0,255]
    
    #  --- TO BE MODIFIED ---
    # Seeker controller. Deliberately does NOT receive hider positions (cf2_poses)
    # so "the seeker doesn't know where the hiders are" is enforced structurally.
    if not hasattr(rmtt_control_fn, "_seeker_cmd_fn"):
        try:
            from .rmtt_seeker import compute_seeker_cmd
        except ImportError:
            # For standalone testing of this module, allow importing from the same directory.
            from rmtt_seeker import compute_seeker_cmd
        rmtt_control_fn._seeker_cmd_fn = compute_seeker_cmd

    vx, vy, vz, led = rmtt_control_fn._seeker_cmd_fn(
        robotNo, robotPose, obstacle_pose, obstacle_size, clock
    )

    # Bidirectional seeker-hider separation in 3D with a hard trigger at 1.0 m.
    cf2_neighbors = [
        (cf2_poses[0, j], cf2_poses[1, j], cf2_poses[2, j])
        for j in range(nbCF2)
    ]
    dvx, dvy, dvz = _compute_drone_repulsion_3d(robotPose, cf2_neighbors)
    vx += dvx
    vy += dvy
    vz += dvz

    trigger_land = False
    # -----------------------

    return vx,vy,vz,trigger_land,led
# ====================================    

# ====================================
# Control function for Crazyflie 2 drones
# should ONLY return (vx,vy,z_dist) for the robot command
# max useable numbers of drones = 3
# ====================================
def cf2_control_fn(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, clock):
    global TAKEOFF_DONE, Time2Takeoff
    nbTB3= len(tb3B_poses[0]) # number of total tb3 robots in the use
    nbTB3W = len(tb3W_poses[0]) # number of total tb3W robots in the use
    nbRMTT = len(rmtt_poses[0]) # number of total dji rmtt drones in the use
    nbCF2 = len(cf2_poses[0]) # number of total cf2 drones in the use
    nbRMEP = len(rmep_poses[0]) # number of total dji rmep in the use
    nbOBSTACLE = len(obstacle_pose[0]) # number of total obstacle positions in the environment
    led = (0,0,0) # led color (r,g,b) in range [0,255]

    #  --- TO BE MODIFIED ---
    vx = 0.0
    vy = 0.0
    z_dist = 1.0
    trigger_takeoff = False # trigger to takeoff the drone (True/False)
    trigger_land = False # trigger to land the drone (True/False)

    if not hasattr(cf2_control_fn, "_takeoff_done_by_robot"):
        cf2_control_fn._takeoff_done_by_robot = {}
    if not hasattr(cf2_control_fn, "_roaming_cmd_fn"):
        try:
            from .cf2_milestone1 import compute_roaming_cmd
        except ImportError:
            # For standalone testing of this module, allow importing from the same directory.
            from cf2_milestone1 import compute_roaming_cmd
        cf2_control_fn._roaming_cmd_fn = compute_roaming_cmd
    if not hasattr(cf2_control_fn, "_is_caught_fn"):
        try:
            from .game_referee import is_caught
        except ImportError:
            from game_referee import is_caught
        cf2_control_fn._is_caught_fn = is_caught

    takeoff_done_by_robot = cf2_control_fn._takeoff_done_by_robot
    for key in list(takeoff_done_by_robot.keys()):
        if key < 1 or key > nbCF2:
            del takeoff_done_by_robot[key]
    if robotNo not in takeoff_done_by_robot:
        takeoff_done_by_robot[robotNo] = False

    if not takeoff_done_by_robot[robotNo] and robotPose[2] < 0.05:
        trigger_takeoff = True
        led = (0, 0, 255)
    elif not takeoff_done_by_robot[robotNo]:
        if robotPose[2] > 0.1:
            takeoff_done_by_robot[robotNo] = True
        led = (0, 0, 255)
    elif cf2_control_fn._is_caught_fn(robotNo):
        trigger_land = True
        led = (255, 0, 0)
    else:
        vx, vy, z_dist, led = cf2_control_fn._roaming_cmd_fn(
            robotNo, robotPose, cf2_poses, rmtt_poses, obstacle_pose, obstacle_size, clock
        )

        # Drone separation while flying: CF2<->CF2 and CF2<->RMTT in full 3D.
        neighbors = []
        self_idx = robotNo - 1
        for j in range(nbCF2):
            if j != self_idx:
                neighbors.append((cf2_poses[0, j], cf2_poses[1, j], cf2_poses[2, j]))
        for j in range(nbRMTT):
            neighbors.append((rmtt_poses[0, j], rmtt_poses[1, j], rmtt_poses[2, j]))

        vz_cmd = float(z_dist) - float(robotPose[2])
        dvx, dvy, dvz = _compute_drone_repulsion_3d(robotPose, neighbors)
        vx += dvx
        vy += dvy
        vz_cmd += dvz
        z_dist = float(robotPose[2]) + vz_cmd
    # -----------------------

    return vx, vy, z_dist, trigger_takeoff, trigger_land, led

# ====================================
# (Ask Supervisor if you need to use these robots)
# Control function for dji rmep robots (omnidirectional robots with gripper)
# should ONLY return (vx,vy,wz) for the robot command
# max useable numbers of robots = 2
# ====================================
def rmep_control_fn(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, clock):
# ====================================
    nbTB3= len(tb3B_poses[0]) # number of total tb3 robots in the use
    nbTB3W = len(tb3W_poses[0]) # number of total tb3W robots in the use
    nbRMTT = len(rmtt_poses[0]) # number of total dji rmtt drones in the use
    nbCF2 = len(cf2_poses[0]) # number of total cf2 drones in the use
    nbRMEP = len(rmep_poses[0]) # number of total dji rmep in the use
    nbOBSTACLE = len(obstacle_pose[0]) # number of total obstacle positions in the environment

    #  --- TO BE MODIFIED ---

    vx = 0.0
    vy = 0.0
    wz = 0.0
    goal = [1.5,1,1.57]
    ex = goal[0] - robotPose[0]
    ey = goal[1] - robotPose[1]
    etheta = goal[2] - robotPose[2]
    # try to avoid using the wz if possible, not reliable 
    
    if abs(ex) > 0.1 or abs(ey) > 0.1 or abs(etheta) > 0.1:
        vx = 0.3 * ex
        vy = 0.3 * ey 
        wz = 0.1 * etheta
    else:
        vx = 0.0
        vy = 0.0
        wz = 0.0
    # -----------------------

    return vx, vy, wz
# ====================================





# ======== ! DO NOT MODIFY ! ============
def tb3B_controller(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, lidar_scan, clock):
    vx,vy = tb3B_control_fn(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, lidar_scan, clock)
    return vx,vy
def tb3W_controller(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, lidar_scan, clock):
    vx,vy = tb3W_control_fn(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, lidar_scan, clock)
    return vx,vy
def rmtt_controller(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, clock):
    vx, vy, vz, trigger_land, led = rmtt_control_fn(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, clock)
    return vx, vy, vz, trigger_land, led
def cf2_controller(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, clock):
    vx, vy, z_dist, trigger_takeoff, trigger_land, led = cf2_control_fn(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, clock)
    return vx, vy, z_dist, trigger_takeoff, trigger_land, led
def rmep_controller(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, clock):
    vx,vy,wz = rmep_control_fn(robotNo, robotPose, tb3B_poses, tb3W_poses, rmtt_poses, cf2_poses, rmep_poses, obstacle_pose, obstacle_size, clock)
    return vx,vy,wz
# =======================================

