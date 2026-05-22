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

# ====================================
# VISION CONE PARAMETERS
# ====================================
FOV_ANGLE = math.radians(60)   # 60° field of view
FOV_RANGE = 2.5                # max detection distance (meters)

# ==============   "GLOBAL" VARIABLES KNOWN BY ALL THE FUNCTIONS ===================
# all variables declared here will be known by functions below
# use keyword "global" inside a function if the variable needs to be modified by the function



global TAKEOFF_DONE, Time2Takeoff
TAKEOFF_DONE = False
Time2Takeoff = 5 # time to wait before takeoff for the cf2 drone (in seconds)

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
# ====================================        
# Control function for dji rmtt drones (HIDER)
# ====================================
    vx, vy, vz = 0.0, 0.0, 0.0
    trigger_land = False

    # Altitude cible stable pour le hider
    target_z = 0.8 
    ez = target_z - robotPose[2]
    vz = 0.5 * ez

    # --- STRATÉGIE DU HIDER : FUITE SOUS CHAMP DE POTENTIEL ---
    fx, fy = 0.0, 0.0

    # 1. Répulsion du Seeker (le drone CF2)
    for i in range(nbCF2):
        seeker_x = cf2_poses[0, i]
        seeker_y = cf2_poses[1, i]
        
        dx = robotPose[0] - seeker_x
        dy = robotPose[1] - seeker_y
        dist = math.hypot(dx, dy)
        
        # Si le chercheur est à moins de 3 mètres, on fuit !
        if dist < 3.0 and dist > 0.01:
            # Force inversement proportionnelle à la distance
            repulsion_gain = 1.5 / (dist ** 2)
            fx += (dx / dist) * repulsion_gain
            fy += (dy / dist) * repulsion_gain

    # 2. Répulsion des bords du terrain (évite que le hider se bloque contre un mur)
    # Limites : X [-2.5, 2.5], Y [-4.5, 4.5]
    dist_wall_critique = 0.6
    if robotPose[0] - (-2.5) < dist_wall_critique: fx += 0.5 / max(0.1, robotPose[0] - (-2.5))
    if 2.5 - robotPose[0] < dist_wall_critique:  fx -= 0.5 / max(0.1, 2.5 - robotPose[0])
    if robotPose[1] - (-4.5) < dist_wall_critique: fy += 0.5 / max(0.1, robotPose[1] - (-4.5))
    if 4.5 - robotPose[1] < dist_wall_critique:  fy -= 0.5 / max(0.1, 4.5 - robotPose[1])

    # 3. Répulsion des obstacles physiques du TP
    for i in range(nbOBSTACLE):
        ox = obstacle_pose[0, i]
        oy = obstacle_pose[1, i]
        # On calcule grossièrement la distance au centre de l'obstacle
        dx = robotPose[0] - ox
        dy = robotPose[1] - oy
        dist_obs = math.hypot(dx, dy)
        if dist_obs < 1.2 and dist_obs > 0.01:
            fx += (dx / dist_obs) * (0.8 / dist_obs)
            fy += (dy / dist_obs) * (0.8 / dist_obs)

    # Si aucune force de fuite (le seeker est loin), le hider patrouille mollement ou reste caché
    if math.hypot(fx, fy) < 0.1:
        # Petit mouvement de ronde lent pour ne pas rester statique
        fx = 0.1 * math.cos(clock * 0.2)
        fy = 0.1 * math.sin(clock * 0.2)
        led = (0, 0, 255) # Bleu : mode discret / caché
    else:
        led = (255, 165, 0) # Orange : Alerte, on fuit !

    # Application des vitesses commandées
    vx = fx
    vy = fy

    return vx, vy, vz, trigger_land, led
# ====================================    


# ====================================
# FIELD OF VIEW DETECTION
# ====================================
def is_in_fov(seeker_pose, target_pose, seeker_direction):

    # Vector seeker -> target
    dx = target_pose[0] - seeker_pose[0]
    dy = target_pose[1] - seeker_pose[1]
    dz = target_pose[2] - seeker_pose[2]

    # Distance to target
    distance = math.sqrt(dx**2 + dy**2 + dz**2)

    # Too far away
    if distance > FOV_RANGE:
        return False

    # Normalize vector
    vx = dx / distance
    vy = dy / distance

    # Dot product
    dot = vx * seeker_direction[0] + vy * seeker_direction[1]

    # Clamp numerical errors
    dot = max(-1.0, min(1.0, dot))

    # Angle between seeker direction and target
    angle = math.acos(dot)

    return angle < (FOV_ANGLE / 2)


# ====================================
# Control function for Crazyflie 2 drones
# should ONLY return (vx,vy,z_dist) for the robot command
# max useable numbers of drones = 3
# ====================================
# ====================================
# Control function for Crazyflie 2 drones
# ====================================
def cf2_control_fn(robotNo, robotPose,
                   tb3B_poses, tb3W_poses,
                   rmtt_poses, cf2_poses,
                   rmep_poses,
                   obstacle_pose, obstacle_size,
                   clock):

    global TAKEOFF_DONE, Time2Takeoff
    nbRMTT = len(rmtt_poses[0])
    led = (0,0,0)
    vx, vy, z_dist = 0.0, 0.0, 1.0
    trigger_takeoff, trigger_land = False, False

    # 1. GESTION DU DECOLLAGE
    if not TAKEOFF_DONE and robotPose[2] < 0.05:
        if robotNo == 1:
            time.sleep(Time2Takeoff)
        trigger_takeoff = True
        TAKEOFF_DONE = True

    elif TAKEOFF_DONE:
        # ====================================
        # STRATÉGIE DU SEEKER : Balayage par Waypoints
        # ====================================
        # On définit une grille de recherche adaptée aux limites du simulateur (-2.5 à 2.5 sur X, -4.5 à 4.5 sur Y)
        waypoints = [
            [-2.0, -3.5],
            [ 2.0, -3.5],
            [ 2.0, -1.5],
            [-2.0, -1.5],
            [-2.0,  1.5],
            [ 2.0,  1.5],
            [ 2.0,  3.5],
            [-2.0,  3.5]
        ]
        
        # Détermination du waypoint actuel basé sur le temps ou une variable d'état.
        # Faute de pouvoir stocker un index persistant proprement sans variable globale complexe, 
        # on utilise le temps de simulation pour faire progresser le drone de façon fluide.
        period = 15.0 # Temps estimé pour atteindre chaque waypoint (en secondes)
        idx = int(clock / period) % len(waypoints)
        goal = waypoints[idx]

        # Calcul de l'erreur de position
        ex = goal[0] - robotPose[0]
        ey = goal[1] - robotPose[1]
        
        # Consigne proportionnelle fluide
        vx = 0.5 * ex
        vy = 0.5 * ey

        # ====================================
        # DIRECTION & DETECTION TARGET
        # ====================================
        # Le simulateur utilise cf2_yaws calculé sur le déplacement pour orienter le cône
        target_detected = False
        for i in range(nbRMTT):
            target_pose = rmtt_poses[:, i]
            
            # Utilisation de la fonction d'origine pour la détection
            # On recrée temporairement le vecteur direction du simulateur pour notre logique locale
            # (Le simulateur recalculera précisément le vrai cône)
            current_yaw = math.atan2(vy, vx) if (abs(vx) > 0.05 or abs(vy) > 0.05) else 0.0
            seeker_direction = [math.cos(current_yaw), math.sin(current_yaw)]

            if is_in_fov(robotPose, target_pose, seeker_direction):
                target_detected = True
                led = (255, 0, 0)
                print(f"[SEEKER] Target {i+1} detected at time {clock:.2f}s")
                vx, vy = 0.0, 0.0
                trigger_land = True
                break

        if not target_detected:
            led = (0, 255, 0)

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


