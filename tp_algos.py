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
    vx = 0.0
    vy = 0.0
    vz = 0.0
    trigger_land = False # trigger to land the drone (True/False)
    goal = [-2,1,1]
    ex = goal[0] - robotPose[0]
    ey = goal[1] - robotPose[1]
    ez = goal[2] - robotPose[2]
    if abs(ex) > 0.2 or abs(ey) > 0.2 or abs(ez) > 0.1:
        vx = 0.5 * ex
        vy = 0.5 * ey
        vz = 0.5 * ez
        led = (255,0,0)
    else:
        vx = 0
        vy = 0
        vz = 0
        led = (0,255,0)
        trigger_land = True
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

    # =========================================================
    # CONTROLEUR HIDE-AND-SEEK (3 drones CF2)
    # Phase 1 (clock < 15s) : formation leader-follower en V
    #   - drone 1 (i=0) : leader, suit une trajectoire elliptique
    #   - drones 2,3   : followers en formation V derrière le leader
    # Phase 2 (clock >= 15s) : dispersion intelligente
    #   - chaque drone rejoint une cachette unique près de l'obstacle
    #   - répulsion inter-drones forte + altitudes différentes (3D)
    # Couche de sécurité TOUJOURS active :
    #   - répulsion inter-drones, anti-obstacles, murs virtuels, vmax
    # =========================================================

    # ---------- Valeurs par défaut ----------
    vx, vy, z_dist          = 0.0, 0.0, 1.0
    trigger_takeoff         = False
    trigger_land            = False
    led                     = (255, 255, 255)

    # ---------- 1) DÉCOLLAGE AUTOMATIQUE ----------
    if robotPose[2] < 0.5:
        trigger_takeoff = True
        return vx, vy, z_dist, trigger_takeoff, trigger_land, led

    # ---------- 2) MISE EN PLACE ----------
    i = robotNo - 1                       # index 0-based du drone courant
    x, y = robotPose[0], robotPose[1]

    T_SEEKER = 15.0                       # arrivée du seeker (s)
    X_MIN, X_MAX = -2.5, 2.5              # limites de l'arène (cf. standalone_sim)
    Y_MIN, Y_MAX = -4.5, 4.5

    # Géométrie de l'obstacle central (mur)
    obs_x  = obstacle_pose[0][0]
    obs_y  = obstacle_pose[1][0]
    obs_sx = obstacle_size[0][0]
    obs_sy = obstacle_size[1][0]

    # Accumulateur de commande (champ de potentiel)
    u_x, u_y = 0.0, 0.0

    # =========================================================
    # PHASE 1 : FORMATION LEADER-FOLLOWER (V-shape)
    # =========================================================
    if clock < T_SEEKER:
        led = (0, 120, 255)  # bleu = mode formation

        # --- Trajectoire elliptique du leader (drone 1) ---
        # Ellipse à gauche du mur pour éviter l'obstacle en permanence.
        e_cx, e_cy = -1.7, 0.0       # centre de l'ellipse
        e_a, e_b   = 0.35, 1.4       # demi-axes (étroite en X, longue en Y)
        T_period   = 22.0            # période d'un tour (s)
        omega      = 2.0 * math.pi / T_period
        phi        = omega * clock

        leader_goal_x = e_cx + e_a * math.cos(phi)
        leader_goal_y = e_cy + e_b * math.sin(phi)

        # Tangente unitaire à l'ellipse -> cap (heading) du leader
        tx = -e_a * math.sin(phi)
        ty =  e_b * math.cos(phi)
        tn = math.hypot(tx, ty) + 1e-9
        head_x, head_y = tx / tn, ty / tn
        # Perpendiculaire (gauche du cap)
        perp_x, perp_y = -head_y, head_x

        # --- Slots de formation en V (back, side) dans le repère du leader ---
        # back  < 0 => derrière le leader
        # side  > 0 => à gauche, side < 0 => à droite
        formation = {
            0: ( 0.00,  0.00),   # leader
            1: (-0.60, +0.45),   # follower gauche arrière
            2: (-0.60, -0.45),   # follower droite arrière
        }
        b_off, s_off = formation.get(i, (-0.45, 0.0))

        if i == 0:
            # Leader suit directement le point de l'ellipse
            target_x = leader_goal_x
            target_y = leader_goal_y
        else:
            # Followers se positionnent par rapport à la POSITION MESURÉE
            # du leader, avec un cap calculé analytiquement (lisse).
            L_x = cf2_poses[0][0]
            L_y = cf2_poses[1][0]
            target_x = L_x + b_off * head_x + s_off * perp_x
            target_y = L_y + b_off * head_y + s_off * perp_y

        # Attraction vers la cible de formation
        k_attr = 1.8
        u_x += k_attr * (target_x - x)
        u_y += k_attr * (target_y - y)

        # Répulsion inter-drones légère (la formation est déjà espacée)
        d_inter_safe = 0.7
        k_inter      = 2.0

        # Altitude commune en formation
        z_dist = 1.0

    # =========================================================
    # PHASE 2 : DISPERSION INTELLIGENTE + CACHE-CACHE
    # =========================================================
    else:
        # Bords du mur (footprint XY)
        OBS_W = obs_x - obs_sx / 2.0
        OBS_E = obs_x + obs_sx / 2.0
        OBS_N = obs_y + obs_sy / 2.0
        OBS_S = obs_y - obs_sy / 2.0

        # --- Assignation déterministe d'une cachette unique par drone ---
        # Idée : chaque drone se cache dans un "coin" différent de l'obstacle.
        # Aucun couple de drones n'est visible depuis la même direction => au
        # plus 2 drones repérables à la fois par le seeker.
        if i == 0:
            # Drone 1 : coin Nord-Ouest du mur
            hide_x = OBS_W - 0.40
            hide_y = OBS_N + 0.65
        elif i == 1:
            # Drone 2 : coin Sud-Ouest du mur
            hide_x = OBS_W - 0.40
            hide_y = OBS_S - 0.65
        else:
            # Drone 3 : doit contourner le mur pour se cacher côté Est.
            # Routage sans état : on choisit la sous-cible selon la position.
            if x > OBS_E + 0.25:
                # Déjà passé à l'Est -> on se pose à la cachette finale
                hide_x = OBS_E + 0.65
                hide_y = obs_y
            elif y < OBS_S - 0.30:
                # Au sud du mur -> on peut filer vers l'Est
                hide_x = OBS_E + 0.65
                hide_y = OBS_S - 0.55
            else:
                # Encore au Nord ou à l'Ouest -> on descend vers le sud d'abord
                hide_x = obs_x
                hide_y = OBS_S - 0.80

        # Attraction adaptative : forte loin de la cachette, douce près d'elle
        d_to_hide = math.hypot(hide_x - x, hide_y - y)
        if d_to_hide < 0.40:
            k_hide = 0.7
            led    = (0, 220, 60)     # vert = caché / en position
        else:
            k_hide = 1.2
            led    = (255, 120, 0)    # orange = en repositionnement

        u_x += k_hide * (hide_x - x)
        u_y += k_hide * (hide_y - y)

        # Répulsion inter-drones forte pour ne pas se regrouper
        d_inter_safe = 1.8
        k_inter      = 3.5

        # Altitudes différenciées => dispersion aussi en 3D (plus dur à voir)
        z_per_drone = {0: 1.30, 1: 0.70, 2: 1.00}
        z_dist = z_per_drone.get(i, 1.0)

    # =========================================================
    # COUCHE DE SÉCURITÉ HIERARCHISÉE (Le Bouclier Absolu)
    # =========================================================
    
    # 1. On bride la volonté de naviguer AVANT les sécurités
    nav_x, nav_y = u_x, u_y
    v_nav = math.hypot(nav_x, nav_y) + 1e-9
    if v_nav > 0.5:  # Vitesse de navigation tranquille
        nav_x = (nav_x / v_nav) * 0.5
        nav_y = (nav_y / v_nav) * 0.5

    rep_x, rep_y = 0.0, 0.0

    # --- (a) MURS VIRTUELS (Priorité Absolue : k = 10.0) ---
    # On utilise (1/d - 1/margin)^2 : la force tend vers l'infini si on touche le mur.
    margin  = 0.50
    k_bound = 10.0
    
    d_left  = x - X_MIN
    d_right = X_MAX - x
    d_down  = y - Y_MIN
    d_up    = Y_MAX - y
    
    if 0 < d_left < margin:  rep_x += k_bound * (1.0/d_left - 1.0/margin)**2
    if 0 < d_right < margin: rep_x -= k_bound * (1.0/d_right - 1.0/margin)**2
    if 0 < d_down < margin:  rep_y += k_bound * (1.0/d_down - 1.0/margin)**2
    if 0 < d_up < margin:    rep_y -= k_bound * (1.0/d_up - 1.0/margin)**2

    # --- (b) OBSTACLES (Priorité Haute : k = 8.0) ---
    d_rep_obs = 0.55
    k_obs = 8.0
    k_slide = 2.0 # Petit bonus pour glisser au lieu de bloquer
    
    for k in range(nbOBSTACLE):
        skx, sky = obstacle_size[0][k], obstacle_size[1][k]
        cxk = max(obstacle_pose[0][k] - skx/2.0, min(x, obstacle_pose[0][k] + skx/2.0))
        cyk = max(obstacle_pose[1][k] - sky/2.0, min(y, obstacle_pose[1][k] + sky/2.0))
        
        rxk, ryk = x - cxk, y - cyk
        dk = math.hypot(rxk, ryk) + 1e-9
        
        if 0 < dk < d_rep_obs:
            mag = k_obs * (1.0/dk - 1.0/d_rep_obs)**2
            nx, ny = rxk/dk, ryk/dk
            
            rep_x += mag * nx
            rep_y += mag * ny
            
            # Glissement
            rep_x += (k_slide * mag) * (-ny)
            rep_y += (k_slide * mag) * (nx)

    # --- (c) ANTI-COLLISION DRONES (Priorité Moyenne : k = 4.0) ---
    d_drone_safe = 0.50
    k_drone = 4.0
    
    for j in range(3):
        if j != i:
            dxj = x - cf2_poses[0][j]
            dyj = y - cf2_poses[1][j]
            dj  = math.hypot(dxj, dyj) + 1e-9
            
            if 0 < dj < d_drone_safe:
                mag = k_drone * (1.0/dj - 1.0/d_drone_safe)**2
                rep_x += mag * (dxj/dj)
                rep_y += mag * (dyj/dj)

    # --- (d) SOMME ET SATURATION FINALE ---
    # La vitesse réelle est la navigation (bridée) + les répulsions (qui peuvent être immenses)
    vx_final = nav_x + rep_x
    vy_final = nav_y + rep_y

    # Saturation de sécurité matérielle (On bride à la vitesse max physique du CF2)
    v_phys_max = 1.0
    v_norm = math.hypot(vx_final, vy_final) + 1e-9
    
    if v_norm > v_phys_max:
        vx = (vx_final / v_norm) * v_phys_max
        vy = (vy_final / v_norm) * v_phys_max
    else:
        vx = vx_final
        vy = vy_final

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

