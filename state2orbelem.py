# =============================================================================
# Created By  : Francesca Covella
# Created Date: Tuesday 06 July 2021
# =============================================================================
import math
import numpy as np
from scipy import optimize
import constants_2bp as c


"""
This script converts

the state vector                            S = [r (km), 
                                                 v (km/s)]

into orbital parameters                     [a (km),
                                            e (-),
                                            i (degrees),
                                            mean anomaly (degrees),
                                            arg_perigee (degrees),
                                            RAAN (degrees)]
"""


def state2orb(r0, v0):
    """
    This function returns the orbital parameters vector.
    Input : initial position and velocity vectors (two 3-by-1 vectors)
    Output: orbital parameter corresponding to the state vector at that time
    given in degrees
    """
    h = np.cross(r0, v0)
    h_norm = np.linalg.norm(h)
    cos_inclination = h[2]/h_norm                  # since h scalar product z = h_norm*1*cos(i) = h_3
    if np.linalg.norm(cos_inclination) >= 1:
        cos_inclination = np.sign(cos_inclination)
    inclination = math.acos(cos_inclination)
    inclination_deg = inclination*180/np.pi
    if inclination == 0 or inclination == np.pi :
        node_line = [1, 0, 0] # None  
        # pick the x-axis as your line of Nodes, which is undefined as the orbital and equatorial plane coincide
        RAAN = 0  # None 
    else :
        node_line = np.cross([0, 0, 1], h)/(np.linalg.norm(np.cross([0, 0, 1], h))) # cross vector is not commutative
        cos_RAAN = node_line[0]
        if np.linalg.norm(cos_RAAN) >= 1:
            cos_RAAN = np.sign(cos_RAAN)
        RAAN = math.acos(cos_RAAN)
    if node_line[1] < 0:
        RAAN = 2*np.pi - RAAN
    RAAN_deg = RAAN*180/np.pi
    # From the Laplace vector equation 
    e = (np.cross(v0, h))/c.GMe - r0/np.linalg.norm(r0)
    e_norm = np.linalg.norm(e)
    if e_norm < math.pow(10, -4):
        # for circular orbits choose r0 as the apse line to count the true anomaly and define the argument of perigee
        cos_arg_perigee = np.dot(r0, node_line)/np.linalg.norm(r0)
        if np.linalg.norm(cos_arg_perigee) >= 1:
            cos_arg_perigee = np.sign(cos_arg_perigee)
        arg_perigee = math.acos(cos_arg_perigee)
        if r0[2] < 0:
            arg_perigee = 2*np.pi - arg_perigee
        # arg_perigee =  # None 
    else :
        cos_arg_perigee = np.dot(e, node_line)/e_norm
        if np.linalg.norm(cos_arg_perigee) >= 1:
            cos_arg_perigee = np.sign(cos_arg_perigee)
        arg_perigee = math.acos(cos_arg_perigee)
        if e[2] < 0: # e1,e2,e3 dot 0,0,1
            arg_perigee = 2*np.pi - arg_perigee
    arg_perigee_deg = arg_perigee*180/np.pi
    perigee = (np.linalg.norm(h)**2/c.GMe) * (1/(1+e_norm))
    apogee  = (np.linalg.norm(h)**2/c.GMe) * (1/(1-e_norm))
    if apogee < 0:
        # in the case of an hyperbolic orbit
        apogee = - apogee
    semi_major_axis = (perigee+apogee)/2
    T = (2*np.pi/math.sqrt(c.GMe)) * math.pow(semi_major_axis, 3/2)  # orbital period (s)
    if e_norm < math.pow(10, -4):
        true_anomaly = 0
    else :
        cos_true_anomaly = np.dot(e, r0)/(e_norm*np.linalg.norm(r0))
        if np.linalg.norm(cos_true_anomaly) >= 1:
            cos_true_anomaly = np.sign(cos_true_anomaly)
        true_anomaly = math.acos(cos_true_anomaly)
        eccentric_anomaly = 2*math.atan(math.sqrt((1-e_norm)/(1+e_norm))*math.tan(true_anomaly/2))
        Ma = eccentric_anomaly - e_norm*math.sin(eccentric_anomaly)
        Ma_deg = np.degrees(Ma)
    u_r  = r0/np.linalg.norm(r0)
    if np.dot(v0, u_r) < 0:
        # past apogee
        true_anomaly = 2*np.pi - true_anomaly
    # true_anomaly_deg = true_anomaly*180/np.pi 
    return semi_major_axis, e_norm, inclination_deg, Ma_deg, arg_perigee_deg, RAAN_deg



# r0 = [-25456.91908256,  30676.65045501,   6930.65059687] 
# v0 = [-2.53170513, -1.94176302, -0.30092661]
# a, e_norm, i, Ma, omega, RAAN = state2orb(r0, v0)
# print(a, e_norm, i, Ma, omega, RAAN)