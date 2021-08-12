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

the orbital parameters                     [a (km),
                                            e (-),
                                            i (degrees),
                                            mean anomaly (degrees),
                                            arg_perigee (degrees),
                                            RAAN (degrees)]

into the state vector                       S = [r (km), 
                                                 v (km/s)]
"""

def kepler_elliptical(E, Ma, e):
    return E - e*math.sin(E) - Ma


def orb2state(a, e_norm, i, mean_anomaly, arg_perigee, RAAN):
    """
    This function returns the state vector of the position and velocity.
    Input : orbital parameter (6 elements, a in km, angles in radians)
    Output: position and velocity vectors (two 3-by-1 vectors) corresponding to the input parameters
    Procedure: based on the conics calculate the semi-latus rectum (p=h**2/MU) to find the norm of the angular
    momentum vector; perform a 313 euler angle rotation
    """
    # based on the type of orbit find the semi-latus rectum to compute the horm of the specific angular momentum
    if e_norm < math.pow(10, -4):
        orb_type = "circular"
        p = a
    elif e_norm > math.pow(10, -4) and e_norm < 1:
        orb_type = "elliptical"
        b = a*math.sqrt(1 - e_norm**2)
        p = b**2/a
    elif e_norm == 1:
        orb_type = "parabolic"
        p = 2*a
    elif e_norm > 1:
        orb_type = "hyperbolic"
        b = a*math.sqrt(e_norm**2 - 1)
        p = b**2/a
    # print(f'The semi-latus rectum is {p} and the orbit is {orb_type}')
    h_norm = math.sqrt(p*c.GMe) 
    # convert angles quantities to radians
    i = np.radians(i)
    Ma = np.radians(mean_anomaly)
    arg_perigee = np.radians(arg_perigee)
    RAAN = np.radians(RAAN) 
    # calculate the rotation matrices
    R3_Om = np.array( [[math.cos(RAAN), math.sin(RAAN), 0], [-math.sin(RAAN), math.cos(RAAN), 0], [0, 0, 1]] )
    R1_i  = np.array( [[1, 0, 0], [0, math.cos(i), math.sin(i)], [0, -math.sin(i), math.cos(i)]] )
    R3_om = np.array( [[math.cos(arg_perigee), math.sin(arg_perigee), 0], [-math.sin(arg_perigee), math.cos(arg_perigee), 0], [0, 0, 1]] )
    # The final transformation matrix from ECI to perifocal coordinate system 
    support_var = R3_om.dot(R1_i).dot(R3_Om)
    # x,y,z in the perifocal coord. system
    x = support_var[0, :]
    y = support_var[1, :]
    z = support_var[2, :]
    e_orb = e_norm * x
    h_orb = h_norm * z
    # calculate the true anomaly
    X0 = 0                                     # radians, initial guess for eccentric anomaly
    eccentric_anomaly = optimize.root(kepler_elliptical, X0, args=(Ma, e_norm), method='lm')
    eccentric_anomaly = eccentric_anomaly.x[0] # radians
    true_anomaly = 2*math.atan( (math.sqrt((1+e_norm)/(1-e_norm))) * math.tan(eccentric_anomaly/2) )  # radians
    # find the norm of r
    r_norm = (h_norm**2/c.GMe) * (1/(1+e_norm*math.cos(true_anomaly)))
    # resolve r in vector form
    r_orb = r_norm*math.cos(true_anomaly)*x + r_norm*math.sin(true_anomaly)*y 
    # find the two components of v 
    u_radial = r_orb/r_norm
    u_normal = np.transpose( np.cross(z, u_radial)/np.linalg.norm(np.cross(z, u_radial)) )
    # resolve v in vector form
    v_orb = (c.GMe/h_norm) * e_norm * math.sin(true_anomaly) * u_radial + \
            (c.GMe/h_norm) * (1+e_norm*math.cos(true_anomaly)) * u_normal
    # return
    return r_orb, v_orb


# r0, v0 = orb2state(42321.28, 0.1882959, 10.9412, 168.1598, 92.0366, 355.8997)
# print(r0, v0)


