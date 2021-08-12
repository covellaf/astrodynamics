# =============================================================================
# Created By  : Francesca Covella
# Created Date: Monday 17 May 2021
# =============================================================================

import math
from numpy import ma
from scipy.integrate import odeint
from scipy import optimize
import numpy as np


# Constants
G = 6.6742*math.pow(10, -20)
M1 = 5.972*math.pow(10, 24)  # Mass of Earth (kg)
M2 = 1000                    # Mass of s/c (kg)
MU = G*(M1 + M2)             # gravitational parammeter (km^3/s^2)
r0 = [8000, 0, 6000]         # km
v0 = [0, 10, 0]              # km/s 
h = np.cross(r0, v0)         # angular momentum vector per unit mass
e = np.cross(v0, h)/MU - r0/np.linalg.norm(r0)
e_norm = np.linalg.norm(e)
PERIGEE = (np.linalg.norm(h)**2/MU) * (1/(1 + e_norm))
APOGEE = (np.linalg.norm(h)**2/MU) * (1/(1 - e_norm))
a = (PERIGEE + abs(APOGEE))/2
THETA_INF = math.acos(-1/e_norm) * 180/(np.pi)

"""
Numerically solving Kepler equation for an hyperbolic trajectory
"""


def kepler_hyper(F, Mh, e):
    return -F + e*math.sinh(F) -Mh

def main():
    # the time from perigee where I want the true anomaly value
    t_desired = pow(10, 10)
    Mh = (MU**2)/(np.linalg.norm(h)**3) * math.pow((e_norm**2 -1), 3/2) * t_desired
    X0 = 0 # radians, initial guess for eccentric anomaly
    eccentric_anomaly = optimize.root(kepler_hyper, X0, args=(Mh, e_norm), method='lm')
    eccentric_anomaly = eccentric_anomaly.x[0]
    eccentric_anomaly_deg = eccentric_anomaly*180/math.pi
    true_anomaly = 2*math.atan( math.sqrt( (1+e_norm)/(e_norm-1) ) * round(math.tanh(eccentric_anomaly/2), 2))
    true_anomaly_deg = true_anomaly*180/math.pi
    print('------------------------\n')
    print('The true anomaly is {} rads'.format(true_anomaly))
    print('                 or {} deg'.format(true_anomaly_deg))
    print('The eccentric anomaly is {} rads'.format(eccentric_anomaly))
    print('                      or {} deg'.format(eccentric_anomaly_deg))
    print('The value of true anomaly at inf is {} degrees.'.format(THETA_INF))
    print('\n------------------------')

if __name__ == "__main__":
    main()