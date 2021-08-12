# =============================================================================
# Created By  : Francesca Covella
# Created Date: Monday 17 May and Thursday 3 June 2021
# =============================================================================
import math
from numpy import ma
from scipy.integrate import odeint
from scipy import optimize
import numpy as np
from Constants import G, MU_earth

"""
Kepler equation for the case of circular, elliptical, parabolic or hyperbolic orbits.
"""

def kepler_elliptical(E, Me, e):
    return E - e*math.sin(E) - Me


def kepler_hyper(F, Mh, e):
    return -F + e*math.sinh(F) -Mh


def t2theta(r0, v0, t):
    """
    If t is given and you want to find the corresponding true anomaly the solution is numerical for
    elliptic and hyperbolic trajectories, while it is analytical for circular and parabolic trajectories.
    For elliptical orbit
    1. given t
    2. compute Me
    3. Numerically solving Kepler's equation for an ellipse (E - esinE - Me = 0)
    4. compute theta (true anomaly)
    For hyperbolic orbit
    1. given t
    2. compute Mh
    3. Numerically solving Kepler's equation for an hyperbola (-F - esinhF - Mh = 0)
    4. compute theta (true anomaly)
    Output in radiants
    """
    # tolerance for parabolic case
    tol = 10**(-3)
    # Constant orbital quantities
    h = np.cross(r0, v0)         # angular momentum vector per unit mass
    e = np.cross(v0, h)/MU_earth - r0/np.linalg.norm(r0)
    h_norm = np.linalg.norm(h)
    e_norm = np.linalg.norm(e)
    if e_norm <= 10**(-5):
        # The calculations are all analytical as the orbital angular velocity is constant
        r_norm = np.linalg.norm(r0)
        T = (2*math.pi/math.sqrt(MU_earth)) * math.pow(r_norm, 3/2)
        OMEGA = 2*math.pi/T
        # t = T/2
        theta = OMEGA * t
        print(f'After time {t} s from perigee, the value of the true anomaly is {theta*180/np.pi} degrees.')
        print('------------------------\n')
    elif e_norm > 10**(-5) and e_norm < 1-tol:
        PERIGEE = (np.linalg.norm(h)**2/MU_earth) * (1/(1 + e_norm))
        APOGEE = (np.linalg.norm(h)**2/MU_earth) * (1/(1 - e_norm))
        a = (PERIGEE + APOGEE)/2
        T = (2*math.pi/math.sqrt(MU_earth)) * math.pow(a, 3/2)
        n = 2*math.pi/T     # mean motion
        Me = n * t          # mean anomaly
        X0 = 0              # radians, initial guess for eccentric anomaly
        eccentric_anomaly = optimize.root(kepler_elliptical, X0, args=(Me, e_norm), method='lm')
        eccentric_anomaly = eccentric_anomaly.x[0]
        eccentric_anomaly_deg = eccentric_anomaly*180/math.pi
        theta = 2*math.atan( (math.sqrt((1+e_norm)/(1-e_norm))) * math.tan(eccentric_anomaly/2) )
        if theta < 0:
            theta = 2*math.pi + theta
        theta_deg = theta*180/math.pi
        print('------------------------\n')
        print('The true anomaly is {} rads'.format(theta))
        print('                 or {} deg'.format(theta_deg))
        print('The eccentric anomaly is {} rads'.format(eccentric_anomaly))
        print('                      or {} deg'.format(eccentric_anomaly_deg))
        print('After time {} s from perigee, the value of the theta is {} degrees.'.format(t, theta_deg))
        print('\n------------------------')
    elif e_norm <= 1+tol or e_norm > 1-tol:
        # Mean parabolic anomaly
        Mp = ((MU_earth**2)/(h_norm**3))*t
        tan_theta_over2 = (3*Mp + math.sqrt( ((3*Mp)**2) +1))**(1/3) - (3*Mp + math.sqrt( ((3*Mp)**2) +1))**(-1/3)
        theta = 2*math.atan(tan_theta_over2)
        print('\n------------------------')
        print(f'After time {t} s from perigee, the value of the true anomaly is {theta*180/np.pi} degrees.')
    elif e_norm > 1+tol:
        PERIGEE = (np.linalg.norm(h)**2/MU_earth) * (1/(1 + e_norm))
        APOGEE = (np.linalg.norm(h)**2/MU_earth) * (1/(1 - e_norm))
        a = (PERIGEE + abs(APOGEE))/2
        THETA_INF = math.acos(-1/e_norm) * 180/(np.pi)
        Mh = (MU_earth**2)/(np.linalg.norm(h)**3) * math.pow((e_norm**2 -1), 3/2) * t
        X0 = 0          # radians, initial guess for eccentric anomaly
        hyper_ecc_anomaly = optimize.root(kepler_hyper, X0, args=(Mh, e_norm), method='lm')
        hyper_ecc_anomaly = hyper_ecc_anomaly.x[0]
        hyper_ecc_anomaly_deg = hyper_ecc_anomaly*180/math.pi
        theta = 2*math.atan( math.sqrt( (1+e_norm)/(e_norm-1) ) * round(math.tanh(hyper_ecc_anomaly/2), 10))
        theta_deg = theta*180/math.pi
        print('------------------------\n')
        print('The true anomaly is {} rads'.format(theta))
        print('                 or {} deg'.format(theta_deg))
        print('The hyperbolic eccentric anomaly is {} rads'.format(hyper_ecc_anomaly))
        print('                      or {} deg'.format(hyper_ecc_anomaly_deg))
        print('The value of true anomaly at inf is {} degrees.'.format(THETA_INF))
        print('\n------------------------')
    return theta


def theta2t(r0, v0, theta):
    """
    If theta [radiants] is given you want to find the corresponding time in seconds since perigee the solution is analytical
    for all conics.
    For an elliptic orbit:
    1. given theta
    2. Analytically compute E (E=E(e, theta))
    3. Analytically compute Me (Me = E - esinE)
    4. Compute time
    Output in seconds
    """
    # tolerance for parabolic case
    tol = 10**(-2)
    # Constant orbital quantities
    h = np.cross(r0, v0)         # angular momentum vector per unit mass
    e = np.cross(v0, h)/MU_earth - r0/np.linalg.norm(r0)
    h_norm = np.linalg.norm(h)
    e_norm = np.linalg.norm(e)
    if e_norm <= 10**(-5):
        # The calculations are all analytical as the orbital angular velocity is constant
        r_norm = np.linalg.norm(r0)
        T = (2*math.pi/math.sqrt(MU_earth)) * math.pow(r_norm, 3/2)
        OMEGA = 2*math.pi/T
        t = theta/OMEGA 
        print(f'A change in theta of {theta*180/np.pi} degrees from perigee, corresponds to {t} s passed.')
        print('------------------------\n')
    elif e_norm > 10**(-5) and e_norm < 1-tol:
        PERIGEE = (np.linalg.norm(h)**2/MU_earth) * (1/(1 + e_norm))
        APOGEE = (np.linalg.norm(h)**2/MU_earth) * (1/(1 - e_norm))
        a = (PERIGEE + APOGEE)/2
        T = (2*math.pi/math.sqrt(MU_earth)) * math.pow(a, 3/2)
        eccentric_anomaly = 2*math.atan((math.sqrt((1-e_norm)/(1+e_norm))) * math.tan(theta/2) )
        # correct E since it comes out of a atan and could be negative
        if eccentric_anomaly < 0:
            eccentric_anomaly = eccentric_anomaly + 2*np.pi
        Me = eccentric_anomaly - e_norm*math.sin(eccentric_anomaly)
        t = (T/(2*np.pi)) * Me
        # t = t/(3600)
        print('------------------------\n')
        print(f'When the true anomaly is {round(theta*180/np.pi, 3)} deg')
        print('The eccentric anomaly is {} rad or {} deg'.format(eccentric_anomaly, round((eccentric_anomaly*180/np.pi), 3)))
        print('A change in theta of {} degrees from perigee, corresponds to {} s passed.'.format(theta*180/np.pi, t))
        print('\n------------------------')
    elif e_norm <= 1+tol or e_norm > 1-tol:
        temp_var = math.tan(theta/2)
        Mp = (1/2)*temp_var + (1/6)*(temp_var**3)
        t = Mp * (h_norm**3)/(MU_earth**2)
        print(f'A change in theta of {theta*180/np.pi} degrees from perigee, corresponds to {t} s passed.')
        print('------------------------\n')
    elif e_norm > 1+tol:
        hyper_ecc_anomaly = math.asinh((math.sin(theta)*math.sqrt(e_norm**2 -1))/(1+ e_norm*math.cos(theta)))
        Mh = e_norm*math.sinh(hyper_ecc_anomaly) - hyper_ecc_anomaly
        t = Mh * (h_norm**3)/(MU_earth**2) * (e_norm**2 -1)**(-3/2)
        print(f'A change in theta of {theta*180/np.pi} degrees from perigee, corresponds to {t} s passed.')
        print('------------------------\n')
    return t


def main():
    """
    Test cases
    """
    # Initial conditions - Circular
    # r0 = [8000, 0, 6000]         # km
    # v0 = [0, math.sqrt(MU_earth/np.linalg.norm(r0)), 0] # km/s  
    # # Initial conditions - Ellipse
    r0 = [8000, 0, 6000]         # km
    v0 = [0, 8, 0]               # km/s 
    # # Initial conditions - Parabola
    # r0 = [8000, 0, 6000]         # km
    # v0 = [0, math.sqrt(2)*math.sqrt(MU_earth/np.linalg.norm(r0)), 0] # km/s  
    # # Initial conditions - Hyperbola
    # r0 = [8000, 0, 6000]         # km
    # v0 = [0, 10, 0]              # km/s 
    # theta_desired = 3.4836476787763977 # 1 * np.pi/2    
    # t_desired = 2*T/3
    true_anomaly = t2theta(r0, v0, t=1000)
    time = theta2t(r0, v0, true_anomaly)

if __name__ == "__main__":
    main()