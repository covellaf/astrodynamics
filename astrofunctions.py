# =============================================================================
# Created By  : Francesca Covella
# Created Date: Saturday 22 May 2021
# =============================================================================

import math
import numpy as np
import pandas as pd
from scipy.integrate import odeint

from Constants import MU_earth, R_earth, J2, omega_E

"""
General useful astrodynamic functions
"""

def state2orb(r0, v0):
    """
    Converts from state vector to orbital parameters
    """
    h = np.cross(r0, v0)
    h_norm = np.linalg.norm(h)
    cos_inclination = h[2]/h_norm       # since h scalar product z = h_norm*1*cos(i) = h_3

    if np.linalg.norm(cos_inclination) >= 1:
        cos_inclination = np.sign(cos_inclination)
    inclination = math.acos(cos_inclination)

    if inclination == 0 or inclination == np.pi :
        node_line = [1, 0, 0] # None  # pick the x-axis as your line of Nodes, which is undefined as the orbital and equatorial plane coincide
        RAAN = 0  # None 
    else :
        node_line = np.cross([0, 0, 1], h)/(np.linalg.norm(np.cross([0, 0, 1], h))) # cross vector is not commutative
        cos_RAAN = node_line[0]
        if np.linalg.norm(cos_RAAN) >= 1:
            cos_RAAN = np.sign(cos_RAAN)
        RAAN = math.acos(cos_RAAN)

    if node_line[1] < 0:
        RAAN = 2*np.pi - RAAN

    # From the Laplace vector equation 
    e = (np.cross(v0, h))/MU_earth - r0/np.linalg.norm(r0)
    e_norm = np.linalg.norm(e)

    if e_norm < math.pow(10, -5):
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

    perigee = (np.linalg.norm(h)**2/MU_earth) * (1/(1+e_norm))
    apogee  = (np.linalg.norm(h)**2/MU_earth) * (1/(1-e_norm))

    if apogee < 0:
        # in the case of an hyperbolic orbit
        apogee = - apogee

    semi_major_axis = (perigee+apogee)/2
    T = (2*np.pi/math.sqrt(MU_earth)) * math.pow(semi_major_axis, 3/2)  # orbital period (s)

    if e_norm < math.pow(10, -5):
        true_anomaly = 0
    else :
        cos_true_anomaly = np.dot(e, r0)/(e_norm*np.linalg.norm(r0))
        if np.linalg.norm(cos_true_anomaly) >= 1:
            cos_true_anomaly = np.sign(cos_true_anomaly)
        true_anomaly = math.acos(cos_true_anomaly)

    u_r  = r0/np.linalg.norm(r0)
    if np.dot(v0, u_r) < 0:
        # past apogee
        true_anomaly = 2*np.pi - true_anomaly   
    return semi_major_axis, e_norm, inclination, RAAN, arg_perigee, true_anomaly


def orb2state(a, e_norm, i, RAAN, arg_perigee, true_anomaly):
    """
    This function takes as input the 6 orbital parameters and returns a state vector of the position and velocity
    a: semi-major axis (km)
    e_norm: the norm of the eccentricity vector
    i: inclination (rad)
    RAAN: right ascension of the ascending node (deg)
    arg_perigee: argument of perigee (deg)
    true_anomaly: theta (deg)
    """
    if e_norm < math.pow(10, -4):
        p = a
    elif e_norm > math.pow(10, -4) and e_norm < 1:
        b = a*math.sqrt(1 - e_norm**2)
        p = b**2/a
    elif e_norm == 1:
        p = 2*a
    elif e_norm > 1:
        b = a*math.sqrt(e_norm**2 - 1)
        p = b**2/a
    # p = h**2 / MU_earth
    h_norm = math.sqrt(p*MU_earth) 

    R3_Om = np.array( [[math.cos(RAAN), math.sin(RAAN), 0], [-math.sin(RAAN), math.cos(RAAN), 0], [0, 0, 1]] )
    R1_i  = np.array( [[1, 0, 0], [0, math.cos(i), math.sin(i)], [0, -math.sin(i), math.cos(i)]] )
    R3_om = np.array( [[math.cos(arg_perigee), math.sin(arg_perigee), 0], [-math.sin(arg_perigee), math.cos(arg_perigee), 0], [0, 0, 1]] )
    support_var = R3_om.dot(R1_i).dot(R3_Om)
    x = support_var[0, :]
    y = support_var[1, :]
    z = support_var[2, :]
    e_orb = e_norm * x
    h_orb = h_norm * z
    r_norm = (h_norm**2/MU_earth) * (1/(1+e_norm*math.cos(true_anomaly)))
    r_orb = r_norm*math.cos(true_anomaly)*x + r_norm*math.sin(true_anomaly)*y 
    u_radial = r_orb/r_norm
    u_normal = np.transpose( np.cross(z, u_radial)/np.linalg.norm(np.cross(z, u_radial)) )
    v_orb = (MU_earth/h_norm) * e_norm * math.sin(true_anomaly) * u_radial + (MU_earth/h_norm) * (1+e_norm*math.cos(true_anomaly)) * u_normal
    return r_orb, v_orb
    

def relacc(S, t):
    """
    Returns the derivative of the state vector considering the unperturbed motion
    """
    x, y, z, x_v, y_v, z_v = S
    dSdt = [x_v, 
            y_v, 
            z_v,
            (-MU_earth/((math.sqrt(x**2 + y**2 + z**2))**3)) * x,
            (-MU_earth/((math.sqrt(x**2 + y**2 + z**2))**3)) * y,
            (-MU_earth/((math.sqrt(x**2 + y**2 + z**2))**3)) * z
            ]
    return dSdt


def relacc_with_J2(S, t):
    """
    Returns the derivative of the state vector considering the unperturbed motion
    superimposed to the J2 effect
    """
    x, y, z, x_v, y_v, z_v = S
    r = math.sqrt(x**2 + y**2 + z**2)
    dSdt_with_J2 = [x_v, 
                    y_v, 
                    z_v,
                    (-MU_earth/(r**3)) * x + ( (3/2)*J2*MU_earth*R_earth**2/r**4 ) * (x/r)*(5*(z**2/r**2) -1),
                    (-MU_earth/(r**3)) * y + ( (3/2)*J2*MU_earth*R_earth**2/r**4 ) * (y/r)*(5*(z**2/r**2) -1),
                    (-MU_earth/(r**3)) * z + ( (3/2)*J2*MU_earth*R_earth**2/r**4 ) * (z/r)*(5*(z**2/r**2) -3)
                    ]
    return dSdt_with_J2


def relacc_with_j2_and_thrust(S, t):
    """
    Returns the derivative of the state vector considering the unperturbed motion
    superimposed to the J2 effect and a thrust vector
    Possibilities:
    1) T_r*r_vec 
    2) T_s*np.cross(h_vec, r_vec) 
    3) T_h*h_vec 
    """
    x, y, z, x_v, y_v, z_v = S
    T_r = 0.001    # km/s^2 radial thrust per unit mass - we are assuming constant mass for now
    T_s = 0.00001  # km/s^2 along-V thrust per unit mass - we are assuming constant mass for now
    T_h = 0.005    # km/s^2 along-H thrust per unit mass - we are assuming constant mass for now

    r = math.sqrt(x**2 + y**2 + z**2)
    v = math.sqrt(x_v**2 + y_v**2 + z_v**2)
    r_vec = np.transpose(np.array([x/r, y/r, z/r]))
    v_vec = np.transpose(np.array([x/r, y/r, z/r]))
    h = np.cross( np.transpose(np.array([x, y, z])), np.transpose(np.array([x_v, y_v, z_v])) )
    h_vec = h/np.linalg.norm(h)
    dSdt_with_J2_and_thrust = [
                    x_v, 
                    y_v, 
                    z_v,
                    ((-MU_earth/(r**3)) * x + ( (3/2)*J2*MU_earth*R_earth**2/r**4 ) * (x/r)*(5*(z**2/r**2) -1)) + T_r*r_vec[0],
                    (-MU_earth/(r**3)) * y + ( (3/2)*J2*MU_earth*R_earth**2/r**4 ) * (y/r)*(5*(z**2/r**2) -1) + T_r*r_vec[1],
                    (-MU_earth/(r**3)) * z + ( (3/2)*J2*MU_earth*R_earth**2/r**4 ) * (z/r)*(5*(z**2/r**2) -3) + T_r*r_vec[2]
                    ]
    return dSdt_with_J2_and_thrust


def density_model(h):
    """
    https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
    input: 
    h: altitude (km)
    T: temperature (°C)
    P: pressure (kPa)
    rho: pressure (kg/m^3)
    The Earth's atmosphere is an extremely thin sheet of air extending from the surface of 
    the Earth to the edge of space. If the Earth were the size of a basketball, a tightly held 
    pillowcase would represent the thickness of the atmosphere. Gravity holds the atmosphere to 
    the Earth's surface. Within the atmosphere, very complex chemical, thermodynamic, and fluid 
    dynamics effects occur. The atmosphere is not uniform; fluid properties are constantly changing 
    with time and place. We call this change the weather.
    Variations in air properties extend upward from the surface of the Earth. 
    The sun heats the surface of the Earth, and some of this heat goes into warming the air 
    near the surface. The heated air is then diffused or convected up through the atmosphere. 
    Thus the air temperature is highest near the surface and decreases as altitude increases. 
    The speed of sound depends on the temperature and also decreases with increasing altitude. 
    The pressure of the air can be related to the weight of the air over a given location. 
    As we increase altitude through the atmosphere, there is some air below us and some air above us. 
    But there is always less air above us than was present at a lower altitude. 
    Therefore, air pressure decreases as we increase altitude. The air density depends on both the temperature 
    and the pressure through the equation of state and also decreases with increasing altitude.
    """
    h = h * 1000 # converting from km to m
    if h > 25000 : # upper stratosphere
        T = -131.21 + 0.00299*h 
        P = 2.488 * math.pow((T+273.1)/216.6, -11.388)
    elif h > 11000 and h < 25000 : # lower stratosphere
        T = -56.46
        P = 22.65 * math.exp(1.73-0.000157*h)
    elif h < 11000 : # troposphere
        T = 15.04 - 0.00649*h
        P = 101.29 * math.pow((T+273.1)/288.08, 5.256)
    rho = P/(0.2869*(T+273.1)) # density kg/m^3
    # rho = rho * 10**9 # density kg/km^3
    return rho


def relacc_with_drag(S, t):
    """
    Returns the derivative of the state vector considering the unperturbed motion
    superimposed to the effect of drag. The drag model is taken from the nasa website.
    """
    x, y, z, x_v, y_v, z_v = S
    r = math.sqrt(x**2 + y**2 + z**2)
    A = 0.25 #m^2 %.9999990323
    Cd = 2
    m = 5 #kg 
    B = Cd*A/m # ballistic coefficient
    v = np.transpose(np.array([x_v, y_v, z_v]))
    v_atm = omega_E * np.cross( np.array([0,0,1]), np.array([x,y,z]) )
    v_rel = v - np.transpose(v_atm)
    dSdt_with_drag = np.empty((6,1))
    dSdt_with_drag = [ x_v, 
                    y_v, 
                    z_v,
                    (-MU_earth/(r**3)) * x + (-((1/2)*density_model(r-R_earth)*B)*np.linalg.norm(v_rel)*v_rel*10**3)[0],
                    (-MU_earth/(r**3)) * y + (-((1/2)*density_model(r-R_earth)*B)*np.linalg.norm(v_rel)*v_rel*10**3)[1],
                    (-MU_earth/(r**3)) * z + (-((1/2)*density_model(r-R_earth)*B)*np.linalg.norm(v_rel)*v_rel*10**3)[2]
                    ]                        
    return dSdt_with_drag


def GVE(OrbEl, t, perturbation):
    """
    inputs:
    distances in km (a)
    angles in radians
    type of perturbation
    outputs:
    variation of orbital parameters
    """
    a, enorm, inclination, RAAN, arg_per, true_anomaly = OrbEl
    p = a*(1-enorm**2)
    hnorm = math.sqrt(p*MU_earth)
    rnorm = p/(1+enorm*math.cos(true_anomaly))
    # From energy equation the magnitude of the (tangent) velocity vector is 
    # energy = -mu/2a = v^2/2 -mu/r --> v = (2(-mu/2a + mu/r))^1/2 
    # semi-major axis in km
    vnorm = math.sqrt(2* (-MU_earth/(2*a) + MU_earth/rnorm))
    # perpendicular component
    vs = hnorm/rnorm
    # from pythagorean theorem
    vr = math.sqrt(vnorm**2 - vs**2)
    A = 0.25 # m^2 %.9999990323
    Cd = 2
    m = 5 # kg 
    B = Cd*A/m # ballistic coefficient
    # Perturbation of interest (pr, ps, pw, in LVLH)
    # Perturbations due to J2
    if perturbation == 'J2':
        pr = (-3/2)*((J2*MU_earth*R_earth**2)/(rnorm**4)) * (1-3*((math.sin(inclination))**2*(math.sin(true_anomaly+arg_per))**2))
        ps = (-3/2)*((J2*MU_earth*R_earth**2)/(rnorm**4)) * (((math.sin(inclination))**2*math.sin(2*(true_anomaly+arg_per))))
        pw = (-3/2)*((J2*MU_earth*R_earth**2)/(rnorm**4)) * ((math.sin(2*inclination)*math.sin(true_anomaly+arg_per)))
    elif perturbation == 'thrust':
    # Perturbations due to continuous thrust 
    #  1) .001 km/s^2 always pushing in the positive radial direction
    #  2) .00001 km/s^2 always pushing in the positive velocity direction
    #  3) .005 km/s^2 always pushing in the positive h (angular momentum) direction
        pr = pr + 0.001
        ps = ps + 0.00001
        pw = pw + 0.005
    elif perturbation == 'drag':
        rho = density_model(rnorm-R_earth)
        pr = -1/2 * rho*B*vnorm*vr * 10**3
        ps = -1/2 * rho*B*vnorm*vs * 10**3
        pw = 0
    # GVEs
    a_dot            = 2*((a**2)*enorm*math.sin(true_anomaly)/hnorm)* pr + 2*(p*(a**2)/(hnorm*rnorm))* ps
    e_dot            = (hnorm/MU_earth)*math.sin(true_anomaly)* pr + (1/(hnorm*MU_earth))*((hnorm**2+ \
                        MU_earth*rnorm)*math.cos(true_anomaly)+MU_earth*enorm*rnorm)* ps
    inclination_dot  = (rnorm/hnorm)*math.cos(true_anomaly+arg_per)* pw
    RAAN_dot         = (rnorm/(hnorm*math.sin(inclination)))*math.sin(true_anomaly+arg_per)* pw
    arg_per_dot      = -(1/(hnorm*enorm))*((hnorm**2/MU_earth)*math.cos(true_anomaly)* pr - 
                        (rnorm+hnorm**2/MU_earth)*math.sin(true_anomaly)* ps) - \
                        ((rnorm*math.sin(true_anomaly+arg_per))/(hnorm*math.tan(inclination)))* pw
    true_anomaly_dot = hnorm / (rnorm**2) + (1/(hnorm*enorm))*( (hnorm**2/MU_earth)*math.cos(true_anomaly)* pr - \
                        (rnorm+hnorm**2/MU_earth)*math.sin(true_anomaly)* ps)
    return a_dot, e_dot, inclination_dot, RAAN_dot, arg_per_dot, true_anomaly_dot


def clohessy_wiltshire(S, t, om):
    """
    Clohessy-Wiltshire differential equations of S/C relative motion
    """
    x, y, z, xdot, ydot, zdot = S
    xddot = 2*om*ydot + 3*(om**2)*x
    yddot = -2*om*xdot
    zddot = -(om**2)*z
    dCWdt = [xdot,
            ydot,
            zdot,
            xddot,
            yddot,
            zddot]
    return dCWdt


def julian_date(date):
    """
    This function takes a date vector and outputs the Julian Calendar Date
    Inputs
    date: date vector with format [year,month,day,hour,minute,second] or [year,month,day]
    Outputs
    d: Julian date (days)
    """
    yr  = date[0]
    mon = date[1]
    day = date[2]
    if len(date) == 6:
        hr  = date[3]
        min = date[4]
        sec = date[5]
        d = 367.0 * yr - math.floor( (7 * (yr + math.floor((mon + 9) / 12.0))) * 0.25 ) + math.floor( 275 * mon / 9.0 ) + \
            day + 1721013.5 + ( (sec/60.0 + min ) / 60.0 + hr ) / 24.0
    else:
        d = 367.0 * yr - math.floor( (7 * (yr + math.floor((mon + 9) / 12.0))) * 0.25 ) + math.floor( 275 * mon / 9.0 ) + \
            day + 1721013.5
    return d


def gstime(jdut1):
    """
    this function finds the greenwich sidereal time (iau-82).
    inputs          description                    range / units
    jdut1       julian date of ut1             days from 4713 bc
    outputs       
      gst       greenwich sidereal time          0 to 2pi rad
    locals        
      temp        temporary variable for reals            rad
      tut1        julian centuries from the
                  jan 1, 2000 12 h epoch (ut1)
    reference:     vallado       2007, 193, Eq 3-43
    """
    twopi = 2.0*np.pi
    deg2rad = np.pi/180.0
    # implementation
    tut1 = ( jdut1 - 2451545.0 ) / 36525.0
    temp = - math.pow(6.2, -6) * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1  + \
            (876600.0 * 3600.0 + 8640184.812866) * tut1 + 67310.54841
    temp = temp*deg2rad/240.0 % twopi 
    # checking for the quadrants
    if ( temp < 0.0 ):
        temp = temp + twopi
    gst = temp
    return gst