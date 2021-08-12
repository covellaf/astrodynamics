# =============================================================================
# Created By  : Francesca Covella
# Created Date: Tuesday 25 May 2021
# =============================================================================

import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from Constants import MU_earth, omega_E, R_earth
from astrofunctions import orb2state, state2orb, relacc_with_J2, GVE, julian_date, gstime, \
                            relacc_with_j2_and_thrust, relacc_with_drag
from astroplotlib import plot3d_planet_orb, plot2d_x_ys

"""
GVE
"""
JulianDate = julian_date( [2017, 3, 3, 12, 00, 00] ) # equivalent to 2017/062/12:00:00.000
gst = gstime(JulianDate)                             # rad Greenwhich location on julian date (longitude in ECI)
# ISS orbital elements at initial time
a = 6788.70030  # semi-major axis [kilometers] 
e_norm = 0.0017144  # eccentricity. 
incl = 51.30505  # inclination [degrees]
RAAN = 284.05125  # right ascension of the ascending node [degrees]
arg_perigee = 128.76998  # argument of perigee [degrees]
theta = 39.75224  # True anomaly [degrees]

# Conversion to state vector from orbital parameters 
p = a*(1-e_norm**2)  # semi-latus rectum (intermediate variable)
q = p/(1+e_norm*math.cos((math.radians(theta)))) # Kepler first law (intermediate variable)
# Converting orbital parameters to ECI cartsian coordinates 
r = np.empty((3, 1))
v = np.empty((3, 1))

# Alternative way to find r and v from the perifocal coord system
# Creating r vector in pqw coordinates
# R_pqw = np.empty((3, 1))
# R_pqw[0, 0] = q*math.cos((math.radians(theta)))
# R_pqw[1, 0] = q*math.sin((math.radians(theta)))
# R_pqw[2, 0] = 0
# Creating v vector in pqw coordinates  
# V_pqw = np.empty((3, 1))
# V_pqw[0, 0] = -(MU_earth/p)**0.5 * math.sin((math.radians(theta)))
# V_pqw[1, 0] =  (MU_earth/p)**0.5 * (e_norm+math.cos((math.radians(theta))))
# V_pqw[2, 0] = 0
# V = math.sqrt(V_pqw[0, 0]**2 + V_pqw[1, 0]**2 + V_pqw[2, 0]**2)
# # Solving for 313 rotation matrices
# R3_Om = np.array( [[math.cos(math.radians(RAAN)), math.sin(math.radians(RAAN)), 0], [-math.sin(math.radians(RAAN)), \
#         math.cos(math.radians(RAAN)), 0], [0, 0, 1]] )
# R1_i  = np.array( [[1, 0, 0], [0, math.cos(math.radians(incl)), math.sin(math.radians(incl))], \
#         [0, -math.sin(math.radians(incl)), math.cos(math.radians(incl))]] )
# R3_om = np.array( [[math.cos(math.radians(arg_perigee)), math.sin(math.radians(arg_perigee)), 0], \
#         [-math.sin(math.radians(arg_perigee)), math.cos(math.radians(arg_perigee)), 0], [0, 0, 1]] )
# support_var = R3_om.dot(R1_i).dot(R3_Om)
# support_var = np.transpose(support_var) # Transposed
# r = np.empty((3, 1))
# v = np.empty((3, 1))
# r[:, 0] = np.matmul(support_var, R_pqw).ravel() # Radius r [km] in ECI Cartesian
# v[:, 0] = np.matmul(support_var, V_pqw).ravel() # Velocity v [km/s] in ECI Cartesian

# Conversion to state vector from orbital parameters
r, v = orb2state(a, e_norm, math.radians(incl), math.radians(RAAN), math.radians(arg_perigee), math.radians(theta))

# Numerical integration of Cartesian State Vector (with J2 effect)
S0 = [*r, *v]
t0 = 0
tf = 7*24*3600                     # s
# delta_t = (tf-t0)/(N-1) 
delta_t = 3                        # s (max time beween consecutive integration points)
N = math.floor((tf-t0)/delta_t - 1)
TSPAN = np.linspace(0, tf, N)     # duration of integration in seconds
print(f'The max step size of the integrator is {delta_t} and the number of steps is {N}')
St = odeint(relacc_with_J2, S0, TSPAN) 
# Numerical integration of Cartesian State Vector (with drag perturbation)
# St = odeint(relacc_with_drag, S0, TSPAN)
pos = St[:, :3]
vel = St[:, -3:]

# Numerical integration of Gauss Variational equations
OrbEl0 = [a, e_norm, math.radians(incl), math.radians(RAAN), math.radians(arg_perigee), math.radians(theta)]
St_gauss = odeint( GVE, OrbEl0, TSPAN, args=('drag', )) 
# fugure 1
plot3d_planet_orb(R_earth, pos, 'with drag')

# Orbital elements, latitude and longitude (to be corrected with Earth's rotation)
dim1, dim2 = np.shape(St)
a_out = np.empty((1, dim1))
enorm = np.empty((1, dim1))
inclination = np.empty((1, dim1))
RAAN = np.empty((1, dim1))
arg_per = np.empty((1, dim1))
true_anomaly = np.empty((1, dim1))
lat = np.empty((1, dim1))
lon = np.empty((1, dim1))

for i in range(dim1):
    a_out[0, i], enorm[0, i], inclination[0, i], RAAN[0, i], arg_per[0, i], true_anomaly[0, i] = state2orb(St[i,:3], St[i,-3:])
    lat[0, i] = math.asin(St[i,2]/np.linalg.norm(St[i,:3]))
    lon[0, i] = math.atan2(St[i,1], St[i,0])

t_hrs = np.empty((1, dim1))
for t, i in zip(TSPAN, range(dim1)):
    t_hrs[0, i] = t/3600

inclination_deg = np.empty((1, dim1))
RAAN_deg = np.empty((1, dim1))
arg_per_deg = np.empty((1, dim1))
theta_deg = np.empty((1, dim1))
lat_deg = np.empty((1, dim1))
for i in range(dim1):
    inclination_deg[0, i] = inclination[0, i] * 180/np.pi
    RAAN_deg[0, i] = RAAN[0, i] * 180/np.pi
    arg_per_deg[0, i] = arg_per[0, i] * 180/np.pi
    theta_deg[0, i] = true_anomaly[0, i] * 180/np.pi
    lat_deg[0, i] = lat[0, i] * 180/np.pi

plot2d_x_ys(t_hrs[0,:], [a_out[0,:], St_gauss[:, 0]], ['blue', 'red'], 
            'time (hrs)', r'semi-major axis $a$ (km)', ['-', '-'], ['2','2'])
plot2d_x_ys(t_hrs[0,:], [enorm[0,:], St_gauss[:, 1]], ['blue', 'red'], 
            'time (hrs)', r'eccentricity $\epsilon$', ['-', '-'], ['2','2'])
plot2d_x_ys(t_hrs[0,:], [inclination_deg[0,:], St_gauss[:, 2]*180/np.pi], ['blue', 'red'], 
            'time (hrs)', r'inclination $i$ (deg)', ['-', '-'], ['2','2'])
plot2d_x_ys(t_hrs[0,:], [RAAN_deg[0,:], St_gauss[:, 3]* 180/np.pi], ['blue', 'red'], 
            'time (hrs)', r'RAAN $\Omega$ (deg)', ['-', '-'], ['2','2'])
plot2d_x_ys(t_hrs[0,:], [arg_per_deg[0,:], St_gauss[:, 4]* 180/np.pi], ['blue', 'red'], 
            'time (hrs)', r'argument of perigee $\omega$ (deg)', ['-','-'], ['2','2'])
plot2d_x_ys(t_hrs[0,:], [theta_deg[0,:], St_gauss[:, 5]* 180/np.pi], ['blue', 'red'], 
            'time (hrs)', r'true anomaly $\theta$ (deg)', ['-','-'], ['2','2'])
