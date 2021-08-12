# =============================================================================
# Created By  : Francesca Covella
# Created Date: Wednesday 26, Thursday 27 May 2021
# =============================================================================

import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from Constants import MU_earth, omega_E
from astrofunctions import orb2state, relacc, clohessy_wiltshire

"""


"""

# Chief (target) orbital elements at initial time
a_c = 6790.70030 # semi-major axis [kilometers] 
e_norm_c = 0 # eccentricity. 
incl_c = 51.30505 # inclination [degrees]
RAAN_c = 284.05125 #   right ascension of the ascending node [degrees]
arg_perigee_c = 128.76998 #  argument of perigee [degrees]
true_anomaly_c = 39.75224 # True anomaly [degrees]

# Conversion to state vector from orbital parameters
r_c, v_c = orb2state(a_c, e_norm_c, math.radians(incl_c), math.radians(RAAN_c), \
                    math.radians(arg_perigee_c), math.radians(true_anomaly_c))
# Angular velocity (omega)
theta_dot_c = math.sqrt(MU_earth/np.linalg.norm(r_c)**3)

# Deputy (cheaser) orbital elements at initial time
a_d = 6788.70030; # semi-major axis [kilometers] 
e_norm_d = 0.001; # eccentricity. 0.001
incl_d = 51.30505; # inclination [degrees]
RAAN_d = 284.05125; #   right ascension of the ascending node [degrees]
arg_perigee_d = 128.76998; #  argument of perigee [degrees]
true_anomaly_d = 39.75224 + 0.001; # True anomaly [degrees]

# Conversion to state vector from orbital parameters
r_d, v_d = orb2state(a_d, e_norm_d, math.radians(incl_d), math.radians(RAAN_d), \
                    math.radians(arg_perigee_d), math.radians(true_anomaly_d))

# CHIEF numerical integration of Cartesian State Vector
Y0_c = [*r_c, *v_c]
t0 = 0
tf = 1*10*3600                     # s
# delta_t = (tf-t0)/(N-1) 
delta_t = 0.5                        # s (max time beween consecutive integration points)
N = math.floor((tf-t0)/delta_t - 1)
TSPAN = np.linspace(0, tf, N)      # duration of integration in seconds
St_c = odeint(relacc, Y0_c, TSPAN) 

# DEPUTY numerical integration of Cartesian State Vector
Y0_d = [*r_d, *v_d]
St_d = odeint(relacc, Y0_d, TSPAN) 

# converting into relative position and velocity in LVLH
dim1, dim2 = np.shape(St_c)
rho  = np.empty((3, dim1))
vrel = np.empty((3, dim1))
rho_norm = np.empty((1, dim1))

for i in range(dim1):

    i_lvlh = St_c[i, :3]/np.linalg.norm(St_c[i, :3])
    j_lvlh = St_c[i, -3:]/np.linalg.norm(St_c[i, -3:])
    k_lvlh = np.cross(i_lvlh, j_lvlh)

    rho[:3, i] = np.matmul(np.array([i_lvlh, j_lvlh, k_lvlh]), 
            np.transpose([np.array(St_d[i, :3]) - np.array(St_c[i, :3])])).ravel()

    rho_norm[0, i] = math.sqrt(rho[0, i]**2 + rho[1, i]**2 + rho[2, i]**2) 

    vrel[:3, i] = (np.matmul( np.array([i_lvlh, j_lvlh, k_lvlh]),
            np.transpose([np.array(St_d[i, -3:]) - np.array(St_c[i, -3:])]) ) - \
            np.cross(theta_dot_c * np.array([[0], [0], [1]]), rho[:3, i], axis=0)).ravel()

# Integrating CW eqq using initial conditions in LVLH computed above
CW0 = [*rho[:3, 0],  *vrel[-3:, 0]]
St_CW = odeint(clohessy_wiltshire, CW0, TSPAN, args=(theta_dot_c,))

cw_norm = np.empty((1, dim1))
for elem in range(np.shape(St_CW)[0]):
    cw_norm[0, elem] = math.sqrt(St_CW[elem, 0]**2 + St_CW[elem, 1]**2 + St_CW[elem, 2]**2) 
    

# Plotting
plt.figure(1)
plt.plot(rho[1,:], rho[0,:], color='green', linewidth='2', label='analytical', marker='*', markevery=[0, -1])
plt.plot(St_CW[:, 1], St_CW[:, 0], color='pink', linewidth='2', label='numerical', marker='*', markevery=[0, -1])
plt.xlabel('along-track y (km)')
plt.ylabel('radial x (km)')
plt.title('clohessy wiltshire')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()

plt.figure(2)
plt.plot(TSPAN/3600, rho_norm.T-cw_norm.T, 'red')
plt.xlabel('time (hrs)')
plt.ylabel('error (km)')
plt.title('clohessy wiltshire num vs analytical')
plt.grid(True)
plt.show()