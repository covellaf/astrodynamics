# =============================================================================
# Created By  : Francesca Covella
# Created Date: Sunday 16 May 2021
# =============================================================================
import math
from numpy.lib.function_base import append
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

""" 
This code integrates the equations of relative motion between two masses m1 and m2 
m1 is the Earth
m2 is a spacecraft
"""

# Constants
G = 6.6742*math.pow(10, -20)
M1 = 5.972*math.pow(10, 24)  # Mass of Earth (kg)
M2 = 1000                    # Mass of s/c (kg)
MU = G*(M1 + M2)             # gravitational parammeter (km^3/s^2)
J2 = 0.00108263              # [-] second zonal harmonic
Re = 6378                    # km
r0 = [8000, 0, 6000]         # km
v0 = [0, math.sqrt(MU/np.linalg.norm(r0)), 0]               # km/s #7 for elliptical
S0 = [*r0, *v0]
tf = 10*24*3600               # s (10 days)
TSPAN = np.linspace(0, tf, 3000)


def relacc(S, t):
    """
    relative acceleration (kepleriam motion)
    define the function to be integrated
    """
    x, y, z, x_v, y_v, z_v = S

    dSdt = [x_v, 
            y_v, 
            z_v,
            (-MU/((math.sqrt(x**2 + y**2 + z**2))**3)) * x,
            (-MU/((math.sqrt(x**2 + y**2 + z**2))**3)) * y,
            (-MU/((math.sqrt(x**2 + y**2 + z**2))**3)) * z
            ]
    return dSdt


def relacc_with_J2(S, t):
    """
    relative acceleration (perturbed motion) considering the J2 effect
    simplified oblatness model: the radius of the earth changes (decreases) by increasing the
    latitude angle, while a change in longitude angle does not affect the radius, meaning that
    for a given latitude the earth radius is a constant, regardless the longitude.
    define the function to be integrated
    """
    x, y, z, x_v, y_v, z_v = S
    r = math.sqrt(x**2 + y**2 + z**2)
    dSdt_with_J2 = [x_v, 
                    y_v, 
                    z_v,
                    (-MU/(r**3)) * x + ( (3/2)*J2*MU*Re**2/r**4 ) * (x/r)*(5*(z**2/r**2) -1),
                    (-MU/(r**3)) * y + ( (3/2)*J2*MU*Re**2/r**4 ) * (y/r)*(5*(z**2/r**2) -1),
                    (-MU/(r**3)) * z + ( (3/2)*J2*MU*Re**2/r**4 ) * (z/r)*(5*(z**2/r**2) -3)
                    ]
    return dSdt_with_J2


def plot_solution(pos, pos_J2):
    """
    plots the solutions on the same plot in order to get a feel of how the J2 effect changes the
    keplerian unperturbed orbit over time.
    Input: positions to be compared
    Output: 3D plot
    """
    fig = plt.figure()
    # ax = plt.gca()
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal') this workes only for 2d plots
    # defining a shpere (Earth)...
    u = np.linspace(0, 2*np.pi, 100)
    w = np.linspace(0, np.pi, 100)
    x = Re * np.outer(np.cos(u), np.sin(w))
    y = Re * np.outer(np.sin(u), np.sin(w))
    z = Re * np.outer(np.ones(np.size(u)), np.cos(w))
    ax.plot3D(pos[:,0], pos[:,1], pos[:,2], color = 'r', label = 'Keplerian orbit')
    ax.plot3D(pos_J2[:,0], pos_J2[:,1], pos_J2[:,2], color = 'g', label = 'Orbit with J2 effect')
    ax.scatter(pos[0,0], pos[0,1], pos[0,2], s = 8, color = 'k', label = 'Initial position')
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.9)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    plt.grid()
    ax.legend()
    plt.show()


def main():
    """
    Functions:
    scipy.integrate.odeint(func, y0, t, args=(), ...)
    Integrate a system of ordinary differential equations.
    Outputs: 
    some interesting facts about the orbit plotted
    """
    St = odeint(relacc, S0, TSPAN)
    pos = St[:, :3]
    vel = St[:, -3:]
    St_with_J2 = odeint(relacc_with_J2, S0, TSPAN) 
    pos_with_J2 = St_with_J2[:, :3]
    vel_with_J2 = St_with_J2[:, -3:]
    # print(np.shape(pos))
    r_norm = []
    for elem in range(np.shape(pos)[0]): 
        r_norm.append( np.linalg.norm( St[elem, :3] ) )
        # the classical way to take the norm of a vector
        # r_norm.append( math.sqrt( St[elem,0]**2 + St[elem,1]**2 + St[elem,2]**2 ))
    min_altitude = min(r_norm) - Re
    speed_at_min = np.linalg.norm( St[np.argmin(r_norm) , 3:6] )
    max_altitude = max(r_norm) - Re
    speed_at_max = np.linalg.norm( St[np.argmax(r_norm) , 3:6] )
    print('___________________________________________________\n')
    print('the minimum altitude is: {} km '.format(min_altitude))
    print('the speed @ minimum altitude is: {} km/s'.format(speed_at_min))
    print('the maximum altitude is: {} km'.format(round(max_altitude, 3)))
    print('the speed @ maximum altitude is: {} km/s'.format(round(speed_at_max, 3)))
    print('___________________________________________________')
    plot_solution(pos, pos_with_J2)
    # Seems J2 has a big effect on the inclination of the orbit...
    # To be continued... (next lecture)


if __name__ == "__main__":
    main()