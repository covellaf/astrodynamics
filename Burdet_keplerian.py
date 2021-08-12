# =============================================================================
# Created By  : Francesca Covella
# Created Date: Tuesdayb 08 June 2021
# =============================================================================

import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from Constants import MU_earth, R_earth
from astrofunctions import relacc

"""
The natural elements were derived by Burdet (1968) from Sperling’s regularisation
"""

# Known orbital parameters
# semi-major axis a

# C = math.sqrt(MU_earth)
# m = 1 #1.5,2
# dt = ...
# r = ...
# ds = C*dt/(r**m)

def Burdet_keplerian(S, t):
    """
    Returns the derivative of the state vector considering the unperturbed motion
    """
    x, y, z, x_dot, y_dot, z_dot, t_fictitious = S
    r = math.sqrt(x**2 + y**2 + z**2)
    C = 1/math.sqrt(MU_earth)
    m = 1.5
    dSdt = [x_dot, 
            y_dot, 
            z_dot,
            (-MU_earth/(r**3)) * x,
            (-MU_earth/(r**3)) * y,
            (-MU_earth/(r**3)) * z,
            (C*t)/(r**m)
            ]
    return dSdt


def main():
    """

    """
    # numerical integration of Cartesian State Vector
    r0 = [8000, 0, 6000]         # km
    v0 = [0, math.sqrt(MU_earth/np.linalg.norm(r0)), 0]               # km/s #7 for elliptical
    t0 = 0
    tf = 1*24*3600                     # s
    # delta_t = (tf-t0)/(N-1) 
    delta_t = 8                        # s (max time beween consecutive integration points)
    N = math.floor((tf-t0)/delta_t - 1)
    TSPAN = np.linspace(0, tf, N)     # duration of integration in seconds
    S0 = [*r0, *v0, ((math.sqrt(MU_earth)/np.linalg.norm(r0)*1.5))*(tf-t0)]
    St = odeint(Burdet_keplerian, S0, TSPAN)
    pos = St[:, :3]
    vel = St[:, 3:6]
    fic_t = St[:, -1]
    r_norm = []
    print(fic_t)
    
    # print(np.shape(pos))
    print(TSPAN)
    for elem in range(np.shape(pos)[0]): 
        r_norm.append( np.linalg.norm( St[elem, :3] ) )

    # plt.plot(fic_t, r_norm)
    # # plt.plot(fic_t, vel)
    # plt.show()

if __name__ == "__main__":
    main()