# =============================================================================
# Created By  : Francesca Covella
# Created Date: May 2021
# =============================================================================

import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Mass properties  
# m1 = 5.972*math.pow(10, 24)  # Mass of Earth (kg)
# m2 = 7.346*math.pow(10, 22)  # Mass of Moon (kg)
m1 = 1.989*math.pow(10, 30)  # Mass of Sun (kg)
m2 = 5.972*math.pow(10, 24)  # Mass of Earth (kg)
# Non dimensional mass, normalised by the unit of mass (m1+m2)
m1_hat = m1/(m1 + m2)
m2_hat = m2/(m1 + m2)
mu_hat = m2_hat          # mu_hat should be <= 0.03852 for L4, L5 to be stable
# Non dimensional lengths, normalised by the distance P1P2=1
# Location of L4 
x0 = 0.5-mu_hat          
y0 = 0.5*math.sqrt(3)
z0 = 0 
# Location of L5
# x0 = 0.5-mu_hat
# y0 = -0.5*math.sqrt(3)
# z0 = 0 
# Initial conditions
p0 = [x0, y0, z0]
# Non dimensional velocity
v0 = [1e-3, 1e-2, 0] 
S0 = [*p0, *v0] # p0 + v0
# Time span of propagation
# Non dimensional time, normalised by 1/omega (angular velocity or rotating ref. frame wrt pseudo-intertial ref. frame)
# omega = 2*pi means one revolution of the second massive body about the common barycenter
t0 = 0
tf = 2*np.pi * 60 #for the earth-sun system roughly 50 years
delta_t = 2
n_steps = math.floor((tf-t0)/delta_t - 1)
# duration of integration in seconds
tspan = np.linspace(0, tf, n_steps)


def r1_func(x, y, z):
    r1_squared = (x + mu_hat)**2 + y**2 + z**2 
    r1 = math.sqrt(r1_squared)
    return r1

def r2_func(x, y, z):
    r2_squared = (1 - x - mu_hat)**2 + y**2 + z**2 
    r2 = math.sqrt(r2_squared)
    return r2

def dU_dx_func(x, r1, r2):
    dU_dx = x - (x+mu_hat)*(1-mu_hat)/r1**3 + (1 - x - mu_hat)*mu_hat/r2**3
    return dU_dx

def dU_dy_func(y, r1, r2):
    dU_dy = y - y*(1-mu_hat)/r1**3 - y*mu_hat/r2**3
    return dU_dy

def dU_dz_func(z, r1, r2):
    dU_dz = - z*(1-mu_hat)/r1**3 - z*mu_hat/r2**3
    return dU_dz

def fun(S, t):
    """
    define the function to be integrated
    """
    x, y, z, x_v, y_v, z_v = S
    r1 = r1_func(x, y, z)
    r2 = r2_func(x, y, z)
    dSdt = [x_v, 
            y_v, 
            z_v, 
            2*y_v +dU_dx_func(x, r1, r2), 
            -2*x_v +dU_dy_func(y, r1, r2), 
            dU_dz_func(z, r1, r2)]
    return dSdt


def plot_solution(pos):
    ax = plt.gca()
    # Plot 2D
    ax.plot(pos[:,0], pos[:,1], color = 'k', label = 's/c @ L4', markevery=[0, -1])
    ax.scatter(x0, 0.5*math.sqrt(3), s = 10, color = 'pink', label = 'L4')
    ax.scatter(x0, -0.5*math.sqrt(3), s = 10, color = 'g', label = 'L5')
    # ax.scatter(- mu_hat, 0, s = 8, color = 'b', label = 'Earth')
    # ax.scatter(1-mu_hat, 0, s = 8, color = 'r', label = 'Moon')
    ax.scatter(- mu_hat, 0, s = 10, color = 'orange', label = 'Sun')
    ax.scatter(1-mu_hat, 0, s = 10, color = 'b', label = 'Earth')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    # Plot 3D
    # ax = plt.axes(projection='3d')
    # ax.plot3D(pos[:,0], pos[:,1], pos[:,2], color = 'k', label = 's/c @ L4')
    # # ax.scatter(x0, 0, 0, s = 8, color = '#e377c2', label = 'L1')
    # # ax.scatter(x0, 0, 0, s = 8, color = '#7f7f7f', label = 'L2')
    # # ax.scatter(x0, 0, 0, s = 8, color = '#bcbd22', label = 'L3')
    # ax.scatter(x0, 0.5*math.sqrt(3), 0, s = 8, color = 'pink', label = 'L4')
    # ax.scatter(x0, -0.5*math.sqrt(3), 0, s = 8, color = 'g', label = 'L5')
    # # ax.scatter(0, 0, 0, s = 4, color = 'g', label = 'Barycenter')
    # # ax.scatter(- mu_hat, 0, 0, s = 8, color = 'b', label = 'Earth')
    # # ax.scatter(1-mu_hat, 0, 0, s = 8, color = 'r', label = 'Moon')
    # ax.scatter(- mu_hat, 0, 0, s = 8, color = 'orange', label = 'Sun')
    # ax.scatter(1-mu_hat, 0, 0, s = 8, color = 'b', label = 'Earth')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    plt.grid(True)
    ax.legend(loc='upper left')
    plt.show()


def main():
    """
    scipy.integrate.odeint(func, y0, t, args=(), ...)
    Integrate a system of ordinary differential equations.
    """
    St = odeint(fun, S0, tspan)
    pos = St[:, :3]
    vel = St[:, -3:]
    plot_solution(pos)
    
if __name__ == "__main__":
    main()