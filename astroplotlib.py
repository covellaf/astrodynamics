# =============================================================================
# CR_earthated By  : Francesca Covella
# Created Date: Saturday 22 May 2021
# =============================================================================

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Constants import MU_earth, R_earth, J2

"""
General useful plotting tools for astrodynamic 
"""

def plot3d_planet_orb(R, position, orb_type):
    """
    Plots a 3D orbit
    Input:
    R: radius of planet of interest
    position: position vector
    orb_type: legend label that specifies which type of model was used to find the position vector
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Create a sphere to display the planet
    u = np.linspace(0, 2*np.pi, 100)
    w = np.linspace(0, np.pi, 100)
    x = R * np.outer(np.cos(u), np.sin(w))
    y = R * np.outer(np.sin(u), np.sin(w))
    z = R * np.outer(np.ones(np.size(u)), np.cos(w))
    ax.scatter(position[0,0], position[0,1], position[0,2], s = 8, color = 'k', label = 'Initial position')
    ax.plot3D(position[:,0], position[:,1], position[:,2], color = 'r', label = f'Orbit: {orb_type}')   
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.9)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    plt.grid()
    ax.legend()
    plt.show()


def plot2d_x_ys(x, ys, line_colors, h_label, v_label, line_styles, line_widths,\
                labels=None, style='seaborn-paper', markers=None):
    """
    Can plot on the same x axis more than one curve, to perform comparisons
    Inputs:
    x: on x axis
    ys: a list for the y axis
    style: 'Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 
    'fast', 'fivethirtyeight', 'ggplot', 'seaborn', 'seaborn-bright', 'seaborn-paper', 'seaborn-pastel', 
    'seaborn-colorblind','seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 
    'seaborn-poster','seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid'
    """
    plt.figure()
    plt.style.use(style)
    if markers == None:
        markers = ['']*len(ys)
    if labels == None:
        labels = ['']*len(ys)
    else:
        plt.legend()
    for idx in range(len(ys)):
        plt.plot(x, 
                ys[idx], 
                color = line_colors[idx],
                linestyle = line_styles[idx], 
                linewidth = line_widths[idx],
                marker = markers[idx],
                label = labels[idx]
                )
    plt.grid(True)
    plt.xlabel(h_label)
    plt.ylabel(v_label)
    plt.show()