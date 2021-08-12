# =============================================================================
# Created By  : Francesca Covella
# Created Date: Wednesday 19, Thursday 20 May 2021
# =============================================================================

import math
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import Constants  

"""
This script plots the normalised deltaVs for Hohmann and bi-elliptic transfer orbits between two coplanar and concentric
circular orbits (1: initial orbit, 4: final orbit)
ra: distance from focus (centre of initial circle) to point A (perigee or first transfer ellipse)
rc: radius of final orbit, alias the distance from the focus to the apogee of the (second) transfer ellipse
gamma: rc/ra 
rb: distance from focus to point B (apogee of first transfer ellipse for bi-elliptic transfer
                                alias apogee of second transfer ellipse for bi-elliptic transfer ---> if rb>rc
                                      perigee of second transfer ellipse for bi-elliptic transfer --> if rb<rc)
beta: rb/ra
NOTE: distances are normalized by  ra (radius of initial orbit)
      velocities are normalized by v1 (initial circular orbit velocity = (MU/ra)^1/2)

There are two possible cases:
CASE 1: rb > rc (transfer ellipse is greater than the final circular orbit)
CASE 2: rb < rc
"""

CASE = 1

if CASE == 1:
    GAMMA = np.linspace(1, 70, num=100)
    BETA  = np.linspace(70, 1000, num=100)
if CASE == 2: 
    GAMMA = np.linspace(10, 70, num=100)
    BETA  = np.linspace(1, 10, num=100)


def plot_it(dV_H, dV_BE):
    """
    Plots the delta v evolution for different ratios of GAMMA.
    GAMMA = 10 means that the final circular orbit has a radius ten times bigger than the
    initial circular orbit
    """
    fig = plt.figure()
    plt.plot(GAMMA, dV_H[0,:], color='b', linewidth=3)
    plt.plot(GAMMA, dV_BE[0, :], color='r', linewidth=3)
    fig.gca().set_xlabel(r'$\gamma$')
    fig.gca().set_ylabel(r'normalized $\Delta$ V')
    plt.grid()
    if CASE == 1:
        fig.gca().set_title(r'$\beta$ > $\gamma$')
    elif CASE == 2:
        fig.gca().set_title(r'$\gamma$ > $\beta$')
    for j in range(1, 100):
        plt.plot(GAMMA, dV_BE[j, :], color='g', linewidth=1)
    plt.legend(['Hohmann', 'Bi-elliptical', r'Bi-elliptical (varying $\beta$)'])
    plt.show()


def calculate_deltaV():
    """
    Calculates the delta v for the Hohman transfer and the bi-elliptic transfer trajectories
    in order to assess which one is the most efficient
    """
    deltaV_H  = np.empty((1, 100))
    deltaV_BE = np.empty((100, 100))
    for i in range(100):
        deltaV_H[0, i] = (1/math.sqrt(GAMMA[i]) - math.sqrt(2)*( (1-GAMMA[i])/(math.sqrt(GAMMA[i]*(1+GAMMA[i]))) ) - 1) 
        for j in range(100):
            if BETA[j] > GAMMA[i]:
                # if beta > gamma (i.e. rb > rc)
                deltaV_BE[j, i] = math.sqrt( 2*(GAMMA[i]+BETA[j])/(GAMMA[i]*BETA[j]) ) - \
                                (1+math.sqrt(GAMMA[i]))/math.sqrt(GAMMA[i]) - \
                                (1-BETA[j])*math.sqrt(2/(BETA[j]*(1+BETA[j])))
            else:
                # if gamma > beta (i.e. rc > rb)
                # WHY IN THE FORMULA GAMMA+BETA INSTEAD OF GAMMA-BETA
                deltaV_BE[j, i] = math.sqrt( 2*(GAMMA[i]+BETA[j])/(GAMMA[i]*BETA[j]) ) + \
                                (1-math.sqrt(GAMMA[i]))/math.sqrt(GAMMA[i]) - \
                                (1-BETA[j])*math.sqrt(2/(BETA[j]*(1+BETA[j])))
    return deltaV_H, deltaV_BE


def main():
    deltaV_H, deltaV_BE = calculate_deltaV()
    plot_it(deltaV_H, deltaV_BE)


if __name__ == "__main__":
    main()