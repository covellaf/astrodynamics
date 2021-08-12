# =============================================================================
# Created By  : Francesca Covella
# Created Date: Saturday 22 May 2021
# =============================================================================

import math
import numpy as np


"""
Constant quantities
"""

G = 6.6742*math.pow(10, -20)          # Gravitational Constant (km^3/(kg*s^2))
R_earth = 6378                        # Radius of Earth (km)
M_earth = 5.972*math.pow(10, 24)      # Mass of Earth (kg)
MU_earth = G * M_earth                # gravitational parammeter (km^3/s^2)
omega_E = 72.9217*math.pow(10, -6)    # Earth's angular velocity (rad/s)
J2 = 0.00108263                       # [-] second zonal harmonic
