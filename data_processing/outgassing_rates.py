import numpy as np
import os


"""
Volume of the laser chamber
---------------------------
V = L1 x L2 x L3 + 5 x pi x r^2 * Lc
S = 2 x (L1 x L2 + L2 x L3 + L3 x L1)  + 5 x (pi * r^2 + pi * d * Lc)
"""
L1, L2, L3, d = 2.54 * np.array([12.0, 13.5, 10.5, 8.0])  # cm
Lc = 2.28  # cm
r = 0.5*d

volume_laser_chamber = L1*L2*L3 + 5.0*(np.pi * r * r * Lc)
surface_laser_chamber = 2.0 * (L1*L2 + L2*L3 + L3*L1) + 5.0 * np.pi * (r * r + d * Lc)

print( '********* Laser Chamber *********')
print(f'V = {volume_laser_chamber:.2f} cm^3 = {volume_laser_chamber*1E-3:.2f} L')
print(f'S = {surface_laser_chamber:.2f} cm^2 = {surface_laser_chamber*1E-4:.2f} m^2')


