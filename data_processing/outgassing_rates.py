import numpy as np
import os

tube_furnace_base_pressure = 35E-3  # Torr
tube_furnace_outgas_pressure = 39E-3  # Torr
tube_furnace_outgas_time = 60.0  # s


"""
Volume and surface of the laser chamber
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

"""
Volume and surface area of the outgassing chamber in the extruder system
"""
d, L = 2.54 * np.array([6.0, 5.866])  # cm
r = 0.5 * d  # cm
volume_extruder_chamber = np.pi * r * r * L
surface_extruder_chamber = np.pi * (d * L + 2.0 * r * r)

print( '********* Extruder Chamber *********')
print(f'V = {volume_extruder_chamber:.2f} cm^3 = {volume_extruder_chamber*1E-3:.2f} L')
print(f'S = {surface_extruder_chamber:.2f} cm^2 = {surface_extruder_chamber*1E-4:.2f} m^2')

"""
24" Tube Furnace
"""
d, L = 2.54 * np.array([0.866, 29.0])  # cm
r = 0.5 * d  # cm
volume_tube_furnace = np.pi * r * r * L
surface_tube_furnace = np.pi * (d * L + 2.0 * r * r)
dP = tube_furnace_outgas_pressure - tube_furnace_base_pressure
outgassing_rate_tube_furnace = dP * volume_tube_furnace * 1E-3 / (tube_furnace_outgas_time * surface_tube_furnace)

print( '********* Tube Furnace *********')
print(f'V = {volume_tube_furnace:.2f} cm^3 = {volume_tube_furnace*1E-3:.2f} L')
print(f'S = {surface_tube_furnace:.2f} cm^2 = {surface_tube_furnace*1E-4:.2f} m^2')
print(f'Outgassing Rate: {outgassing_rate_tube_furnace:.3E} Torr * L / (s * cm^2) = '
      f'{outgassing_rate_tube_furnace*1E4:.3E} Torr * L / (s * m^2)')

