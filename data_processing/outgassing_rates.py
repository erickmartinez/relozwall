import numpy as np
import os

"""
With GC/Graphite/Phenolic Resin sample
"""
tube_furnace_w_sample_base_pressure = 35E-3  # Torr
tube_furnace_w_sample_outgas_pressure = 39E-3  # Torr
tube_furnace_w_sample_outgas_time = 60.0  # s

tube_furnace_20220512_base_pressure = 33E-3  # Torr
tube_furnace_20220512_outgas_pressure = 36E-3  # Torr
tube_furnace_20220512_outgas_time = 60.0  # s

laser_chamber_w_sample_base_pressure = 4E-3  # Torr
laser_chamber_w_sample_outgas_pressure = 5E-3  # Torr
laser_chamber_w_sample_outgas_time = 11.0 * 60.0  # s


def outgassing_rate(base_pressure_torr, pressure_at_given_time_torr, time_delta, volume_liter, surface_area_cm2) -> float:
      dP = pressure_at_given_time_torr - base_pressure_torr
      return dP * volume_liter / (
                    time_delta * surface_area_cm2)


"""
Volume and surface of the laser chamber
---------------------------
V = L1 x L2 x L3 + 5 x pi x r^2 * Lc
S = 2 x (L1 x L2 + L2 x L3 + L3 x L1)  + 5 x (pi * d * Lc)
"""
L1, L2, L3, d = 2.54 * np.array([12.0, 13.5, 10.5, 8.0])  # cm
Lc = 2.28  # cm
r = 0.5*d

volume_laser_chamber = L1*L2*L3 + 5.0*(np.pi * r * r * Lc)
surface_laser_chamber = 2.0 * (L1*L2 + L2*L3 + L3*L1) + 5.0 * np.pi * (d * Lc)

print( '********* Laser Chamber *********')
print(f'V = {volume_laser_chamber:.2f} cm^3 = {volume_laser_chamber*1E-3:.2f} L')
print(f'S = {surface_laser_chamber:.2f} cm^2 = {surface_laser_chamber*1E-4:.2f} m^2')
outgassing_rate_laser_chamber_w_R3N20_1 = outgassing_rate(
      base_pressure_torr=laser_chamber_w_sample_base_pressure,
      pressure_at_given_time_torr=laser_chamber_w_sample_outgas_pressure,
      time_delta=laser_chamber_w_sample_outgas_time,
      volume_liter=volume_laser_chamber*1E-3,
      surface_area_cm2=surface_laser_chamber
)
print(f'Outgassing Rate R3N20 2022/05/12: {outgassing_rate_laser_chamber_w_R3N20_1:.3E} Torr * L / (s * cm^2) = '
      f'{outgassing_rate_laser_chamber_w_R3N20_1 * 1E4:.3E} Torr * L / (s * m^2)')

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
dP = tube_furnace_w_sample_outgas_pressure - tube_furnace_w_sample_base_pressure
outgassing_rate_tube_furnace_w_sample = outgassing_rate(
      base_pressure_torr=tube_furnace_w_sample_base_pressure,
      pressure_at_given_time_torr=tube_furnace_w_sample_outgas_pressure,
      time_delta=tube_furnace_w_sample_outgas_time,
      volume_liter=volume_tube_furnace*1E-3,
      surface_area_cm2=surface_tube_furnace
)
outgassing_rate_tube_furnace = outgassing_rate(
      base_pressure_torr=tube_furnace_20220512_base_pressure,
      pressure_at_given_time_torr=tube_furnace_20220512_outgas_pressure,
      time_delta=tube_furnace_20220512_outgas_time,
      volume_liter=volume_tube_furnace*1E-3,
      surface_area_cm2=surface_tube_furnace
)

print( '********* Tube Furnace *********')
print(f'V = {volume_tube_furnace:.2f} cm^3 = {volume_tube_furnace*1E-3:.2f} L')
print(f'S = {surface_tube_furnace:.2f} cm^2 = {surface_tube_furnace*1E-4:.2f} m^2')
print(f'Outgassing Rate 2022/05/12: {outgassing_rate_tube_furnace:.3E} Torr * L / (s * cm^2) = '
      f'{outgassing_rate_tube_furnace * 1E4:.3E} Torr * L / (s * m^2)')
print(f'Outgassing Rate With 3/8" and 5/8" GC/Graphite/Phenolic Sample: {outgassing_rate_tube_furnace_w_sample:.3E} Torr * L / (s * cm^2) = '
      f'{outgassing_rate_tube_furnace_w_sample * 1E4:.3E} Torr * L / (s * m^2)')
diff = outgassing_rate_tube_furnace_w_sample - outgassing_rate_tube_furnace
print(f'Difference: {diff:.3E} Torr * L / (s * cm^2) = {diff*1E4:.3E} Torr * L / (s * cm^2)')

