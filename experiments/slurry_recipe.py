import numpy as np

def reactant_mass(
        total_mass: float, spheres_pct:float, binder_pct:float, filler_pct:float,
        **kwargs
) -> dict:
    large_sphere_pct = kwargs.get('large_sphere_pct', 1.0)
    if (spheres_pct > 1.0) or (binder_pct > 1.0) or (filler_pct > 1.0):
        spheres_pct /= 100.0
        binder_pct /= 100.0
        filler_pct /= 100.0
    if spheres_pct + binder_pct + filler_pct != 1.0:
        raise ValueError("Percentages do not add to 100%")
    large_sphere_pct = large_sphere_pct if large_sphere_pct < 1.0 else large_sphere_pct / 100.0
    small_sphere_pct = 1.0 - large_sphere_pct
    ratios = {}
    ratios['spheres_mass'] = total_mass * spheres_pct
    ratios['binder_mass'] = total_mass * binder_pct
    ratios['filler_mass'] = total_mass * filler_pct
    ratios['large_spheres_mass'] = ratios['spheres_mass'] * large_sphere_pct
    ratios['small_spheres_mass'] = ratios['spheres_mass'] * small_sphere_pct
    return ratios
