import numpy as np

def calculate_ion_beam_interaction_area(diameter, height, theta_degrees):
    """
    Calculate the effective area of ion beam interaction with a vertically aligned cylinder.
    This represents the physical cross-sectional area that intercepts the ion beam.

    Parameters:
    diameter (float): Diameter of the cylinder
    height (float): Height of the cylinder
    theta_degrees (float): Angle in degrees between cylinder axis and ion beam direction

    Returns:
    tuple: (total_interaction_area, top_face_area, curved_surface_area)
    """
    # Convert angle to radians
    theta = np.radians(theta_degrees)

    # Radius of cylinder
    radius = diameter / 2

    # Calculate ion beam interaction areas

    # 1. Top circular face - flux-weighted effective area
    # When beam hits at angle theta, effective area = original_area * cos(theta)
    # This is also equal to the geometric projection of the circle
    top_face_area = np.pi * radius ** 2 * abs(np.cos(theta))

    # 2. Curved cylindrical surface - only the half facing the beam
    # The curved surface has a "frontal cross-section" visible to the beam
    # This is NOT the full circumference, but only the portion facing the beam
    # Effective area = (π * radius) * height * sin(theta)
    # where π * radius is the "frontal width" (half circumference)
    curved_surface_area = np.pi * radius * height * abs(np.sin(theta))

    # 3. Bottom face is completely shadowed - no contribution
    bottom_face_area = 0

    # Total ion beam interaction area
    total_interaction_area = top_face_area + curved_surface_area

    return total_interaction_area, top_face_area, curved_surface_area