from typing import List, Tuple

def get_sample_up_string() -> List[Tuple[float, float, float]]:
    """
    Returns a hard-coded sample Ultimate Pit (UP) string.
    The string is a closed loop of 3D coordinates (x, y, z).
    """
    # Sample points representing a simple pit boundary (e.g., a rough rectangle/polygon)
    points = [
        (0.0, 0.0, 100.0),
        (100.0, 0.0, 100.0),
        (150.0, 50.0, 100.0),
        (150.0, 150.0, 100.0),
        (50.0, 150.0, 100.0),
        (0.0, 100.0, 100.0)
    ]

    # Ensure closed loop
    if points[0] != points[-1]:
        points.append(points[0])

    return points
