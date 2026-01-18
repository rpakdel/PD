from typing import List, Tuple
import math

def get_sample_up_string(num_points: int = 72, radius: float = 200.0, center: Tuple[float, float] = (0.0, 0.0), elevation: float = 100.0) -> List[Tuple[float, float, float]]:
    """
    Returns a generated sample Ultimate Pit (UP) string representing a circular pit.
    The string is a closed loop of 3D coordinates (x, y, z).

    Args:
        num_points: Number of points in the circle (excluding closure).
        radius: Radius of the pit string.
        center: (x, y) center of the pit.
        elevation: Z coordinate for the string.
    """
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y, elevation))

    # Ensure closed loop
    points.append(points[0])

    return points
