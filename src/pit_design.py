import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import shapely.geometry as sg
import shapely.ops as so
import shapely

@dataclass
class PitDesignParams:
    bench_height: float
    batter_angle_deg: float
    berm_width: float
    target_elevation: float
    # Future: dictionary for sector-based rules

@dataclass
class BenchGeometry:
    bench_id: int
    z_crest: float
    crest_poly: sg.Polygon
    z_toe: float
    toe_poly: sg.Polygon
    is_valid: bool = True
    message: str = ""

def inset_polygon(poly: sg.Polygon, distance: float) -> sg.Polygon:
    """
    Insets a polygon by a given distance.
    Distance should be positive for inward offset (inset).
    Shapely buffer uses negative distance for erosion (inset).
    """
    if not poly.is_valid:
        poly = poly.buffer(0)

    # Shapely buffer: positive expands, negative shrinks (insets)
    # So we use -distance
    res = poly.buffer(-distance, join_style='mitre') # mitre style preserves sharp corners better?

    # If the result is a MultiPolygon, we need to decide what to do.
    # For PoC, we might just take the largest one or keep all?
    # Agents.md says: "Prefer largest-area polygon as main pit continuation, and log warning"

    if res.is_empty:
        return res

    if isinstance(res, sg.MultiPolygon):
        # Find largest polygon by area
        largest = max(res.geoms, key=lambda p: p.area)
        return largest

    return res

def generate_pit_benches(
    up_points: List[Tuple[float, float, float]],
    params: PitDesignParams
) -> Tuple[List[BenchGeometry], Dict[str, Any]]:
    """
    Generates pit benches from an Ultimate Pit (UP) string.

    Returns:
        List of BenchGeometry objects
        Diagnostics dictionary
    """

    # 1. Convert UP points to Shapely Polygon (XY only)
    # Assuming up_points is a closed loop or we close it
    # We take the Z from the first point as the starting elevation if needed,
    # but strictly speaking, UP string might have varying Z.
    # Agents.md: "P0 = UP polygon at z0 (top design elevation reference)"
    # It also says: "If z not provided, treat UP as plan-view boundary at a reference elevation"
    # For this PoC, we'll assume the UP string defines the starting crest.
    # We will use the average Z of the UP string as z0, or just the first one?
    # Agents.md: "User inputs z0 OR derive z0 from topo sampling".
    # Since we don't have topo sampling yet, let's use the average Z of the UP string as the starting crest Z.

    if not up_points:
        return [], {"error": "No UP points provided"}

    avg_z = sum(p[2] for p in up_points) / len(up_points)
    current_crest_z = avg_z

    # Create polygon from XY
    poly_coords = [(p[0], p[1]) for p in up_points]
    current_crest_poly = sg.Polygon(poly_coords)

    if not current_crest_poly.is_valid:
        current_crest_poly = current_crest_poly.buffer(0)

    benches = []
    bench_id = 1

    # Precompute horizontal offsets
    # H_face = BH / tan(theta)
    # theta in degrees, convert to radians
    theta_rad = math.radians(params.batter_angle_deg)
    if theta_rad <= 0 or theta_rad >= math.pi/2:
        # Avoid division by zero or invalid angle
        # Fallback to vertical wall if 90 or invalid
        h_face = 0.0
    else:
        h_face = params.bench_height / math.tan(theta_rad)

    bw = params.berm_width

    diagnostics = {
        "start_z": current_crest_z,
        "h_face": h_face,
        "h_berm": bw,
        "bench_log": []
    }

    while current_crest_z > params.target_elevation:
        # Calculate target toe Z
        current_toe_z = current_crest_z - params.bench_height

        # Check if we went past target elevation (last partial bench?)
        # For now, we do full benches. If remaining height < bench_height, we might stop or do a partial.
        # Let's stop if we go below target_elevation.
        # Or if the next toe is below target, should we clamp it?
        # Agents.md says "num_benches OR target_elevation".
        # Let's generate until toe_z < target_elevation.

        # 1. Generate Toe Polygon from Current Crest
        # Offset inwards by H_face
        toe_poly = inset_polygon(current_crest_poly, h_face)

        if toe_poly.is_empty:
            diagnostics["bench_log"].append(f"Bench {bench_id}: Toe generation failed (empty). Stopping.")
            break

        # 2. Generate Next Crest Polygon from Toe (for the berm)
        # Offset inwards by BW
        next_crest_poly = inset_polygon(toe_poly, bw)

        # Store this bench
        b = BenchGeometry(
            bench_id=bench_id,
            z_crest=current_crest_z,
            crest_poly=current_crest_poly,
            z_toe=current_toe_z,
            toe_poly=toe_poly
        )
        benches.append(b)

        diagnostics["bench_log"].append(f"Bench {bench_id}: generated at z={current_crest_z:.1f} to {current_toe_z:.1f}")

        if next_crest_poly.is_empty:
             diagnostics["bench_log"].append(f"Bench {bench_id}: Next crest generation failed (empty berm offset). Stopping.")
             break

        # Update for next iteration
        current_crest_poly = next_crest_poly
        current_crest_z = current_toe_z # Next crest starts at same level as current toe (berm is flat)
        bench_id += 1

        # Safety break to prevent infinite loops if something is wrong
        if bench_id > 1000:
            diagnostics["bench_log"].append("Max benches limit reached.")
            break

    return benches, diagnostics
