from typing import List, Tuple, Optional, Dict, Any, Set
import math
import shapely.geometry as sg
import shapely.ops as so
import numpy as np
import heapq

from design_params import BenchGeometry, RampParams, Slice, PitDesignParams
from pit_design import offset_polygon, clean_polygons, get_design_params_at_elevation

# --- Helper Functions ---

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def get_heading(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate heading from p1 to p2"""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def dist2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_pit_polygon_at_z(z: float, benches: List[BenchGeometry], pit_design_params: PitDesignParams) -> sg.Polygon:
    """
    Interpolates the pit polygon (hole boundary) at elevation z.
    Requires pit_design_params to accurately calculate face angle interpolation.
    """
    if not benches:
        return sg.Polygon()

    # Find global max crest and min toe
    max_crest = max(b.z_crest for b in benches)
    min_toe = min(b.z_toe for b in benches)

    if z > max_crest:
         top_benches = [b for b in benches if abs(b.z_crest - max_crest) < 1e-3]
         if top_benches:
             return so.unary_union(top_benches[0].crest_polys)
         return sg.Polygon()

    if z < min_toe:
        bottom_benches = [b for b in benches if abs(b.z_toe - min_toe) < 1e-3]
        if bottom_benches:
             return so.unary_union(bottom_benches[0].toe_polys)
        return sg.Polygon()

    for i, bench in enumerate(benches):
        if bench.z_toe <= z <= bench.z_crest:
            try:
                angle_deg, _ = get_design_params_at_elevation(z, pit_design_params)
                if angle_deg > 89.9:
                    offset_dist = 0.0
                else:
                    delta_z = bench.z_crest - z
                    offset_dist = delta_z / math.tan(math.radians(angle_deg))

                crests = so.unary_union(bench.crest_polys)
                if not crests.is_valid: crests = crests.buffer(0)

                pit_poly = crests.buffer(-offset_dist, join_style='mitre')
                return pit_poly

            except ValueError:
                return so.unary_union(bench.toe_polys)

        # Check for GAP (Berm) between benches
        # Logic: If z < z_toe of bench[i] AND z > z_crest of bench[i+1] (if exists)
        # Assumes benches are sorted or we find the "bench above"

    # More robust check: Find the bench immediately ABOVE z
    # Since we iterated and didn't find "Face", we check if we are below a toe.

    # Sort benches by z_crest descending
    sorted_benches = sorted(benches, key=lambda b: b.z_crest, reverse=True)

    for i in range(len(sorted_benches)):
        curr = sorted_benches[i]

        # Check Face (already done roughly above but let's be strict)
        if curr.z_toe <= z <= curr.z_crest:
             # Logic duplicated from above or call helper?
             # Let's trust the loop above handled faces.
             pass

        # Check if we are in the berm below this bench
        # We are below curr.z_toe
        if z < curr.z_toe:
            # Check if we are above next bench crest
            if i + 1 < len(sorted_benches):
                next_bench = sorted_benches[i+1]
                if z > next_bench.z_crest:
                    # In the Gap!
                    # Return Toe of current bench
                    return so.unary_union(curr.toe_polys)

    return sg.Polygon()

def get_z_at_xy(x: float, y: float, benches: List[BenchGeometry]) -> Optional[float]:
    """
    Determines the elevation of the pit surface at (x, y).
    Returns None if (x,y) is outside the pit definition (e.g. above top crest).
    """
    if not benches:
        return None

    point = sg.Point(x, y)
    sorted_benches = sorted(benches, key=lambda b: b.z_crest, reverse=True)

    top_crest = so.unary_union(sorted_benches[0].crest_polys)

    # Use covers to include boundary
    if not top_crest.covers(point):
        if top_crest.distance(point) > 1e-3:
            return None

    for i, bench in enumerate(sorted_benches):
        crest_poly = so.unary_union(bench.crest_polys)
        if crest_poly.covers(point):
            toe_poly = so.unary_union(bench.toe_polys)
            if toe_poly.covers(point):
                if i == len(sorted_benches) - 1:
                     return bench.z_toe
                else:
                    continue
            else:
                d_crest = crest_poly.boundary.distance(point)
                d_toe = toe_poly.boundary.distance(point)

                total_d = d_crest + d_toe
                if total_d < 1e-6:
                    return bench.z_crest

                return bench.z_crest - (bench.z_crest - bench.z_toe) * (d_crest / total_d)
        else:
             if i > 0:
                 return sorted_benches[i-1].z_toe
             else:
                 return None

    return None

def create_grid(
    benches: List[BenchGeometry],
    grid_size: float
) -> Tuple[np.ndarray, float, float, float, float, int, int]:
    # Placeholder
    return np.array([]), 0, 0, 0, 0, 0, 0

# --- Solver Logic ---

class GridNode:
    def __init__(self, r: int, c: int, z: float, g_cost: float, h_cost: float, parent=None):
        self.r = r
        self.c = c
        self.z = z # This is RAMP elevation
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost

def solve_ramp(
    slices: List[Slice],
    start_point: Tuple[float, float],
    target_z: float,
    ramp_params: RampParams,
    benches: Optional[List[BenchGeometry]] = None
) -> Tuple[List[Tuple[float, float, float]], Dict[str, Any]]:
    """
    Raster-based Least Cost Path Analysis (A*) with Wall Following
    """
    if not benches:
        return [], {"error": "Benches data required"}

    diagnostics = {}

    # 1. Grid Setup
    grid_size = max(1.0, ramp_params.z_step) # Use z_step as proxy or add xy_grid param

    # Determine top crest (UP Limit)
    top_bench = max(benches, key=lambda b: b.z_crest)
    top_z = top_bench.z_crest
    top_poly = so.unary_union(top_bench.crest_polys)

    minx, miny, maxx, maxy = top_poly.bounds
    pad = 20.0
    minx -= pad; miny -= pad; maxx += pad; maxy += pad

    cols = int(math.ceil((maxx - minx) / grid_size))
    rows = int(math.ceil((maxy - miny) / grid_size))

    diagnostics["grid"] = f"{cols}x{rows} (Size: {grid_size}m)"

    # Safety check for grid size
    if cols * rows > 1000000:
        grid_size = max(grid_size, math.sqrt((maxx-minx)*(maxy-miny)/1000000))
        cols = int(math.ceil((maxx - minx) / grid_size))
        rows = int(math.ceil((maxy - miny) / grid_size))
        diagnostics["grid_adjusted"] = f"{cols}x{rows} (Size: {grid_size}m)"

    # 2. Start
    start_c = int((start_point[0] - minx) / grid_size)
    start_r = int((start_point[1] - miny) / grid_size)

    # Get Start Z (Surface Z)
    start_z_surf = get_z_at_xy(start_point[0], start_point[1], benches)
    if start_z_surf is None:
        start_z_surf = top_z

    # Start Node Z matches surface
    start_node = GridNode(start_r, start_c, start_z_surf, 0.0, 0.0)

    open_set = []
    heapq.heappush(open_set, start_node)

    visited = {} # (r, c) -> g_cost (track cheapest way to reach cell)
    visited[(start_r, start_c)] = 0.0

    # Neighbors (8-way)
    neighbors = [
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]

    best_node = start_node
    min_dist_to_target_z = abs(start_z_surf - target_z)

    nodes_expanded = 0
    max_nodes = 50000

    # Directions for ramp descent logic
    is_downward = (target_z < start_z_surf)
    grade_factor = -1.0 if is_downward else 1.0

    # Determine average batter angle for Ideal Offset estimation
    # Just take params at top Z?
    # Better: get angle at current Z.
    # We will approximate inside loop or pre-fetch a representative angle.
    # Let's assume uniform 75 deg for heuristic to save time, or look up.
    # Using specific angle at each depth is better.

    while open_set:
        current = heapq.heappop(open_set)
        nodes_expanded += 1

        # Check finish
        is_finished = False
        if is_downward:
            if current.z <= target_z: is_finished = True
        else:
            if current.z >= target_z: is_finished = True

        if is_finished:
            best_node = current
            break

        # Track best progress
        dist_z = abs(current.z - target_z)
        if dist_z < min_dist_to_target_z:
            min_dist_to_target_z = dist_z
            best_node = current

        if nodes_expanded > max_nodes:
            break

        for dr, dc in neighbors:
            nr, nc = current.r + dr, current.c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            px = minx + (nc + 0.5) * grid_size
            py = miny + (nr + 0.5) * grid_size

            # Constraint: Must be within UP string (Top Crest)
            # Use geometric check
            p_geom = sg.Point(px, py)
            if not top_poly.covers(p_geom):
                # Allow slight tolerance?
                 if top_poly.distance(p_geom) > 1e-3:
                     continue

            # Calculate New Z (Ramp Descent)
            step_dist = math.hypot(dr, dc) * grid_size
            new_z = current.z + step_dist * ramp_params.grade_max * grade_factor

            # Cost Calculation
            # 1. Distance Cost
            move_cost = step_dist

            # 2. Wall Following Penalty
            # Ideal offset from Top Crest at depth Z
            # depth = top_z - new_z (if Downward)
            # offset = depth / tan(angle)

            # Get angle at new_z
            # Use default or lookup
            # We don't have direct access to params object easily here unless we pass it or assume constant.
            # But get_z_at_xy uses benches.
            # We can infer from benches structure or just use a fixed heuristic weight.
            # Let's assume ~75 degrees or derive from bench.

            # Better: Calculate distance to 'Top Crest' boundary
            dist_to_crest = top_poly.boundary.distance(p_geom)

            # Calculate Ideal Distance
            if is_downward:
                depth = top_z - new_z
            else:
                depth = abs(new_z - start_z_surf) # Approx for Upward?

            # Assume 75 deg (approx 3.73 tan)
            # ideal_dist = depth / tan(75) approx depth / 3.73
            ideal_dist = depth / 3.73

            deviation = abs(dist_to_crest - ideal_dist)

            # Penalty Weight
            # Deviation in meters.
            # If deviation is large, we are far from "Wall".
            # Penalty = deviation * weight
            WALL_WEIGHT = 2.0
            move_cost += deviation * WALL_WEIGHT

            new_g = current.g_cost + move_cost

            if (nr, nc) in visited and visited[(nr, nc)] <= new_g:
                continue

            visited[(nr, nc)] = new_g

            h_cost = abs(new_z - target_z) / ramp_params.grade_max

            child = GridNode(nr, nc, new_z, new_g, h_cost, parent=current)
            heapq.heappush(open_set, child)

    # Reconstruct
    path = []
    curr = best_node
    while curr:
        px = minx + (curr.c + 0.5) * grid_size
        py = miny + (curr.r + 0.5) * grid_size
        path.append((px, py, curr.z))
        curr = curr.parent

    path = path[::-1]

    path = simplify_path(path, tolerance=1.0)

    diagnostics["nodes_expanded"] = nodes_expanded
    return path, diagnostics

def simplify_path(points: List[Tuple[float, float, float]], tolerance: float) -> List[Tuple[float, float, float]]:
    if len(points) < 3: return points

    dmax = 0.0
    index = 0
    end = len(points) - 1
    p1 = np.array(points[0])
    p2 = np.array(points[end])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0: line_dir = np.zeros(3)
    else: line_dir = line_vec / line_len

    for i in range(1, end):
        p = np.array(points[i])
        v = p - p1
        proj = np.dot(v, line_dir)
        if proj <= 0: dist = np.linalg.norm(v)
        elif proj >= line_len: dist = np.linalg.norm(p - p2)
        else: dist = np.linalg.norm(v - line_dir * proj)
        if dist > dmax: index = i; dmax = dist

    if dmax > tolerance:
        rec1 = simplify_path(points[:index+1], tolerance)
        rec2 = simplify_path(points[index:], tolerance)
        return rec1[:-1] + rec2
    else:
        return [points[0], points[end]]

def generate_ramp_corridor(
    centerline: List[Tuple[float, float, float]],
    ramp_width: float
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    if not centerline or len(centerline) < 2: return [], []
    left_edge, right_edge = [], []
    half_width = ramp_width / 2.0

    for i in range(len(centerline)):
        if i == 0: p1, p2 = centerline[i], centerline[i+1]
        elif i == len(centerline) - 1: p1, p2 = centerline[i-1], centerline[i]
        else: p1, p2 = centerline[i-1], centerline[i+1]

        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        mag = math.hypot(dx, dy)
        if mag < 1e-6: nx, ny = 0, 0
        else: nx, ny = -dy / mag, dx / mag

        p = centerline[i]
        left_edge.append((p[0] + nx * half_width, p[1] + ny * half_width, p[2]))
        right_edge.append((p[0] - nx * half_width, p[1] - ny * half_width, p[2]))

    return left_edge, right_edge

def create_slices(
    benches: List[BenchGeometry],
    ramp_params: RampParams,
    pit_design_params: PitDesignParams
) -> List[Slice]:
    if not benches: return []
    top_z = max(b.z_crest for b in benches)
    bottom_z = min(b.z_toe for b in benches)
    z_levels = []
    curr_z = top_z
    while curr_z >= bottom_z - 1e-6:
        z_levels.append(curr_z)
        curr_z -= ramp_params.z_step
    slices = []
    clearance = (ramp_params.ramp_width / 2.0) + ramp_params.safety_margin
    for z in z_levels:
        pit_poly = get_pit_polygon_at_z(z, benches, pit_design_params)
        if pit_poly is None or pit_poly.is_empty:
             slices.append(Slice(z=float(z), pit_poly=sg.Polygon(), free_poly=sg.Polygon()))
             continue
        if not pit_poly.is_valid: pit_poly = pit_poly.buffer(0)

        # Include ditch_allowance in clearance
        total_clearance = clearance + ramp_params.ditch_allowance

        free_poly = pit_poly.buffer(-total_clearance, join_style='mitre')
        if not free_poly.is_valid: free_poly = free_poly.buffer(0)
        slices.append(Slice(z=float(z), pit_poly=pit_poly, free_poly=free_poly))
    return slices
