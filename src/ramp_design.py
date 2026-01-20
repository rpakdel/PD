from typing import List, Tuple, Optional, Dict, Any
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

    For Downward pit (Crest Z > Toe Z):
    - Bench i: z_crest -> z_toe (Face)
    - Berm: z_toe (of i) -> z_crest (of i+1) (Berm)
    """
    if not benches:
        return sg.Polygon()

    # Sort benches by z_crest descending just in case
    # Assume downward generation logic for now (top-down)
    # If upward, benches are still stored with z_crest > z_toe.
    # We rely on z levels.

    # 1. Check if Z is above the top bench crest
    top_bench = benches[0]
    if z > top_bench.z_crest:
        # Return the top crest polygon (the "rim")
        return so.unary_union(top_bench.crest_polys)

    # 2. Iterate through benches to find where Z falls
    for i, bench in enumerate(benches):
        # Case A: Z is on the Face (between crest and toe)
        # Note: z_crest >= z >= z_toe
        if bench.z_toe <= z <= bench.z_crest:
            # Interpolate on the face
            try:
                # Get design params for this elevation to know the angle
                angle_deg, _ = get_design_params_at_elevation(z, pit_design_params)

                # Check for near-vertical wall
                if angle_deg > 89.9:
                    offset_dist = 0.0
                else:
                    delta_z = bench.z_crest - z
                    offset_dist = delta_z / math.tan(math.radians(angle_deg))

                # The hole boundary is the Crest Inset by offset_dist
                # Buffer negative to shrink the hole (move wall inward)?
                # Wait.
                # Bench Crest is the OUTER boundary of the hole at that level.
                # As we go down, the hole gets SMALLER (walls come in).
                # So we INSET the Crest.

                crests = so.unary_union(bench.crest_polys)
                if not crests.is_valid: crests = crests.buffer(0)

                pit_poly = crests.buffer(-offset_dist, join_style='mitre')
                return pit_poly

            except ValueError:
                # Fallback if params issue
                return so.unary_union(bench.toe_polys)

        # Case B: Z is below this bench's toe, but potentially in the Berm before the next bench
        if z < bench.z_toe:
            # Check if there is a next bench
            if i + 1 < len(benches):
                next_bench = benches[i+1]
                if z > next_bench.z_crest:
                    # Z is in the Berm zone (Between Bench i Toe and Bench i+1 Crest)
                    # The hole boundary is defined by the Toe of Bench i (upper wall limit)
                    # The material starts at the Toe of Bench i and goes out.
                    # The berm is "ground".
                    # So the valid "air" polygon is the Toe of Bench i.

                    # Actually, if we are on the berm, the wall is at the Toe of the bench ABOVE.
                    # So the hole is the Toe of bench i.
                    return so.unary_union(bench.toe_polys)
            else:
                # Below the last bench toe.
                # The pit bottom is defined by the last toe.
                return so.unary_union(bench.toe_polys)

    # If we fall through (should be covered by "Below last bench toe"), return last toe
    return so.unary_union(benches[-1].toe_polys)


def create_slices(
    benches: List[BenchGeometry],
    ramp_params: RampParams,
    pit_design_params: PitDesignParams
) -> List[Slice]:
    """
    Generates slices for the ramp solver.
    """
    if not benches:
        return []

    # Determine Z range
    # Benches might be ordered top-down or bottom-up depending on generation
    # Robustly find max and min Z
    top_z = max(b.z_crest for b in benches)
    bottom_z = min(b.z_toe for b in benches)

    # Generate Z levels
    # We want to start exactly at top_z and go down.

    # Num steps
    total_height = top_z - bottom_z
    num_steps = int(math.ceil(total_height / ramp_params.z_step))

    # We can just iterate
    z_levels = []
    curr_z = top_z
    # We use a small epsilon for float comparison to include bottom_z if exact hit
    while curr_z >= bottom_z - 1e-6:
        z_levels.append(curr_z)
        curr_z -= ramp_params.z_step

    # Ensure we include the bottom if it's close or missed
    # Check if we have any levels
    if not z_levels:
         # Should not happen given logic above, unless top_z < bottom_z (impossible here)
         pass
    elif abs(z_levels[-1] - bottom_z) > 1e-3:
        # If we stopped above bottom_z, check if we need to add bottom_z
        # (Usually logic above covers it, but floating point drift might stop slightly before)
        # However, the loop continues while >= bottom_z.
        # If we are effectively at bottom_z, we stop.
        pass

    slices = []

    # Pre-calculate clearance
    # Clearance = Ramp Half Width + Safety Margin + Ditch
    clearance = (ramp_params.ramp_width / 2.0) + ramp_params.safety_margin + ramp_params.ditch_allowance

    for z in z_levels:
        pit_poly = get_pit_polygon_at_z(z, benches, pit_design_params)

        if pit_poly is None or pit_poly.is_empty:
            continue

        # Create Free Space Polygon (Erosion)
        if not pit_poly.is_valid:
            pit_poly = pit_poly.buffer(0)

        free_poly = pit_poly.buffer(-clearance, join_style='mitre')

        # Clean up free_poly
        if not free_poly.is_valid:
            free_poly = free_poly.buffer(0)

        # We store it even if empty (solver needs to know it's blocked)
        slices.append(Slice(z=float(z), pit_poly=pit_poly, free_poly=free_poly))

    return slices

# --- Solver Logic ---

class SearchState:
    def __init__(self, slice_idx: int, point: Tuple[float, float, float], heading: float, cost: float, parent=None):
        self.slice_idx = slice_idx
        self.point = point
        self.heading = heading
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost

def solve_ramp(
    slices: List[Slice],
    start_point: Tuple[float, float],
    target_z: float,
    ramp_params: RampParams
) -> Tuple[List[Tuple[float, float, float]], Dict[str, Any]]:
    """
    Backtracking / Beam Search Solver for Ramp Centerline.
    """
    if not slices:
        return [], {"error": "No slices"}

    # 1. Initialize
    # Find start slice (closest to start Z, usually top)
    start_slice_idx = 0
    # Or find slice closest to start_point Z?
    # Assuming start_point is 2D (x,y) on the top slice (index 0)

    # If start_point is 3D, find matching slice
    # But usually user clicks on top.

    current_z = slices[0].z
    start_node = SearchState(0, (start_point[0], start_point[1], current_z), 0.0, 0.0)

    # Priority Queue for Beam Search
    # Using a list of candidates per level to implement Beam Search
    # Current level states
    current_states = [start_node]

    # Memoization: Key = (slice_idx, quantized_x, quantized_y, quantized_heading) -> min_cost
    visited = {}

    diagnostics = {
        "nodes_expanded": 0,
        "depth_reached": 0
    }

    # Beam width
    BEAM_WIDTH = 20

    final_state = None

    # Iterate through levels (slices)
    # We want to go from slices[0] down to target_z

    target_slice_idx = -1
    for i, s in enumerate(slices):
        if s.z <= target_z:
            target_slice_idx = i
            break

    if target_slice_idx == -1:
        target_slice_idx = len(slices) - 1 # Go as deep as possible

    for i in range(target_slice_idx):
        diagnostics["depth_reached"] = i
        next_states = []

        slice_curr = slices[i]
        slice_next = slices[i+1]

        # Determine step length limits based on grade
        dz = abs(slice_curr.z - slice_next.z)
        if dz < 1e-3: continue # Skip if z mismatch

        # L = dz / grade
        # min/max
        l_target = dz / ramp_params.grade_max
        l_min = l_target * (1.0 - ramp_params.horizontal_tol)
        l_max = l_target * (1.0 + ramp_params.horizontal_tol)

        # Process current states
        for state in current_states:
            diagnostics["nodes_expanded"] += 1

            # Pruning based on visited
            # Quantize state
            q_x = round(state.point[0] / 2.0) * 2.0 # 2m grid
            q_y = round(state.point[1] / 2.0) * 2.0
            q_h = round(state.heading / (math.pi/16)) # 11.25 deg buckets

            key = (i, q_x, q_y, q_h)
            if key in visited and visited[key] <= state.cost:
                continue
            visited[key] = state.cost

            # Generate Candidates
            # 1. Determine heading range
            # If i=0, we can go any direction (or towards center?)
            # If i>0, constrained by min_radius

            headings = []
            if i == 0:
                # Initial headings: try 8 directions
                for k in range(8):
                    headings.append(k * math.pi / 4)
            else:
                # Constrained by previous heading and radius
                # Turn angle limit: theta_max approx 2 * asin( L / 2R )
                # L approx l_target
                # clamp argument to [-1, 1]
                val = l_target / (2 * ramp_params.min_radius)
                if val > 1.0: val = 1.0
                theta_max = 2 * math.asin(val)

                # Sample headings within [current_heading - theta, current_heading + theta]
                num_samples = 5
                step = (2 * theta_max) / (num_samples - 1) if num_samples > 1 else 0
                for k in range(num_samples):
                    h = state.heading - theta_max + k * step
                    headings.append(normalize_angle(h))

            for h in headings:
                # Sample radii (distances)
                dists = [l_min, l_target, l_max]
                for d in dists:
                    # Calculate new point
                    nx = state.point[0] + d * math.cos(h)
                    ny = state.point[1] + d * math.sin(h)
                    nz = slice_next.z
                    p_next = (nx, ny, nz)

                    # Hard Constraints

                    # 1. Inside Free Space of NEXT slice
                    p_geom = sg.Point(nx, ny)
                    if not slice_next.free_poly.contains(p_geom):
                        continue

                    # 2. Segment Validity (check midpoint/samples)
                    # Simple check: midpoint
                    mx = (state.point[0] + nx) / 2
                    my = (state.point[1] + ny) / 2
                    m_geom = sg.Point(mx, my)
                    # Check against INTERPOLATED slice?
                    # Approximation: check against curr and next free poly intersection?
                    # Or just next (conservative if walls are vertical/steep)
                    if not slice_next.free_poly.contains(m_geom):
                         continue

                    # 3. Calculate Cost
                    # Cost = path length + penalties
                    # Penalty for curvature?
                    # Penalty for proximity to wall?

                    # Distance to wall
                    dist_to_wall = slice_next.free_poly.boundary.distance(p_geom)
                    # We want to maximize distance to wall (stay centered) -> minimize -dist
                    # But not too much, spiral needs to hug wall?
                    # Actually for spiral we want to follow the pit shape.

                    # Switchback vs Spiral Logic
                    # If Switchback:
                    # Check if we are reversing direction (large turn angle)
                    # If so, check if we are in a "Turn Bulb"
                    # For now, just allow if radius constraint met?
                    # Min radius check handles local curvature.

                    turn_angle = abs(normalize_angle(h - state.heading)) if i > 0 else 0
                    is_switchback = turn_angle > math.pi / 2

                    if is_switchback and ramp_params.mode != "switchback":
                        # Prevent sharp turns if not in switchback mode
                        continue

                    # Simple cost: accumulated distance
                    new_cost = state.cost + d

                    # Heuristic: curvature penalty
                    if i > 0:
                        new_cost += turn_angle * 10.0 # Penalty for turning

                    child = SearchState(i+1, p_next, h, new_cost, parent=state)
                    next_states.append(child)

        # Beam Pruning
        # Sort by cost and keep top N
        if not next_states:
            # Dead end at this level
            break

        next_states.sort(key=lambda x: x.cost)
        current_states = next_states[:BEAM_WIDTH]

    # Reconstruct Path
    if current_states:
        # Best state at deepest level reached
        # If we didn't reach target_slice_idx, we take what we have
        best_state = current_states[0]

        path = []
        curr = best_state
        while curr:
            path.append(curr.point)
            curr = curr.parent
        return path[::-1], diagnostics
    else:
        return [], diagnostics

def generate_ramp_corridor(
    centerline: List[Tuple[float, float, float]],
    ramp_width: float
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """
    Generates left and right edges of the ramp corridor from the centerline.
    """
    if not centerline or len(centerline) < 2:
        return [], []

    left_edge = []
    right_edge = []

    half_width = ramp_width / 2.0

    for i in range(len(centerline)):
        # Calculate tangent
        if i == 0:
            p1 = centerline[i]
            p2 = centerline[i+1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
        elif i == len(centerline) - 1:
            p1 = centerline[i-1]
            p2 = centerline[i]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
        else:
            # Average tangent
            p_prev = centerline[i-1]
            p_next = centerline[i+1]
            dx = p_next[0] - p_prev[0]
            dy = p_next[1] - p_prev[1]

        mag = math.hypot(dx, dy)
        if mag < 1e-6:
            # Degenerate segment, keep previous normal or zero
            nx, ny = 0, 0
        else:
            # Normal vector (-dy, dx)
            nx = -dy / mag
            ny = dx / mag

        # Left point: Center + Normal * HalfWidth
        p = centerline[i]
        lx = p[0] + nx * half_width
        ly = p[1] + ny * half_width
        lz = p[2]

        rx = p[0] - nx * half_width
        ry = p[1] - ny * half_width
        rz = p[2]

        left_edge.append((lx, ly, lz))
        right_edge.append((rx, ry, rz))

    return left_edge, right_edge
