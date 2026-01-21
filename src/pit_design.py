import math
from typing import List, Tuple, Dict, Any, Union, Optional
import shapely.geometry as sg
import shapely.ops as so
import pyclipper
from design_params import DesignBlock, PitDesignParams, Mesh3D, BenchGeometry, RampParams
from ramp_geometry import RampGenerator

# Integer scaling for Clipper robustness
SCALE = 1000

def get_design_params_at_elevation(z: float, params: PitDesignParams) -> Tuple[float, float]:
    """
    Returns (batter_angle_deg, berm_width) for a given elevation z.
    Checks variable_params first. If z is within a block, returns that block's params.
    Otherwise returns default params.

    If variable_params are provided, STRICTLY enforces that z must be within a block.
    """
    if params.variable_params:
        for block in params.variable_params:
            if block.z_start <= z <= block.z_end:
                return block.batter_angle_deg, block.berm_width

        # If we have variable params but no match, raise error (Mutual Exclusivity)
        raise ValueError(f"No design parameters defined for elevation {z:.1f}")

    return params.batter_angle_deg, params.berm_width

def to_clipper(poly: sg.Polygon) -> List[List[Tuple[int, int]]]:
    """
    Converts a Shapely Polygon to a list of Clipper paths (scaled integers).
    Ensures Outer is CCW and Holes are CW (standard for pyclipper Offset with Holes?).
    Wait, pyclipper docs say: "Outer polygons should be defined by a clockwise vertex order".
    BUT PyclipperOffset docs say: "Positive offsets expand outer polygons and contract inner polygons."
    My experiments showed:
    - Input CCW Outer -> Positive Offset Expands.
    - Input CW Hole -> Positive Offset Contracts (Shrinks hole).

    So we want:
    - Outer: CCW (Orientation True)
    - Hole: CW (Orientation False)

    Shapely 2.0:
    - exterior is CCW.
    - interiors are CCW.

    So we keep Exterior as is (CCW).
    We REVERSE Interiors to be CW.
    """
    paths = []

    # Exterior
    ext_coords = list(poly.exterior.coords)
    # Remove last point if duplicate (Shapely is closed, Clipper takes open list of points implies closed)
    if ext_coords[0] == ext_coords[-1]:
        ext_coords.pop()

    ext_scaled = [(int(round(x * SCALE)), int(round(y * SCALE))) for x, y in ext_coords]

    # Ensure CCW for Exterior (Area > 0)
    # pyclipper.Orientation(True) = CCW
    if not pyclipper.Orientation(ext_scaled):
        ext_scaled.reverse()

    paths.append(ext_scaled)

    # Interiors
    for interior in poly.interiors:
        int_coords = list(interior.coords)
        if int_coords[0] == int_coords[-1]:
            int_coords.pop()

        int_scaled = [(int(round(x * SCALE)), int(round(y * SCALE))) for x, y in int_coords]

        # Ensure CW for Hole (Area < 0 implies Orientation False)
        if pyclipper.Orientation(int_scaled):
            int_scaled.reverse()

        paths.append(int_scaled)

    return paths

def from_clipper(paths: List[List[Tuple[int, int]]]) -> List[sg.Polygon]:
    """
    Converts a list of Clipper paths back to Shapely Polygons.
    Uses Orientation to distinguish Outer (CCW) from Holes (CW).
    Reconstructs parent-child relationships for holes.
    """
    if not paths:
        return []

    # Separate Shells (CCW) and Holes (CW)
    shells = []
    holes = []

    for path in paths:
        # Convert back to float
        coords = [(x / SCALE, y / SCALE) for x, y in path]
        if len(coords) < 3:
            continue

        # Orientation: True=CCW (Shell), False=CW (Hole)
        if pyclipper.Orientation(path):
            shells.append(sg.Polygon(coords))
        else:
            holes.append(sg.Polygon(coords))

    if not shells:
        return []

    # We use Shapely to subtract holes from shells
    # Union all shells
    shell_union = so.unary_union(shells)

    if holes:
        hole_union = so.unary_union(holes)
        # Difference
        res = shell_union.difference(hole_union)
    else:
        res = shell_union

    if res.is_empty:
        return []

    if isinstance(res, sg.MultiPolygon):
        return list(res.geoms)
    elif isinstance(res, sg.Polygon):
        return [res]
    else:
        return []

def remove_spikes(poly: sg.Polygon, eps: float = 0.01) -> sg.Polygon:
    """
    Removes vertices that are closer than eps to the previous vertex.
    """
    if poly.is_empty:
        return poly

    def filter_coords(coords):
        if not coords:
            return []
        new_coords = [coords[0]]
        for i in range(1, len(coords)):
            p = coords[i]
            prev = new_coords[-1]
            dist = math.sqrt((p[0]-prev[0])**2 + (p[1]-prev[1])**2)
            if dist > eps:
                new_coords.append(p)
        # Check closure
        if len(new_coords) > 2:
            # Check last against first
             p = new_coords[-1]
             first = new_coords[0]
             dist = math.sqrt((p[0]-first[0])**2 + (p[1]-first[1])**2)
             if dist <= eps: # Close enough to close
                 new_coords.pop()
                 new_coords.append(first)
             elif new_coords[0] != new_coords[-1]:
                 new_coords.append(first)
        return new_coords

    ext = filter_coords(list(poly.exterior.coords))
    if len(ext) < 4: # Triangle is 4 points (closed)
        return sg.Polygon() # Collapsed

    interiors = []
    for inner in poly.interiors:
        inner_coords = filter_coords(list(inner.coords))
        if len(inner_coords) >= 4:
            interiors.append(inner_coords)

    return sg.Polygon(ext, interiors).buffer(0)

def clean_polygons(polys: List[sg.Polygon]) -> List[sg.Polygon]:
    """
    Cleans a list of polygons by performing a Union using Clipper.
    """
    if not polys:
        return []

    valid_polys = [p for p in polys if not p.is_empty and p.is_valid]
    if not valid_polys:
        return []

    pc = pyclipper.Pyclipper()

    for p in valid_polys:
        paths = to_clipper(p)
        pc.AddPaths(paths, pyclipper.PT_SUBJECT, True)

    # Execute Union
    # PFT_NONZERO is standard
    try:
        solution = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    except Exception:
        # Fallback to shapely if clipper fails
        merged = so.unary_union(valid_polys)
        if isinstance(merged, sg.MultiPolygon):
            return list(merged.geoms)
        elif isinstance(merged, sg.Polygon):
            return [merged]
        return []

    return from_clipper(solution)


def offset_polygon(poly: sg.Polygon, distance: float) -> List[sg.Polygon]:
    """
    Offsets a polygon by a given distance using Clipper.
    Positive distance expands, negative distance shrinks (insets).
    """
    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty:
        return []

    pco = pyclipper.PyclipperOffset()

    paths = to_clipper(poly)
    pco.AddPaths(paths, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)

    dist_scaled = distance * SCALE

    try:
        solution = pco.Execute(dist_scaled)
    except Exception:
        return []

    return from_clipper(solution)

def inset_polygon(poly: sg.Polygon, distance: float) -> List[sg.Polygon]:
    """
    Insets a polygon by a given distance (shrinks).
    Returns a list of Polygons (handles splits).
    """
    return offset_polygon(poly, -distance)

def mesh_from_polygon_difference(
    outer_poly: sg.Polygon,
    inner_polys: List[sg.Polygon],
    z_outer: float,
    z_inner: float
) -> Mesh3D:
    """
    Generates a 3D mesh for the area between outer_poly and inner_polys.
    """
    if not inner_polys:
        diff_poly = outer_poly
    else:
        # Use Clipper for difference to be consistent
        pc = pyclipper.Pyclipper()
        outer_paths = to_clipper(outer_poly)
        pc.AddPaths(outer_paths, pyclipper.PT_SUBJECT, True)

        for p in inner_polys:
            inner_paths = to_clipper(p)
            pc.AddPaths(inner_paths, pyclipper.PT_CLIP, True)

        try:
            solution = pc.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
            diff_polys = from_clipper(solution)
        except Exception:
            diff_polys = []

        if not diff_polys:
            return Mesh3D()

        diff_poly = sg.MultiPolygon(diff_polys)

    if diff_poly.is_empty:
        return Mesh3D()

    triangles = so.triangulate(diff_poly)
    valid_triangles = [t for t in triangles if diff_poly.contains(t.centroid)]

    if not valid_triangles:
        return Mesh3D()

    vertices_map = {}
    vertices_list = []
    faces_list = []

    def get_vertex_index(x, y):
        key = (round(x, 6), round(y, 6))
        if key not in vertices_map:
            pt = sg.Point(x, y)
            d_outer = outer_poly.boundary.distance(pt)
            d_inner = float('inf')
            if inner_polys:
                 d_inner = min(p.boundary.distance(pt) for p in inner_polys)

            if d_inner == float('inf'):
                 z = z_outer
            else:
                total_d = d_outer + d_inner
                if total_d < 1e-6:
                     z = z_outer if d_outer < d_inner else z_inner
                else:
                     z = z_outer * (d_inner / total_d) + z_inner * (d_outer / total_d)

            vertices_map[key] = len(vertices_list)
            vertices_list.append((x, y, z))

        return vertices_map[key]

    for tri in valid_triangles:
        coords = list(tri.exterior.coords)
        idx0 = get_vertex_index(coords[0][0], coords[0][1])
        idx1 = get_vertex_index(coords[1][0], coords[1][1])
        idx2 = get_vertex_index(coords[2][0], coords[2][1])
        faces_list.append((idx0, idx1, idx2))

    return Mesh3D(vertices=vertices_list, faces=faces_list)


def generate_pit_benches(
    up_points: List[Tuple[float, float, float]],
    params: PitDesignParams,
    ramp_params: Optional[RampParams] = None,
    ramp_start_index: int = 0
) -> Tuple[List[BenchGeometry], Dict[str, Any]]:
    """
    Generates pit benches and surfaces.
    """
    if not up_points:
        return [], {"error": "No UP points provided"}

    # Pre-process UP ring
    avg_z = sum(p[2] for p in up_points) / len(up_points)
    poly_coords = [(p[0], p[1]) for p in up_points]

    # 1. Close ring
    if poly_coords[0] != poly_coords[-1]:
        poly_coords.append(poly_coords[0])

    up_poly = sg.Polygon(poly_coords)

    # 2. Remove spikes (eps=0.01)
    up_poly = remove_spikes(up_poly, eps=0.01)

    if not up_poly.is_valid:
        up_poly = up_poly.buffer(0)

    # Initialize Ramp Generator if params provided
    ramp_gen = None
    if ramp_params:
        ramp_gen = RampGenerator(ramp_params)
        # Find start point on UP ring
        # Use index if valid, else 0
        idx = ramp_start_index % len(poly_coords)
        ramp_gen.set_start_point(poly_coords[idx])

    # 3. Ensure consistent orientation is handled by to_clipper automatically

    benches = []
    bench_id = 1

    if params.design_direction == "Upward":
        current_toe_polys = [up_poly]
        current_toe_z = avg_z

        diagnostics = {
            "start_z": current_toe_z,
            "direction": "Upward",
            "bench_log": []
        }

        if current_toe_z >= params.target_elevation:
            msg = f"Start elevation ({current_toe_z:.1f}) is already >= Target elevation ({params.target_elevation:.1f}). No benches generated."
            diagnostics["error"] = msg
            diagnostics["bench_log"].append(msg)
            return [], diagnostics

        while current_toe_z < params.target_elevation:
            current_crest_z = current_toe_z + params.bench_height
            z_mid = (current_toe_z + current_crest_z) / 2.0
            try:
                angle_deg, bw = get_design_params_at_elevation(z_mid, params)
            except ValueError as e:
                diagnostics["error"] = str(e)
                diagnostics["bench_log"].append(f"Error: {str(e)}")
                break

            theta_rad = math.radians(angle_deg)
            if theta_rad <= 0 or theta_rad >= math.pi/2:
                h_face = 0.0
            else:
                h_face = params.bench_height / math.tan(theta_rad)

            bench_geom = BenchGeometry(
                bench_id=bench_id,
                z_crest=current_crest_z,
                z_toe=current_toe_z
            )

            next_level_toe_polys = []

            for poly in current_toe_polys:
                bench_geom.toe_polys.append(poly)
                crest_polys_list = offset_polygon(poly, h_face)
                if not crest_polys_list:
                    continue

                # Apply Ramp Cut (Upward)
                # For Upward, we are filling. Ramp adds to the fill? Or cuts into the fill?
                # Assuming Ramp is a road on the side of the fill (Dump).
                # It adds volume (expands the crest).
                if ramp_gen:
                    # Select largest crest if multiple (heuristic)
                    # For Upward, we usually have one main dump body.
                    ramp_target_poly = max(crest_polys_list, key=lambda p: p.area)
                    ramp_poly, _, ramp_mesh = ramp_gen.generate_ramp_segment(
                        ramp_target_poly, current_toe_z, current_crest_z, ramp_params.grade_max
                    )
                    # Union ramp to crests
                    crest_polys_list = [p.union(ramp_poly) for p in crest_polys_list]
                    bench_geom.ramp_mesh.extend(ramp_mesh)


                bench_geom.crest_polys.extend(crest_polys_list)

                # Mesh Face
                # Use clean union for crests derived from this toe
                crest_union = clean_polygons(crest_polys_list)

                for crest_poly in crest_union:
                    face_mesh = mesh_from_polygon_difference(
                        crest_poly, [poly], current_crest_z, current_toe_z
                    )
                    bench_geom.face_mesh.extend(face_mesh)

                for crest_poly in crest_polys_list:
                    next_toes = offset_polygon(crest_poly, bw)
                    for next_toe in next_toes:
                        berm_mesh = mesh_from_polygon_difference(
                            next_toe, [crest_poly], current_crest_z, current_crest_z
                        )
                        bench_geom.berm_mesh.extend(berm_mesh)
                    next_level_toe_polys.extend(next_toes)

            benches.append(bench_geom)

            # PoC rule: keep largest?
            current_toe_polys = clean_polygons(next_level_toe_polys)

            # Stats
            total_crest_area = sum(p.area for p in bench_geom.crest_polys)
            diagnostics["bench_log"].append(
                f"Bench {bench_id}: {len(bench_geom.crest_polys)} crests ({total_crest_area:.0f} m2)"
            )

            current_toe_z = current_crest_z
            bench_id += 1

            if bench_id > 1000:
                diagnostics["bench_log"].append("Max benches limit reached.")
                break

    else:
        # Downward
        current_crest_polys = [up_poly]
        current_crest_z = avg_z

        diagnostics = {
            "start_z": current_crest_z,
            "direction": "Downward",
            "bench_log": []
        }

        if current_crest_z <= params.target_elevation:
            msg = f"Start elevation ({current_crest_z:.1f}) is already <= Target elevation ({params.target_elevation:.1f}). No benches generated."
            diagnostics["error"] = msg
            diagnostics["bench_log"].append(msg)
            return [], diagnostics

        while current_crest_z > params.target_elevation:
            if not current_crest_polys:
                diagnostics["bench_log"].append("Pit bottomed out (no polygons left).")
                break

            current_toe_z = current_crest_z - params.bench_height
            z_mid = (current_crest_z + current_toe_z) / 2.0
            try:
                angle_deg, bw = get_design_params_at_elevation(z_mid, params)
            except ValueError as e:
                diagnostics["error"] = str(e)
                diagnostics["bench_log"].append(f"Error: {str(e)}")
                break

            theta_rad = math.radians(angle_deg)
            if theta_rad <= 0 or theta_rad >= math.pi/2:
                h_face = 0.0
            else:
                h_face = params.bench_height / math.tan(theta_rad)

            bench_geom = BenchGeometry(
                bench_id=bench_id,
                z_crest=current_crest_z,
                z_toe=current_toe_z
            )

            next_level_crest_polys = []

            for poly in current_crest_polys:
                bench_geom.crest_polys.append(poly)
                toe_polys_list = inset_polygon(poly, h_face)

                # RAMP INTEGRATION
                if ramp_gen:
                    # We are going Downward.
                    # Ramp travels from Crest (current_crest_z) to Toe (current_toe_z).
                    # We need to cut the ramp into the "Toe Polygon".
                    # Normal Toe Polygon is 'poly' inset by 'h_face'.
                    # Ramp Polygon expands this hole.

                    # We trace along 'poly' (the Crest).
                    # Actually, the ramp runs along the wall. The wall starts at Crest.
                    ramp_poly, _, ramp_mesh = ramp_gen.generate_ramp_segment(
                        poly, current_crest_z, current_toe_z, ramp_params.grade_max
                    )

                    # Merge ramp_poly into toe_polys
                    # Since toe_polys are Holes (Void), and Ramp is Void, we Union.
                    if toe_polys_list:
                         toe_polys_list = [p.union(ramp_poly) for p in toe_polys_list]
                         bench_geom.ramp_mesh.extend(ramp_mesh)
                    else:
                         # If no toe (e.g. pinch out), the ramp might still exist?
                         # Usually if pinch out, pit ends. But ramp might extend slightly.
                         # For now, if no toe, no ramp integration.
                         pass

                if not toe_polys_list:
                    continue

                bench_geom.toe_polys.extend(toe_polys_list)

                # Mesh Face
                # Face is Difference(Crest, Toe).
                # Since Toe is now larger (includes Ramp), the Face will have a slot cut out.
                face_mesh = mesh_from_polygon_difference(
                    poly, toe_polys_list, current_crest_z, current_toe_z
                )
                bench_geom.face_mesh.extend(face_mesh)

                for toe_poly in toe_polys_list:
                    next_crests = inset_polygon(toe_poly, bw)

                    # Note: We do NOT add ramp to next_crests (Berm level).
                    # The ramp usually traverses the Face.
                    # On the Berm, the ramp continues?
                    # "Project to Next Bench... Repeat".
                    # The `ramp_gen` maintains state (end point).
                    # The next iteration (next bench) will pick up from the end point.
                    # But between Toe and Next Crest is a Berm.
                    # Does the ramp travel across the Berm?
                    # Usually Ramp is continuous.
                    # If Berm Width > 0, the ramp must cross it.
                    # BUT `ramp_gen` generated a segment for `h_bench` drop.
                    # The distance travelled covers the Face drop.
                    # If we have a flat Berm, does the ramp drop? No, it's flat?
                    # "Ensure the switchback has a 'flat' section".
                    # Usually the ramp continues descending.
                    # If the ramp continues descending across the berm width, we need to handle that.
                    # However, in this simplified model:
                    # We treat (Face + Berm) as one step?
                    # Or we just assume the Ramp logic handles the vertical drop across the face.
                    # Let's assume the Ramp is "Face-only" for now, or that the Berm is cut by the next iteration's start?

                    # Issue: The ramp end point is at the Toe.
                    # The next ramp segment starts at the Toe?
                    # If there is a Berm, the next Crest is "inward".
                    # The Ramp End Point (at Toe) is "outward" (on the ramp cut).
                    # So the next iteration will start from that outward point.
                    # And trace along the Toe (which includes the ramp cut).
                    # So it should naturally work!

                    berm_mesh = mesh_from_polygon_difference(
                        toe_poly, next_crests, current_toe_z, current_toe_z
                    )
                    bench_geom.berm_mesh.extend(berm_mesh)
                    next_level_crest_polys.extend(next_crests)

            benches.append(bench_geom)

            # PoC Rule: Keep Largest (Downward split handling)
            candidates = clean_polygons(next_level_crest_polys)
            if candidates:
                # Select largest
                largest = max(candidates, key=lambda p: p.area)
                if len(candidates) > 1:
                    discarded = len(candidates) - 1
                    diagnostics["bench_log"].append(f"Bench {bench_id}: kept largest polygon, discarded {discarded} smaller components.")
                current_crest_polys = [largest]
            else:
                current_crest_polys = []

            # Stats
            total_crest_area = sum(p.area for p in bench_geom.crest_polys)
            diagnostics["bench_log"].append(
                f"Bench {bench_id}: {len(bench_geom.crest_polys)} crests ({total_crest_area:.0f} m2)"
            )

            current_crest_z = current_toe_z
            bench_id += 1

            if bench_id > 1000:
                diagnostics["bench_log"].append("Max benches limit reached.")
                break

    return benches, diagnostics
