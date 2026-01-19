import math
from typing import List, Tuple, Dict, Any, Union, Optional
import shapely.geometry as sg
import shapely.ops as so
from design_params import DesignBlock, PitDesignParams, Mesh3D, BenchGeometry

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

def clean_polygons(polys: List[sg.Polygon]) -> List[sg.Polygon]:
    """
    Cleans a list of polygons by performing a unary_union.
    This handles merging overlapping polygons and simplifying topology.
    Returns a list of disjoint Polygons.
    """
    if not polys:
        return []

    # Filter out invalid or empty
    valid_polys = [p for p in polys if not p.is_empty and p.is_valid]
    if not valid_polys:
        return []

    merged = so.unary_union(valid_polys)

    if merged.is_empty:
        return []

    if isinstance(merged, sg.MultiPolygon):
        return list(merged.geoms)
    elif isinstance(merged, sg.Polygon):
        return [merged]
    else:
        # GeometryCollection or other (shouldn't happen with polygons input)
        return []

def offset_polygon(poly: sg.Polygon, distance: float) -> List[sg.Polygon]:
    """
    Offsets a polygon by a given distance.
    Positive distance expands, negative distance shrinks (insets).
    Returns a list of Polygons (handles splits/merges).
    """
    if not poly.is_valid:
        poly = poly.buffer(0)

    res = poly.buffer(distance, join_style='mitre')

    if res.is_empty:
        return []

    if isinstance(res, sg.MultiPolygon):
        return list(res.geoms)
    elif isinstance(res, sg.Polygon):
        return [res]
    else:
        return []

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
    # 1. Create the difference polygon (surface with holes)
    # Union inner polygons first
    if not inner_polys:
        # If no inner polygons (e.g. bottom of pit), maybe we just mesh the outer?
        # But usually this function is called for Face (Crest-Toe) or Berm (Toe-Crest).
        # If inner is empty, it means the offset collapsed.
        # We can mesh the whole outer polygon (cap).
        diff_poly = outer_poly
    else:
        inner_union = sg.MultiPolygon(inner_polys).buffer(0) # clean union
        diff_poly = outer_poly.difference(inner_union)

    if diff_poly.is_empty:
        return Mesh3D()

    # 2. Triangulate
    # shapely.ops.triangulate triangulates the vertices of the geometry
    triangles = so.triangulate(diff_poly)

    # 3. Filter triangles that are strictly inside the difference polygon
    # Use representative_point() or centroid
    valid_triangles = [t for t in triangles if diff_poly.contains(t.centroid)]

    if not valid_triangles:
        return Mesh3D()

    # 4. Build Mesh3D
    # We need to map (x,y) to indices and determine Z
    vertices_map = {} # (x,y) -> index
    vertices_list = []
    faces_list = []

    def get_vertex_index(x, y):
        # Rounding to handle float precision issues
        key = (round(x, 6), round(y, 6))
        if key not in vertices_map:
            # Determine Z
            pt = sg.Point(x, y)

            # Simple distance check
            # Note: Boundary of MultiPolygon might be complex
            d_outer = outer_poly.boundary.distance(pt)

            # Distance to inner boundaries
            d_inner = float('inf')
            if inner_polys:
                 # Check against original inner list boundaries
                 d_inner = min(p.boundary.distance(pt) for p in inner_polys)

            if d_inner == float('inf'):
                 # No inner, must be on outer or inside (cap)
                 # If it's a cap, maybe we want it flat at z_outer?
                 z = z_outer # Or z_inner if we are closing bottom?
                 # If this is "Face" and inner collapsed, it's a pinch out.
                 # Usually we pinch out to z_toe.
                 # Let's assume cap at z_outer?
                 z = z_outer
            else:
                # Interpolate Z
                total_d = d_outer + d_inner
                if total_d < 1e-6:
                     # On boundary or very close
                     z = z_outer if d_outer < d_inner else z_inner
                else:
                     # Linear interpolation
                     ratio = d_inner / total_d # Closer to outer (d_outer small) -> ratio close to 0 -> ???
                     # Wait. If close to outer, d_outer is 0. We want Z_outer.
                     # Formula: z = z_outer * (d_inner/total) + z_inner * (d_outer/total)
                     # Check: if d_outer=0 -> z = z_outer * 1 + z_inner * 0 = z_outer. Correct.
                     z = z_outer * (d_inner / total_d) + z_inner * (d_outer / total_d)

            vertices_map[key] = len(vertices_list)
            vertices_list.append((x, y, z))

        return vertices_map[key]

    for tri in valid_triangles:
        coords = list(tri.exterior.coords)
        # coords has 4 points (closed loop), take first 3
        # Check winding? Shapely triangles are usually CCW?
        # We'll take first 3.
        idx0 = get_vertex_index(coords[0][0], coords[0][1])
        idx1 = get_vertex_index(coords[1][0], coords[1][1])
        idx2 = get_vertex_index(coords[2][0], coords[2][1])
        faces_list.append((idx0, idx1, idx2))

    return Mesh3D(vertices=vertices_list, faces=faces_list)


def generate_pit_benches(
    up_points: List[Tuple[float, float, float]],
    params: PitDesignParams
) -> Tuple[List[BenchGeometry], Dict[str, Any]]:
    """
    Generates pit benches and surfaces.
    """
    if not up_points:
        return [], {"error": "No UP points provided"}

    # Initial Crest (UP)
    avg_z = sum(p[2] for p in up_points) / len(up_points)
    poly_coords = [(p[0], p[1]) for p in up_points]
    up_poly = sg.Polygon(poly_coords)

    if not up_poly.is_valid:
        up_poly = up_poly.buffer(0)

    benches = []
    bench_id = 1

    if params.design_direction == "Upward":
        # Upward generation: UP string is Toe of bottom bench
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

        # We loop while we are below the target elevation (assuming target is Top)
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

            # We need to collect next level toes (for next iteration)
            next_level_toe_polys = []

            # 1. Process Faces (Toe -> Crest) - Expansion
            for poly in current_toe_polys:
                bench_geom.toe_polys.append(poly)

                # Expand Toe to Crest
                # Use offset_polygon with positive distance
                crest_polys_list = offset_polygon(poly, h_face)

                if not crest_polys_list:
                    # Should not happen with positive buffer unless geometry invalid
                    continue

                bench_geom.crest_polys.extend(crest_polys_list)

                # Mesh Face (Crest - Toe)
                # Outer = Crest (High Z), Inner = Toe (Low Z)
                # Note: mesh_from_polygon_difference(outer, inner, z_outer, z_inner)
                # For each crest polygon resulting from this toe, we mesh the difference.
                # However, offset_polygon might merge or split.
                # Ideally we mesh the area between 'poly' (Toe) and 'crest_polys_list'.
                # But 'crest_polys_list' is the outer boundary.
                # We need to map which crest corresponds to which toe?
                # Actually, simpliest is: Mesh (Union(Crests) - Toe).
                # But we handle one 'poly' (Toe) at a time.
                # The 'crest_polys_list' is the expansion of *this* toe poly.
                # So the difference between Union(crest_polys_list) and poly is the face.

                # Union of crests for this toe (usually just one, but buffer handles topology)
                crest_union = so.unary_union(crest_polys_list)
                if isinstance(crest_union, sg.Polygon):
                    crests_for_mesh = [crest_union]
                elif isinstance(crest_union, sg.MultiPolygon):
                    crests_for_mesh = list(crest_union.geoms)
                else:
                    crests_for_mesh = [] # Empty?

                # We want to mesh the area: Crests - Toe
                # So outer = Crests, inner = Toe.
                # Since mesh_from_polygon_difference takes one outer and list of inners,
                # we iterate over crests.

                for crest_poly in crests_for_mesh:
                    face_mesh = mesh_from_polygon_difference(
                        crest_poly, [poly], current_crest_z, current_toe_z
                    )
                    bench_geom.face_mesh.extend(face_mesh)

                # 2. Process Berms (Crest -> Next Toe)
                for crest_poly in crest_polys_list:
                    next_toes = offset_polygon(crest_poly, bw)

                    # Mesh Berm (Next Toe - Crest)
                    # Berm is flat at current_crest_z?
                    # Wait, in downward: Berm is between Toe and Next Crest at z_toe.
                    # In Upward: Berm is between Crest and Next Toe at z_crest.

                    # Next Toes (Outer) - Crest (Inner)
                    for next_toe in next_toes:
                        berm_mesh = mesh_from_polygon_difference(
                            next_toe, [crest_poly], current_crest_z, current_crest_z
                        )
                        bench_geom.berm_mesh.extend(berm_mesh)

                    next_level_toe_polys.extend(next_toes)

            benches.append(bench_geom)

            # Diagnostic stats
            total_crest_area = sum(p.area for p in bench_geom.crest_polys)
            total_toe_area = sum(p.area for p in bench_geom.toe_polys)
            diagnostics["bench_log"].append(
                f"Bench {bench_id}: {len(bench_geom.crest_polys)} crests ({total_crest_area:.0f} m2), "
                f"{len(bench_geom.toe_polys)} toes ({total_toe_area:.0f} m2)."
            )

            # CLEAN UP and MERGE for next iteration
            # This is crucial for Upward to merge expanding bubbles
            current_toe_polys = clean_polygons(next_level_toe_polys)
            current_toe_z = current_crest_z
            bench_id += 1

            if bench_id > 1000:
                diagnostics["bench_log"].append("Max benches limit reached.")
                break

    else:
        # Downward generation (Default)
        # Active polygons for current crest level
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

            # We need to collect next level crests
            next_level_crest_polys = []

            # 1. Process Faces (Crest -> Toe)
            for poly in current_crest_polys:
                bench_geom.crest_polys.append(poly)

                # Generate Toes
                toe_polys_list = inset_polygon(poly, h_face)

                if not toe_polys_list:
                    # Pit pinched out at this face
                    continue

                bench_geom.toe_polys.extend(toe_polys_list)

                # Mesh Face
                face_mesh = mesh_from_polygon_difference(
                    poly, toe_polys_list, current_crest_z, current_toe_z
                )
                bench_geom.face_mesh.extend(face_mesh)

                # 2. Process Berms (Toe -> Next Crest)
                for toe_poly in toe_polys_list:
                    next_crests = inset_polygon(toe_poly, bw)

                    # Mesh Berm (flat)
                    berm_mesh = mesh_from_polygon_difference(
                        toe_poly, next_crests, current_toe_z, current_toe_z
                    )
                    bench_geom.berm_mesh.extend(berm_mesh)

                    next_level_crest_polys.extend(next_crests)

            benches.append(bench_geom)

            # Diagnostic stats
            total_crest_area = sum(p.area for p in bench_geom.crest_polys)
            total_toe_area = sum(p.area for p in bench_geom.toe_polys)
            diagnostics["bench_log"].append(
                f"Bench {bench_id}: {len(bench_geom.crest_polys)} crests ({total_crest_area:.0f} m2), "
                f"{len(bench_geom.toe_polys)} toes ({total_toe_area:.0f} m2)."
            )

            # CLEAN UP and MERGE/SPLIT cleanly for next iteration
            # Even for downward, this simplifies topology (e.g. if islands merge - though unlikely in shrinkage)
            # But it ensures valid geometry.
            current_crest_polys = clean_polygons(next_level_crest_polys)
            current_crest_z = current_toe_z
            bench_id += 1

            if bench_id > 1000:
                diagnostics["bench_log"].append("Max benches limit reached.")
                break

    return benches, diagnostics
