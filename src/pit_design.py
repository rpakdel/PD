import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
import shapely.geometry as sg
import shapely.ops as so

@dataclass
class PitDesignParams:
    bench_height: float
    batter_angle_deg: float
    berm_width: float
    target_elevation: float

@dataclass
class Mesh3D:
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[Tuple[int, int, int]] = field(default_factory=list)

    def extend(self, other: 'Mesh3D'):
        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        for f in other.faces:
            self.faces.append((f[0] + offset, f[1] + offset, f[2] + offset))

@dataclass
class BenchGeometry:
    bench_id: int
    z_crest: float
    z_toe: float
    crest_polys: List[sg.Polygon] = field(default_factory=list)
    toe_polys: List[sg.Polygon] = field(default_factory=list)
    face_mesh: Mesh3D = field(default_factory=Mesh3D)
    berm_mesh: Mesh3D = field(default_factory=Mesh3D)
    diagnostics: List[str] = field(default_factory=list)

def inset_polygon(poly: sg.Polygon, distance: float) -> List[sg.Polygon]:
    """
    Insets a polygon by a given distance.
    Returns a list of Polygons (handles splits).
    """
    if not poly.is_valid:
        poly = poly.buffer(0)

    # Shapely buffer: positive expands, negative shrinks (insets)
    res = poly.buffer(-distance, join_style='mitre')

    if res.is_empty:
        return []

    if isinstance(res, sg.MultiPolygon):
        return list(res.geoms)
    elif isinstance(res, sg.Polygon):
        return [res]
    else:
        # GeometryCollection or other?
        return []

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
                 # No, if it's Face (Crest->Toe) and Toe is empty, it's a cone ending at a point?
                 # But inset_polygon returned empty, so we don't have a point.
                 # Actually if inner is empty, diff_poly is outer.
                 # If this is "Face", we probably shouldn't be here if offset failed?
                 # But let's assume flat for robustness if unsure.
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

    # Active polygons for current crest level
    current_crest_polys = [up_poly]
    current_crest_z = avg_z

    benches = []
    bench_id = 1

    diagnostics = {
        "start_z": current_crest_z,
        "bench_log": []
    }

    # Precompute offsets
    theta_rad = math.radians(params.batter_angle_deg)
    if theta_rad <= 0 or theta_rad >= math.pi/2:
        h_face = 0.0
    else:
        h_face = params.bench_height / math.tan(theta_rad)

    bw = params.berm_width

    while current_crest_z > params.target_elevation:
        if not current_crest_polys:
            diagnostics["bench_log"].append("Pit bottomed out (no polygons left).")
            break

        current_toe_z = current_crest_z - params.bench_height

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
                # Maybe mesh a cone to the center?
                # For now, just skip meshing face if toe is empty (vertical cliff to nothing?)
                # Actually if toe is empty, it means the face intersection eliminated the polygon.
                # Effectively it's a "conical pit bottom" somewhere between crest and toe Z.
                # We could try to estimate depth, but for now we leave it open or mesh "cap".
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
                # Note: if next_crests is empty, the pit bottom is this toe polygon.
                # We can mesh the berm as "Toe - Empty" = whole Toe polygon (flat floor).

                berm_mesh = mesh_from_polygon_difference(
                    toe_poly, next_crests, current_toe_z, current_toe_z
                )
                bench_geom.berm_mesh.extend(berm_mesh)

                next_level_crest_polys.extend(next_crests)

        benches.append(bench_geom)
        diagnostics["bench_log"].append(f"Bench {bench_id}: {len(bench_geom.crest_polys)} crests, {len(bench_geom.toe_polys)} toes.")

        current_crest_polys = next_level_crest_polys
        current_crest_z = current_toe_z
        bench_id += 1

        if bench_id > 1000:
            diagnostics["bench_log"].append("Max benches limit reached.")
            break

    return benches, diagnostics
