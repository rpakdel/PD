from typing import List, Tuple, Optional, Union
import math
import shapely.geometry as sg
import shapely.ops as so
import numpy as np
from design_params import RampParams, Mesh3D

class RampGenerator:
    def __init__(self, ramp_params: RampParams):
        self.params = ramp_params
        self.current_point: Optional[Tuple[float, float]] = None
        self.current_heading: float = 0.0 # Radians
        self.direction: int = -1 # -1 for CW (Standard Pit), 1 for CCW

    def set_start_point(self, point: Tuple[float, float]):
        self.current_point = point

    def trace_polygon(self, poly: sg.Polygon, start_pt: Tuple[float, float], dist: float) -> sg.LineString:
        """
        Traces the polygon boundary from start_pt for a given distance.
        Handles wrapping around the ring.
        """
        line = poly.exterior
        if not line.is_closed:
             # Should be closed for a polygon
             pass

        total_length = line.length
        if total_length == 0:
            return sg.LineString([start_pt, start_pt])

        start_dist = line.project(sg.Point(start_pt))

        points = []
        points.append(start_pt)

        remaining = dist
        cursor_dist = start_dist

        # We assume "Downward" pit generation usually implies CW spiral?
        # Or we just follow the line direction (CCW) and flip if needed.
        # Let's support moving along the defined order (CCW).
        # If user wants CW, we'd need to invert logic or geometry.
        # For now, let's just move in the direction of vertex order (CCW).

        while remaining > 0:
            # Distance to end of line
            dist_to_end = total_length - cursor_dist

            if remaining <= dist_to_end:
                target = cursor_dist + remaining
                p = line.interpolate(target)
                points.append((p.x, p.y))
                remaining = 0
            else:
                # Wrap around
                # Add point at end/start (they are same)
                p_end = line.interpolate(total_length)
                points.append((p_end.x, p_end.y))

                remaining -= dist_to_end
                cursor_dist = 0.0
                # Continue loop

        return sg.LineString(points)

    def generate_ramp_segment(
        self,
        reference_poly: sg.Polygon,
        z_top: float,
        z_bottom: float,
        grade: float,
        direction: str = "Downward"
    ) -> Tuple[sg.Polygon, Tuple[float, float], Mesh3D]:
        """
        Generates the ramp polygon and 3D mesh for the vertical interval [z_top, z_bottom].
        Returns the ramp polygon (to be unioned), the new end point, and the 3D mesh.
        """
        if self.current_point is None:
             self.current_point = reference_poly.exterior.coords[0]

        # Snap current point to poly
        start_pt = self.current_point
        p_geom = sg.Point(start_pt)
        dist_poll = reference_poly.exterior.project(p_geom)
        start_pt_snapped = reference_poly.exterior.interpolate(dist_poll)
        start_pt = (start_pt_snapped.x, start_pt_snapped.y)

        # Calculate Length required
        if grade < 0.001: grade = 0.10

        delta_z = abs(z_top - z_bottom)
        length = delta_z / grade

        # Trace path (Centerline)
        # Note: If we want the ramp to be fully "cut", we should offset the path inward.
        # But integrating offset path into loop is tricky.
        # For now, let's keep centerline on crest, but maybe widen inward more?
        # Current: buffer centered.
        centerline = self.trace_polygon(reference_poly, start_pt, length)

        # Create Ramp Polygon
        half_width = self.params.ramp_width / 2.0
        ramp_poly = centerline.buffer(half_width, cap_style=2, join_style=2)

        # Update state
        end_pt = centerline.coords[-1]
        self.current_point = end_pt

        # --- Mesh Generation ---
        mesh = Mesh3D()

        # Discretize centerline
        # centerline is a LineString. coords gives us points.
        # We need to assign Z to these points linearly from z_top to z_bottom.

        cl_coords = list(centerline.coords)
        if len(cl_coords) < 2:
            return ramp_poly, end_pt, mesh

        # Calculate cumulative distances
        dists = [0.0]
        for i in range(1, len(cl_coords)):
            p1 = cl_coords[i-1]
            p2 = cl_coords[i]
            d = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            dists.append(dists[-1] + d)

        total_len = dists[-1]
        if total_len == 0:
             return ramp_poly, end_pt, mesh

        # Generate left/right edges
        # Ideally we use offsets. But simple normal extrusion is easier for mesh construction.

        left_points = []
        right_points = []

        for i, (x, y) in enumerate(cl_coords):
            # Determine Z
            # Fraction of distance traveled
            frac = dists[i] / total_len

            # If Downward: z goes from z_top to z_bottom
            # z_top > z_bottom
            z = z_top - (z_top - z_bottom) * frac

            # Determine Normal
            if i < len(cl_coords) - 1:
                dx = cl_coords[i+1][0] - x
                dy = cl_coords[i+1][1] - y
            elif i > 0:
                dx = x - cl_coords[i-1][0]
                dy = y - cl_coords[i-1][1]
            else:
                dx, dy = 1, 0 # Default

            mag = math.hypot(dx, dy)
            if mag == 0: nx, ny = 0, 0
            else: nx, ny = -dy/mag, dx/mag

            # Left/Right points
            # Left is +Normal, Right is -Normal
            lx = x + nx * half_width
            ly = y + ny * half_width
            rx = x - nx * half_width
            ry = y - ny * half_width

            left_points.append((lx, ly, z))
            right_points.append((rx, ry, z))

        # Triangulate
        # Strip of quads (2 triangles each)

        vertices = []
        faces = []

        # We can just dump all points into vertices list
        # Order: L0, R0, L1, R1, ...
        # i goes from 0 to N-1
        # Vertices: 2*i (Left), 2*i+1 (Right)

        for i in range(len(cl_coords)):
            vertices.append(left_points[i])
            vertices.append(right_points[i])

            if i > 0:
                # Quad between i-1 and i
                # Indices:
                l_prev = 2*(i-1)
                r_prev = 2*(i-1) + 1
                l_curr = 2*i
                r_curr = 2*i + 1

                # Tri 1: L_prev, R_prev, L_curr
                faces.append((l_prev, r_prev, l_curr))
                # Tri 2: R_prev, R_curr, L_curr
                faces.append((r_prev, r_curr, l_curr))

        mesh = Mesh3D(vertices=vertices, faces=faces)

        return ramp_poly, end_pt, mesh
