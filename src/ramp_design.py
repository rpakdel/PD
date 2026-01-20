from typing import List, Tuple, Optional
import math
import shapely.geometry as sg
import shapely.ops as so
import numpy as np

from design_params import BenchGeometry, RampParams, Slice
from pit_design import offset_polygon, clean_polygons

def get_pit_polygon_at_z(z: float, benches: List[BenchGeometry]) -> sg.Polygon:
    """
    Interpolates the pit polygon (hole boundary) at elevation z.
    Assumes 'benches' are sorted by elevation (descending for Downward).

    For Downward pit:
    - Bench i has z_crest (top) and z_toe (bottom).
    - Between z_crest and z_toe is the Face (sloping).
    - Between z_toe (of i) and z_crest (of i+1) is the Berm (flat).

    We need to handle the case where z is exactly on a berm or inside a face.
    """
    if not benches:
        return sg.Polygon()

    # Find which bench or berm we are on
    # Sort benches by z_crest descending
    # Assuming benches are generated in order and consistent

    # We scan to find the interval
    # 1. Above top crest? -> Top Crest (or UP string)
    # 2. Below bottom toe? -> Bottom Toe

    # Let's find the bench where z falls into [z_toe, z_crest] OR [z_crest_next, z_toe_prev]

    # Note: BenchGeometry store z_crest and z_toe.
    # For Downward: z_crest > z_toe.

    # Search
    for i, bench in enumerate(benches):
        if z > bench.z_crest:
            if i == 0:
                # Above the first bench crest -> Return first bench crest
                # (Ideally we should have the UP string as the absolute boundary above first crest)
                return clean_polygons(bench.crest_polys)[0] if bench.crest_polys else sg.Polygon()
            else:
                # Between benches? (Should be covered by prev bench toe logic, but let's see)
                continue

        if bench.z_toe <= z <= bench.z_crest:
            # On the face of bench i
            # To properly interpolate, we'd need the params.
            # For now, if no params available, we fallback to returning the TOE polygon (conservative).
            # This assumes the hole is at least as big as the toe.
            return so.unary_union(bench.toe_polys)

        if z < bench.z_toe:
            # Below this bench toe.
            # Check if we are above the next bench crest (Berm).
            if i + 1 < len(benches):
                next_bench = benches[i+1]
                if z > next_bench.z_crest:
                    # On the berm between bench i and i+1.
                    # The hole is defined by the Toe of bench i.
                    # (Which should be wider than Crest of bench i+1).
                    # Wait. The "hole" is the void.
                    # The material is outside.
                    # The berm is "material".
                    # So the hole boundary at the berm level is the Toe of bench i.
                    # (The wall stands at the toe).
                    # So for z in (next_bench.z_crest, bench.z_toe), the polygon is bench.toe_polys.

                    # Note: We return a SINGLE polygon (or Multi).
                    # We should union them.
                    return so.unary_union(bench.toe_polys)
            else:
                # Below the last bench toe.
                # Just return the last toe.
                return so.unary_union(bench.toe_polys)

    # If z is below all benches (should be caught by loop), return last toe
    return so.unary_union(benches[-1].toe_polys) if benches else sg.Polygon()


def create_slices(
    benches: List[BenchGeometry],
    ramp_params: RampParams,
    pit_design_params: 'PitDesignParams' = None # We might need this for accurate slope interpolation
) -> List[Slice]:
    """
    Generates slices for the ramp solver.
    """
    if not benches:
        return []

    # Determine Z range
    # Start from top (first bench crest) down to target (last bench toe)
    # Or use user specified start/end in ramp_params?
    # Usually we start at the top of the pit.

    top_z = benches[0].z_crest
    bottom_z = benches[-1].z_toe

    # Generate Z levels
    z_levels = np.arange(top_z, bottom_z, -ramp_params.z_step)

    slices = []

    # We need a robust way to get the polygon.
    # Passing pit_design_params allows us to re-calculate the batter angle for accurate interpolation.

    from pit_design import get_design_params_at_elevation

    for z in z_levels:
        # 1. Get Pit Polygon (Hole Boundary)
        # We find the active bench
        active_bench = None
        for b in benches:
            # Check if z is within this bench's vertical extent (including the berm below it)
            # Actually, let's just find the bench face it belongs to.
            if b.z_toe <= z <= b.z_crest:
                active_bench = b
                break
            # Check berm area (below toe, above next crest)
            # If z is in berm, we associate with the bench above (since its toe defines the wall)
            if z < b.z_toe:
                 # Check next bench
                 pass

        pit_poly = None

        if active_bench:
            # On Face
            # Interpolate
            if pit_design_params:
                try:
                    # Get angle at this Z
                    angle, _ = get_design_params_at_elevation(z, pit_design_params)
                    # We need to know how far down from crest we are
                    delta_z = active_bench.z_crest - z
                    if angle > 89.9:
                        offset_dist = 0
                    else:
                        offset_dist = delta_z / math.tan(math.radians(angle))

                    # Inset the crest
                    # Note: inset_polygon returns a list. Union them.
                    crests = so.unary_union(active_bench.crest_polys)
                    # We inset the crest (hole gets smaller)
                    # buffer(-dist)
                    pit_poly = crests.buffer(-offset_dist, join_style='mitre')

                except ValueError:
                    # Fallback if params fail
                    pit_poly = so.unary_union(active_bench.toe_polys)
            else:
                 pit_poly = so.unary_union(active_bench.toe_polys)
        else:
            # Must be on a berm or below last bench
            # Find the bench above
            bench_above = None
            for b in benches:
                if z < b.z_toe:
                    bench_above = b
                else:
                    break # We found the bench above z (b.z_toe > z) ? No.
                    # We want the lowest bench whose toe is above z.

            # Actually, simply:
            # Iterate benches. If z > z_crest, it's above.
            # If z in [z_toe, z_crest], it's face.
            # If z < z_toe, it might be in the berm of this bench.

            # Re-logic:
            poly_found = False
            for i, b in enumerate(benches):
                if b.z_toe <= z <= b.z_crest:
                    # Face
                    if pit_design_params:
                        angle, _ = get_design_params_at_elevation(z, pit_design_params)
                        delta_z = b.z_crest - z
                        offset_dist = delta_z / math.tan(math.radians(angle)) if angle < 89.9 else 0
                        crests = so.unary_union(b.crest_polys)
                        pit_poly = crests.buffer(-offset_dist)
                    else:
                        pit_poly = so.unary_union(b.toe_polys) # Fallback
                    poly_found = True
                    break
                elif z < b.z_toe:
                    # Could be in the berm below this bench
                    # Check next bench
                    if i + 1 < len(benches):
                        if z > benches[i+1].z_crest:
                            # On Berm
                            pit_poly = so.unary_union(b.toe_polys)
                            poly_found = True
                            break
                    else:
                        # Below last bench
                        pit_poly = so.unary_union(b.toe_polys)
                        poly_found = True
                        break

            if not poly_found:
                # Above top bench?
                if z > benches[0].z_crest:
                     pit_poly = so.unary_union(benches[0].crest_polys)
                else:
                     pit_poly = sg.Polygon()

        if pit_poly is None or pit_poly.is_empty:
             continue

        # 2. Calculate Free Polygon (Eroded)
        # clearance = ramp_width / 2 + safety_margin + ditch
        clearance = (ramp_params.ramp_width / 2.0) + ramp_params.safety_margin + ramp_params.ditch_allowance

        # Buffer negative (inset)
        free_poly = pit_poly.buffer(-clearance)

        slices.append(Slice(z=float(z), pit_poly=pit_poly, free_poly=free_poly))

    return slices
