import pytest
import shapely.geometry as sg
from ramp_geometry import RampGenerator
from design_params import RampParams

def test_ramp_gen_straight():
    # Square polygon 100x100
    poly = sg.Polygon([(0,0), (100,0), (100,100), (0,100), (0,0)])

    params = RampParams(ramp_width=10, grade_max=0.10)
    gen = RampGenerator(params)

    gen.set_start_point((0,0))

    # Drop 10m. Length = 100m.
    # Should traverse bottom edge (100m) exactly.
    ramp_poly, end_pt, ramp_mesh = gen.generate_ramp_segment(poly, 100, 90, 0.10)

    assert abs(end_pt[0] - 100.0) < 0.1
    assert abs(end_pt[1] - 0.0) < 0.1
    assert ramp_poly.area > 0

    # Check mesh
    assert len(ramp_mesh.vertices) > 0
    assert len(ramp_mesh.faces) > 0

    # Check Z range of vertices
    zs = [v[2] for v in ramp_mesh.vertices]
    assert max(zs) <= 100.0 + 1e-3
    assert min(zs) >= 90.0 - 1e-3

    # Drop another 10m. Should go up right edge.
    ramp_poly_2, end_pt_2, ramp_mesh_2 = gen.generate_ramp_segment(poly, 90, 80, 0.10)

    assert abs(end_pt_2[0] - 100.0) < 0.1
    assert abs(end_pt_2[1] - 100.0) < 0.1

def test_ramp_gen_spiral():
    # Circular polygon (approx)
    # buffer(100) creates a circle centered at 0,0 with radius 100
    poly = sg.Point(0,0).buffer(100, resolution=16)

    params = RampParams(ramp_width=10, grade_max=0.10)
    gen = RampGenerator(params)

    # We need a start point on the polygon.
    # Point(100,0) is roughly on the boundary.
    start_pt = (100, 0)
    gen.set_start_point(start_pt)

    # Drop 10m. Length = 100m.
    ramp_poly, end_pt, ramp_mesh = gen.generate_ramp_segment(poly, 100, 90, 0.10)

    assert ramp_poly.area > 0
    # Check end point is approximately 100m away along perimeter
    # Distance from start
    dist = sg.Point(start_pt).distance(sg.Point(end_pt))
    # Chord length for arc 100 on radius 100 is approx 2*R*sin(theta/2)
    # Theta = 1 rad. 2*100*sin(0.5) = 200*0.479 = 95.8
    assert 90 < dist < 100

    assert len(ramp_mesh.vertices) > 0
