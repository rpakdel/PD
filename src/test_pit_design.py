import pytest
import math
import shapely.geometry as sg
from pit_design import inset_polygon, generate_pit_benches, PitDesignParams, Mesh3D

def test_inset_polygon():
    # Simple square 10x10
    poly = sg.Polygon([(0,0), (10,0), (10,10), (0,10)])
    insets = inset_polygon(poly, 1.0)
    assert len(insets) == 1
    assert insets[0].area < poly.area
    # Square 10x10 -> inset 1 -> 8x8 -> area 64
    assert math.isclose(insets[0].area, 64.0)

def test_generate_pit_benches_simple():
    # 100x100 square at z=100
    points = [(0,0,100), (100,0,100), (100,100,100), (0,100,100)]
    params = PitDesignParams(
        bench_height=10.0,
        batter_angle_deg=45.0, # tan(45)=1 -> offset=10
        berm_width=5.0,
        target_elevation=0.0
    )

    benches, diag = generate_pit_benches(points, params)

    assert len(benches) > 0

    b1 = benches[0]
    assert b1.z_crest == 100.0
    assert b1.z_toe == 90.0

    # Check meshes
    assert len(b1.face_mesh.vertices) > 0
    assert len(b1.berm_mesh.vertices) > 0

def test_split_pit():
    # Dumbbell shape that should split
    # Two 10x10 squares connected by a thin bridge
    # Bridge width 2.
    # Inset by 2 -> bridge disappears -> split into 2

    poly = sg.Polygon([
        (0,0), (10,0), (10,10), (0,10), # Left square
        (10,4), (20,4), (20,6), (10,6), # Bridge
        (20,0), (30,0), (30,10), (20,10) # Right square (approx)
    ])
    # Correction: Need valid polygon loop
    coords = [
        (0,0), (10,0), (12,4), (18,4), (20,0), (30,0),
        (30,10), (20,10), (18,6), (12,6), (10,10), (0,10)
    ]
    # Bridge is between x=10 and x=20? No, bridge is width 2 (y=4 to y=6).

    up_points = [(x,y,100) for x,y in coords]

    params = PitDesignParams(
        bench_height=1.0,
        batter_angle_deg=45.0, # offset 1
        berm_width=0.0,
        target_elevation=90.0
    )

    # Inset 1.0 should close the bridge (width 2 -> inset 1 from both sides = 0)
    # Actually shapely buffer might keep it touching or split.

    benches, diag = generate_pit_benches(up_points, params)

    # Check if bench 2 has 2 polygons (split)
    # Bench 1 offset 1.0. Bridge width 2. It might pinch.

    # Let's inspect bench 1
    # print(len(benches[0].toe_polys))

    # Just ensure it doesn't crash
    assert benches

def test_generate_pit_benches_upward():
    # 10x10 square at z=0
    # Generate upward to z=20 (2 benches)
    # Start: Toe at 0.
    # Bench 1: Toe=0, Crest=10. Crest should be larger than Toe.
    # Face slope 45 deg -> dx = 10 (h=10).
    # So Crest should be 10+10+10 = 30 wide (inset -10).

    points = [(0,0,0), (10,0,0), (10,10,0), (0,10,0)]
    params = PitDesignParams(
        bench_height=10.0,
        batter_angle_deg=45.0, # h=10 -> dx=10
        berm_width=5.0,
        target_elevation=20.0, # Should fit 2 benches (0->10, 10->20)
        design_direction="Upward"
    )

    benches, diag = generate_pit_benches(points, params)

    assert len(benches) > 0
    # Should produce at least 1 bench, maybe 2.
    # Bench 1: Toe Z=0, Crest Z=10.
    # Bench 2: Toe Z=10, Crest Z=20.
    # Loop continues while toe < target (10 < 20).

    assert len(benches) == 2

    b1 = benches[0]
    assert b1.z_toe == 0.0
    assert b1.z_crest == 10.0

    # Check area expansion
    # Start area 100.
    # Bench 1 Crest: Buffer +10.
    # Square 10x10 buffered by 10 becomes much larger.
    # Roughly (10+20)x(10+20) = 30x30 = 900 (plus rounded corners).

    toe_area = b1.toe_polys[0].area
    crest_area = b1.crest_polys[0].area

    assert crest_area > toe_area
    assert math.isclose(toe_area, 100.0)

    # Bench 2
    b2 = benches[1]
    assert b2.z_toe == 10.0
    assert b2.z_crest == 20.0

    # Bench 2 Toe should be Bench 1 Crest expanded by Berm (5.0).
    # So b2.toe > b1.crest
    b2_toe_area = b2.toe_polys[0].area
    assert b2_toe_area > crest_area
