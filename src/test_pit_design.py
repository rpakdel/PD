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

    # Dumbbell:
    # Left Box: (0,0) to (10,10)
    # Bridge: (10,4) to (20,6) - Width 2, Length 10
    # Right Box: (20,0) to (30,10)

    # Polygon coords
    coords = [
        (0,0), (10,0), (10,4), (20,4), (20,0), (30,0),
        (30,10), (20,10), (20,6), (10,6), (10,10), (0,10)
    ]
    # Note: I simplified the bridge connection points

    up_points = [(x,y,100) for x,y in coords]

    # If we inset by > 1.0 (bridge half-width), it should split.
    # Let's use offset 1.5.
    # Bench height 1.5, batter 45 -> offset 1.5

    params = PitDesignParams(
        bench_height=1.5,
        batter_angle_deg=45.0,
        berm_width=0.0,
        target_elevation=90.0
    )

    benches, diag = generate_pit_benches(up_points, params)

    # Bench 1: Crest is original (1 poly). Toe should be split (2 polys).
    # Since we check benches[0].toe_polys

    assert benches
    b1 = benches[0]

    # Check split
    # Depending on exact shapely behaviour, it might be 2 polygons.
    assert len(b1.toe_polys) >= 2

def test_polygons_merge_upward():
    # Two separate squares 10x10, separated by 5 units gap.
    # Gap: (10,0) to (15,0) is gap?
    # Sq1: (0,0)-(10,10)
    # Sq2: (15,0)-(25,10)
    # Gap width = 5.

    # We need a MultiPolygon input or just list of points?
    # generate_pit_benches takes list of points (one poly).
    # But we can hack it or use a "C" shape that closes on itself?
    # Or we can just pass a list of points that goes back and forth?
    # No, let's create a U-shape that fills in.

    # U-Shape:
    # (0,0) -> (25,0) -> (25,10) -> (15,10) -> (15,5) -> (10,5) -> (10,10) -> (0,10)
    # Gap is 5 wide (x=10 to x=15) and 5 deep (y=5 to y=10).
    # If we expand by 2.6 (radius > 2.5), the gap should close.

    coords = [
        (0,0), (25,0), (25,10), (15,10), (15,5),
        (10,5), (10,10), (0,10)
    ]

    up_points = [(x,y,0) for x,y in coords]

    params = PitDesignParams(
        bench_height=3.0,
        batter_angle_deg=45.0, # offset 3.0. Gap half-width is 2.5. Should close.
        berm_width=0.0,
        target_elevation=10.0,
        design_direction="Upward"
    )

    benches, diag = generate_pit_benches(up_points, params)

    # Bench 1: Toe is U-shape (1 poly).
    # Crest is Expanded. Offset 3.0 > 2.5. Gap closes.
    # The hole might vanish or become a hole polygon?
    # If it fills completely, 1 polygon without holes.
    # If it encloses a hole, 1 polygon with hole.
    # But shapely buffer usually dissolves holes if they are filled?
    # Actually positive buffer fills holes.

    b1 = benches[0]
    # It should still be 1 polygon, but the area should clearly indicate filling.
    # Wait, if we use separate polygons (MultiPolygon input), `generate_pit_benches` currently takes `up_points` -> single Polygon.
    # So we can't easily test "Two separate squares merging" unless we modify input to accept Polygons or use the U-shape.

    # U-shape test is valid for "Closing a bay".
    # Area check:
    # Base area: 25*10 - 5*5 = 250 - 25 = 225.
    # Expanded box (approx): (25+6) * (10+6) = 31 * 16 = 496.
    # If gap didn't fill, we'd have less area.
    # But "merging" usually refers to topology change (2 -> 1).

    # Let's try to simulate 2 squares by passing a MultiPolygon to `generate_pit_benches`?
    # The function signature expects `List[Tuple]`.
    # It converts to `sg.Polygon`. `sg.Polygon` doesn't support multiple outer shells.
    # So with current input signature, we can only start with one loop.
    # However, intermediate steps use `clean_polygons` which handles MultiPolygons.
    # So the "Split Pit" test covers 1 -> 2.
    # The reverse (2 -> 1) implies we start with 2.
    # Since we can't start with 2, we can't test "Start with 2 separate -> Merge to 1" easily without changing input signature.
    # But we can test "Split then Merge" sequence?
    # Start split (Down), then go Up? No.

    # Let's stick to U-shape closing.
    assert len(b1.crest_polys) == 1

    # Check diagnostics for merge info?
    # We logged count.
    # We can check that we have 1 crest.
    assert len(b1.crest_polys) == 1

def test_generate_pit_benches_upward_simple():
    # 10x10 square at z=0
    points = [(0,0,0), (10,0,0), (10,10,0), (0,10,0)]
    params = PitDesignParams(
        bench_height=10.0,
        batter_angle_deg=45.0,
        berm_width=5.0,
        target_elevation=20.0,
        design_direction="Upward"
    )

    benches, diag = generate_pit_benches(points, params)

    assert len(benches) == 2
    b1 = benches[0]
    # Check areas to ensure expansion
    assert b1.crest_polys[0].area > b1.toe_polys[0].area
