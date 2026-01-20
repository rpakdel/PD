import pytest
import math
import shapely.geometry as sg
from design_params import PitDesignParams, RampParams, BenchGeometry
from ramp_design import create_slices, get_pit_polygon_at_z, solve_ramp, generate_ramp_corridor

@pytest.fixture
def sample_benches():
    # Create a simple pit: 2 benches
    # Z: 100 (Crest 1) -> 90 (Toe 1) -> 90 (Crest 2) -> 80 (Toe 2)
    # Uniform square pit

    b1 = BenchGeometry(
        bench_id=1,
        z_crest=100.0,
        z_toe=90.0,
        crest_polys=[sg.box(-100, -100, 100, 100)],
        toe_polys=[sg.box(-90, -90, 90, 90)]
    )

    b2 = BenchGeometry(
        bench_id=2,
        z_crest=90.0,
        z_toe=80.0,
        crest_polys=[sg.box(-85, -85, 85, 85)], # Assume berm width 5
        toe_polys=[sg.box(-75, -75, 75, 75)]
    )

    return [b1, b2]

@pytest.fixture
def ramp_params():
    return RampParams(
        ramp_width=10.0,
        grade_max=0.1,
        z_step=2.5,
        safety_margin=1.0,
        ditch_allowance=1.0
    )

@pytest.fixture
def pit_design_params():
    return PitDesignParams(
        bench_height=10.0,
        batter_angle_deg=45.0, # Easy calc: tan(45)=1. H_face = 10/1 = 10.
        berm_width=5.0,
        target_elevation=80.0
    )

def test_get_pit_polygon_at_z(sample_benches, pit_design_params):
    # Test at z=95 (Face of Bench 1)
    # 45 deg angle -> offset = (100-95)/tan(45) = 5.
    # Crest is 100 wide. Offset 5 -> 95 wide box.

    poly_95 = get_pit_polygon_at_z(95.0, sample_benches, pit_design_params)
    assert not poly_95.is_empty

    # Check bounds
    minx, miny, maxx, maxy = poly_95.bounds
    assert abs(maxx - 95.0) < 0.1
    assert abs(maxy - 95.0) < 0.1

    # Test at z=85 (Berm of Bench 1? No, Bench 2 Face)
    # Bench 1 Toe is at 90. Bench 2 Crest is at 90.
    # Wait, in sample_benches setup:
    # B1: 100->90. Toe box 90.
    # B2: 90->80. Crest box 85.
    # So there is a berm from box 90 to box 85 at z=90.
    # At z=95 (Face B1), we expect box 95. (Correct)

    # Test Face B2 at z=85.
    # Crest B2 is 85 wide. z=90.
    # z=85 is 5m down. Offset 5.
    # 85 - 5 = 80 wide box.
    poly_85 = get_pit_polygon_at_z(85.0, sample_benches, pit_design_params)
    minx, miny, maxx, maxy = poly_85.bounds
    assert abs(maxx - 80.0) < 0.1

def test_get_pit_polygon_berm_logic(sample_benches, pit_design_params):
    # If we had a gap between benches...
    # Current setup: B1 Toe Z=90, B2 Crest Z=90. No vertical gap.
    # Let's Modify B2 to start at 89.

    b1 = sample_benches[0]
    b2 = sample_benches[1]
    b2.z_crest = 88.0 # Gap from 90 to 88

    # At z=89 (in berm)
    # Should return Toe of B1 (box 90)
    poly_89 = get_pit_polygon_at_z(89.0, sample_benches, pit_design_params)
    minx, miny, maxx, maxy = poly_89.bounds
    assert abs(maxx - 90.0) < 0.1

def test_create_slices(sample_benches, ramp_params, pit_design_params):
    slices = create_slices(sample_benches, ramp_params, pit_design_params)

    assert len(slices) > 0

    # z range: 100 down to 80.
    # z_step 2.5: 100, 97.5, 95.0, 92.5, 90.0, ... 80.0
    # Approx 9 slices.

    # Check slice at z=95
    s95 = next((s for s in slices if abs(s.z - 95.0) < 0.1), None)
    assert s95 is not None
    assert not s95.pit_poly.is_empty

    # Pit poly at 95 is box 95 (from prev test)
    # Ramp width 10, safety 1, ditch 1 -> Clearance = 7.
    # Free poly = 95 - 7 = 88 wide box.

    if not s95.free_poly.is_empty:
        minx, miny, maxx, maxy = s95.free_poly.bounds
        assert abs(maxx - 88.0) < 0.5
    else:
        pytest.fail("Free polygon shouldn't be empty for this large pit")

def test_create_slices_no_benches(pit_design_params):
    slices = create_slices([], RampParams(10, 0.1), pit_design_params)
    assert slices == []

def test_solve_ramp_basic(sample_benches, ramp_params, pit_design_params):
    # Setup slices
    slices = create_slices(sample_benches, ramp_params, pit_design_params)

    # Start Point on top (Z=100)
    # Pit is 100x100 box. Free space approx 93x93.
    # Start at (50, 50) inside.
    start = (50.0, 50.0)
    target_z = 80.0

    # Run solver
    path, diag = solve_ramp(slices, start, target_z, ramp_params)

    # Check if path found
    if not path:
        pytest.fail(f"Solver failed: {diag}")

    assert len(path) > 1
    assert path[0][0] == 50.0
    assert path[0][1] == 50.0
    assert path[0][2] == 100.0 # Start Z

    # Check if we went down
    assert path[-1][2] < 100.0

def test_solve_ramp_switchback_mode(sample_benches, ramp_params, pit_design_params):
    slices = create_slices(sample_benches, ramp_params, pit_design_params)
    start = (50.0, 50.0)
    target_z = 80.0

    # Test strict spiral mode (no switchbacks allowed)
    ramp_params.mode = "spiral"
    path_spiral, _ = solve_ramp(slices, start, target_z, ramp_params)
    assert len(path_spiral) > 0

    # Test switchback mode
    ramp_params.mode = "switchback"
    path_sb, _ = solve_ramp(slices, start, target_z, ramp_params)
    assert len(path_sb) > 0

def test_generate_ramp_corridor():
    # Simple straight ramp
    centerline = [(0, 0, 100), (10, 0, 99), (20, 0, 98)]
    width = 10.0

    left, right = generate_ramp_corridor(centerline, width)

    assert len(left) == 3
    assert len(right) == 3

    # Check offset
    # Line along X. Normal is along Y (0, 1).
    # Left: y + 5. Right: y - 5.

    assert abs(left[0][1] - 5.0) < 1e-6
    assert abs(right[0][1] - (-5.0)) < 1e-6

    # Check Z
    assert left[0][2] == 100
