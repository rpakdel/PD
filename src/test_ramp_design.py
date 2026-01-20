import pytest
import math
import shapely.geometry as sg
from design_params import PitDesignParams, RampParams, BenchGeometry
from ramp_design import create_slices, get_pit_polygon_at_z

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

def test_get_pit_polygon_at_z(sample_benches):
    # Test at z=95 (Face of Bench 1)
    # Interpolation not implemented perfectly yet in test fixture setup logic inside get_pit_polygon_at_z
    # But let's check basic retrieval

    # We didn't pass params, so it should fallback to toe
    poly_95 = get_pit_polygon_at_z(95.0, sample_benches)
    assert not poly_95.is_empty
    # Fallback is union of toe polys
    assert poly_95.equals(sample_benches[0].toe_polys[0])

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
    assert not s95.free_poly.is_empty

    # Check clearance
    # Ramp width 10, safety 1, ditch 1 -> Clearance = 5 + 1 + 1 = 7.
    # Free poly should be offset by -7 from pit poly.
    # Area check approx
    assert s95.free_poly.area < s95.pit_poly.area

def test_create_slices_no_benches():
    slices = create_slices([], RampParams(10, 0.1))
    assert slices == []
