from pit_design import generate_pit_benches, get_design_params_at_elevation
from design_params import PitDesignParams, DesignBlock
from data_loader import get_sample_up_string
import pytest

def test_scenario_a_missing_coverage():
    """
    Scenario A: Ranges 0-100. Target 200 (Upward).
    UP starts at 100.
    Generation goes 100 -> 200.
    Ranges only cover 0-100.
    Expect: ValueError (No design parameters defined for elevation 105).
    """
    up_string = get_sample_up_string(elevation=100.0)
    blocks = [
        DesignBlock(z_start=0.0, z_end=100.0, batter_angle_deg=10.0, berm_width=5.0)
    ]
    params = PitDesignParams(
        bench_height=10.0,
        batter_angle_deg=75.0,
        berm_width=5.0,
        target_elevation=200.0,
        design_direction="Upward",
        variable_params=blocks
    )

    benches, diagnostics = generate_pit_benches(up_string, params)

    assert "error" in diagnostics
    assert "No design parameters defined" in diagnostics["error"]

def test_scenario_b_variable_params_success():
    """
    Scenario B: Ranges 100-200. Target 200 (Upward).
    Ranges cover the area.
    Expect: Variable parameters (10 deg) used, not uniform (75 deg).
    """
    up_string = get_sample_up_string(elevation=100.0)
    blocks = [
        DesignBlock(z_start=100.0, z_end=200.0, batter_angle_deg=10.0, berm_width=5.0)
    ]
    params = PitDesignParams(
        bench_height=10.0,
        batter_angle_deg=75.0,
        berm_width=5.0,
        target_elevation=200.0,
        design_direction="Upward",
        variable_params=blocks
    )

    benches, diagnostics = generate_pit_benches(up_string, params)

    # Ensure no error
    if "error" in diagnostics:
        pytest.fail(f"Generation failed: {diagnostics['error']}")

    assert len(benches) > 0

    # Check geometry of first bench (100 -> 110)
    # 10 deg batter angle.
    first_crest = benches[0].crest_polys[0]
    minx, miny, maxx, maxy = first_crest.bounds
    width = maxx - minx

    # Initial UP string radius = 200 (Diameter 400).
    # Height 10m.
    # Angle 10 deg. Tan(10) = 0.176. Offset = 10/0.176 = 56.7.
    # New Radius = 200 + 56.7 = 256.7. Diameter = 513.4.

    # If 75 deg. Tan(75) = 3.73. Offset = 10/3.73 = 2.68.
    # New Radius = 200 + 2.68 = 202.68. Diameter = 405.36.

    print(f"Scenario B Crest Width: {width:.2f}")
    assert width > 500.0, f"Expected shallow angle width > 500, got {width}. This means uniform params were used."

def test_scenario_c_uniform_fallback():
    """
    Scenario C: No variable params.
    Expect: Uniform parameters (75 deg) used.
    """
    up_string = get_sample_up_string(elevation=100.0)
    # No blocks
    params = PitDesignParams(
        bench_height=10.0,
        batter_angle_deg=75.0,
        berm_width=5.0,
        target_elevation=200.0,
        design_direction="Upward",
        variable_params=[]
    )

    benches, diagnostics = generate_pit_benches(up_string, params)
    assert len(benches) > 0

    first_crest = benches[0].crest_polys[0]
    minx, miny, maxx, maxy = first_crest.bounds
    width = maxx - minx

    print(f"Scenario C Crest Width: {width:.2f}")
    assert width < 420.0, f"Expected steep angle width < 420, got {width}"
