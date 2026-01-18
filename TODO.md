# Pit Design PoC - Implementation Todo List

## Phase 0: Skeleton
- [x] Create Streamlit app scaffolding and module layout.
- [x] Show placeholder Plotly 3D axes.

## Phase 1: Load + display UP string (core)
- [x] Load UP string from JSON/hard-coded points.
- [x] Ensure closed loop.
- [x] Plot UP polyline in 3D.
- [x] Add index selector + highlight.

## Phase 2: Generate pit benches from UP (core)
- [x] Create `src/pit_design.py` as the central engine.
- [x] Implement `inset_polygon` function using Shapely/PyClipper.
- [x] Implement bench generation logic:
    - [x] Calculate horizontal offsets based on bench height, batter angle, and berm width.
    - [x] Loop to generate nested bench polygons until target depth/elevation.
    - [x] Handle polygon validity and robustness (MultiPolygons).
- [x] Integrate `pit_design` into `app.py`.
- [x] Visualize bench outlines in `pit_viz.py`.

## Phase 3: Export bench strings (core)
- [ ] Create export function for bench strings (CSV/JSON).
- [ ] Add download buttons in Streamlit UI.

## Phase 4: Import + display DXF topo TIN (important)
- [ ] Implement DXF import using `ezdxf` in `data_loader.py`.
- [ ] Filter by hard-coded layer name.
- [ ] Visualize topo mesh in `pit_viz.py`.

## Phase 5: Align pit design with topo (optional but valuable)
- [ ] Sample topo at UP vertices.
- [ ] Add Z0 alignment option.

## Phase 6: Robustness improvements (recommended)
- [ ] Improve polygon offset robustness (handle self-intersections).
- [ ] Add detailed diagnostics.

## Phase 7: Sector-based slopes / berms (if needed)
- [ ] Implement sector-based parameters.
