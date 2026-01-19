# Pit Design PoC - Implementation Todo List

## Phase 0: Skeleton
- [x] Create Streamlit app scaffolding and module layout.
- [x] Show placeholder Plotly 3D axes.

## Phase 1: Load + display UP string (core)
- [x] Load UP string from JSON/hard-coded points.
- [x] Ensure closed loop.
- [x] Plot UP polyline in 3D.
- [x] Add index selector + highlight.

## Phase 2: Pit Design Engine & Surface Generation (Core Priority)
- [x] Create `src/pit_design.py` as the central engine.
- [x] Implement `inset_polygon` function using Shapely/PyClipper.
- [x] Implement bench generation logic (Strings).
- [ ] **Upgrade**: Handle MultiPolygons (split pits) properly (don't just take largest).
- [ ] **New**: Generate Triangulated Mesh (Faces & Berms) for "complete pit" representation.
- [ ] **New**: Verify geometry validity (normals, winding).

## Phase 3: Visualization (Core Priority)
- [x] Line visualization (Crest/Toe).
- [ ] **New**: Mesh visualization in Plotly (using `go.Mesh3d`).

## Phase 4: Export (Core Priority)
- [ ] Create export function for bench strings (CSV/JSON).
- [ ] Create export function for Pit Surface (DXF/OBJ/STL).
- [ ] Add download buttons in Streamlit UI.

## Phase 5: DXF Topo Import (Lower Priority)
- [ ] Implement DXF import using `ezdxf` in `data_loader.py`.
- [ ] Filter by hard-coded layer name.
- [ ] Visualize topo mesh in `pit_viz.py`.

## Phase 6: Advanced/Robustness
- [ ] Align pit design with topo (sample Z0).
- [ ] Sector-based slopes.
- [ ] Detailed diagnostics.
