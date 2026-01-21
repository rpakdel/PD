# agents.md (Focused) — Pit Wall/Bench Generation from Ultimate Pit (UP) String + Ramp Generation

This is a **small, algorithm-focused** guide to avoid context bloat. It specifies:
1) how to generate **pit walls + benches** from an Ultimate Pit string, and  
2) **exactly how to generate ramps** (in-pit haul roads) and “cut” them into the pit geometry, using an **open-source Boolean/offset engine**.

---

## 0) Core Libraries (Open Source)

### Boolean + Offset (recommended “engine”)
- **Clipper/Clipper2 via `pyclipper`** (MIT): robust polygon **offsetting** and **Boolean** operations (union/diff/intersection/xor). ([github.com](https://github.com/fonttools/pyclipper?utm_source=openai))  
  Use this as the *primary* geometry engine for inset/offset + booleans.

### Optional helpers
- **Shapely/GEOS**: useful for validation, area, and misc geometry, but do **not** rely on it for repeated offsets if you already have Clipper. (Keep stack small.)

---

## 1) Inputs & Definitions

### 1.1 Ultimate Pit (UP) String
- A polyline intended to represent the **ultimate pit limit in plan (XY)**.
- Must be convertible into a **simple polygon ring** (closed, non-self-crossing ideally).
- If UP has Z: treat Z as metadata; pit generation is driven primarily in **XY + vertical rules**.

### 1.2 Pit Design Parameters (uniform for PoC)
- `BH` = bench height (m)
- `θ_face` = face/batter angle (define clearly: **from horizontal** OR **from vertical**)
- `BW` = catch berm width (m)
- `z_top` = reference top elevation (manual or derived)
- Termination: `z_bottom` or `num_benches`

**Horizontal setback per bench** (uniform case):
- If `θ_face` is **from horizontal**: `H_face = BH / tan(θ_face)`
- If `θ_face` is **from vertical**: `H_face = BH * tan(θ_face)`
- Total inset distance per bench polygon step: `d = H_face + BW`

### 1.3 Outputs (minimum viable)
For each bench `k`:
- `z_crest[k] = z_top - k*BH`
- `z_toe[k]   = z_top - (k+1)*BH`
- `crest_ring_xy[k]`  (closed ring)
- `toe_ring_xy[k]`    (closed ring)  
Plus diagnostics: polygon area, component count, early termination reason.

---

## 2) Pit Wall + Bench Generation Algorithm (from UP String)

### 2.1 Pre-process UP ring (critical)
1. **Close** the ring if last point != first point.
2. **Remove spikes/duplicates**: drop consecutive points closer than `eps` (e.g., 0.01 m).
3. Ensure consistent orientation (clockwise vs ccw) for your offset semantics.
4. Validate:
   - If ring self-intersects: attempt a repair (PoC: fail with a clear message unless you implement a “clean” pass).

### 2.2 Use Clipper-style integer scaling
Clipper operates in integer space for robustness.
- Choose `SCALE = 1000` (mm) or `10000` depending on coordinate magnitude.
- Convert `(x,y)` → `(round(x*SCALE), round(y*SCALE))`
- Convert distances (meters) → `round(d * SCALE)`

### 2.3 Bench loop generation (uniform slopes)
Let `P0` be the UP polygon ring (int coords), representing the **crest at z_top**.

For `k = 0..K-1`:
1. `crest[k] = Pk`
2. Compute inset distance `d_int = round((H_face + BW) * SCALE)`
3. `Pk+1 = offset_inward(Pk, d_int)` using Clipper offset
4. If `Pk+1` is empty → stop (pit pinched out)
5. If `Pk+1` returns MultiPolygon:
   - **PoC rule**: select the **largest-area** polygon as continuation; store others as “discarded components” with warning.
6. `toe[k] = Pk+1`
7. Continue until `z_bottom` reached or offset fails.

**Notes**
- The above generates **nested bench outlines**; it is the backbone of pit wall/bench geometry.
- You can later derive:
  - berm “platform” polygons (between toe/crest)
  - wall faces (connect rings vertically between elevations)

### 2.4 Wall surface (optional but recommended for 3D)
For each consecutive ring pair `(crest[k], toe[k])`:
- Resample rings to comparable vertex counts (PoC: uniform spacing along perimeter).
- Create quad strips and triangulate (two triangles per quad).
- Assign `z_crest[k]` to crest vertices and `z_toe[k]` to toe vertices.

---

## 3) Where the Boolean Engine is Used (Pit)
- **Offset/inset** for bench progression: Clipper offset. ([angusj.com](https://angusj.com/clipper2/Docs/Overview.htm?utm_source=openai))
- **Boolean difference** for removing ramp cuts from berm/platform polygons (Section 5).
- **Union** if you merge multiple components or add design features.

---

## 4) Ramp Generation — What “Exactly” Means Here

You will generate:
1) a **ramp centerline** (3D polyline), then  
2) a **ramp corridor** (2D polygon per elevation/bench, or a swept polygon), then  
3) **Boolean-cut** that corridor out of pit berm/platform polygons (and optionally re-triangulate).

This matches common “toe + ramp strings first” style workflows used in mine design tooling. ([docs.dataminesoftware.com](https://docs.dataminesoftware.com/StudioPM/Latest/STUDIO_OP/open%20pit%20mine%20design.htm?utm_source=openai))

---

## 5) Ramp Generation Algorithm (Recommended, documented approach)

### 5.1 Approach: Raster-based Least-Cost Path + geometric post-processing
A widely published method for open-pit haul road layout is:
- Compute a road layout with **raster-based least-cost path analysis**,  
- simplify the zigzag path (e.g., Douglas–Peucker),  
- then modify to satisfy **curvature radius constraints**,  
- and finally create a 3D road model combining with pit/bench geometry. ([mdpi.com](https://www.mdpi.com/2076-3417/7/7/747/htm?utm_source=openai))

This is the **most implementable “exact algorithm”** for a PoC because it’s explicit, testable, and doesn’t require guessing a spiral by hand.

### 5.2 Inputs (ramp)
- Start point: on/near top access (often on/near the UP crest); index-selection is fine for choosing a seed.
- End point: bottom access point (pit bottom) or a target elevation.
- Design constraints:
  - `grade_max` (e.g., 10%)
  - `width` (m)
  - `min_turn_radius` (m)
  - optional: “prefer wall-following” weight
- Operational guidance for switchbacks/curves (constraints to enforce):
  - Switchback inside tire-path radius guideline commonly set ≥ **150%** of the truck’s minimum turning circle inner clearance radius (rule-of-thumb guidance). ([researchgate.net](https://www.researchgate.net/publication/330495905_GUIDELINES_AND_CONSIDERATIONS_FOR_OPEN_PIT_DESIGNERS?utm_source=openai))

### 5.3 Build the planning surface (“where ramps are allowed”)
1. Create a raster grid over the pit area (cell size 2–10 m for PoC).
2. For each cell, compute:
   - elevation `z` from your pit surface model (bench/wall mesh) or a simplified DEM
   - slope magnitude vs neighbors
3. Create a **feasibility mask**:
   - disallow cells outside pit (for in-pit ramp) or outside allowed corridor band
   - disallow cells too close to walls if you need clearance

### 5.4 Define the cost function (per move)
For each move from cell `i` → `j`:
- base distance cost: `dist(i,j)`
- slope penalty: increases with grade (vertical factor)
- turning penalty: increases with heading change (horizontal factor)  
This matches the “terrain slope + rotational angle” cost modeling described for haul road layout optimization. ([mdpi.com](https://www.mdpi.com/2076-3417/7/7/747/htm?utm_source=openai))

### 5.5 Solve least-cost path
- Run Dijkstra/A* on the grid to get a polyline path (sequence of grid centers).
- Output is a “raw” path, often zigzaggy.

### 5.6 Simplify the path
- Apply Douglas–Peucker simplification (tolerance in meters).
- Preserve endpoints and critical constraint points.

### 5.7 Enforce minimum curvature radius (geometry rewrite)
Iterate over vertices:
1. Compute local turn angle and implied radius (or fit circle through 3 points).
2. If radius < `min_turn_radius`:
   - replace the corner with a circular arc (tangent continuous)
   - or spread the corner by inserting intermediate points
3. Re-check until all corners pass.

This “modify by reflecting curvature radius suggested in road design guides” is explicitly part of published workflow. ([mdpi.com](https://www.mdpi.com/2076-3417/7/7/747/htm?utm_source=openai))

### 5.8 Assign elevations + enforce grade
Turn the 2D path into 3D:
1. Sample/compute ground elevation `z_ground(s)` along the path on the pit surface model.
2. Generate a design profile `z_design(s)` such that:
   - `|dz/ds| <= grade_max`
   - endpoint elevation targets are met
3. If `z_design` departs from `z_ground`, treat that as “cut/fill” within the pit model (PoC: allow “cut” only).

### 5.9 Build ramp corridor polygon(s)
- In XY, offset the centerline by `width/2` to left and right (buffer).
- Use Clipper offset to build a clean polygon corridor (scaled ints).
- For switchbacks, widen corridor by an additional allowance (PoC: width + 0.5–1.0 truck widths, if you model truck width). ([researchgate.net](https://www.researchgate.net/publication/330495905_GUIDELINES_AND_CONSIDERATIONS_FOR_OPEN_PIT_DESIGNERS?utm_source=openai))

### 5.10 Boolean-cut ramps into the pit benches
For each bench/platform polygon (the “walkable/berm” area):
1. Compute corridor footprint at that bench elevation (or use same XY corridor for all benches it crosses).
2. `platform_cut = platform_polygon - corridor_polygon` using Clipper **Difference**
3. Store:
   - updated platform ring(s)
   - ramp edge strings (intersection boundaries) for export/visualization

---

## 6) Slot Ramp vs Spiral Ramp (PoC guidance)
- **Slot ramp**: behaves like a trench and is *not* tightly controlled by pit perimeter shape; it widens at berm crossings and can be converted to spiral later (tooling documentation describes this conceptual difference). ([webhelp.micromine.com](https://webhelp.micromine.com/mm/latest/English/Content/mmpit/IDH_VX_PIT_SLOT_RAMP.htm?utm_source=openai))  
- **Spiral ramp**: can still be produced by the same LCPA method by adding a “wall-following preference” in the cost function and constraining feasible cells to a band near the wall.

For PoC: implement **one planner** (LCPA) and switch “slot vs spiral” by:
- feasibility mask shape (freeform vs wall-adjacent band)
- cost weights (turn penalty, wall-following penalty)

---

## 7) Minimal Deliverables (so agents don’t overbuild)
1) Pit benches from UP string via repeated inset offsets (Clipper).  
2) Ramp centerline via grid least-cost path + simplify + curvature fix + grade profile. ([mdpi.com](https://www.mdpi.com/2076-3417/7/7/747/htm?utm_source=openai))  
3) Ramp corridor polygon via offset + Boolean difference into bench/platform polygons (Clipper). ([angusj.com](https://angusj.com/clipper2/Docs/Overview.htm?utm_source=openai))  
4) Export strings:
- bench crest/toe per level
- ramp centerline (xyz)
- ramp corridor edges (xy + z tags)

---

## 8) Agent Notes (common failure modes)
- **Scaling**: choose SCALE and stick to it everywhere for Clipper.
- **MultiPolygons**: offsets will split concave pits; define a deterministic rule (largest-area) and log warnings.
- **Early termination**: inset may collapse before target depth—report depth achieved.
- **Curvature enforcement**: do not just “smooth”; explicitly enforce **min radius** per design constraint pass. ([researchgate.net](https://www.researchgate.net/publication/277759950_Guidelines_for_Mine_Haul_Road_Design?utm_source=openai))
