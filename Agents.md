# Open Pit Mine Design Agent — Coding Instructions & Technical Documentation

This document is a **build-ready specification** for implementing an AI-assisted open-pit design component inside an existing CAD application. It is organized as:

1. **System scope & invariants** (what the agent must guarantee)
2. **Data model** (inputs/outputs and required derived representations)
3. **Pipeline architecture** (stages, feedback loops, and failure modes)
4. **Ramp centerline solver (constraints + backtracking)** — primary focus
5. **Optional ILP integration** (economic pit + ramp space reservation)
6. **CAD generation** (strings, surfaces, corridor, validation)
7. **Interfaces, persistence, UI hooks**
8. **Validation, testing, telemetry, and debugging playbook**

The goal is to produce a ramp design that is *geometrically feasible*, *domain-aware*, and *CAD-ready*, with deterministic, testable behavior.

---

## 0) Non-negotiable invariants (agent must enforce)

### Geometry / safety invariants
- **Grade**: ramp centerline grade must satisfy `g_min ≤ Δz / Δxy ≤ g_max` (or your chosen convention), within tolerance.
- **Minimum turn radius**: no local curvature tighter than `R_min`.
- **Clearance**: the *ramp corridor* (not just the centerline) must fit inside the pit void with required offsets:
  - corridor half-width = `ramp_width/2 + berm + ditch + safety_margin (+ domain_margin)`
- **No self-intersection** of the ramp corridor (or if allowed, must be flagged and resolved).
- **Connectivity**: one continuous path from entry elevation to target elevation.

### Engineering invariants (project-specific; make explicit)
- **Bench height** and bench configuration (face angle, berm width) are respected in generated strings/meshes.
- **Spiral vs switchback** behavior follows user mode:
  - Spiral: monotonic progression around pit (no frequent reversals)
  - Switchback: reversals occur only in designated turn bulbs with sufficient space.

### Software invariants
- Runs deterministically from the same inputs + random seed.
- Produces a full audit log: parameters, constraint checks, and reason for failure.

---

## 1) Required inputs and normalized data model

### 1.1 Inputs (minimum)
- **Block model** (regular grid or arbitrary blocks):
  - `block_id`, centroid `(x,y,z)`, dimensions, `domain_id`, `value`, (optional) precedence info
- **Bench strings** (crest/toe polylines per bench elevation):
  - can be multiple loops per level
- **Slope rules** domain-based:
  - `domain_id -> (face_angle, berm_width, interramp_angle, geotech_clearance_margin, etc.)`
- **Ramp parameters**:
  - `ramp_width`, `grade_max`, `grade_min` (optional), `R_min`, `crossfall`, `turning_template` (switchback), `z_step` (solver step), tolerances
- **No-go volumes/polygons** (optional):
  - e.g., geotech exclusion, infrastructure, water

### 1.2 Canonical internal representations
Define these core objects and make them immutable after creation (easier debugging):

```ts
type Vec2 = { x: number, y: number }
type Vec3 = { x: number, y: number, z: number }

type Polyline2 = Vec2[]
type Polyline3 = Vec3[]
type Polygon2 = Vec2[]            // closed; last point need not repeat first
type MultiPolygon2 = Polygon2[]   // disjoint components, holes handled separately (see below)

type BenchLoop = {
  z: number,
  crestLoops: Polygon2[],  // outer + inner loops possible
  toeLoops: Polygon2[]
}

type DomainRules = {
  faceAngleDeg: number,
  bermWidth: number,
  wallMargin: number,      // extra clearance for ramp near that domain wall
  // add more as needed
}

type RampParams = {
  rampWidth: number,
  safetyMargin: number,
  ditchAllowance: number,
  gradeMax: number,        // e.g. 0.10
  gradeMin?: number,       // optional, usually 0
  minRadius: number,       // plan-view radius
  zStep: number,           // e.g. 1.0m
  horizontalTol: number,   // e.g. 0.15 (±15% around target step length)
  segmentSamples: number,  // e.g. 7
  mode: "spiral" | "switchback",
  seed: number
}
```

### 1.3 Derived geometry caches (performance-critical)
You will need per-slice feasible regions:

```ts
type Slice = {
  z: number,
  pitPoly: MultiPolygon2,      // pit opening at this z (inside is "air")
  freePoly: MultiPolygon2,     // pitPoly eroded by clearance => where centerline may lie
  components: { id: number, poly: Polygon2 }[], // connected components of freePoly
  // optional: distance field grid for heuristics
}
```

---

## 2) Pipeline architecture (staged + feedback loop)

### Stage A — Preprocess pit geometry
1. Validate bench strings (closed, non-self-intersecting; fix winding).
2. Build *pit opening* polygons per elevation slice `z_k`:
   - Use bench surfaces if available; otherwise interpolate between bench loops.
3. Apply no-go areas: `pitPoly := pitPoly - noGo(z_k)`
4. Compute clearance eroded region:
   - `clearance = ramp_width/2 + safetyMargin + ditchAllowance + domainMargin(z_k, location)`
   - Start with a conservative constant clearance if domain-varying erosion is not yet implemented.
   - `freePoly = offset(pitPoly, -clearance)`
5. Compute connected components per slice.

**Output**: `Slice[]` covering from entry z down to target z.

### Stage B — Ramp centerline solve (primary deliverable)
Run a constrained backtracking/beam/A* hybrid that returns a 3D polyline centerline.

### Stage C — Smooth + enforce constraints (constrained smoothing)
Fit tangential arcs / clothoid-like transitions where possible, but re-check constraints *against free space* after every modification.

### Stage D — Generate CAD geometry
- Generate ramp corridor (offset centerline by half-width; add berm/ditch template).
- Generate toe/crest strings and bench surfaces.
- Stitch pit surface to topo surface (trim/patch, or solid boolean if your CAD kernel supports solids robustly).

### Stage E — Validate + iterate
If constraints fail (clearance, grade, radius), either:
- re-run Stage B with different seed/parameters, or
- widen clearance assumptions / adjust turn bulbs, then retry.

---

## 3) Ramp centerline solver (constraints + backtracking)

This is the module you indicated you’re keen on. The solver should be written as a standalone library component with no CAD dependencies.

### 3.1 Core idea
Solve for points `{p_k}` where `z_{k+1} = z_k - zStep` and `p_k.xy ∈ freePoly(z_k)`.

The grade constraint sets the target horizontal distance per step:
- `L_target = zStep / gradeMax`
- `L_min = L_target * (1 - horizontalTol)`
- `L_max = L_target * (1 + horizontalTol)`

### 3.2 Hard constraints
At candidate step `p_prev -> p_curr -> p_next`:

1. **Inside free space**:
   - `p_next.xy ∈ freePoly(z_next)`
2. **Segment inside free space** (approx):
   - sample `t = 0..1` at `segmentSamples` points
   - check each sample point is inside the corresponding `freePoly` (by interpolating z → nearest slice)
3. **Grade**:
   - `L_min ≤ dist2D(p_curr, p_next) ≤ L_max`
4. **Min radius** (plan-view):
   - Compute turn angle `φ` between segments and ensure `R(φ) ≥ minRadius`
   - Use robust formula:
     - Let `a = p_curr.xy - p_prev.xy`, `b = p_next.xy - p_curr.xy`
     - `φ = acos( clamp( dot(a,b) / (|a||b|), -1, 1 ) )`
     - With step length `L = (|a|+|b|)/2`, estimate `R ≈ L / (2 sin(φ/2))`
5. **Mode-specific rules**:
   - Spiral: enforce monotonic progression in polar angle (or along a guide loop arc-length)
   - Switchback: only allow large heading reversals inside “turn bulb” regions

### 3.3 Soft objectives (candidate ranking)
Rank candidates by a weighted cost (lower is better):

- distance to guide curve (spiral: offset pit boundary; switchback: selected wall band)
- boundary clearance (prefer interior to avoid near-wall issues)
- curvature penalty (prefer smoother)
- progress metric (prefer getting closer to target region / desired angular progress)

### 3.4 Candidate generation
Given `p_curr`, generate candidates on the next slice:

1. Determine preferred heading `θ_pref`
2. Determine allowed heading deviation `±θ_max` from radius constraint:
   - approximate `θ_max = 2 * asin( L_target / (2*minRadius) )` (cap to a sane max)
3. Sample `Nθ` angles and `Nr` radii:
   - `θ_i = θ_pref + linspace(-θ_max, θ_max, Nθ)`
   - `r_j ∈ {L_min, L_target, L_max}` or a small set
4. Candidate:
   - `q.xy = p_curr.xy + r_j * (cos θ_i, sin θ_i)`
   - `q.z = z_next`

Filter by hard constraints; then keep top `B` candidates (“beam width”).

### 3.5 Backtracking engine (beam-DFS with branch-and-bound)
Use a DFS recursion but:
- keep a global `bestCost`
- prune paths whose `costSoFar` exceeds `bestCost`
- store a memoization table keyed by `(sliceIndex, componentId, headingBin)` to avoid revisiting dominated states.

**State:**
```ts
type SearchState = {
  k: number,                 // slice index
  prev?: Vec3,
  curr: Vec3,
  heading: number,           // radians
  cost: number,
  spiralS?: number,          // arc-length on guide (spiral)
  legId?: number             // switchback leg index
}
```

**Pseudo-code:**
```ts
function solveRamp(slices, start, target, params): Polyline3 | Fail {
  best = null
  bestCost = +inf
  memo = new Map<Key, number>() // stores best cost seen for key

  function rec(state, path):
    if state.cost >= bestCost: return

    if reachedTarget(state.curr, target):
      best = path.clone()
      bestCost = state.cost
      return

    key = makeKey(state)
    if memo.has(key) && memo.get(key)! <= state.cost: return
    memo.set(key, state.cost)

    candidates = generateCandidates(state, slices, params)
    for q in candidates:
      if !hardConstraintsOk(state.prev, state.curr, q): continue
      if !modeConstraintsOk(state, q): continue
      if !forwardCheck(q): continue

      rec(nextState(state, q), path + [q])

  rec(initState(start), [start])
  return best ?? Fail("no feasible ramp found", diagnostics)
}
```

### 3.6 Forward checking (major pruning)
Implement these *early*:

- **Component survivability**: if the connected component containing `curr` does not exist (or disconnects) at deeper slices, prune.  
  (Precompute component mapping or simply check that at least one feasible component remains that overlaps a “reachable corridor”.)
- **Spiral monotonicity**: if spiral angle decreases beyond tolerance, prune.
- **Switchback turn bulbs**: if attempting reversal outside a bulb, prune.

### 3.7 Switchback turn bulbs (implementation)
A “turn bulb” is a region at a given elevation slice where a U-turn of radius `minRadius` and corridor width can fit.

Start simple:
- user draws/chooses turn points (or zones) on a plan view
- each becomes a circular/oval polygon expanded by `(minRadius + corridorHalfWidth + margin)`
- bulb feasibility check: bulb polygon must be subset of `freePoly(z_k)` (approx by sampling points on bulb boundary).

Then automate later:
- detect wide-enough areas in `freePoly` using distance-to-boundary field and pick local maxima.

---

## 4) Domain-based slopes and clearance (how to integrate)

### 4.1 Minimal viable approach (recommended first)
- Use **constant clearance** (conservative max over domains) when building `freePoly`.
- Domain slopes remain relevant for **bench surfaces** and overall pit geometry, not ramp solving.

### 4.2 Improved approach (phase 2)
Add spatially varying clearance:
- Create a 2D grid over each slice.
- For each cell, compute:
  - inside pit?
  - distance to pit boundary
  - nearest boundary segment’s domain → `domainMargin`
  - cell is feasible if `distance ≥ baseClearance + domainMargin`
- Extract polygon contours from feasible mask → `freePoly`

This also gives you a distance field for candidate ranking (“stay away from boundaries”).

---

## 5) Optional Stage: Economic optimization (ILP/MIP) integration

If you keep an ILP stage, define it so it **does not conflict** with the centerline solver.

### 5.1 Recommended split of responsibilities
- ILP finds an economically optimal pit shell (or pushback) **without** detailed ramp geometry.
- Ramp solver finds a feasible ramp within that shell.
- If ramp fails, feed back a **space reservation constraint** and re-solve ILP.

### 5.2 Space reservation constraint (practical)
Instead of modeling the ramp as a one-block path, reserve a “ramp corridor allowance” by:
- requiring additional extraction of a corridor band along selected sectors/benches, or
- penalizing blocks that would choke off connectivity in `freePoly`.

This avoids an overly rigid ILP ramp model that later becomes invalid after smoothing.

---

## 6) CAD generation & integration instructions

### 6.1 From centerline → ramp corridor strings
Given centerline `C(s)`:

1. Resample at fixed station spacing (e.g., 2–5 m).
2. Compute local tangent and normal in plan.
3. Generate left/right edge polylines:
   - `L = C + n * corridorHalfWidth`
   - `R = C - n * corridorHalfWidth`
4. Build ramp surface as a ruled surface between L and R, assign elevation from centerline grade (or project to designed ramp plane).

### 6.2 Bench strings and wall surfaces
- Generate toe/crest strings per bench.
- Triangulate between strings to form bench faces and berms.
- Ensure consistent winding and normals.

### 6.3 Pit-to-topography stitching (avoid vague “CSG subtraction”)
Preferred robust method:
- Trim topo surface by crest boundary (plan projection).
- Insert pit surface patch inside the trimmed region.
- Stitch edges (tolerance-based) and validate watertightness if required.

Use solid booleans only if your CAD kernel is stable with large triangulated solids.

---

## 7) Public module API (what you expose to the CAD app)

### 7.1 Core service entry point
```ts
type SolveRequest = {
  benchLoops: BenchLoop[],
  blockModel: BlockModel,
  domainRules: Record<string, DomainRules>,
  rampParams: RampParams,
  start: Vec3,
  targetZ: number,
  modeOptions: { spiral?: {...}, switchback?: {...} },
  noGo?: NoGoDefinition[],
}

type SolveResult = {
  centerline: Polyline3,
  corridor?: { left: Polyline3, right: Polyline3 },
  diagnostics: Diagnostics,
  validation: ValidationReport
}

function designPitRamp(req: SolveRequest): SolveResult | SolveFailure
```

### 7.2 Diagnostics (must-have)
Include:
- why solver failed (first violated constraint category)
- slice index where failure occurred
- stats: nodes expanded, prunes by category, time spent
- best partial path length and depth reached

This is essential for production usability.

---

## 8) Validation suite (automated)

### 8.1 Geometry checks
- Centerline inside `freePoly` at all sampled points
- Corridor (offset edges) inside `pitPoly` (or inside offset free volume)
- Grade distribution: min/max/percentiles
- Radius distribution: detect segments below `Rmin`
- Self-intersection test (plan view) for centerline and corridor boundaries
- Switchback landing length and bulb containment (if enabled)

### 8.2 Deterministic unit tests
- Synthetic pits: circle pit (spiral should succeed), rectangle pit (switchback should succeed)
- Narrow pit: must fail with correct diagnostic
- Domain clearance: increases should reduce feasible region (monotonic behavior)

### 8.3 Regression harness
Store golden outputs (centerline stations) for a few reference datasets; tolerate small numeric drift.

---

## 9) Implementation roadmap (incremental milestones)

### Milestone 1 — Slices + feasibility polygons
- Implement slice generation from bench strings
- Implement polygon offset (erosion) and point-in-polygon
- Visual debug rendering in CAD (draw freePoly outlines)

### Milestone 2 — Basic spiral solver (no domain-varying clearance)
- Implement backtracking/beam solver
- Implement spiral guide curve (offset pit boundary)
- Output centerline polyline

### Milestone 3 — Switchback mode
- Add turn bulb definition + enforcement
- Add landing segment rules (length, optional reduced grade)

### Milestone 4 — Constrained smoothing + fillets
- Add arc replacement for corners
- Revalidation loop to ensure constraints remain satisfied

### Milestone 5 — Corridor + CAD surfaces
- Generate corridor edges, ramp surface
- Stitch pit surface to topo

### Milestone 6 — Domain-varying clearance + feedback loop to ILP
- Add distance field per slice
- Add variable clearance and “retry with relaxed/adjusted” strategies
- Add ILP space reservation loop if required

# agents.md — Pit Design PoC from Ultimate Pit String (Python + Streamlit + DXF TIN Topography)

This document guides coding agents (Codex/Gemini/Claude/etc.) to implement a Proof-of-Concept (PoC) application where the **core requirement is pit design derived from an Ultimate Pit (UP) string**. It reflects the decisions and constraints discussed in this session, corrected to emphasize **pit design first** (ramps are optional/secondary).

---

## 1) Core Goal (PoC)

Build a **server-side Python** application with a **Streamlit** UI that:

1. Imports **topography from DXF** containing **TIN triangles** on a **known/hard-coded layer**.
2. Loads an **Ultimate Pit (UP) string** (a polyline / “string” representing the ultimate pit limit).
3. Uses the UP string as the controlling geometry to generate a **pit design**:
   - Produce **bench-by-bench pit strings** (crest/toe outlines) derived from the UP string.
   - Apply configurable geotechnical parameters (bench height, batter angles, berm widths, slope sectors, etc. as available).
   - Output a coherent **3D pit surface representation** (at least bench strings; optionally a triangulated surface).
4. Provides a **simple interactive 3D viewer** (rotation/zoom/pan sufficient).
5. Allows the user to select a **point index along the UP string** and highlights it in 3D for intuitive orientation/debug (no 3D picking required).
6. Exposes export options (bench strings as CSV/JSON; optional DXF export later).

> Ramp design may exist later, but it is **not the core requirement** for this PoC. The PoC’s “solver” is a **pit design generator** from the UP string.

---

## 2) Non-Goals (out of scope for PoC)

- Full CAD-grade editing, snapping, or interactive drafting.
- Browser-only execution (Streamlit is client-server).
- Full geotechnical rigor (e.g., inter-ramp slope analytics, kinematic checks) beyond configurable geometric rules.
- Complex constraint optimization of pushbacks / phase design (future).
- Universal DXF handling (we assume a known layer for TIN triangles; UP string may come from JSON or be hard-coded).

---

## 3) Key Concepts & Terminology (align the team)

- **Ultimate Pit (UP) string**: a polyline representing the final pit limit (commonly in plan view). In this PoC we treat it as the boundary control from which benches are derived.
- **Bench**: a vertical interval (bench height `BH`) with a horizontal catch berm, defined by slope/batter angles and berm widths.
- **Crest / Toe strings**:
  - Crest: the top edge of a bench face.
  - Toe: the bottom edge of a bench face.
- **Design from UP**: generate a set of nested outlines by “offsetting” the pit boundary per bench, based on geometry rules.

---

## 4) High-Level Architecture

### 4.1 Components

**A) Streamlit UI**
- Dataset controls: load sample DXF topo + sample UP string (hard-coded path).
- Parameters: bench height, overall slope or batter angle, berm widths, design depth/target elevation, optional sector angles.
- UP string point-index selection + highlight (debug/intuition).
- “Generate pit design” button.
- 3D viewer + exports + diagnostics.

**B) Data Ingest / Preprocessing**
- DXF topo (TIN triangles) import using `ezdxf`.
- UP string load (JSON/CSV/hard-coded). Optionally DXF polyline import later, but not required now.

**C) Pit Design Engine (core)**
- Converts UP string to a valid planar polygon.
- Generates bench outlines down to target depth by repeated, parameterized offsets:
  - inward offsets for descending benches
  - manage berm creation and bench face projection
- Handles geometry robustness (self-intersections, MultiPolygons) and provides warnings/fallbacks.

**D) Visualization**
- Plot topography TIN mesh.
- Plot UP string and highlighted point index.
- Plot generated bench crest/toe strings per elevation (lines).
- Optional: generate a pit surface mesh from bench strings for nicer 3D rendering.

**E) Export**
- Bench strings per elevation (crest/toe) as:
  - CSV (x,y,z, bench_id, string_type)
  - JSON
- Optional later: DXF export of generated strings.

---

## 5) Tech Stack (agreed direction)

### 5.1 Core
- Python 3.10+
- Streamlit (UI)
- Plotly (3D visualization)
- NumPy (arrays, math)

### 5.2 Geometry
- **Shapely 2.x** (GEOS): robust 2D polygon operations (buffer/offset, union, validity checks)
- Optional: **pyclipper** for robust polygon offsetting (esp. large offsets, sharp angles, numeric stability)

### 5.3 DXF Import (Topography)
- **ezdxf**: open source, recommended, reads DXF TIN triangles from known layer

---

## 6) Input Data Model

### 6.1 Ultimate Pit String (UP)
Representation (minimum):
- `name: str`
- `points: Nx(2 or 3)` (x,y) or (x,y,z)

Assumptions:
- If z not provided, treat UP as plan-view boundary at a reference elevation (e.g., topo intersection or a user-specified reference).
- UP must form a closed loop for polygon creation; if not closed, close it.

### 6.2 Topography DXF (TIN)
- DXF path is hard-coded (PoC)
- TIN triangles are `3DFACE` on a hard-coded layer name `TIN_LAYER`

### 6.3 Design vertical limits
- Choose one:
  - `target_elevation` (absolute)
  - `target_depth` (relative)
  - `num_benches`

---

## 7) Core Algorithm: Pit Design from UP String (Primary Requirement)

### 7.1 Overview
Given:
- UP polygon in XY
- Bench height `BH`
- Berm width `BW` (catch berm)
- Face angle / batter `θ` (degrees) OR slope ratio
- Optional: sector-based parameters (different θ/BW by azimuth)

We generate a sequence of bench polygons:
- `P0 = UP polygon` at `z0` (top design elevation reference)
- For each bench `k = 1..K`:
  - Compute horizontal set-back for bench face:
    - `H_face = BH / tan(θ)` (if θ measured from horizontal)
      - If θ is from vertical, adjust accordingly (`H_face = BH * tan(θ_from_vertical)`).
  - Add berm width:
    - `H_total = H_face + BW` (or berm applied separately depending on conventions)
  - Generate next bench polygon:
    - `Pk = inset(P_{k-1}, H_total)` using robust offsetting
  - Assign elevation `z_k = z_{k-1} - BH`

Outputs:
- Crest strings: `P_{k-1}` at z_{k-1}
- Toe strings: `Pk` at z_k (depending on how you define crest/toe)
- Optionally produce both and label clearly.

> Important: Define and document angle conventions early. Many mining contexts define batter angle from horizontal (e.g., 70°) while some define slope angle differently.

### 7.2 Geometry robustness requirements
- `inset()` can produce:
  - empty geometry (design exhausted / too narrow)
  - MultiPolygon (split pit due to concavities)
- PoC handling:
  - Prefer largest-area polygon as main pit continuation, and log warning
  - Or keep all components and treat as multiple pits (more complex)
- Always validate polygons:
  - If invalid: attempt `buffer(0)` fix
  - If still invalid: fail with readable error

### 7.3 Sector-based slopes (optional but recommended for realism)
If needed, allow slope parameters vary by azimuth:
- Divide UP boundary into sectors (e.g., 0–90°, 90–180°, etc.)
- Apply different offset distances locally
- Implementation complexity increases substantially (true variable offset)
- PoC approach:
  - Start with uniform parameters
  - Add sectorization later as Phase 6/7

### 7.4 Linking to Topography (optional for PoC)
Two typical uses:
- Determine `z0` from topo: sample topo elevation at UP points and take median/mean
- Clip pit design to topo surface for visuals (not required for bench strings)

PoC guidance:
- Keep pit design vertical reference simple:
  - user inputs `z0` OR
  - derive `z0` from topo sampling at UP vertices (nice-to-have)

---

## 8) Visualization Requirements (Plotly 3D)

### 8.1 Must-have
- Topography mesh (TIN) in muted color with opacity.
- UP string polyline (bold).
- Highlighted UP point at selected index (large marker; optional vertical flag).
- Generated benches:
  - Render each bench polygon as a closed 3D line at its elevation
  - Use color ramp by depth

### 8.2 Nice-to-have
- Generate pit shell mesh by triangulating between successive bench polygons (lofting).
  - PoC can omit this; lines are acceptable and faster.

### 8.3 Performance
TIN can be huge:
- Provide `max_faces` cap for visualization
- Offer “show topo” toggle
- Provide pit-area cropping:
  - compute bbox of UP polygon and pad
  - only keep triangles whose centroid falls in bbox

---

## 9) DXF Topography Import (TIN triangles)

### 9.1 Requirements
- Use `ezdxf`.
- Filter entities by **hard-coded layer name**.
- Extract TIN triangles (primarily `3DFACE`).
- Deduplicate vertices with tolerance.
- Provide stats (faces loaded, skipped, bbox).

### 9.2 Error handling
- If layer not found / no triangles: show readable error and allow pit design to continue without topo (optional toggle).

---

## 10) User Interaction: UP string point index selection (debug/intuition)

Even though pit design is the core, keep the index selection feature:

- UI:
  - dropdown for string name (if multiple)
  - slider + numeric input for index
- Show:
  - x, y, z (if available)
  - station (cumulative distance)
- Visualization:
  - highlight marker
  - optional label “Index: i”

Purpose:
- Helps users confirm the UP string is loaded/oriented correctly.
- Aids debugging of sector-based parameters later.

---

## 11) Incremental Development Plan (agents must follow)

### Phase 0 — Skeleton
- Create Streamlit app scaffolding and module layout.
- Show placeholder Plotly 3D axes.

Acceptance:
- App runs.

### Phase 1 — Load + display UP string (core)
- Load UP string from JSON/hard-coded points.
- Ensure closed loop.
- Plot UP polyline in 3D (use constant z or supplied z).
- Add index selector + highlight.

Acceptance:
- Index highlight works and updates live.

### Phase 2 — Generate pit benches from UP (core)
- Implement uniform-parameter inset design:
  - inputs: BH, θ, BW, num_benches OR target elevation
- Output bench polygons per level.
- Visualize bench outlines as 3D lines.

Acceptance:
- Bench strings appear nested and descend as expected.
- Clear reporting if inset collapses/empties.

### Phase 3 — Export bench strings (core)
- Export bench outlines:
  - CSV rows: bench_id, string_type, x, y, z, point_order
  - JSON structure with arrays

Acceptance:
- Downloads work and data is structured.

### Phase 4 — Import + display DXF topo TIN (important)
- Implement DXF TIN import (ezdxf, hard-coded layer).
- Render topo mesh (with decimation cap).

Acceptance:
- Topo visible in 3D; app remains responsive.

### Phase 5 — Align pit design with topo (optional but valuable)
- Sample topo at UP vertices to infer z0 (optional).
- Add toggle: “Use topo-derived z0” vs manual z0.

Acceptance:
- Bench elevations reasonable relative to topo.

### Phase 6 — Robustness improvements (recommended)
- Handle MultiPolygons:
  - choose largest component OR keep multiple with warning
- Fix invalid polygons.
- Improve inset stability:
  - consider pyclipper if needed

Acceptance:
- Works on real UP strings with concavities.

### Phase 7 — Sector-based slopes / berms (if needed)
- Add ability to specify slope/berm by azimuth sector.
- Start with coarse sectors (N/E/S/W).
- Provide debug visualization of sectors on UP boundary.

Acceptance:
- Different wall geometries reflect sector settings.

---

## 12) Module Responsibilities (recommended)


Key separation:
- `pit_design.py` must be the central “engine”.
- `viz.py` must not contain business logic; it only visualizes provided geometry.

---

## 13) Pit Design Engine — Detailed Behavior Spec

### 13.1 Inputs
- `up_points` (Nx2 or Nx3)
- `z0` (float) — top reference elevation (manual or derived)
- `BH` bench height (float)
- `theta` batter/face angle (float) with clearly defined convention
- `BW` berm width (float)
- Termination:
  - `num_benches` OR `target_elevation`

### 13.2 Outputs
- `benches`: list of objects:
  - `bench_id`
  - `z_crest`
  - `crest_polygon` (XY ring)
  - `z_toe`
  - `toe_polygon` (XY ring)
- `diagnostics`:
  - offsets used
  - areas per bench
  - warnings for MultiPolygon/empties/fixes

### 13.3 Inset function
Must:
- take Polygon/MultiPolygon
- inset by distance `d` (positive distance meaning inward)
- return:
  - Polygon/MultiPolygon
  - plus warnings if split/empty

### 13.4 Angle conventions (must be explicit)
Agent must implement one and document it in UI:
- Option A: `theta_from_horizontal` (common)
  - `H_face = BH / tan(theta)`
- Option B: `theta_from_vertical`
  - `H_face = BH * tan(theta)`

Provide helper text and sanity checks.

---

## 14) Caching Strategy (Streamlit)

Cache expensive operations:
- DXF TIN load + dedup + decimation
- UP string load
- Pit bench generation (keyed on parameters)

Use:
- `st.cache_data` for pure data transforms.

---

## 15) Diagnostics & Quality Controls

Must include:
- Basic bbox overlap check:
  - warn if UP bbox far outside topo bbox (likely coordinate mismatch)
- Polygon validity:
  - report invalid and attempt fix
- Termination:
  - if inset becomes empty before reaching target -> stop and report depth reached

Debug mode toggle:
- show inset distances per bench
- show areas and number of components (MultiPolygon)
- show timing

---

## 16) Acceptance Criteria (Pit Design PoC)

- [ ] Loads UP string and shows it in 3D
- [ ] Index selection highlights a UP point and shows coordinates/station
- [ ] Generates bench-by-bench pit strings from UP using bench geometry rules
- [ ] Visualizes nested bench outlines at correct elevations
- [ ] Imports DXF topo TIN from hard-coded layer and visualizes it (with performance cap)
- [ ] Exports bench strings to CSV/JSON
- [ ] Error messages are readable and helpful

---

## 17) Decision Summary (from conversation, corrected)

- **Primary requirement**: **pit design from the ultimate pit string**.
- Start point selection is by **index** on a known string, highlighted in 3D (no 3D picking).
- **Server-side Python** is preferred for robust geometry/math.
- Topography DXF contains **TIN triangles**, and the **layer name can be hard-coded**.
- Use open-source **ezdxf** for DXF import.
- Visualization is **simple** (rotate/zoom) using Plotly in Streamlit.

---
