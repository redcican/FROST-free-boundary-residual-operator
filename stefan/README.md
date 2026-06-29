# FROST benchmark — Stefan solidification (2D, grain coalescence)

A **time-dependent, flux-driven** free-boundary benchmark: several solid grains grow into a melt by
the **Stefan condition** and physically **merge** (N→1). Adapted from FBNO's `stefan_data`, with the
2D front speed **calibrated to FBNO's measured 1D physics** (#1) and **validated** against the 300
FBNO interface paths in the no-coalescence limit (#2).

Pipeline (run single-threaded, `OPENBLAS_NUM_THREADS=1 …`):
1. `python calibrate_from_fbno.py` → `fbno_calibration.json` + `fbno_calibration.npz` (reads FBNO `stefan_data.npy` once)
2. `python gen_stefan.py [n]` → `stefan.npy` (+ summary, preview)
3. `python viz_stefan_2d.py` → `stefan_preview.png`;  `python viz_stefan_3d.py` → `stefan_3d.png`
4. `python validate_vs_fbno_1d.py` → `stefan_vs_fbno_1d.png`

## How this adapts FBNO's `stefan_data`
FBNO's Stefan data (`../../FBNO/data/stefan_data`) is a **1D** melting front `s(t)` — a single moving
point, driven by the Stefan flux condition `ds/dt = -(k/ρL)·∂T/∂x|ₛ + cooling`. A 1D front **cannot
change topology**, and FBNO's single diffeomorphism cannot represent coalescence.

We keep the **same Stefan physics** — the free boundary is the `T=0` (melting) isotherm, its velocity
set by the heat-flux jump ÷ latent heat — but **lift it to a 2D fixed grid** via the **enthalpy
method** (itself FROST's "level set on a fixed background box" idea). Several solid seeds then grow
and **coalesce**, so the number of solid components drops over time = the topology change FBNO can't do.

This makes Stefan distinct from the other FROST benchmarks: the front is **flux-driven** (Stefan
condition + latent-heat recalescence, two-phase temperature field), not growth-rate-driven by a mean
field (`tumour_merge`) and not geometric (`obstacle`). It is the **physically-emergent** topology
change — grain coalescence during solidification is real, unlike the artificial tumour split we dropped.

## Model (enthalpy method, c=1, melting temp T=0)
`H = T + L·f_liquid`:
- `H < 0` → solid, `T = H`, `f_s = 1`
- `0 ≤ H ≤ L` → mushy (interface), `T = 0`, `f_s = 1 − H/L`  ← latent-heat buffer
- `H > L` → liquid, `T = H − L`, `f_s = 0`

Heat equation `∂H/∂t = D·∇²T`, explicit FTCS on a fixed 128×128 box `[-1,1]²` (`r = D·dt/h² = 0.2`).
The melt starts at the melting point (`T=0`, `H=L`); `n` **cold solid nuclei** (Dirichlet `H = T_cold`)
are the heat sinks. The melt around them loses heat, the `T=0` front advances, grains merge. Outer box
boundary insulated (Neumann). This is the **one-phase Stefan problem** on a fixed grid.

Free boundary `Γ = {T=0} = {f_s = 0.5}`; `φ = signed_distance` (negative inside solid).

Per sample varied: `n_seeds ∈ {2,3,4}`, seed inter-edge `gap` and latent heat `L`, with the **Stefan
number `St = |T_cold|/L` and the gap drawn from FBNO-calibrated ranges** (see below). The run length
is set to **2× the steps-to-coalescence**, so the N→1 merge lands mid-window. 15 frames per run.

## #1 Calibration to FBNO's 1D physics
`calibrate_from_fbno.py` reads FBNO `stefan_data.npy` (300 × 1D) and extracts, in a shared
dimensionless frame (length = fraction of characteristic length; time = τ∈[0,1] over the run):
forward front excursion (p50 `0.082`, p90 `0.185`, max `0.255`), normalized speed envelope
(p5/p50/p95 = `-0.276/0.059/0.458`), and an effective **Stefan number** per excursion quantile
(median `0.18`; active fronts p75/p95 = `0.69`/`1.73`). FBNO's *median* run is near-static; the fronts
that actually move and would coalesce live in the upper quantiles, so we calibrate to that **active
regime**. The generator then draws `St ∈ [0.69, 1.60]` and seed `gap ∈ [0.24, 0.37]` (≈ 2× FBNO
forward excursion, so each approaching front travels a FBNO-like distance to merge). 1D temperature
fields are **not** reused as 2D samples (dimensionally incompatible) — only the physics is.

## Format (`stefan.npy`, object array of dicts)
`phi (15,128,128) f32` (signed distance to front), `u (15,128,128) f32` (temperature T),
`mask (15,128,128) bool` (solid region), `boundary_components (15,)` (per-frame front polylines in
`[-1,1]` for 3D rendering), `seeds (n,2)`, `params [n_seeds, L, T_cold, seed_radius, D, gap]`,
`St` (Stefan number), `fo (15,)` (Fourier number per frame), `r_eq (15,)` (area-equivalent front
radius), `n_components (15,)` (solid components per frame, decreasing), `merge_step` (first frame at 1 comp).

## Validation & visualization
80 solidification runs, each a genuine N→1 coalescence (`n_components[0] ≥ 2`, `n_components[-1] = 1`);
seed-count distribution in `stefan_summary.json`.
- `stefan_preview.png` — temperature `T` + front `Γ = {T=0}` over 5 times: separate cold grains grow,
  neck together, and fuse into one solid (cold seed cores stay visible). The standalone visualizer
  prefers a non-clipped 3/4-seed coalescence and uses one shared temperature colorbar.
- `stefan_3d.png` — flat **2D front-evolution** figure (replaces the earlier 3D space-time render):
  *(left)* the free boundary `Γ(t)={T=0}` overlaid for every timestep, coloured by time — the separate
  grain fronts sweep outward and the contours **coalesce N→1**; *(right)* the solidification
  arrival-time map (frame index each point first froze) — grains as nested time bands, the merge
  showing as the ridge where two fronts meet (crimson = final front).

## #2 Validation vs FBNO's 1D Stefan (`stefan_vs_fbno_1d.png`)
A **single isolated seed** (no coalescence) is the degenerate slice that should behave like FBNO's 1D
front. `validate_vs_fbno_1d.py` runs it at three calibrated Stefan numbers and checks, against the
300 FBNO interface paths (from `fbno_calibration.npz`, so the 2.3 GB file is never reloaded):
- **Stefan √Fo similarity:** `r − r₀ ∝ √Fo` with **R² = 0.999** — our enthalpy front is a correct
  one-phase Stefan front; higher St → smaller Fo to the same travel (faster front).
- **Excursion / speed match:** peak radial travel reaches FBNO's p95 excursion (`0.215`), and **79%**
  of frames' normalized speed lie inside FBNO's measured `[p5,p95]` band. Our monotonic similarity
  front rides the *upper (active)* part of FBNO's band — expected, since FBNO's fronts oscillate
  (cooling) and deviate from pure similarity.
- Realized over the 80-sample set: per-front merge travel **[0.12, 0.18]**, inside FBNO's forward
  excursion p55–p90 **[0.08, 0.19]** (`stefan_summary.json → calibration`).

## How FROST uses it
- **Topology-change operator (C1):** learn (seed layout / `L` / `T_cold`) → `(T-field, φ(t))` as a
  level-set/DEQ rollout where the front velocity obeys the **Stefan flux condition** — the genuinely
  flux-coupled free boundary, complementing `tumour_merge` (growth-driven) and `obstacle` (steady).
- The N→1 coalescence is the **physically-emergent** topology change that motivates FROST over FBNO's
  single diffeomorphism. (Future extension: anisotropic Gibbs–Thomson → dendritic tip-splitting.)
