# FROST benchmark — channel (turbulent-flow design, solid-fluid free boundary)

A **steady, real-CFD** free-boundary benchmark adapted from the Neural-Topology-Optimization (NTO)
dataset. The free boundary is the **solid–fluid interface** of a channel baffle, the field is the
transported **concentration** `C`, and the baffle **topology varies across designs** (1, 2, 3+
disconnected pieces) — the design space FROST's topology-optimization loop (C2) explores.

`python gen_channel.py [n]` → `channel.npy` (+ summary, `channel_preview.png`). 2D only — the channel
field is naturally read as a planar map, so no 3D height-surface view is kept for this benchmark.

## How this adapts the NTO data
NTO (`../../neural-topology-optimization/`) optimizes a 2D turbulent channel (0.02×0.01 m): a
DeepONet surrogate predicts concentration/pressure for a baffle layout, and a design loop reshapes the
baffle to flatten the outlet concentration variance vs pressure drop. Its dataset is **1074 CFD
designs** on a regular **201×101** grid, each storing the **fluid fraction `γ∈[0,1]`** (γ=0 solid baffle,
γ=1 fluid), concentration `C`, and pressure `p`.

FROST reframing ("modify the NTO data to our problem"): represent each design as a FROST
**(field, level-set)** pair — the **solid–fluid interface `{γ=0.5}` is a free boundary**, encoded as a
signed-distance level set `φ` (negative inside solid), paired with `u = C`. This is the **steady,
real-CFD analogue of the `obstacle` benchmark** (topology varies across samples), but with genuine
turbulent advection–diffusion physics instead of a synthetic Poisson problem — and it is the dataset
the C2 design experiment runs on.

## Model / processing
Each CFD CSV (`udm-3`=γ, `uds-0-scalar`=C, `total-pressure`=p) is rasterized to the native 201×101
grid (the nodes form a perfect structured grid — no interpolation). `P = p/2000 + 0.26` (paper's
normalization). Free boundary `Γ = {γ=0.5}`; solid mask `γ<0.5`; `φ = signed_distance` (negative inside
solid). Degenerate designs (solid fraction outside `(0.02, 0.40)` — empty or all-solid) are skipped.
Samples are stratified across inlet velocities `v ∈ {0.1,…,0.9} m/s`.

## Format (`channel.npy`, object array of dicts)
`phi (201,101) f32` (signed distance to Γ), `u (201,101) f32` (concentration C), `P (201,101) f32`
(normalized pressure), `gamma (201,101) f32` (fluid fraction), `mask (201,101) bool` (solid baffle),
`params [v]` (inlet velocity), `n_components` (number of solid baffles), `source` (original CSV name).

## Validation & visualization
96 designs; topology distribution (solid-baffle components) and velocity counts in
`channel_summary.json` (here: components {1:40, 2:44, 3:12}, ~11 designs per velocity).
- `channel_preview.png` — concentration `C` (jet) + free boundary `Γ` (white) for four designs
  spanning 1/2/3 baffles and low/high velocity: the inlet plume is deflected by the baffles, and the
  solid-baffle topology (1→2→3 components) varies across designs.

## How FROST uses it
- **Steady forward operator (C1):** learn `(γ / v) → (C, φ)` — the field + free-boundary map, like the
  `obstacle` benchmark but on real turbulent CFD with cross-design topology variation.
- **Design / inverse (C2):** this *is* NTO's design space. With a frozen differentiable FROST operator
  as the forward model, the baffle (a level set) is optimized for outlet-concentration uniformity vs
  pressure drop — FROST-Design replacing NTO's heuristic active-learning loop, with the free-boundary
  topology free to change during optimization.

## Provenance
Built from the NTO CFD dataset (Kou et al., *Advanced Science* 2025). The paper's main result
(**Figure 5**) is reproduced from the same data in
`../../neural-topology-optimization/reproduce/` (`figure5_direct_data.png`).
