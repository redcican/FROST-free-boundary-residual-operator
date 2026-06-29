# FROST benchmark — obstacle problem (steady free-boundary equilibrium)

The **well-posed, steady** free-boundary problem used to *develop and debug FROST's equilibrium
solve* before the time-dependent tumour merge. The reproductions showed the equilibrium/fixed-point
is the hard part of FROST (from-scratch local operators gave spurious fixed points); the obstacle
problem is the clean monotone testbed.

`python gen_obstacle.py [n]` → `obstacle.npy` (+ summary, preview).  `python viz_obstacle_2d.py`
→ `obstacle_preview.png`; `python viz_obstacle_3d.py` → `obstacle_3d.png`.  Run single-threaded
(`OPENBLAS_NUM_THREADS=1 …`).

## Why it fits FROST
The obstacle problem is solved by a **constrained fixed point**
`u ← max(χ, mean_neighbours(u))`, `u=0` on the boundary — i.e. *exactly* the kind of equilibrium
FROST's DEQ targets, but **steady and monotone (guaranteed to converge)**. So FROST's equilibrium
machinery (DEQ solve + implicit-diff training + contraction) can be built and verified here first.

The **free boundary is the contact set boundary** `Γ = ∂{u = χ}`. With an obstacle made of two
Gaussian bumps, the contact set has **two components** when the bumps are far/low and **one** when
they are close/high — a *steady* topology family (the kind a single diffeomorphism cannot track).

## Model
Fixed box `D = [-1,1]²` (128×128). Obstacle `χ = h₁·G(·−c₁) + h₂·G(·−c₂) − base` (two Gaussian
bumps minus a threshold, so `χ>0` only near the bump cores). Solve the obstacle problem (least
super-harmonic majorant of `χ` with `u=0` on `∂D`) by **projected red-black SOR**. Contact set
`{u≈χ, χ>0}`; `φ = signed_distance(contact)` (negative inside); `Γ = {φ=0}`.

Per sample varied: bump half-separation `d∈[0.16,0.46]`, heights `∈[0.45,0.75]`, width, threshold.
**Steady** — one equilibrium per sample (no time axis), unlike the tumour benchmark.

## Format (`obstacle.npy`, object array of dicts)
`u (128,128) f32` (membrane field), `chi (128,128) f32` (obstacle), `contact (128,128) bool`,
`phi (128,128) f32` (signed distance to Γ), `params [c1x,c2x,h1,h2,w,base]`, `n_components` (1 or 2
contact regions).

## Validation & visualization
80 obstacle solutions; contact-component counts in `obstacle_summary.json` (mix of 1 and 2).
- `obstacle_preview.png` — membrane field `u` + free boundary Γ for several samples (2 separate
  contact regions vs 1 merged).
- `obstacle_3d.png` — membrane `u(x,y)` as a height surface (rising over the bumps, flat where it
  contacts), with Γ overlaid in crimson; a 2×3 grid spanning 1- and 2-region contact sets. (3D here
  is field-as-height, since the problem is steady — not space-time like the tumour benchmark.)

## How FROST uses it
- **Debug the equilibrium solve (C1):** learn the map (obstacle/params) → `(u, φ)` as a DEQ fixed
  point — the same machinery as the tumour operator, but on a monotone, convergent steady problem
  where correctness is easy to check.
- The contact-set topology family (1↔2 components) also exercises the level-set representation in a
  steady setting before the time-dependent merge.
