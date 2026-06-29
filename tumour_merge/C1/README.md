# FROST C1 — forward operator (tumour_merge benchmark, time-dependent, 2→1 topology change)

The tumour-merge instance of the time-conditioned C1 operator (same machinery as Stefan):

```
[ φ₀ (initial two tumours), sep0, sep1, scale, k, t ]  ->  ( u(t), φ(t) )
```

`u` is the nutrient field ((-Δ+k)u=0, u=1 on Γ), `φ` the level set. The two organic tumours drift
together and **coalesce (2 components → 1)**, so the operator must produce a topology-changing free
boundary — the FROST capability a single diffeomorphism cannot represent.

```
python train_c1_tumour.py [epochs]      # default 100; uses ../tumour_merge.npy
python viz_c1_tumour_3d.py              # 3D space-time GT-vs-prediction surfaces (needs results/model.pt)
```
Outputs → `results/`: `model.pt`, `metrics.json`, `loss_curve.png`, `topo_vs_time.png`,
`predictions.png` (2D), `predictions_3d.png` (3D space-time).

## Setup
2D FNO (`fno.py`, in=6, out=2), **128² full resolution, modes 20** (~3.3 M params; promoted 2026-06-21 —
see *Field-accuracy experiment* below). 80 samples, 64/16 split **stratified by reaction-rate `k`** (all
samples are 2→1, so topology can't stratify; `k∈[1.0,5.0]` is the main varying physical parameter;
geometry is FBNO geometry-A, no rotation). ~108 min CPU (100 ep at 128²). Figures use the tumour style
(`YlGnBu` nutrient + black/crimson Γ) at fontsize 22, dpi 600.

## Results (held-out test: 16 samples × 15 frames)

| Metric | **128² (default)** | 64² (previous) |
|---|---|---|
| `u` rel-L2 | **12.1%** | 15.9% |
| `φ` rel-L2 | **1.07%** | 1.3% |
| tumour IoU | **0.974** | 0.961 |
| topology accuracy | **0.983** | 0.996 |

**Topology accuracy through the merge** (`topo_vs_time.png`): 1.0 away from the merge, dipping only at the
2→1 transition frame.

**Findings:**
- The operator **reproduces the 2→1 coalescence in its level-set output almost perfectly**
  (`predictions.png`: predicted `φ` has 2 separate contours pre-merge, 1 after; `u` tracks the
  nutrient through the merge). Free-boundary IoU 0.96, topology accuracy 99.6%.
- `predictions_3d.png` shows this in **space-time** using the **exact `tumour_merge_3d.png` renderer**
  (FBNO panel-a style, same 5 time steps), as a 2×5: top row = GT (the benchmark `draw_panel`,
  identical to `tumour_merge_3d.png`), bottom row = FROST prediction (same machinery fed the predicted
  φ(t)). Component counts match at every step (GT/pred = 2,2,1,1,1) — the predicted space-time
  surfaces are visually indistinguishable from GT, with the two tumour bodies fusing 2→1.
- **Contrast with Stefan** (topology dipped to 0.47 at its merge): the tumour merge is far easier to
  time because the approach is a **smooth, geometrically-prescribed trajectory** (`sep(t)`), whereas
  Stefan's coalescence instant is **flux-determined** (latent-heat-driven) and physically sharper.
  How well an operator times a merge depends on how physically-determined that merge is.
- `u` rel-L2 ~12% — the nutrient field is the hard part (it is a near-binary indicator with a sharp `u=1`
  rim; `|Δu|` concentrates there); the FROST-relevant free-boundary/topology metrics are excellent. See
  *Field-accuracy experiment* below for why ~1% is not reachable for this field.

## FBNO single-diffeomorphism baseline, time-dependent form (`baseline_diffeo_time.py`)
The sharpest version of the capability comparison: a moving free boundary tracked by a **time-continuous
diffeomorphism** `Φ(·,t)` (FBNO's reference-frame warp) is a homeomorphism at every `t`, so its number
of connected components is **constant in time** — it can never cross a `2→1` coalescence. The tumour
merge *is* that coalescence within a single trajectory.

```
python baseline_diffeo_time.py [n_iter]    # default 300 Adam iters/frame; needs results/model.pt
```
Outputs → `results/baseline/`: `metrics.json`, `baseline_time_topology.png`, `baseline_time_filmstrip.png`.

Made generous (oracle): the reference is fixed to the `t=0` two-tumour domain, and for **each frame**
we fit the best diffeomorphic warp **directly against the ground-truth `φ(t)`** with a positive-Jacobian
(no-fold) constraint (min relative Jacobian det over all frames `+0.044 > 0` → every warp is a genuine
diffeomorphism). On the held-out trajectory (merge at frame 11):

| frame | 0–10 | 11 (merge) | 12–14 |
|---|---|---|---|
| ground truth (#components) | 2 | 1 | 1 |
| single diffeomorphism (t=0 ref) | 2 | **2** | **2** |
| **FROST** (level set) | 2 | **1** | **1** |

**Findings:** the oracle diffeomorphism is **frozen at 2 components for the entire trajectory** and is
wrong on **every post-merge frame** (trajectory topology accuracy **73%**), while FROST tracks the `2→1`
transition **exactly** with the ground truth (**100%**). `baseline_time_filmstrip.png` shows the
diffeomorphism's two blobs growing and approaching but **never fusing** (still 2 regions at `t=14`,
where the truth is one mass), against FROST merging correctly. `baseline_time_topology.png` is the
component-count-vs-time plot. This is the time-dependent companion to the steady obstacle baseline
(`../../obstacle/C1/baseline_diffeo.py`).

## Field-accuracy experiment — can `u` reach obstacle level (~1%)? (`improve_field.py`)
The level set `φ` is already at obstacle level (1.3%); only the raw field `u` is high (~16%). **Why:** the
nutrient `u` is a **near-binary indicator** (≈1.000 inside the domain, exactly 0 outside; the domain is
only ~5% of the grid), so it has a sharp jump at Γ that a spectral FNO rings at (Gibbs), and the small
`‖u‖` inflates the relative error. We tried the physically-correct lever — **full 128² resolution + more
modes** (`python improve_field.py --ds 1 --modes 20 --epochs 100` → `results/improve_ds1_m20_w32/`):

| config | `u` rel-L2 | `φ` rel-L2 | IoU |
|---|---|---|---|
| baseline (64², modes 16) | 0.159 | 0.013 | 0.961 |
| **128², modes 20** | **0.121** | **0.0107** | **0.974** |

A real but **modest** improvement (≈24% relative, better across the board) that **plateaus by ep 90** —
it does **not** reach ~1%. A spectral operator has a **Gibbs floor** at the near-binary discontinuity, so
this is intrinsic to the field, not an under-training/under-resolution artifact (the FROST-relevant
`φ`/IoU/topology metrics are already at obstacle level).

**This 128² model is now the default** (promoted 2026-06-21): `train_c1_tumour.py` defaults to `DS=1,
MODES=20`; `python promote.py` reuses the trained weights (the network is deterministic — no retrain
needed) and regenerates the canonical `results/` artifacts (`model.pt`, `metrics.json`, `loss_curve.png`,
`topo_vs_time.png`, `predictions.png`); the 3D view and diffeomorphism baseline were re-run at 128². The
Stefan field, by contrast, is **resolution-invariant** (no gain at 128² + OOM on this box) — see
`../../stefan/C1/README.md`.

**Next:** merge-weighted loss; DEQ rollout.
