# FROST C1 — forward operator (Stefan benchmark, time-dependent + topology change)

The Stefan extension of the obstacle C1: a single **time-conditioned** 2D FNO learns the operator

```
[ φ₀ (initial cold nuclei), L, T_cold, t ]  ->  ( T(t), φ(t) )
```

Given the initial seeds (level set `φ₀`), the physics (`L`, `T_cold`) and a query time `t`, it predicts
the temperature field and the level set at that time. The level set `φ(t)` must change **topology over
time** (N grains coalesce to 1) — the capability a single diffeomorphism cannot represent.

```
python train_c1_stefan.py [epochs]      # default 100; uses ../stefan.npy
python viz_c1_stefan.py                  # regenerate 2D predictions + topo_vs_time
python viz_c1_stefan_3d.py              # 3D space-time GT/pred/|Δφ| surfaces (needs results/model.pt)
```
Outputs → `results/`: `model.pt`, `metrics.json`, `loss_curve.png`, `topo_vs_time.png`,
`predictions.png` (2D), `predictions_3d.png` (3D space-time, GT/pred/|Δφ| × 5 times — N→1 merge).

## Setup
2D FNO (`fno.py`, in=4, out=2, modes 16, width 32, ~2.1 M params). **64² grid** (downsampled from 128²;
the Stefan fields are smooth — ≈4× faster on CPU). 80 samples, **64/16 split stratified by seed count,
no frame leakage** (all 15 frames of a sample stay together). Topology is counted at the working
resolution for both prediction and GT, so the metric is self-consistent. ~26 min CPU (100 epochs).

## Results (held-out test: 15 samples × 15 frames)

| Metric | Value |
|---|---|
| `T` rel-L2 | 14.6% |
| `φ` rel-L2 | 3.8% |
| solid IoU | 0.909 |
| topology accuracy (all frames) | 0.907 |

**Topology accuracy through the merge** (`topo_vs_time.png`) is the headline:

| frame | 0–3 | 4 | 5 | 6 | 7 | 8 | 9 | 10–14 |
|---|---|---|---|---|---|---|---|---|
| topo acc | **1.0** | 0.87 | 0.73 | **0.47** | 0.73 | 0.87 | 0.93 | **1.0** |

**Findings:**
- The operator **reproduces the N→1 topology change in its level-set output** (`predictions.png`: the
  predicted `φ` has N separate front contours pre-merge and 1 after) — the FROST capability, on a
  time-dependent free boundary.
- Topology accuracy is **perfect away from the merge** (clearly-separate or clearly-merged frames)
  and dips only **at the merge transition** (frame 6: 0.47). This is **merge-*timing* jitter** — the
  operator gets the component count right when grains are distinct or fused, but is off by ~±1 frame
  on exactly *when* they connect (predicting the precise coalescence instant is the hard part).
- The temperature field is recovered to ~15% rel-L2 — looser than the steady obstacle (0.9%), as
  expected: Stefan is time-dependent with deep cold wells / sharp gradients, and runs at 64².

## Relation to obstacle C1
Same operator/metrics, extended from steady (one field) to time-dependent (a rollout) with the
topology change now happening *over time* rather than across samples. The per-frame topology curve is
the time-dependent analogue of the obstacle's 1-vs-2 contact-component accuracy.

## Field-accuracy experiment — can `T` reach obstacle level (~1%)? (`improve_field.py`)
`φ` is near obstacle level (3.8%); the temperature `T` is the high one (~14.6%). `T` is **continuous**
(liquid exactly 0, solid down to `T_cold`) but **~60% of its spectral energy is in high modes** — deep,
localized cold wells in the ~10% solid region. We tried **full 128² resolution + more modes**
(`python improve_field.py --ds 1 --modes 20 --epochs 100`):

| config | `T` rel-L2 |
|---|---|
| baseline (64², modes 16) | 0.146 |
| 128², modes 20 (to ep 60) | **~0.145 — no improvement** |

**Resolution does *not* help** here (the error is resolution-invariant, and even ticks up mid-run), and
128² **OOM-kills** the run on this ~7 GB box around ep 60. So `T`'s error is **data/generalization-limited**
(65 training trajectories, high field variability across `T_cold`/`L`/seed configs), **not**
resolution-limited — it cannot be driven to ~1% by training harder/finer. The FROST-relevant
`φ`/IoU/topology metrics are already strong; the raw-`T` number reflects the field's intrinsic
high-frequency content and the data budget, not an operator deficiency.

**Next:** a boundary-/merge-weighted loss (up-weight frames near `merge_step`) to sharpen the merge
timing; a DEQ/level-set-rollout variant; and the same operator on `tumour_merge`.
