# FROST C1 — forward operator (channel, real CFD)

The channel **inverts** the other benchmarks: the baffle (free boundary) is the **input design**, and the
operator predicts the resulting flow fields. So C1-channel is the **differentiable forward model** that
C2 (FROST-Design) optimizes through:

```
G : (baffle, v) -> (C, P)        C = concentration, P = pressure, v = inlet velocity
```

```
python gen_channel_train.py [per_velocity]                 # Stage 1: 450 designs from the CFD CSVs
python train_c1_channel.py --rep {phi,gamma} --split {random,topo} [--epochs 100]
python viz_compare.py                                      # operator_comparison.png
```
Run with ≤4 BLAS threads (`TORCH_THREADS=4 …`) — 8 threads OOM'd the ~7 GB box mid-training.

## Setup
2D FNO (`fno.py`, in=4, out=2, ~2.1 M params), **100×50** grid (downsampled from 201×101). 450 designs,
topology {1:169, 2:214, 3:67}. Two studies via flags:
- **`--rep`** = baffle encoding: `phi` (level set `[φ,∂xφ,∂yφ,v]`, FROST) vs `gamma` (density `[γ,∂xγ,∂yγ,v]`, NTO-style) — same channel count, only the encoding differs.
- **`--split`** = `random` (stratified 80/20) vs `topo` (train 1–2 baffles, **test 3-baffle** = topology extrapolation).

## Results (2×2 study, held-out test)

| config | C rel-L2 | P rel-L2 | C by topology (1 / 2 / 3) |
|---|---|---|---|
| φ random | 0.164 | 0.048 | 0.053 / 0.206 / 0.317 |
| γ random | 0.164 | 0.046 | 0.063 / 0.196 / 0.323 |
| φ topo→3 | 0.254 | 0.081 | — / — / 0.254 |
| γ topo→3 | 0.263 | 0.071 | — / — / 0.263 |

**Findings (honest):**
1. **A2 — φ vs γ encoding: essentially equal** (C 0.164 = 0.164; φ marginally better only on topology
   extrapolation, 0.254 vs 0.263). The level-set conditioning hypothesis **did not pan out** here — for
   this operator/data the smooth distance field gives no real advantage over NTO's raw density. Reported
   as a clean negative result.
2. **A3 — topology generalization works:** trained on 1–2 baffles, the operator predicts unseen
   **3-baffle** flow at C rel-L2 **0.25** — comparable to its in-distribution 2-baffle accuracy (~0.20).
   It extrapolates to more-complex topology without catastrophic failure (real-CFD support for FROST's
   "handles topology" claim).
3. Accuracy degrades with baffle count (1c 5% ≪ 2c 20% ≪ 3c 32%); P (≈5%) is far easier than C (≈16%) —
   pressure is smooth/global, concentration has sharp plumes. Both expected and physical.

Figures: `results/<cfg>/predictions.png` (C/P GT/pred/|error|, one design per topology),
`results/operator_comparison.png` (A2 + A3 bar charts), per-config `loss_curve.png`.

The operator's main role is as the **frozen differentiable forward model for C2** — see `../C2_inverse/`.
