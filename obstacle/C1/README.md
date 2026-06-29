# FROST C1 — forward operator (obstacle benchmark)

The first FROST **forward operator** experiment: a single neural operator learns the map
`χ → (u, φ)` — predicting both the membrane field `u` **and** the level set `φ` (signed distance to
the contact-set free boundary `Γ = ∂{u=χ}`) in one shot. The obstacle problem is the **steady,
monotone equilibrium**, so it is the clean first test of the operator before the time-dependent
benchmarks.

```
python train_c1_obstacle.py [epochs]      # default 400; uses ../obstacle.npy
python viz_c1_3d.py                        # 3D GT-vs-prediction surfaces (needs results/model.pt)
```
Outputs → `results/`: `model.pt`, `metrics.json`, `loss_curve.png`, `predictions.png`,
`predictions_3d.png`, `train.log`.
(Run with default threads — FNO is FFT/BLAS-heavy, unlike the single-threaded data generators.)

## Model
Compact 2D **FNO** (`fno.py`): input `χ` (+ 2 grid-coordinate channels) → output `(u, φ)`,
`modes=16, width=32, 4 layers` (~2.1 M params). Trained with relative-L² loss on `u` and `φ`
(AdamW + cosine schedule). Inputs/targets standardized by train statistics; metrics are denormalized.

## Protocol
- 80 samples, **stratified** 80/20 split by contact topology (so the held-out test set has both
  1- and 2-component cases).
- Metrics on the held-out test set:
  - `u_relL2`, `phi_relL2` — field and level-set accuracy;
  - `contact_IoU` — IoU of the predicted contact set `{φ<0}∩{χ>0}` vs ground truth;
  - `topology_acc` — fraction of test cases whose predicted contact set has the **correct number of
    components** (1 vs 2) — the free-boundary *topology* FBNO's single diffeomorphism cannot vary;
  - **head-to-head:** `φ-head` contact (explicit level set) vs the `u-threshold` baseline
    (read the contact set off the predicted field as `u−χ<tol`). Tests whether an explicit level-set
    output recovers the free boundary better than thresholding a predicted field.
  - per-topology breakdown (1-comp vs 2-comp).

## Why this is the right first C1
The obstacle equilibrium is monotone/convergent, so any failure is the *operator's*, not the
solver's — exactly the clean setting the benchmark was built for. It establishes the field +
level-set forward map and the free-boundary/topology metrics that the time-dependent operators
(`stefan`, `tumour_merge`) will reuse. The **DEQ / implicit-differentiation** form (the obstacle
problem *is* the projected fixed point `u ← max(χ, mean_nbr(u))`) is built in `deq.py` +
`train_c1_deq.py` — see the DEQ results section below.

## Results (400 epochs, ~19 min CPU, 64/16 split)
Held-out test (16 samples; full numbers in `results/metrics.json`):

| Metric | Value |
|---|---|
| `u` rel-L2 | **0.91%** |
| `φ` rel-L2 | **0.85%** |
| contact IoU — **φ-head** | **0.954** |
| contact IoU — `u`-threshold baseline | 0.832 |
| **topology accuracy** (1-vs-2 comp) | **100%** (16/16) |
| per-topology IoU (1-comp / 2-comp) | 0.934 / 0.961 |

**Findings:** (1) the forward operator recovers field + free boundary to ~1% with **100% topology
accuracy** on held-out instances, including every 2-component case — the free-boundary topology a
single diffeomorphism cannot vary; (2) the **explicit level-set head beats thresholding the predicted
field by +12 IoU points** (0.954 vs 0.832) — the concrete payoff of predicting `φ` directly.
See `results/predictions.png` (2D, a 1- and a 2-component held-out case — χ, u GT/pred, **|Δu|**,
φ GT/pred, **|Δφ|**, contact GT-vs-pred; error panels in `magma` with colorbars),
`results/predictions_3d.png` (3D: membrane surface `u(x,y)` GT vs predicted — Γ lifted onto each
surface, red = pred, blue = GT — plus an **absolute-error surface** coloured by `|Δu|`), and
`results/loss_curve.png`. Error concentrates near the contact rim / peaks (the free boundary), with
max `|Δu|` ≈ 0.003–0.004 on these cases.

## FBNO single-diffeomorphism baseline — the capability comparison (`baseline_diffeo.py`)
The result that makes the topology number land: **why** a single diffeomorphism (FBNO's
reference-frame warp) cannot do what the FROST level-set operator does.

```
python baseline_diffeo.py [n_iter]        # default 300 Adam iters per oracle fit; needs results/model.pt
```
Outputs → `results/baseline/`: `metrics.json`, `baseline_topology.png`, `baseline_qualitative.png`.

**The argument.** A diffeomorphism is a homeomorphism, so it **preserves the number of connected
components**. The obstacle contact set has **both** topologies in one family (2 separate regions when
the bumps are far/low; 1 merged peanut when close/high), so a single fixed reference template — which
a diffeomorphism can only stretch, never split or fuse — cannot cover the family.

**Why it is airtight (oracle + verified diffeomorphism).** For every held-out instance we fit the
*best* diffeomorphic warp of a fixed reference (the class medoid) **directly against the ground-truth
level set** — an oracle, hence an **upper bound on any learned diffeomorphism operator (FBNO
included)**. The warp is a band-limited control-grid displacement with a positive-Jacobian (no-fold)
constraint; the **minimum relative Jacobian determinant over every fit is `+0.044 > 0`**, so each warp
is a genuine diffeomorphism. The topology ceiling is therefore **structural**, not under-fitting.

Held-out test (16 cases; 4 one-comp, 12 two-comp), by target topology:

| method | topo acc — 1-comp | topo acc — 2-comp | IoU — 1-comp | IoU — 2-comp |
|---|---|---|---|---|
| single diffeomorphism, **1-comp** ref | **1.00** | **0.00** | 0.88 | 0.54 |
| single diffeomorphism, **2-comp** ref | **0.00** | **1.00** | 0.55 | 0.88 |
| **FROST** (level-set operator) | **1.00** | **1.00** | **0.93** | **0.96** |

**Findings:** each single-diffeomorphism reference is perfect on its *own* topology class and **exactly
0%** on the other — no single reference covers the family — and even the oracle best-fit IoU collapses
off-class (0.88 → ~0.54). FROST is correct on **both** classes. `baseline_qualitative.png` shows it
directly: against a 2-component target the 1-comp reference stretches into a connected **dumbbell**
(still 1 region — it cannot break the neck), and against a 1-component target the 2-comp reference
stays **two disjoint blobs** (it cannot fuse), while FROST recovers the right topology in both.
`baseline_topology.png` is the grouped-bar summary. The **time-dependent** form of this comparison
(a 2→1 coalescence *within one trajectory*, which a time-continuous diffeomorphism cannot represent)
is in `../../tumour_merge/C1/baseline_diffeo_time.py`.

## Boundary-weighted loss (`--bw β`)
Adds a weighted relative-L2 term on φ with per-cell weight `w = exp(−(φ_gt/σ)²/2)` (σ=0.05) that
up-weights the free boundary Γ={φ=0}. `β=0` is the baseline; `β>0` saves to `results/bw_β/`.

| variant | contact IoU | topo | φ rel-L2 | u rel-L2 |
|---|---|---|---|---|
| baseline (β=0) | 0.9542 | 1.0 | **0.0085** | **0.0091** |
| β=1 | 0.9565 | 1.0 | 0.0089 | 0.0113 |
| β=3 | **0.9592** | 1.0 | 0.0098 | 0.0130 |

Sharpens the interface (IoU +0.5 pts at β=3, concentrated in the harder 1-component cases
0.934→0.948) at a small cost to global field accuracy. Topology stays 100%. **Marginal with a clear
trade-off** — the baseline IoU (0.95) left little headroom, so this is a low-priority refinement.

## DEQ / implicit-diff upgrade (`deq.py`, `train_c1_deq.py`)
The **equilibrium** form of the operator: a fixed-point cell `f_θ(z; χ)` whose equilibrium
`z* = f_θ(z*; χ)` decodes to `(u, φ)`. Forward = **Anderson** fixed-point solve; backward = **implicit
function theorem** (adjoint fixed-point solve via an autograd hook), so training is **O(1) memory in
solver depth** — no unrolling. This mirrors the physics (the obstacle problem is itself a fixed point).

Stability: a generic equilibrium cell can lose contractivity mid-training (the fixed-point solve then
diverges) — observed once here (great until ~ep 120, then blow-up). Fixed with **gradient clipping +
keep-best checkpointing**; the forward residual then stays ~9e-4 throughout. (Jacobian regularization
was tried but its 2nd-order graph collides with the implicit-backward hook.)

Results (160 epochs, best @ep 140; held-out test):

| Metric | **DEQ** (302k params) | FNO (2.1M params) |
|---|---|---|
| `u` rel-L2 | 2.1% | 0.9% |
| `φ` rel-L2 | 1.3% | 0.85% |
| contact IoU (φ-head) | 0.886 | 0.954 |
| topology accuracy | 93.8% (15/16) | 100% (16/16) |
| obstacle fixed-pt residual | 0.0036 | 0.0025 |

**Findings:** the implicit-diff DEQ reaches **near-FNO accuracy with ~7× fewer parameters** and
constant-memory training, and **provably converges to an equilibrium** (`results/deq_convergence.png`:
residual → 1e-3 in ~27 Anderson iters). The FNO stays a bit sharper (more params + direct
supervision), including a slightly lower obstacle residual — so "the DEQ satisfies the physics better"
did **not** hold; the more accurate model has the smaller residual. Outputs → `results/deq_model.pt`,
`results/deq_metrics.json`, `results/deq_convergence.png`, and (via `python viz_c1_deq.py`)
`results/deq_predictions.png` + `results/deq_predictions_3d.png` (same 2D/3D GT-vs-pred layout as the
FNO; DEQ error ~2–3× the FNO's, max `|Δu|` ≈ 0.008–0.011, still concentrated at the contact rim).

## Data-efficiency / few-shot (`few_shot.py`)
Validates FROST's **data-efficient** headline claim (proposal §9 exp 2): FBNO is data-hungry (~3,000
sims); FROST's operator should generalize from O(1–10). We train the χ→(u,φ) operator on
`K ∈ {1,2,4,8,16,32,64}` samples (random subsets, small-K averaged over 3 draws) and evaluate on the
**fixed** 16-sample held-out set.

```
python few_shot.py [--epochs 300]   # -> results/few_shot/{few_shot_curve.png, few_shot_metrics.json}
```

| K | field `u` rel-L2 | `φ` rel-L2 | contact IoU | topology acc | vs FBNO (~3000 sims) |
|---|---|---|---|---|---|
| 1 | 21.4% | 0.143 | 0.34 | 0.40 | 3000× fewer |
| 2 | 19.9% | 0.086 | 0.56 | 0.69 | 1500× |
| 4 | 13.1% | 0.057 | 0.67 | **0.92** | 750× |
| 8 | 8.9% | 0.040 | 0.71 | 0.92 | 375× |
| 16 | 3.8% | 0.024 | 0.73 | 0.94 | 190× |
| 32 | **1.5%** | 0.013 | 0.91 | **1.00** | 94× |
| 64 (full) | 0.99% | 0.0088 | 0.946 | 1.00 | 47× |

**Findings (`few_shot_curve.png`):**
- The **free-boundary/topology metrics saturate first**: topology accuracy reaches **0.92 with just K=4**
  samples and **1.00 by K=32**; the level set `φ` is already 4–6% by K=4–8. The capability FROST is built
  for needs **single-digit** simulations.
- The **raw field** converges a bit later: within **2× of full-data field error by K=32** (1.5% vs 0.99%),
  single-digit % by K=8.
- Against FBNO's ~3,000 sims, **every useful operating point is 100–750× more data-efficient** — a
  conservative bound, since this is the *global* FNO; the local-stencil operator (§3/§4.2, one sim →
  10⁵–10⁶ stencils) would be more efficient still.

## Local-stencil operator & why FROST needs the DEQ (`local_stencil.py`, `train_local_stencil.py`)
We built the operator the proposal specifies (§3/§4.2/§6): a small MLP on a geometry-conditioned stencil
`[4-neighbour u, χ, φ, ∇φ] → u(x)`, trained from the **~16k stencils of K simulations**, with inference by
**fixed-point iteration** (red-black projected SOR) so the contact-set free boundary *emerges* from the
equilibrium. The result is an instructive **negative-with-explanation**:

```
python train_local_stencil.py [--epochs 60]   # -> results/local_stencil/{local_stencil_finding.png, metrics.json}
```

| measurement | value |
|---|---|
| one-step rel-L2 (full train set) | **0.0014** — the local map `max(χ, mean_nbr)` is learned near-perfectly |
| one-step rel-L2 (from K=1 sim) | 0.107 — a few sims suffice for the local map (locality ⇒ many stencils) |
| **warm-start FPI drift from the GT solution** | **→ 0.64** — the GT solution is **not** the operator's fixed point |
| **cold-start FPI field rel-L2** | **1.37** (spurious fixed point; IoU 0.20) |
| FROST **DEQ** (`deq.py`) field rel-L2 | **0.021** |
| global **FNO** field rel-L2 | **0.009** |

**Finding (`local_stencil_finding.png`):** the operator is **locally accurate but globally spurious**.
Started *at* the true solution, the FPI **drifts away**; cold-started it converges to a **spurious fixed
point** with a 137% field error. The cause is **elliptic fixed-point conditioning**: the tiny one-step bias
is amplified by ≈ 1/(1−ρ), ρ→1. This rigorously reproduces — and explains — the spurious-fixed-point
pathology our One-shot reproduction first hit, now on the obstacle, and it is **robust to trajectory
training**.

**Consequence (this is the point):** a free-boundary *equilibrium* operator must be trained **through** the
fixed point, not by one-step supervision. That is exactly the **DEQ** (`deq.py`): implicit differentiation
makes the equilibrium correct *by construction* (it converges to u 2.1%), and a *global* spectral cell
keeps the equilibrium solve fast (a purely-local cell would also iterate slowly on an elliptic problem). So
this experiment is the **empirical justification for FROST's §4.3 DEQ formulation** — the local-stencil
operator motivates the equilibrium operator rather than replacing it.
