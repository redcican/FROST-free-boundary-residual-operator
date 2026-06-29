# FROST C2 — Stefan target-front control (the moving-boundary design demo)

The headline Part-B result (`FROST_proposal.md` §4.4, §12.3 step 3): **control a free boundary that
changes topology**, by differentiating the frozen **time-conditioned** Stefan operator (C1)

```
G:  [ φ₀(seed layout), L, T_cold, t ]  →  ( T(t), φ(t) )
```

We optimize the **controls** so the solidification front hits a target. Because the front **coalesces**
(N seeds → 1 grain), the control is steering a *topology-changing* interface — exactly what the
single-diffeomorphism baseline (TODO b) **cannot even represent**, so this inverse problem is only
well-posed for a topology-capable operator like FROST.

```
python control_c2_stefan.py        # needs ../C1/results/model.pt + ../stefan.npy
```
Outputs → `results/`: `control_metrics.json`, `control_trajectory.png`, `control_targets.png`,
`control_authority.png`.

Controls are kept **on-manifold by construction** (the seed layout is parameterized as disks; T_cold,
seed radius clamped to the trained ranges) so we don't repeat the off-manifold exploit of the earlier C2
runs. A cheap **physics-consistency self-certificate** (predicted solid must be cold, liquid warm) is
reported — the Stefan analogue of the obstacle residual / channel ΔP check from step 2.

## Scenario A — shape control (the headline)
Drive the final-time solid region `{φ(t=1)<0}` to a target shape `M*` (achievable: `M*` is a real
sample's final, merged front) by optimizing the **seed layout + T_cold + seed radius** through the
operator.

| target | IoU(final, target) | topology (achieved/target) | physics-inconsistency |
|---|---|---|---|
| sample 0 (3 seeds) | **0.98** | **1 / 1** | 0.003 |
| sample 3 (3 seeds) | **0.98** | **1 / 1** | 0.003 |

`control_trajectory.png` is the money figure: the optimized layout's **3 separate seeds (t=0) grow and
merge 3→1 into the target** (green dashed) by t=14 — the controller is steering the front *through* a
topology change. `control_targets.png` overlays the controlled final front (red) on the target (blue)
with the optimized seeds (yellow). The physics-consistency certificate stays `~0.003` (the controlled
solutions are physical / on-manifold).

## Scenario B — control authority (an honest finding)
Which knob actually moves the front? Sweeping each control alone (`control_authority.png`):

| knob | final-solid-fraction authority |
|---|---|
| **seed layout** (spread ×) | **0.111** |
| T_cold (cooling) | 0.007 |

**The seed layout is ~16× the control authority of T_cold.** T_cold barely moves the *final* extent
because this benchmark's time window is calibrated to the merge time (the generator made all fronts
travel a similar distance regardless of Stefan number), so St's independent effect on the final front was
calibrated out. The **layout is the effective control** — which is what scenario A exploits. (T_cold would
regain authority for an *un-normalized-time* target, e.g. controlling the merge *instant*; not pursued
here.)

## Why this completes the inverse side
This is §12.3 step 3 — the moving / topology-changing design demo — and the capstone of FROST's Part B:
- **obstacle** (`../../obstacle/C2_inverse/`): IFT-through-equilibrium design + active trust-region loop
  (physics-residual trust signal);
- **channel** (`../../channel/C2_inverse/`): differentiable design + physics-violation trust region on
  real CFD;
- **stefan** (here): control of a **topology-changing moving boundary** through the time-dependent
  operator.

Together they show FROST's design side across steady, real-CFD, and time-dependent topology-changing free
boundaries — the capability a diffeomorphism operator structurally lacks.
```
../C1/  →  the frozen time-conditioned forward operator this controller differentiates
```
