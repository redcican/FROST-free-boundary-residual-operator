# FROST C2 — FROST-Design (channel inverse design)

NTO's topology-optimization task done with FROST's **differentiable forward operator**: a learnable
baffle is optimized by gradient descent **through the frozen C1 operator** `G:(γ,v)→(C,P)` to minimize
NTO's objective, averaged over inlet velocities:

```
J(v) = σ²_C + ω·ΔP + λ·max(0, V_target − V_solid)
   σ²_C = mean(|C − mean C|²)            (concentration uniformity / mixing)
   ΔP   = mean(P_inlet) − mean(P_outlet) (pressure drop)
   volume inequality keeps a baffle present
```

Design parameterization (matches NTO): `γ = sigmoid(α·(θ − mean θ))`. θ is optimized at **coarse
resolution + a TV penalty** so the design stays band-limited (compact baffles, not pixel speckle). The
**γ-conditioned operator** is the forward model (the design produces γ directly → fully differentiable;
a φ-operator would need a non-differentiable distance transform each step).

```
python design_c2_channel.py        # needs ../C1/results/gamma_random/model.pt
```
Outputs → `results/`: `design_metrics.json`, `design_fields.png`, `objective_vs_velocity.png`.

## Results (optimized vs smooth channel, mean over v∈{0.1…0.9})

| | σ²_C | ΔP |
|---|---|---|
| smooth (γ≡1) | 0.0142 | +0.25 |
| optimized ω=0.1 | **0.0081** (1.8×) | **+0.21** (physical) |
| optimized ω=1.0 | 0.0050 (2.8×) | −0.05 |
| optimized ω=10 | 0.0037 (3.8×) | −0.34 |

`objective_vs_velocity.png` is the **NTO Fig 5c analogue**: every optimized design lowers concentration
variance below the smooth channel at all velocities (most at low v) — the mixing-improvement story,
reproduced through FROST's operator. `design_fields.png` shows the optimized baffle + its more-uniform
concentration field vs smooth.

## Two honest findings (both paper-relevant)
1. **The differentiable-design machinery works:** optimizing through the frozen operator reduces the
   concentration objective **1.8–3.8×** vs the smooth channel, consistently across velocities — FROST's
   "design by differentiating the equilibrium operator" demonstrated on real-CFD-trained physics.
2. **It exposes exactly the failure mode FROST's proposed C2 is meant to fix.** As the pressure weight ω
   rises, the optimizer drives **ΔP to unphysical negative values** (P_out > P_in) — it games the frozen
   operator off-manifold, where `G`'s pressure response is unreliable. Only ω=0.1 (which barely weights
   pressure) stays physical. *Without* a band-limit the design collapses to **pixel speckle** with
   ΔP≈−0.95 (the first unregularized run). This empirically motivates FROST's proposed enhancement: an
   **active-acquisition + trust-region certificate** `|J_FROST − J_true| ≤ ε` so the design optimizer is
   only trusted where the operator is verified accurate.

# Completion — the trust region (`trust_region_channel.py`)

The design above (`design_c2_channel.py`) exposed the exploit but had no trust region. This completes it
with the §4.5 mechanism, transferring the obstacle C2_inverse step-2 lesson to the channel. The obstacle
used its **physics fixed-point residual** as a self-certifying trust signal; the channel has **no cheap CFD
solver**, so we use the operator's own predicted fields — a design where `G` predicts **non-physical** flow
is a design where it is extrapolating wrongly:

```
viol(d) = relu(−ΔP)  +  mean(relu(−C) + relu(C−1))      # forward flow must lose pressure; C∈[0,1]
```

This is cheap, operator-only, and the channel analogue of the obstacle residual (real CFD: ΔP∈[0.10,0.61],
**0 % negative**; C∈[0,1] exactly).

```
python trust_region_channel.py     # needs ../C1/{results/gamma_random/model.pt, channel_train.npy}
```
Outputs → `results/`: `trust_metrics.json`, `trust_signal_validation.png`, `trust_design_comparison.png`,
`trust_objective_vs_velocity.png`.

## Finding 3 — the trust signal is validated on the **450 real-CFD designs** (no re-solve)
On every real design the operator's predicted ΔP is **accurate and physical** (on the `y=x` line, all
ΔP≥0) and `viol ≈ 0` (mean **1e-4**, max 2.5e-3; mean C-error 0.09). The naive ω-exploit predicts **ΔP<0**
— impossible for forward flow — so `viol` is large. **`viol` separates trustworthy from untrustworthy
designs without a single CFD solve** (`trust_signal_validation.png`). This is the channel's substitute for
the obstacle's true-solver certificate.

## Finding 4 — trust-steering **fixes** the exploit (mixing kept, physics restored)
Minimizing `J_NTO + λ·viol` (λ scaled with ω so the physics floor dominates):

| ω | naive ΔP | naive `viol` | **trust** ΔP | **trust** `viol` | trust σ²_C (vs smooth 0.014) |
|---|---|---|---|---|---|
| 1 | −0.05 | 0.05 | **+0.012** | **0.000** | 0.0057 (2.5×) |
| 10 | −0.34 (speckle baffle) | 0.34 | **+0.007** | **0.000** | 0.0057 (2.5×) |

The naive ω=10 design is **pixel speckle** with `ΔP=−0.36`; trust-steering turns it into a **clean,
physical baffle** with `ΔP=+0.006`, **keeping the ~2.5× mixing improvement** over the smooth channel
(`trust_design_comparison.png`, `trust_objective_vs_velocity.png`). The exploit's *slightly* lower σ²_C
was bought off-manifold where `G` is unreliable; the trust design earns its gain honestly.

## Status / honest caveat
This completes the channel C2 with the **physics-violation trust signal + trust-steered design +
self-certification** — the channel parallel of the obstacle's step-2 loop, adapted to a benchmark with no
cheap solver. The remaining piece (shared by any operator-design loop) is the **true CFD acquisition**:
designs that fail `viol ≤ τ` would be re-solved in Fluent and folded back in. The loop is structured for
it; the expensive Fluent solve itself is out of scope here (it is exactly why the cheap physics-violation
self-certificate matters).
