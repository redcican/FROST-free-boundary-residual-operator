# FROST C2 — inverse design by differentiating the equilibrium (obstacle)

The first FROST inverse-design result built the way the proposal prescribes
(`FROST_proposal.md` §4.4–4.5, §12.3 step 1): instead of NTO's "backprop through a **frozen
feed-forward** operator" (what the channel C2 did), we **differentiate the DEQ equilibrium**. The
design variable enters the fixed point, and the design gradient

```
dJ/dd = ∂J/∂(u*,φ*) · (I − ∂T_θ/∂(u,φ))⁻¹ · ∂T_θ/∂d        (the IFT adjoint, §4.4)
```

is obtained **for free** by the DEQ's implicit-function-theorem backward — the *same* adjoint that
trains it. In code: make the obstacle `χ` a differentiable leaf, run the frozen DEQ forward, call
`J.backward()`.

**Why the obstacle is the right first case.** Its forward operator is the strongest in the suite
(~1%), and — crucially — it has a **cheap exact solver** (projected SOR,
`gen_obstacle.solve_obstacle`). So we can actually compute the **trust-region certificate**
`|J_FROST − J_true|` at the optimized design — the very check the channel C2 had to leave out of
scope (CFD re-solve too expensive).

```
python design_c2_obstacle.py [steps]      # default 150; needs ../C1/results/{deq_model.pt, model.pt}
```
Outputs → `results/`: `design_metrics.json`, `design_targets.png`, `design_convergence.png`,
`design_certificate.png`.

## Task
**Free-boundary targeting.** Given a target contact footprint `M*` (taken from a *real* obstacle, so it
is achievable), design `χ` so the equilibrium contact set `{φ*<0}` matches `M*`. Objective
`J = mean( (sigmoid(−φ*/τ) − M*)² )`, τ = 0.03. Two targets: a **merged** 1-component footprint and a
**separated** 2-component footprint. Run three ways, and certify every optimized design with the true
solver:
- **DEQ/IFT** — through the DEQ equilibrium (the principled §4.4 adjoint);
- **FNO/ff** — through the frozen feed-forward FNO (the NTO-style baseline);
- **DEQ/field** — a band-limited free `χ` field (off the data manifold), to probe the certificate.

## Results (150 design steps)

| target | method | IoU(op/target) | **IoU(true/target)** | topology op/true | **certificate gap** `|J_F−J_t|` |
|---|---|---|---|---|---|
| merged (1-comp) | **DEQ/IFT** | 0.96 | 0.70 | **1 / 1** | 0.003 |
| merged (1-comp) | FNO/ff | — | 0.86 | 1 / 1 | 0.001 |
| separated (2-comp) | **DEQ/IFT** | 0.87 | **0.93** | **2 / 2** | 0.000 |
| separated (2-comp) | FNO/ff | — | 0.90 | 2 / 2 | 0.000 |
| merged (1-comp) | **DEQ/field (off-manifold)** | — | — | — | **0.643** |

## Findings
1. **IFT-through-equilibrium design works (§4.4 realized).** Differentiating the *frozen* DEQ
   equilibrium drives a design to the target free boundary with the **correct topology** (1-comp and
   2-comp), converging in ~60 steps (`design_convergence.png`). The design gradient is the exact
   equilibrium sensitivity from the same IFT backward that trains the DEQ — no unrolling, constant
   memory. This is the principled upgrade of the channel C2's frozen-feed-forward design.
2. **On-manifold designs are *certified*.** Every parametric design has a **tiny** certificate gap
   (≤ 0.003): the true solver agrees with the operator where the design stays on the data manifold —
   the operator is trustworthy there (`design_certificate.png`, blue/grey bars).
3. **The off-manifold field design exposes — and now *quantifies* — the failure mode.** A band-limited
   `χ` field **fools** the DEQ (`J_FROST` 0.031, the operator "thinks" it matched) while the true
   solver completely disagrees (`J_true` 0.674) — a certificate gap of **0.643, ~250× the parametric
   gap**. Because the obstacle has a cheap exact solver, we can **measure** the gap the channel C2 could
   only hypothesize. This is the direct, quantified empirical case for the proposed
   **active-acquisition + trust-region C2** (§4.5): trust the design optimizer only inside
   `|J_FROST − J_true| ≤ ε`, and acquire a true solve where it is not.
4. **Honest nuance — realized design quality tracks operator accuracy.** The DEQ is a *less accurate*
   operator than the FNO (C1: DEQ contact IoU 0.886 vs FNO 0.954), so on the harder **merged** target
   the FNO-designed obstacle verifies a bit better (true IoU 0.86 vs DEQ 0.70), while on the
   **separated** target the DEQ wins (0.93 vs 0.90). The IFT adjoint is the principled *gradient* for an
   equilibrium operator; the operator's own accuracy still bounds the achievable design — which is
   exactly why the certificate (finding 2–3) matters.

# Step 2 — the active trust-region loop, with the *right* trust signal (`active_trust_region.py`)

§4.5 proposes closing the loop with an uncertainty signal `σ_J` from a **deep ensemble**. We built it,
tested the literal proposal first, and found a clean, important result that **refines the proposal**.

```
python active_trust_region.py [--members 4 --rounds 6 --lam-res 8 --tau 0.05 ...]   # ensemble cached in results/ensemble/
```
Outputs → `results/`: `active_metrics.json`, `active_trust_signals.png`, `active_convergence.png`,
`active_designs.png`.

## Finding A — deep-ensemble `σ_J` **fails**; the FROST physics residual **works**
A trust signal must be large exactly where the operator is wrong (large true gap `|J_FROST−J_true|`). On
a sweep of designs spanning on-manifold (parametric) to off-manifold (free-`χ` exploits of increasing
severity), with a **frozen** 4-FNO ensemble:

| trust signal | corr. with true gap | behaviour |
|---|---|---|
| ensemble disagreement `σ_J` | **−0.36** | tiny (~1e-4) and **lowest at the *most severe* exploits** — the members agree *most confidently exactly where they are most wrong* (the textbook OOD failure of same-data/same-architecture ensembles) |
| **FROST fixed-point residual** `‖u−max(χ,mean_nbr u)‖/‖u‖` | **+0.97** | 0.007 on-manifold → 0.5 at severe exploits; tracks the gap, **needs no ensemble and no true solver**, and *is* the FROST equilibrium residual |

`active_trust_signals.png` shows it directly (left: `σ_J` vs gap, no trend; right: residual vs gap,
clean monotonic). **The signal that defines the FROST operator is also its self-certificate.**

## Finding B — residual self-certification fixes the off-manifold exploit
The loop steers the design by `J_mean + λ·residual`, **self-certifies** by `residual ≤ τ` (no true solver
needed), and only when a candidate is outside the trust region does it acquire the true solve and
fine-tune (then shrink τ). On the merged target:

| | designed `χ` residual | true gap `|J_F−J_t|` | self-certified? |
|---|---|---|---|
| round 0 — **naive** (J-only, off-manifold) | 0.144 | 0.058 | ✗ (residual > τ) → acquire + fine-tune |
| round 1 — **residual-steered** | **0.002** | **0.037** | ✓ (residual ≤ τ) |

The residual drops **~70×** below the trust boundary while the true gap also falls
(`active_convergence.png`); the self-certified design no longer fools the operator
(`active_designs.png`: round-0's true contact has spurious off-target regions, round-1's is on-target).
*Honest caveat:* a free `χ`-field is a hard design space, so the self-certified field design's footprint
IoU is modest (~0.27); the trust mechanism guarantees the operator is **reliable** at the design (small
gap), not that the field design is optimal. The high-quality *and* certified design is the **parametric**
one from step 1 (IoU 0.70, gap 0.003) — step 2's contribution is the **trust signal + self-certification**
that gates any design.

## Status vs the proposal
- **§12.3 step 1** — IFT-through-equilibrium design — **DONE** (above).
- **§12.3 step 2 / §4.5** — trust-region + acquisition — **DONE, with a refinement**: the loop is built and
  closes, and the key lesson is that the **physics residual replaces ensemble `σ_J`** as the trust signal
  for free-boundary operators (a FROST-native, self-certifying alternative the proposal should adopt;
  `σ_J`/PNO remain options where no cheap residual exists).
- **§12.3 step 3** — moving-boundary control (Stefan target-front) — **still open** (the headline Part-B demo).
```
../C1/  →  the frozen forward operators (FNO + DEQ) this design loop differentiates
```
