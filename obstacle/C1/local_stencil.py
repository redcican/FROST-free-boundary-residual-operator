"""
FROST C1 — the LOCAL-STENCIL operator + L_fb (the operator the proposal actually specifies, §3/§4.2/§6).

Unlike the global FNO (which regresses the whole field χ→(u,φ)), this is the One-shot-style local
operator, geometry-conditioned and equipped with the free-boundary residual:

  ĝ_θ( {u(x') : x'∈stencil}, χ(x), φ(x), ∇φ(x) )  →  u(x)

A small MLP on a LOCAL stencil, applied per grid point. Two consequences the proposal claims:
  • LOCALITY ⇒ FEW-SHOT: one 128² simulation yields ~16k stencils, so the operator can train from K=1 sim.
  • Inference = FIXED-POINT ITERATION of the local operator to equilibrium; the contact-set FREE BOUNDARY
    {u=χ} EMERGES from the equilibrium (it is not regressed). The obstacle problem is the right testbed:
    its interior update IS the local stencil u←max(χ, mean_nbr u), monotone ⇒ the FPI provably converges
    (avoiding the spurious-fixed-point failure our One-shot reproduction hit on a nonlinear PDE).

L_fb: the free-boundary condition as a loss — an interface-band-weighted residual near Γ={φ=0} plus the
obstacle complementarity constraint u≥χ. This module defines the operator, stencil builder, and FPI;
`train_local_stencil.py` runs the experiments (FPI convergence, few-shot, ablations, vs the global FNO).
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scipy.ndimage import distance_transform_edt

N = 128
H = 2.0 / (N - 1)
CONTACT_TOL = 2e-3


# ---------------------------------------------------------------- geometry
def signed_distance(mask):
    """Signed distance to the contact set (negative inside) — matches gen_obstacle."""
    return (distance_transform_edt(~mask) - distance_transform_edt(mask)) * H


def contact_of(u, chi):
    return ((u - chi) < CONTACT_TOL) & (chi > 0.0)


# ---------------------------------------------------------------- stencils (one sim -> ~16k examples)
def build_stencils(u, chi, phi, geom=True):
    """Per-interior-point stencil features. geom=True appends (φ, ∂xφ, ∂yφ) -> geometry-conditioned.
    Returns X (M, D) and the per-point free-boundary weight w (M,) for L_fb."""
    ul, ur = u[:-2, 1:-1], u[2:, 1:-1]
    ud, uu = u[1:-1, :-2], u[1:-1, 2:]
    cc = chi[1:-1, 1:-1]
    feats = [ul, ur, ud, uu, cc]
    if geom:
        pc = phi[1:-1, 1:-1]
        px = (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2 * H)
        py = (phi[1:-1, 2:] - phi[1:-1, :-2]) / (2 * H)
        feats += [pc, px, py]
    X = np.stack(feats, -1).reshape(-1, len(feats)).astype(np.float32)
    w = np.exp(-((phi[1:-1, 1:-1] / 0.05) ** 2) / 2.0).reshape(-1).astype(np.float32)   # interface band
    return X, w


# ---------------------------------------------------------------- local operator (MLP)
class LocalOp(nn.Module):
    def __init__(self, in_dim, width=64, depth=3):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------- fixed-point inference
_II, _JJ = np.indices((N, N))
_INTERIOR = (_II > 0) & (_II < N - 1) & (_JJ > 0) & (_JJ < N - 1)
_RED = _INTERIOR & ((_II + _JJ) % 2 == 0)
_BLACK = _INTERIOR & ((_II + _JJ) % 2 == 1)


def fpi(model, stats, chi, geom=True, max_iter=800, tol=1e-6, omega=1.5, recompute_geom_every=10,
        u0=None, track=None):
    """Iterate the local operator to equilibrium by RED-BLACK projected over-relaxation (mirrors the true
    solver's projected SOR, so it converges in O(N) sweeps, not O(N²)). The contact-set free boundary
    EMERGES; u=0 on ∂, u≥χ enforced. `u0` warm-starts (e.g. the GT solution, to test fixed-point drift);
    `track` (a GT field) records field rel-L2 per sweep. Returns u*, residual history, #sweeps."""
    xm, xs, ym, ys = stats
    u = (np.maximum(chi, 0.0) if u0 is None else u0.copy()).astype(np.float32)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    drift = []
    phi = signed_distance(contact_of(u, chi)) if geom else np.zeros_like(chi, np.float32)
    res_hist = []; it = 0
    for it in range(max_iter):
        if geom and it % recompute_geom_every == 0:
            phi = signed_distance(contact_of(u, chi))
        u_old = u.copy()
        for color in (_RED, _BLACK):                                 # Gauss-Seidel ordering (uses latest values)
            X, _ = build_stencils(u, chi, phi, geom=geom)
            with torch.no_grad():
                pred = (model((torch.from_numpy(X) - xm) / xs) * ys + ym).numpy()
            full = np.zeros((N, N), np.float32); full[1:-1, 1:-1] = pred.reshape(N - 2, N - 2)
            u_gs = (1 - omega) * u + omega * full                    # over-relaxation
            u = np.where(color, np.maximum(u_gs, chi), u).astype(np.float32)
            u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
        r = float(np.max(np.abs(u - u_old)))
        res_hist.append(r)
        if track is not None:
            drift.append(float(np.linalg.norm(u - track) / (np.linalg.norm(track) + 1e-8)))
        if r < tol:
            break
    return u, (drift if track is not None else res_hist), it + 1


def obstacle_residual(u, chi):
    """Relative projected-fixed-point residual ‖u − max(χ, mean_nbr u)‖/‖u‖ (the physics check)."""
    nb = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
    r = u[1:-1, 1:-1] - np.maximum(chi[1:-1, 1:-1], nb)
    return float(np.linalg.norm(r) / (np.linalg.norm(u[1:-1, 1:-1]) + 1e-8))
