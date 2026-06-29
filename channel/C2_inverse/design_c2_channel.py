"""
Stage 4 — FROST-Design (C2) on the channel: optimize the baffle through the FROZEN forward operator.

This is NTO's topology-optimization task done with FROST's differentiable operator. A learnable field
θ(x,y) defines the baffle density  γ = sigmoid(α·(θ − mean θ));  γ is fed (with its gradients + the
inlet velocity) to the FROZEN γ-conditioned operator G:(γ,v)→(C,P) from C1. We minimise NTO's
objective by gradient descent through G (implicit/auto-diff design):

    J(v) = σ²_C  +  ω·ΔP  +  λ·max(0, V_target − V_solid)        averaged over inlet velocities
      σ²_C = mean(|C − mean C|²)      (concentration uniformity / mixing)
      ΔP   = mean(P_inlet) − mean(P_outlet)   (area-weighted pressure drop)
      volume inequality keeps a baffle present (else the optimum is "no baffle")

We sweep ω (like NTO Fig 5) and compare the optimised objective vs the SMOOTH channel (γ≡1) across
velocities (the Fig 5c analogue). NOTE: the objective is OPERATOR-PREDICTED — verifying an optimised
design needs a CFD re-solve (NTO used active learning for this); that is out of scope here and stated.

Outputs -> results/ : design_metrics.json, design_fields.png, objective_vs_velocity.png
Run:  python design_c2_channel.py
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "C1"))
from fno import FNO2d

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
torch.manual_seed(0); np.random.seed(0)
OP = os.path.join(HERE, "..", "C1", "results", "gamma_random", "model.pt")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
FS, FIG_DPI = 22, 600
VELS = [0.1, 0.3, 0.5, 0.7, 0.9]
ALPHA = 6.0                                              # sigmoid sharpness for γ
V_TARGET = 0.06                                          # min solid-volume fraction (keep a baffle)
LAMBDA = 50.0                                            # volume-constraint weight
COARSE = (24, 12)                                        # design optimized at coarse res -> band-limited
TV = 3.0                                                 # total-variation penalty (keeps γ in-distribution)


def load_operator():
    ck = torch.load(OP, map_location="cpu")
    H, W = ck["grid"]
    model = FNO2d(modes=16, width=32, in_c=4, out_c=2, n_layers=4)
    model.load_state_dict(ck["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    s = ck["stats"]
    return model, s["xm"], s["xs"], s["ym"].reshape(2), s["ys"].reshape(2), H, W


def gamma_from_theta(theta_coarse, H, W):
    """Coarse design field -> bilinear upsample -> band-limited γ = sigmoid(α(θ-mean θ))."""
    up = F.interpolate(theta_coarse[None, None], size=(H, W), mode="bilinear", align_corners=True)[0, 0]
    return torch.sigmoid(ALPHA * (up - up.mean()))


def tv(theta):
    return (theta[1:, :] - theta[:-1, :]).abs().mean() + (theta[:, 1:] - theta[:, :-1]).abs().mean()


def operator_fields(model, gamma, v, stats):
    """γ (H,W) + velocity -> (C, P) via the frozen operator (denormalized)."""
    xm, xs, ym, ys = stats
    gx, gy = torch.gradient(gamma)
    vfield = torch.full_like(gamma, float(v))
    X = torch.stack([gamma, gx, gy, vfield], -1)[None]   # (1,H,W,4)
    out = model((X - xm) / xs)[0] * ys + ym              # (H,W,2)
    return out[..., 0], out[..., 1]


def objective(C, P, gamma, omega):
    sigma2 = torch.mean((C - C.mean()) ** 2)             # concentration uniformity
    dP = P[0, :].mean() - P[-1, :].mean()                # inlet(x=0) - outlet(x=-1) drop
    v_solid = torch.mean(1.0 - gamma)                    # solid fraction
    vol_pen = torch.clamp(V_TARGET - v_solid, min=0.0)
    return sigma2 + omega * dP + LAMBDA * vol_pen, sigma2.item(), dP.item(), v_solid.item()


def smooth_baseline(model, stats, H, W):
    """Smooth channel (γ≡1, no baffle): operator-predicted σ²_C and ΔP per velocity."""
    g1 = torch.ones(H, W)
    rows = {}
    with torch.no_grad():
        for v in VELS:
            C, P = operator_fields(model, g1, v, stats)
            rows[v] = (float(torch.mean((C - C.mean()) ** 2)), float(P[0, :].mean() - P[-1, :].mean()))
    return rows


def optimize(model, stats, H, W, omega, iters=400, lr=5e-2):
    Hc, Wc = COARSE
    # seed two soft blobs (coarse) so the optimizer starts from a plausible, in-distribution baffle
    xs_ = torch.linspace(-1, 1, Hc)[:, None]; ys_ = torch.linspace(-1, 1, Wc)[None, :]
    theta0 = -3.0 * torch.exp(-(((xs_ + 0.2) ** 2 + ys_ ** 2) / 0.05)) \
             - 3.0 * torch.exp(-(((xs_ - 0.2) ** 2 + ys_ ** 2) / 0.05))
    theta = theta0.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    hist = []
    for it in range(iters):
        opt.zero_grad(); total = 0.0
        for v in VELS:
            gamma = gamma_from_theta(theta, H, W)
            C, P = operator_fields(model, gamma, v, stats)
            J, *_ = objective(C, P, gamma, omega)
            total = total + J
        total = total / len(VELS) + TV * tv(theta)         # band-limit + smoothness
        total.backward(); opt.step(); hist.append(total.item())
    rows = {}
    with torch.no_grad():
        gamma = gamma_from_theta(theta, H, W)
        for v in VELS:
            C, P = operator_fields(model, gamma, v, stats)
            _, s2, dP, vs = objective(C, P, gamma, omega)
            rows[v] = (s2, dP)
    return theta.detach(), gamma.detach(), rows, hist


def main():
    if not os.path.exists(OP):
        print("gamma_random operator not found yet:", OP); return
    model, xm, xs, ym, ys, H, W = load_operator()
    stats = (xm, xs, ym, ys)
    print(f"operator loaded, grid {H}x{W}")
    t0 = time.time()

    base = smooth_baseline(model, stats, H, W)
    omegas = [0.1, 1.0, 10.0]
    designs = {}
    for om in omegas:
        theta, gamma, rows, hist = optimize(model, stats, H, W, om)
        designs[om] = dict(gamma=gamma.numpy(), rows=rows, hist=hist)
        s2m = np.mean([rows[v][0] for v in VELS]); dpm = np.mean([rows[v][1] for v in VELS])
        print(f"  ω={om}: mean σ²_C {s2m:.4f}  mean ΔP {dpm:.4f}  (smooth "
              f"σ²_C {np.mean([base[v][0] for v in VELS]):.4f})", flush=True)

    metrics = {"velocities": VELS, "alpha": ALPHA, "v_target": V_TARGET, "omegas": omegas,
               "train_time_s": round(time.time() - t0, 1),
               "smooth": {str(v): {"sigma2_C": base[v][0], "dP": base[v][1]} for v in VELS},
               "optimized": {str(om): {str(v): {"sigma2_C": designs[om]["rows"][v][0],
                                                "dP": designs[om]["rows"][v][1]} for v in VELS}
                             for om in omegas}}
    json.dump(metrics, open(os.path.join(OUT, "design_metrics.json"), "w"), indent=2)

    # --- objective-vs-velocity curves (Fig 5c analogue): σ²_C and ΔP, optimized ω's vs smooth
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    a1.plot(VELS, [base[v][0] for v in VELS], "k-o", lw=2.5, ms=8, label="smooth")
    a2.plot(VELS, [base[v][1] for v in VELS], "k-o", lw=2.5, ms=8, label="smooth")
    for om in omegas:
        a1.plot(VELS, [designs[om]["rows"][v][0] for v in VELS], "-o", lw=2.5, ms=7, label=f"ω={om}")
        a2.plot(VELS, [designs[om]["rows"][v][1] for v in VELS], "-o", lw=2.5, ms=7, label=f"ω={om}")
    a1.set_xlabel("inlet velocity v", fontsize=FS); a1.set_ylabel("σ²_C", fontsize=FS)
    a2.set_xlabel("inlet velocity v", fontsize=FS); a2.set_ylabel("ΔP", fontsize=FS)
    for ax in (a1, a2):
        ax.tick_params(labelsize=FS - 5); ax.grid(alpha=0.3); ax.legend(fontsize=FS - 6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "objective_vs_velocity.png"), dpi=FIG_DPI); plt.close(fig)

    # --- optimized baffle + resulting fields (ω=1, v=0.5) vs smooth
    om = 1.0; g = torch.from_numpy(designs[om]["gamma"])
    ext = (0, 0.02, 0, 0.01)
    fig, axs = plt.subplots(2, 3, figsize=(16, 6))
    with torch.no_grad():
        for r, (gg, lab) in enumerate([(torch.ones(H, W), "smooth"), (g, f"optimized (ω={om})")]):
            C, P = operator_fields(model, gg, 0.5, stats)
            axs[r, 0].imshow(gg.numpy().T, origin="lower", extent=ext, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
            axs[r, 0].set_ylabel(lab, fontsize=FS - 2)
            axs[r, 1].imshow(C.numpy().T, origin="lower", extent=ext, cmap="jet", vmin=0, vmax=1, aspect="equal")
            axs[r, 2].imshow(P.numpy().T, origin="lower", extent=ext, cmap="viridis", aspect="equal")
            for c, t in zip(range(3), ["baffle γ", "concentration C", "pressure P"]):
                if r == 0:
                    axs[r, c].set_title(t, fontsize=FS - 2)
                axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "design_fields.png"), dpi=FIG_DPI); plt.close(fig)
    print(f"saved -> results/  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
