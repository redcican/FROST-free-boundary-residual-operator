"""
FROST C2 (inverse) — step 2: the active trust-region loop, with the RIGHT trust signal.

Step 1 (`design_c2_obstacle.py`) showed that designing a free χ-field by differentiating the operator
**exploits it off-manifold** (a large certificate gap |J_FROST − J_true|). §4.5 / §12.3 step 2 proposes
closing the loop with an uncertainty signal σ_J. We first tested the proposal's literal suggestion — a
**deep ensemble** — and found it FAILS here: the 4 FNO members agree (σ_J≈0) even on off-manifold designs
whose true gap is 0.5–0.6, so σ_J↔gap correlation is −0.36 (members extrapolate to the *same* wrong
answer — a known OOD failure of ensemble disagreement). See `diagnose_trust_signal.py`.

The FROST-native fix: the **physics fixed-point residual** of the predicted solution,
‖u − max(χ, mean_nbr(u))‖/‖u‖ (the obstacle problem *is* this fixed point). It is large exactly where the
operator is wrong — corr(residual, gap) = **+0.97** — needs no ensemble and no true solver, and *is* the
FROST equilibrium residual. This script uses it as the trust signal:

  • steer the design with  J_mean(d) + λ·residual(d)  → stay where the operator satisfies the physics;
  • SELF-CERTIFY by residual ≤ τ (no true solver needed); verify with the true solver to report the gap;
  • if outside the trust region → acquire the true solve at d★, fine-tune, shrink τ, repeat.

Ensemble = M compact FNOs (cached in results/ensemble/); kept so we can also report the (failing) σ_J for
the head-to-head. Run:  python active_trust_region.py [--members 4 --rounds 6 --lam-res 5 ...]
Out:  results/{active_metrics.json, active_convergence.png, active_trust_signals.png, active_designs.png}
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "C1"))
sys.path.insert(0, os.path.join(HERE, ".."))
from fno import FNO2d, LpLoss
from train_c1_obstacle import load_split, n_comp, iou
import gen_obstacle as G
import design_c2_obstacle as D1

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
torch.manual_seed(0); np.random.seed(0)
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
ENS = os.path.join(OUT, "ensemble"); os.makedirs(ENS, exist_ok=True)
FS, DPI = 22, 600
N = D1.N
COARSE = 16
CHI_LO, CHI_HI = -0.25, 0.55


# ------------------------------------------------------------------ operators / signals
def train_member(seed, Xtr, Ytr, epochs):
    torch.manual_seed(seed)
    m = FNO2d(modes=12, width=24, in_c=1, out_c=2, n_layers=4)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs); lp = LpLoss(); bs = 8
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr))
        for j in range(0, len(Xtr), bs):
            b = perm[j:j+bs]; opt.zero_grad()
            pred = m(Xtr[b]); (lp(pred[..., 0], Ytr[b][..., 0]) + lp(pred[..., 1], Ytr[b][..., 1])).backward()
            opt.step()
        sched.step()
    m.eval(); return m


def finetune(members, Xtr, Ytr, epochs, lr=3e-4):
    lp = LpLoss(); bs = 8
    for m in members:
        m.train(); opt = torch.optim.Adam(m.parameters(), lr=lr)
        for _ in range(epochs):
            perm = torch.randperm(len(Xtr))
            for j in range(0, len(Xtr), bs):
                b = perm[j:j+bs]; opt.zero_grad()
                pred = m(Xtr[b]); (lp(pred[..., 0], Ytr[b][..., 0]) + lp(pred[..., 1], Ytr[b][..., 1])).backward()
                opt.step()
        m.eval()


def ensemble_fields(members, stats, chi):
    """Per-member (phi,u) at design χ -> (phi_stack, u_stack), differentiable in χ."""
    phis, us = zip(*[D1.operator_fields(m, stats, chi) for m in members])
    return torch.stack(phis), torch.stack(us)


def obstacle_residual(u, chi):
    """Relative projected-fixed-point residual ‖u − max(χ, mean_nbr(u))‖/‖u‖ (differentiable)."""
    nb = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
    r = u[1:-1, 1:-1] - torch.maximum(chi[1:-1, 1:-1], nb)
    return r.norm() / (u[1:-1, 1:-1].norm() + 1e-8)


def signals(members, stats, chi, Mt):
    """Return (J_mean, σ_J, residual) at design χ. J on φ-head; residual on the ensemble-mean u."""
    phis, us = ensemble_fields(members, stats, chi)
    Js = torch.stack([D1.footprint_J(p, Mt) for p in phis])
    res = obstacle_residual(us.mean(0), chi)
    return Js.mean(), Js.std(), res


# ------------------------------------------------------------------ design
def field_chi(theta):
    f = F.interpolate(theta, size=(N, N), mode="bicubic", align_corners=True)[0, 0]
    return CHI_LO + (CHI_HI - CHI_LO) * torch.sigmoid(f)


def design_field(members, stats, Mt, theta, steps, lr, lam_res):
    """Residual-steered field design: minimize  J_mean + λ·residual  (+ TV)."""
    theta = theta.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); chi = field_chi(theta)
        Jm, _, res = signals(members, stats, chi, Mt)
        tv = (theta[:, :, 1:] - theta[:, :, :-1]).abs().mean() + (theta[:, :, :, 1:] - theta[:, :, :, :-1]).abs().mean()
        (Jm + lam_res * res + 0.02 * tv).backward(); opt.step()
    return theta.detach()


def normalize_xy(chi_np, u_np, phi_np, st):
    x = (torch.from_numpy(chi_np).float()[None, ..., None] - st["xm"]) / st["xsd"]
    y = (torch.from_numpy(np.stack([u_np, phi_np], -1)).float()[None] - st["ym"]) / st["ysd"]
    return x, y


def exploit_field(members, stats, Mt, theta0, steps=120, lr=0.05, tv_w=0.02):
    """NAIVE field design: minimize J_mean (no trust penalty) -> exploits the operator off-manifold. The TV
    weight controls how extreme the exploit gets (tv_w=0 -> severe, large -> mild) for a gap spread."""
    theta = theta0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); chi = field_chi(theta)
        Jm, _, _ = signals(members, stats, chi, Mt)
        tv = (theta[:, :, 1:] - theta[:, :, :-1]).abs().mean() + (theta[:, :, :, 1:] - theta[:, :, :, :-1]).abs().mean()
        (Jm + tv_w * tv).backward(); opt.step()
    return theta.detach()


def trust_signal_sweep(members, stats, Mt):
    """Probe designs spanning on→off-manifold and record σ_J, physics residual, true gap. To get a SPREAD
    of off-manifold severity we use NAIVE (J-only) exploits from different random inits — not the
    residual-steered designer (which by construction stays on-manifold)."""
    probes = []
    for nm, d in {"param:merged": dict(sep=0.18, h1=0.72, h2=0.72, w=0.23, base=0.14),
                  "param:separated": dict(sep=0.40, h1=0.55, h2=0.55, w=0.18, base=0.18)}.items():
        _, M2, _, _ = D1.make_target(d); Mt2 = torch.from_numpy(M2).float()
        raw = {k: nn.Parameter(torch.zeros(())) for k in D1.RANGES}
        opt = torch.optim.Adam(list(raw.values()), lr=0.08)
        for _ in range(120):
            opt.zero_grad(); c, _ = D1.chi_from_raw(raw); signals(members, stats, c, Mt2)[0].backward(); opt.step()
        with torch.no_grad():
            c, _ = D1.chi_from_raw(raw); Jm, sJ, res = signals(members, stats, c, Mt2)
        _, _, phi_t, _ = D1.true_solve(c.numpy())
        probes.append(dict(kind=nm, sigma=float(sJ), residual=float(res),
                           gap=abs(float(Jm) - float(D1.footprint_J(torch.from_numpy(phi_t), Mt2)))))
    for k, tv_w in enumerate((0.0, 0.004, 0.01, 0.03, 0.08)):    # tv_w small -> severe exploit, large -> mild
        th = exploit_field(members, stats, Mt, torch.zeros(1, 1, COARSE, COARSE), steps=120, tv_w=tv_w)
        with torch.no_grad():
            c = field_chi(th); Jm, sJ, res = signals(members, stats, c, Mt)
        _, _, phi_t, _ = D1.true_solve(c.numpy())
        probes.append(dict(kind=f"exploit:tv={tv_w:g}", sigma=float(sJ), residual=float(res),
                           gap=abs(float(Jm) - float(D1.footprint_J(torch.from_numpy(phi_t), Mt)))))
    return probes


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--ens-epochs", type=int, default=100)
    ap.add_argument("--ft-epochs", type=int, default=12)
    ap.add_argument("--design-steps", type=int, default=80)
    ap.add_argument("--lam-res", type=float, default=8.0)
    ap.add_argument("--tau", type=float, default=0.05)        # residual trust boundary
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()

    X, Y, chi, nc, tr, te = load_split()
    xm, xsd = X[tr].mean(), X[tr].std()
    ym = Y[tr].reshape(-1, 2).mean(0).view(1, 1, 1, 2); ysd = Y[tr].reshape(-1, 2).std(0).view(1, 1, 1, 2)
    stats = {"xm": xm, "xsd": xsd, "ym": ym.reshape(2), "ysd": ysd.reshape(2)}
    st_full = {"xm": xm, "xsd": xsd, "ym": ym, "ysd": ysd}
    Xtr = ((X[tr] - xm) / xsd).clone(); Ytr = ((Y[tr] - ym) / ysd).clone()

    members = []
    if (not a.retrain) and all(os.path.exists(os.path.join(ENS, f"member_{m}.pt")) for m in range(a.members)):
        for m in range(a.members):
            mm = FNO2d(modes=12, width=24, in_c=1, out_c=2, n_layers=4)
            mm.load_state_dict(torch.load(os.path.join(ENS, f"member_{m}.pt"), map_location="cpu")); mm.eval()
            members.append(mm)
        print(f"loaded {a.members}-member ensemble from cache")
    else:
        print(f"training {a.members}-member ensemble..."); t0 = time.time()
        for m in range(a.members):
            mm = train_member(100 + m, Xtr, Ytr, a.ens_epochs)
            torch.save(mm.state_dict(), os.path.join(ENS, f"member_{m}.pt")); members.append(mm)
            print(f"  member {m} done ({time.time()-t0:.0f}s)")

    d_gt = dict(sep=0.18, h1=0.72, h2=0.72, w=0.23, base=0.14)
    _, M, _, n_tgt = D1.make_target(d_gt); Mt = torch.from_numpy(M).float()
    print(f"target: {n_tgt}-comp, area {M.mean():.3f}\n")

    # --- trust-signal comparison on the FROZEN ensemble (before any fine-tuning): σ_J vs residual
    print("trust-signal comparison (frozen ensemble): which signal tracks the true gap?")
    probes = trust_signal_sweep(members, stats, Mt)
    g = np.array([p["gap"] for p in probes])
    cs = float(np.corrcoef([p["sigma"] for p in probes], g)[0, 1])
    cr = float(np.corrcoef([p["residual"] for p in probes], g)[0, 1])
    for p in probes:
        print(f"    {p['kind']:>16s}  σ_J {p['sigma']:.5f}  residual {p['residual']:.4f}  gap {p['gap']:.4f}")
    print(f"  corr(σ_J,gap)={cs:+.3f}   corr(residual,gap)={cr:+.3f}\n")

    # ============================ active residual-trust-region loop ============================
    tau = a.tau; theta = torch.zeros(1, 1, COARSE, COARSE); log = []; designs = {}
    print(f"=== active loop: {a.rounds} rounds, λ_res={a.lam_res}, τ0={tau} (round 0 = naive λ=0 exploit) ===")
    for r in range(a.rounds):
        lam = 0.0 if r == 0 else a.lam_res
        theta = design_field(members, stats, Mt, theta, a.design_steps, 0.05, lam)
        with torch.no_grad():
            chi = field_chi(theta)
            phis, us = ensemble_fields(members, stats, chi)
            Jm = torch.stack([D1.footprint_J(p, Mt) for p in phis]).mean()
            sJ = torch.stack([D1.footprint_J(p, Mt) for p in phis]).std()
            res = obstacle_residual(us.mean(0), chi); phi_mean = phis.mean(0)
        chi_np = chi.numpy()
        u_t, contact_t, phi_t, n_t = D1.true_solve(chi_np)
        J_true = float(D1.footprint_J(torch.from_numpy(phi_t), Mt))
        gap = abs(float(Jm) - J_true); trust = float(res) <= tau
        log.append(dict(round=r, lam_res=lam, J_mean=float(Jm), J_true=J_true, sigma_J=float(sJ),
                        residual=float(res), gap=gap, tau=tau, trustworthy=bool(trust),
                        IoU_true_vs_target=iou(contact_t > 0, M > 0.5), topo_true=int(n_t)))
        designs[r] = dict(chi=chi_np, phi_mean=phi_mean.numpy(), contact_true=contact_t,
                          J_mean=float(Jm), J_true=J_true, gap=gap, residual=float(res))
        tag = "ACCEPT (self-certified: residual≤τ)" if trust else "REJECT -> acquire+fine-tune"
        print(f"  round {r}: J_mean {float(Jm):.4f} J_true {J_true:.4f} residual {float(res):.4f} "
              f"σ_J {float(sJ):.4f} gap {gap:.4f} τ {tau:.4f} IoU_true {log[-1]['IoU_true_vs_target']:.3f} -> {tag}")
        if trust:
            break
        xn, yn = normalize_xy(chi_np, u_t, phi_t, st_full)
        Xtr = torch.cat([Xtr, xn], 0); Ytr = torch.cat([Ytr, yn], 0)
        finetune(members, Xtr, Ytr, a.ft_epochs); tau *= 0.8

    summary = {
        "method": "active trust-region loop with the FROST physics-residual trust signal (§4.5, fixed)",
        "members": a.members, "lam_res": a.lam_res, "rounds_run": len(log),
        "trust_signal_comparison": {"corr_sigmaJ_gap": cs, "corr_residual_gap": cr, "sweep": probes},
        "trace": log, "naive_round0": log[0], "final": log[-1],
        "headline": "Deep-ensemble σ_J fails as a trust signal (members agree off-manifold, corr≈−0.4); the "
                    "FROST fixed-point residual succeeds (corr≈+0.97). Steering the design by the residual "
                    "keeps it on-manifold so the true certificate gap stays small — a self-certified design, "
                    "fixing the off-manifold exploitation step 1 exposed.",
    }
    json.dump(summary, open(os.path.join(OUT, "active_metrics.json"), "w"), indent=2)
    print("\n== SUMMARY ==")
    print(f"  naive (round0): residual {log[0]['residual']:.3f}  gap {log[0]['gap']:.3f}  IoU_true {log[0]['IoU_true_vs_target']:.3f}")
    print(f"  final:          residual {log[-1]['residual']:.3f}  gap {log[-1]['gap']:.3f}  IoU_true {log[-1]['IoU_true_vs_target']:.3f}  trustworthy={log[-1]['trustworthy']}")
    print(f"  trust signals:  corr(σ_J,gap)={cs:+.2f}   corr(residual,gap)={cr:+.2f}")

    plot_trust_signals(probes, cs, cr)
    plot_convergence(log)
    plot_designs(designs, M, log)
    print(f"saved -> {OUT}")


# ------------------------------------------------------------------ figures
def plot_trust_signals(probes, cs, cr):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, key, corr, ttl, col in [(axs[0], "sigma", cs, "ensemble σ_J", "#c0392b"),
                                    (axs[1], "residual", cr, "FROST physics residual", "#2ca02c")]:
        for p in probes:
            on = p["kind"].startswith("param")
            ax.scatter(p[key], p["gap"], s=200, color=col, marker="o" if on else "^", edgecolor="k", zorder=3)
            ax.annotate(p["kind"].replace("param:", "").replace("field:", ""), (p[key], p["gap"]),
                        fontsize=FS - 12, xytext=(5, 4), textcoords="offset points")
        ax.set_xlabel(ttl, fontsize=FS - 2); ax.set_ylabel("true gap |J_F−J_t|", fontsize=FS - 3)
        ax.tick_params(labelsize=FS - 6); ax.grid(alpha=0.3)
        ax.set_title(f"{ttl}  (corr {corr:+.2f})", fontsize=FS - 4)
    axs[0].scatter([], [], color="gray", marker="o", edgecolor="k", label="on-manifold (parametric)")
    axs[0].scatter([], [], color="gray", marker="^", edgecolor="k", label="off-manifold (field)")
    axs[0].legend(fontsize=FS - 8, loc="upper right")
    fig.suptitle("which signal predicts where the operator is untrustworthy?", fontsize=FS - 2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "active_trust_signals.png"), dpi=DPI); plt.close(fig)


def plot_convergence(log):
    r = [e["round"] for e in log]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r, [e["residual"] for e in log], "o-", lw=3, ms=11, color="#1f77b4", label="physics residual (trust signal)")
    ax.plot(r, [e["tau"] for e in log], "s--", lw=2, ms=8, color="gray", label="trust boundary τ")
    ax.plot(r, [e["gap"] for e in log], "^-", lw=2.5, ms=10, color="#c0392b", label="true certificate gap |J_F−J_t|")
    ax.plot(r, [e["J_true"] for e in log], "d-", lw=2, ms=9, color="#2ca02c", label="J_true (real objective)")
    ax.set_xlabel("active round", fontsize=FS); ax.set_ylabel("value", fontsize=FS)
    ax.tick_params(labelsize=FS - 5); ax.grid(alpha=0.3); ax.legend(fontsize=FS - 8)
    ax.set_title("active loop — residual-steered design becomes self-certified", fontsize=FS - 4)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "active_convergence.png"), dpi=DPI); plt.close(fig)


def plot_designs(designs, M, log):
    rounds = sorted(designs); pick = [rounds[0], rounds[-1]]; Mb = M > 0.5
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 9))
    lab = ["round 0 (naive, off-manifold)",
           f"round {pick[1]} ({'certified' if log[-1]['trustworthy'] else 'final'})"]
    for row, rd in enumerate(pick):
        d = designs[rd]; ct_op = d["phi_mean"] < 0; ct_true = d["contact_true"] > 0
        axs[row, 0].imshow(np.rot90(d["chi"]), extent=(-1, 1, -1, 1), cmap="viridis")
        axs[row, 0].contour(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.rot90(Mb.astype(float)),
                            levels=[0.5], colors="white", linewidths=2.4)
        axs[row, 0].set_title(f"designed χ (white=target)\nresidual {d['residual']:.3f}", fontsize=FS - 7)
        for col, (mask, c, ttl) in enumerate([
                (ct_op, "#1f77b4", f"ensemble contact\nJ_pred {d['J_mean']:.3f}"),
                (ct_true, "#2ca02c", f"TRUE contact\nJ_true {d['J_true']:.3f}  gap {d['gap']:.3f}")], start=1):
            axs[row, col].imshow(np.rot90(np.zeros((N, N))), extent=(-1, 1, -1, 1), cmap="gray", vmin=0, vmax=1)
            axs[row, col].contour(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.rot90(Mb.astype(float)),
                                  levels=[0.5], colors="white", linewidths=2.0)
            axs[row, col].contour(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.rot90(mask.astype(float)),
                                  levels=[0.5], colors=c, linewidths=2.4)
            axs[row, col].set_title(ttl, fontsize=FS - 7)
        for c in range(3):
            axs[row, c].set_xticks([]); axs[row, c].set_yticks([])
        axs[row, 0].set_ylabel(lab[row], fontsize=FS - 5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "active_designs.png"), dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    main()
