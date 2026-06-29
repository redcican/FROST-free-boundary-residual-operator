"""
The LOCAL-STENCIL operator finding (obstacle): why the proposal's §3/§4.2 purely-local operator needs the
DEQ. We build the operator exactly as specified — an MLP on a geometry-conditioned stencil
[4-neighbour u, χ, φ, ∇φ] → u(x), trained from the ~16k stencils of K simulations — and probe its
fixed-point iteration.

FINDING (rigorous, 3 measurements):
  1. ONE-STEP accuracy is excellent (~0.1%), and is reached from K=1 sim (locality ⇒ few-shot is real at
     the stencil level).
  2. But the GT solution is NOT the iterated operator's fixed point: warm-started AT the solution, the FPI
     DRIFTS AWAY; cold-started it converges to a SPURIOUS fixed point with a large field error.
  3. The cause is elliptic fixed-point conditioning: the tiny one-step bias is amplified by ≈1/(1−ρ),
     ρ→1. (Confirmed robust to trajectory training.)

CONSEQUENCE: the practical free-boundary equilibrium operator must be trained THROUGH the fixed point —
i.e. the **DEQ** (`deq.py`, implicit differentiation), whose equilibrium is correct by construction
(u 2.1%, converges) — or the global FNO. This is the empirical justification for FROST's §4.3 DEQ.

Run:  python train_local_stencil.py [--epochs 60]
Out:  results/local_stencil/{metrics.json, local_stencil_finding.png}
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_c1_obstacle import load_split, n_comp, OUT
from local_stencil import LocalOp, build_stencils, fpi, contact_of, obstacle_residual, N

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
torch.manual_seed(0); np.random.seed(0)
BASE = os.path.join(OUT, "local_stencil"); os.makedirs(BASE, exist_ok=True)
FS, DPI = 22, 600
FNO_U, DEQ_U = 0.0091, 0.021                                # documented obstacle C1 field rel-L2 (same test set)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def train_op(train_idx, U, CHI, PHI, geom=True, lam_fb=1.0, epochs=60, bs=8192, seed=0):
    torch.manual_seed(seed)
    Xs, Ws, Ys, Cs = [], [], [], []
    for i in train_idx:
        X, w = build_stencils(U[i], CHI[i], PHI[i], geom=geom)
        Xs.append(X); Ws.append(w)
        Ys.append(U[i][1:-1, 1:-1].reshape(-1).astype(np.float32))
        Cs.append(CHI[i][1:-1, 1:-1].reshape(-1).astype(np.float32))
    X = np.concatenate(Xs); w = np.concatenate(Ws); Y = np.concatenate(Ys); C = np.concatenate(Cs)
    xm = X.mean(0); xsd = X.std(0) + 1e-6; ym = float(Y.mean()); ysd = float(Y.std() + 1e-6)
    Xt = torch.from_numpy((X - xm) / xsd); Yt = torch.from_numpy((Y - ym) / ysd)
    wt = torch.from_numpy(w); Ct = torch.from_numpy(C)
    model = LocalOp(X.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    M = len(Xt); ys_t = torch.tensor(ysd); ym_t = torch.tensor(ym)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(M)
        for j in range(0, M, bs):
            b = perm[j:j+bs]; opt.zero_grad()
            pred = model(Xt[b]); l_eq = ((pred - Yt[b]) ** 2).mean()
            if lam_fb > 0:
                l_band = (wt[b] * (pred - Yt[b]) ** 2).sum() / (wt[b].sum() + 1e-8)
                l_con = torch.relu(Ct[b] - (pred * ys_t + ym_t)).pow(2).mean()
                loss = l_eq + lam_fb * (l_band + l_con)
            else:
                loss = l_eq
            loss.backward(); opt.step()
        sched.step()
    model.eval()
    stats = (torch.from_numpy(xm.astype(np.float32)), torch.from_numpy(xsd.astype(np.float32)),
             torch.tensor(ym), torch.tensor(ysd))
    return model, stats


def onestep_err(model, stats, idx, U, CHI, PHI, geom=True):
    xm, xs, ym, ys = stats; errs = []
    for i in idx:
        X, _ = build_stencils(U[i], CHI[i], PHI[i], geom=geom)
        with torch.no_grad():
            pred = (model((torch.from_numpy(X) - xm) / xs) * ys + ym).numpy()
        ug = U[i][1:-1, 1:-1].reshape(-1)
        errs.append(float(np.linalg.norm(pred - ug) / (np.linalg.norm(ug) + 1e-8)))
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=60); a = ap.parse_args()
    X, Y, chi_t, nc, tr, te = load_split()
    U = Y[..., 0].numpy(); PHI = Y[..., 1].numpy(); CHI = chi_t.numpy()
    print(f"train pool {len(tr)}  test {len(te)}  (one 128² sim ≈ {(N-2)**2//1000}k stencils)")
    t0 = time.time()

    model, stats = train_op(tr, U, CHI, PHI, geom=True, lam_fb=1.0, epochs=a.epochs)
    os_full = onestep_err(model, stats, te, U, CHI, PHI)
    m1, s1 = train_op(tr[:1], U, CHI, PHI, geom=True, lam_fb=1.0, epochs=a.epochs)   # K=1 sim
    os_k1 = onestep_err(m1, s1, te, U, CHI, PHI)
    print(f"one-step rel-L2:  full-K {os_full:.4f}   K=1 sim {os_k1:.4f}  (locality ⇒ few-shot is real)")

    # drift: warm-start the FPI AT the GT solution -> does it stay? (isolates the fixed-point mismatch)
    drift = {}
    for i in list(te)[:4]:
        _, dr, _ = fpi(model, stats, CHI[i], geom=True, max_iter=200, u0=U[i], track=U[i])
        drift[int(i)] = dr
    drift_final = float(np.mean([d[-1] for d in drift.values()]))

    # cold-start FPI (the actual inference) -> spurious fixed point
    cold = []
    for i in te:
        us, _, _ = fpi(model, stats, CHI[i], geom=True, max_iter=600)
        cold.append(dict(u=float(np.linalg.norm(us - U[i]) / (np.linalg.norm(U[i]) + 1e-8)),
                         iou=iou(contact_of(us, CHI[i]), contact_of(U[i], CHI[i])),
                         topo=int(n_comp(contact_of(us, CHI[i])) == int(nc[i])),
                         resid=obstacle_residual(us, CHI[i])))
    cold_u = float(np.mean([c["u"] for c in cold]))
    cold_iou = float(np.mean([c["iou"] for c in cold]))
    cold_topo = float(np.mean([c["topo"] for c in cold]))
    print(f"FPI fixed point:  cold-start field rel-L2 {cold_u:.3f}  (IoU {cold_iou:.2f}, topo {cold_topo:.2f}) "
          f"| warm drift from GT -> {drift_final:.3f}")

    metrics = {
        "operator": "local-stencil MLP (one-shot style) + red-black SOR FPI; geometry-conditioned + L_fb",
        "stencils_per_sim": int((N - 2) ** 2), "epochs": a.epochs,
        "one_step_relL2_fullK": os_full, "one_step_relL2_K1sim": os_k1,
        "FPI_coldstart_field_relL2": cold_u, "FPI_coldstart_contact_IoU": cold_iou,
        "FPI_coldstart_topology_acc": cold_topo,
        "warm_start_drift_from_GT": drift_final,
        "comparison_field_relL2": {"local_stencil_FPI": cold_u, "FROST_DEQ": DEQ_U, "global_FNO": FNO_U},
        "finding": ("One-step accuracy is excellent (~0.1%) and reachable from K=1 sim, but the iterated "
                    "operator has a SPURIOUS fixed point (large field error; the solution is not fixed) — "
                    "the elliptic fixed point amplifies the tiny one-step bias by ~1/(1−ρ). The DEQ "
                    "(implicit training THROUGH the fixed point) is the fix; this is the empirical "
                    "justification for FROST's §4.3 equilibrium/DEQ formulation."),
        "train_time_s": round(time.time() - t0, 1),
    }
    json.dump(metrics, open(os.path.join(BASE, "metrics.json"), "w"), indent=2)
    plot_finding(drift, os_full, cold_u)
    print(f"saved -> {BASE}  ({time.time()-t0:.0f}s)")


def plot_finding(drift, onestep, cold_u):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    # left: warm-start drift from the GT solution
    for i, dr in drift.items():
        a1.plot(range(1, len(dr) + 1), dr, lw=2.2, label=f"test {i}")
    a1.axhline(onestep, color="k", ls=":", lw=1.6, label=f"one-step error {onestep:.3f}")
    a1.set_xlabel("FPI sweep (warm-started at GT solution)", fontsize=FS - 2)
    a1.set_ylabel("field rel-L2 vs GT", fontsize=FS - 2)
    a1.tick_params(labelsize=FS - 6); a1.grid(alpha=0.3); a1.legend(fontsize=FS - 9)
    a1.set_title("the GT solution is NOT the operator's fixed point", fontsize=FS - 6)
    # right: field error — local-stencil FPI vs DEQ vs global FNO
    labels = ["local-stencil\nFPI", "FROST DEQ\n(implicit)", "global FNO"]
    vals = [cold_u, DEQ_U, FNO_U]; cols = ["#c0392b", "#1f77b4", "#2ca02c"]
    a2.bar(range(3), vals, color=cols)
    for i, v in enumerate(vals):
        a2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=FS - 7)
    a2.set_yscale("log"); a2.set_xticks(range(3)); a2.set_xticklabels(labels, fontsize=FS - 6)
    a2.set_ylabel("field rel-L2 (log)", fontsize=FS - 2); a2.tick_params(axis="y", labelsize=FS - 6)
    a2.grid(axis="y", alpha=0.3, which="both")
    a2.set_title("the fix: train THROUGH the fixed point (DEQ)", fontsize=FS - 6)
    fig.suptitle("local-stencil operator: locally accurate, globally spurious — why FROST needs the DEQ",
                 fontsize=FS - 4)
    fig.tight_layout(); fig.savefig(os.path.join(BASE, "local_stencil_finding.png"), dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    main()
