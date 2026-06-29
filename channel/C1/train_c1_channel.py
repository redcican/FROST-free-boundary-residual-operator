"""
Stage 2/3 — FROST C1 forward operator on the channel benchmark:  G: (baffle, v) -> (C, P).

Unlike obstacle/stefan/tumour (where the free boundary is a SOLVED output), here the baffle is the
INPUT design and the operator predicts the resulting flow fields. Two studies via flags:

  --rep phi   : condition on the LEVEL SET  [φ, ∂xφ, ∂yφ, v]      (FROST: smooth distance-to-interface)
  --rep gamma : condition on the DENSITY    [γ, ∂xγ, ∂yγ, v]      (NTO-style baseline)
                (same channel count; only the baffle encoding differs)

  --split random : stratified 80/20 by topology
  --split topo   : train on 1- & 2-baffle designs, TEST on 3-baffle designs (topology extrapolation)

Outputs -> results/<rep>_<split>/ : model.pt, metrics.json, loss_curve.png, predictions.png
Run:  python train_c1_channel.py --rep phi --split random [--epochs 100]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno import FNO2d, LpLoss

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
torch.manual_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "channel_train.npy")
FS, FIG_DPI = 22, 600


def build(rep, split, test_frac=0.2):
    d = np.load(DATA, allow_pickle=True)
    H, W = d[0]["u"].shape
    base = "phi" if rep == "phi" else "gamma"
    X, Y, nc, vel = [], [], [], []
    for s in d:
        b = s[base].astype(np.float32)
        gx, gy = np.gradient(b)
        v = np.full_like(b, float(s["params"][0]))
        X.append(np.stack([b, gx.astype(np.float32), gy.astype(np.float32), v], -1))
        Y.append(np.stack([s["u"], s["P"]], -1).astype(np.float32))
        nc.append(int(s["n_components"])); vel.append(float(s["params"][0]))
    X = np.array(X); Y = np.array(Y); nc = np.array(nc)
    rng = np.random.default_rng(0)
    if split == "topo":                                  # train 1&2 baffles, test 3 baffles
        tr = np.where(nc < 3)[0]; te = np.where(nc == 3)[0]
    else:                                                # stratified random by topology
        tr, te = [], []
        for c in np.unique(nc):
            idx = np.where(nc == c)[0]; rng.shuffle(idx)
            k = max(1, int(round(len(idx) * test_frac)))
            te += list(idx[:k]); tr += list(idx[k:])
        tr, te = np.array(sorted(tr)), np.array(sorted(te))
    return torch.from_numpy(X), torch.from_numpy(Y), nc, np.array(tr), np.array(te), H, W


def evaluate(model, X, Y, nc, idx, stats):
    xm, xs, ym, ys = stats
    rows = []
    with torch.no_grad():
        for e in idx:
            pred = (model(((X[e:e+1] - xm) / xs))[0] * ys + ym).numpy()
            C_p, P_p = pred[..., 0], pred[..., 1]
            C_g, P_g = Y[e, ..., 0].numpy(), Y[e, ..., 1].numpy()
            rows.append(dict(
                nc=int(nc[e]),
                C_l2=float(np.linalg.norm(C_p - C_g) / (np.linalg.norm(C_g) + 1e-8)),
                P_l2=float(np.linalg.norm(P_p - P_g) / (np.linalg.norm(P_g) + 1e-8))))
    return rows


def agg(rows, key, sub=None):
    v = [r[key] for r in rows if (sub is None or r["nc"] == sub)]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", choices=["phi", "gamma"], default="phi")
    ap.add_argument("--split", choices=["random", "topo"], default="random")
    ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()
    out = os.path.join(HERE, "results", f"{a.rep}_{a.split}"); os.makedirs(out, exist_ok=True)

    X, Y, nc, tr, te, H, W = build(a.rep, a.split)
    print(f"[{a.rep}/{a.split}] train {len(tr)}  test {len(te)}  grid {H}x{W}  "
          f"test topo {dict(zip(*np.unique(nc[te], return_counts=True)))}")

    xm = X[tr].reshape(-1, 4).mean(0).view(1, 1, 1, 4); xs = X[tr].reshape(-1, 4).std(0).view(1, 1, 1, 4) + 1e-6
    ym = Y[tr].reshape(-1, 2).mean(0).view(1, 1, 1, 2); ys = Y[tr].reshape(-1, 2).std(0).view(1, 1, 1, 2) + 1e-6
    stats = (xm, xs, ym.reshape(2), ys.reshape(2))
    Xn, Yn = (X - xm) / xs, (Y - ym) / ys

    model = FNO2d(modes=16, width=32, in_c=4, out_c=2, n_layers=4)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lp = LpLoss()
    Xtr, Ytr = Xn[tr], Yn[tr]
    bs = 16; hist = []
    print(f"FNO {n_par/1e3:.0f}k  epochs {a.epochs}")
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(tr)); tot = 0.0
        for j in range(0, len(tr), bs):
            b = perm[j:j+bs]; opt.zero_grad()
            pred = model(Xtr[b])
            loss = lp(pred[..., 0], Ytr[b][..., 0]) + lp(pred[..., 1], Ytr[b][..., 1])
            loss.backward(); opt.step(); tot += loss.item() * len(b)
        sched.step(); hist.append(tot / len(tr))
        if ep % 15 == 0 or ep == a.epochs - 1:
            r = evaluate(model, X, Y, nc, te, stats)
            print(f"  ep {ep:4d}  train {hist[-1]:.4f}  test C {agg(r,'C_l2'):.4f}  P {agg(r,'P_l2'):.4f}", flush=True)

    rte = evaluate(model, X, Y, nc, te, stats)
    metrics = {"rep": a.rep, "split": a.split, "n_params": int(n_par), "epochs": a.epochs,
               "train_time_s": round(time.time() - t0, 1), "n_train": int(len(tr)), "n_test": int(len(te)),
               "test": {"C_relL2": agg(rte, "C_l2"), "P_relL2": agg(rte, "P_l2"),
                        "by_topology": {int(c): {"C_relL2": agg(rte, "C_l2", c), "P_relL2": agg(rte, "P_l2", c),
                                                 "n": int((nc[te] == c).sum())} for c in np.unique(nc[te])}}}
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    torch.save({"model": model.state_dict(), "stats": {"xm": xm, "xs": xs, "ym": ym, "ys": ys},
                "rep": a.rep, "grid": [H, W]}, os.path.join(out, "model.pt"))
    np.save(os.path.join(out, "hist.npy"), np.array(hist, np.float32))
    print(f"\n[{a.rep}/{a.split}] TEST C {metrics['test']['C_relL2']:.4f}  P {metrics['test']['P_relL2']:.4f}")

    plt.figure(figsize=(7.5, 5)); plt.semilogy(hist, lw=2)
    plt.xlabel("epoch", fontsize=FS); plt.ylabel("train rel-L2 (C+P)", fontsize=FS)
    plt.xticks(fontsize=FS - 4); plt.yticks(fontsize=FS - 4); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "loss_curve.png"), dpi=FIG_DPI); plt.close()

    plot_predictions(model, X, Y, nc, te, stats, out, a)
    print(f"saved -> results/{a.rep}_{a.split}/")


def plot_predictions(model, X, Y, nc, te, stats, out, a):
    xm, xs2, ym, ys = stats; ym = ym.reshape(2); ys = ys.reshape(2)
    picks = []
    for c in (1, 2, 3):
        cand = [e for e in te if nc[e] == c]
        if cand:
            picks.append(cand[0])
    ext = (0, 0.02, 0, 0.01)
    fig, axs = plt.subplots(len(picks), 6, figsize=(22, 2.6 * len(picks)))
    if len(picks) == 1:
        axs = axs[None, :]
    for r, e in enumerate(picks):
        with torch.no_grad():
            pred = (model(((X[e:e+1] - xm) / xs2))[0] * ys + ym).numpy()
        C_g, P_g = Y[e, ..., 0].numpy(), Y[e, ..., 1].numpy()
        C_p, P_p = pred[..., 0], pred[..., 1]
        mask = X[e, ..., 0].numpy() < 0 if a.rep == "phi" else X[e, ..., 0].numpy() < 0.5
        panels = [(C_g, "C GT", "jet", 0, 1), (C_p, "C pred", "jet", 0, 1),
                  (np.abs(C_g - C_p), f"|ΔC| {np.abs(C_g-C_p).max():.2f}", "magma", 0, None),
                  (P_g, "P GT", "viridis", None, None), (P_p, "P pred", "viridis", None, None),
                  (np.abs(P_g - P_p), f"|ΔP| {np.abs(P_g-P_p).max():.2f}", "magma", 0, None)]
        for cc, (img, ttl, cmap, vmin, vmax) in enumerate(panels):
            ax = axs[r, cc]
            im = ax.imshow(np.ma.masked_where(mask, img).T if "Δ" not in ttl else img.T,
                           origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            if "Δ" in ttl:
                fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).ax.tick_params(labelsize=FS - 8)
            ax.set_title(ttl, fontsize=FS - 4); ax.set_xticks([]); ax.set_yticks([])
        axs[r, 0].set_ylabel(f"{nc[e]} baffle(s)", fontsize=FS - 4)
    fig.tight_layout(); fig.savefig(os.path.join(out, "predictions.png"), dpi=FIG_DPI); plt.close(fig)


if __name__ == "__main__":
    main()
