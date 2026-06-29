"""
FROST C1 — forward operator on the Stefan benchmark (time-dependent, topology-changing).

Mirrors the obstacle C1 experiment, extended to time + topology change. A single time-conditioned
2D FNO learns the operator

    [ φ₀ (initial seeds), L, T_cold, t ]  ->  ( T(t), φ(t) )

i.e. given the initial cold nuclei (level set φ₀), the physics (latent heat L, nucleus temperature
T_cold) and a query time t, predict the temperature field T and the level set φ at that time. The
level set φ(t) changes TOPOLOGY over time (N grains coalesce to 1), so the operator must produce a
topology-changing free boundary — the capability a single diffeomorphism cannot represent.

Key question: does the operator track the field, the free boundary Γ={T=0}, and the N→1 topology
through the merge event, on held-out instances? Reports field/level-set accuracy, solid IoU, and
topology accuracy per frame (incl. through the merge).

Outputs -> C1/results/ : model.pt, metrics.json, loss_curve.png, predictions.png, topo_vs_time.png
Run:  python train_c1_stefan.py [epochs]
"""
import os, sys, json, time
import numpy as np
import torch
from scipy.ndimage import label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno import FNO2d, LpLoss

torch.set_num_threads(8)
torch.manual_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "stefan.npy")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
CONN = np.ones((3, 3), int)


DS = 2                                                       # downsample 128 -> 64 (smooth fields; ~4x faster)


def load_split(test_frac=0.2):
    """Build time-conditioned (sample,frame) examples; split by SAMPLE (no frame leakage)."""
    d = np.load(DATA, allow_pickle=True)
    N = len(d); T = d[0]["phi"].shape[0]; H = d[0]["phi"].shape[1] // DS
    rng = np.random.default_rng(0)
    ns = np.array([int(x["params"][0]) for x in d])
    tr_s, te_s = [], []
    for c in np.unique(ns):                                  # stratify by seed count
        idx = np.where(ns == c)[0]; rng.shuffle(idx)
        k = max(1, int(round(len(idx) * test_frac)))
        te_s += list(idx[:k]); tr_s += list(idx[k:])
    tr_s, te_s = np.array(sorted(tr_s)), np.array(sorted(te_s))

    tnorm = np.linspace(0, 1, T).astype(np.float32)
    X = np.zeros((N * T, H, H, 4), np.float32)               # [phi0, L, T_cold, t]
    Y = np.zeros((N * T, H, H, 2), np.float32)               # [T(t), phi(t)]
    meta = np.zeros((N * T, 3), np.float32)                  # sample, frame, merge_step
    for i, s in enumerate(d):
        phi0 = s["phi"][0][::DS, ::DS]; L = float(s["params"][1]); Tc = float(s["params"][2])
        for t in range(T):
            e = i * T + t
            X[e, ..., 0] = phi0; X[e, ..., 1] = L; X[e, ..., 2] = Tc; X[e, ..., 3] = tnorm[t]
            Y[e, ..., 0] = s["u"][t][::DS, ::DS]; Y[e, ..., 1] = s["phi"][t][::DS, ::DS]
            meta[e] = [i, t, s["merge_step"]]
    exp = lambda S: np.concatenate([np.arange(i * T, i * T + T) for i in S])
    return (torch.from_numpy(X), torch.from_numpy(Y), meta, T,
            exp(tr_s), exp(te_s), tr_s, te_s)


def n_comp(mask):
    return int(label(mask, structure=CONN)[1])


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def evaluate(model, X, Y, meta, idx, stats):
    xm, xs, ym, ys = stats
    rows = []
    with torch.no_grad():
        for e in idx:
            pred = (model(((X[e:e+1] - xm) / xs))[0] * ys + ym).numpy()
            T_p, phi_p = pred[..., 0], pred[..., 1]
            T_g, phi_g = Y[e, ..., 0].numpy(), Y[e, ..., 1].numpy()
            sp, sg = phi_p < 0, phi_g < 0
            rows.append(dict(
                t=int(meta[e, 1]),
                u_l2=float(np.linalg.norm(T_p - T_g) / (np.linalg.norm(T_g) + 1e-8)),
                phi_l2=float(np.linalg.norm(phi_p - phi_g) / (np.linalg.norm(phi_g) + 1e-8)),
                iou=iou(sp, sg), topo=int(n_comp(sp) == n_comp(sg))))   # both at working resolution
    return rows


def agg(rows, key, frame=None):
    v = [r[key] for r in rows if (frame is None or r["t"] == frame)]
    return float(np.mean(v)) if v else None


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    X, Y, meta, T, tr, te, tr_s, te_s = load_split()
    print(f"train {len(tr_s)} samples ({len(tr)} frames)  test {len(te_s)} samples ({len(te)} frames)")

    xm = X[tr].reshape(-1, 4).mean(0).view(1, 1, 1, 4); xs = X[tr].reshape(-1, 4).std(0).view(1, 1, 1, 4)
    ym = Y[tr].reshape(-1, 2).mean(0).view(1, 1, 1, 2); ys = Y[tr].reshape(-1, 2).std(0).view(1, 1, 1, 2)
    stats = (xm, xs, ym.reshape(2), ys.reshape(2))
    Xn, Yn = (X - xm) / xs, (Y - ym) / ys

    model = FNO2d(modes=16, width=32, in_c=4, out_c=2, n_layers=4)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lp = LpLoss()
    Xtr, Ytr = Xn[tr], Yn[tr]
    bs = 32; hist = []
    print(f"FNO params {n_par/1e3:.0f}k  epochs {epochs}")
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(tr)); tot = 0.0
        for j in range(0, len(tr), bs):
            b = perm[j:j+bs]; opt.zero_grad()
            pred = model(Xtr[b])
            loss = lp(pred[..., 0], Ytr[b][..., 0]) + lp(pred[..., 1], Ytr[b][..., 1])
            loss.backward(); opt.step(); tot += loss.item() * len(b)
        sched.step(); hist.append(tot / len(tr))
        if ep % 15 == 0 or ep == epochs - 1:
            r = evaluate(model, X, Y, meta, te, stats)
            print(f"  ep {ep:4d}  train {hist[-1]:.4f}  test u_l2 {agg(r,'u_l2'):.4f} "
                  f"phi_l2 {agg(r,'phi_l2'):.4f} IoU {agg(r,'iou'):.3f} topo {agg(r,'topo'):.3f}", flush=True)

    rte = evaluate(model, X, Y, meta, te, stats)
    per_frame_topo = {t: agg(rte, "topo", t) for t in range(T)}
    metrics = {
        "n_params": int(n_par), "epochs": epochs, "train_time_s": round(time.time() - t0, 1),
        "n_train_samples": int(len(tr_s)), "n_test_samples": int(len(te_s)), "frames": T,
        "test": {"u_relL2": agg(rte, "u_l2"), "phi_relL2": agg(rte, "phi_l2"),
                 "solid_IoU": agg(rte, "iou"), "topology_acc": agg(rte, "topo"),
                 "topology_acc_per_frame": per_frame_topo},
    }
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    torch.save({"model": model.state_dict(),
                "stats": {"xm": xm, "xs": xs, "ym": ym, "ys": ys}}, os.path.join(OUT, "model.pt"))
    print("\n== TEST =="); print(json.dumps({k: v for k, v in metrics["test"].items()
                                             if k != "topology_acc_per_frame"}, indent=2))

    # loss curve
    plt.figure(figsize=(7.5, 5)); plt.semilogy(hist, lw=2)
    plt.xlabel("epoch", fontsize=FS); plt.ylabel("train rel-L2 (T+φ)", fontsize=FS)
    plt.xticks(fontsize=FS - 4); plt.yticks(fontsize=FS - 4)
    plt.title("FROST C1 stefan — training loss", fontsize=FS); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "loss_curve.png"), dpi=FIG_DPI); plt.close()

    # topology accuracy vs time
    plt.figure(figsize=(8.5, 5.8))
    plt.plot(range(T), [per_frame_topo[t] for t in range(T)], "o-", lw=2.5, ms=9)
    plt.xlabel("frame t", fontsize=FS); plt.ylabel("topology accuracy", fontsize=FS); plt.ylim(-0.05, 1.05)
    plt.xticks(fontsize=FS - 4); plt.yticks(fontsize=FS - 4)
    plt.title("topology accuracy through the merge", fontsize=FS); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "topo_vs_time.png"), dpi=FIG_DPI); plt.close()

    plot_predictions(model, X, Y, meta, te_s, T, stats)
    print("saved -> results/{metrics.json, model.pt, loss_curve.png, topo_vs_time.png, predictions.png}")


FS = 22          # figure font size (project preference: > 20)
FIG_DPI = 600    # project preference: save figures at 600 dpi


def plot_predictions(model, X, Y, meta, te_s, T, stats):
    """stefan_preview.png style (Blues_r T + black/crimson Γ); 6 rows = T/φ GT, pred, |error|."""
    xm, xs2, ym, ys = stats
    d = np.load(DATA, allow_pickle=True)
    cand = [i for i in te_s if int(d[i]["params"][0]) >= 3]
    i = (cand or list(te_s))[0]
    ms = int(d[i]["merge_step"])
    times = [0, max(1, ms - 2), ms, min(T - 1, ms + 3), T - 1]
    H = X.shape[1]; gx = np.linspace(-1, 1, H)
    Tnorm = Normalize(float(d[i]["params"][2]), 0.0)

    Tg, Tp, Pg, Pp = [], [], [], []
    for t in times:
        e = i * T + t
        with torch.no_grad():
            pred = (model(((X[e:e+1] - xm) / xs2))[0] * ys + ym).numpy()
        Tg.append(Y[e, ..., 0].numpy()); Tp.append(pred[..., 0])
        Pg.append(Y[e, ..., 1].numpy()); Pp.append(pred[..., 1])
    dT = [np.abs(a - b) for a, b in zip(Tg, Tp)]; dP = [np.abs(a - b) for a, b in zip(Pg, Pp)]
    vT, vP = max(x.max() for x in dT), max(x.max() for x in dP)
    Pnorm = Normalize(min(p.min() for p in Pg + Pp), max(p.max() for p in Pg + Pp))

    def field(ax, img, phi, cmap, norm):
        ax.imshow(img.T, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap, norm=norm, interpolation="bilinear")
        ax.contour(gx, gx, phi.T, levels=[0], colors="black", linewidths=3.0)
        ax.contour(gx, gx, phi.T, levels=[0], colors="crimson", linewidths=1.4)
        ax.set_xticks([]); ax.set_yticks([])

    def err(ax, img, vmax):
        im = ax.imshow(img.T, origin="lower", extent=(-1, 1, -1, 1), cmap="magma", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([]); return im

    fig, axs = plt.subplots(6, 5, figsize=(15.5, 18.5))
    for c, t in enumerate(times):
        field(axs[0, c], Tg[c], Pg[c], "Blues_r", Tnorm); axs[0, c].set_title(f"t={t}  ({n_comp(Pg[c] < 0)}c)", fontsize=FS)
        field(axs[1, c], Tp[c], Pp[c], "Blues_r", Tnorm)
        imT = err(axs[2, c], dT[c], vT)
        field(axs[3, c], Pg[c], Pg[c], "coolwarm", Pnorm)
        field(axs[4, c], Pp[c], Pp[c], "coolwarm", Pnorm)
        imP = err(axs[5, c], dP[c], vP)
    for im, row in [(imT, 2), (imP, 5)]:
        cb = fig.colorbar(im, ax=list(axs[row, :]), fraction=0.022, pad=0.01)
        cb.ax.tick_params(labelsize=FS - 4)
    # row labels on the far left, tumour-3d style (bold rotated fig.text)
    for r, lab in enumerate(["T GT", "T pred", "|ΔT|", "φ GT", "φ pred", "|Δφ|"]):
        pos = axs[r, 0].get_position()
        fig.text(pos.x0 - 0.022, pos.y0 + pos.height / 2, lab, rotation=90,
                 va="center", ha="center", fontsize=FS, fontweight="bold")
    fig.savefig(os.path.join(OUT, "predictions.png"), dpi=FIG_DPI); plt.close(fig)


if __name__ == "__main__":
    main()
