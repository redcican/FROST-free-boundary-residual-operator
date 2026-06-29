"""
FROST C1 — forward operator on the tumour_merge benchmark (time-dependent, 2->1 topology change).

Same time-conditioned operator as the Stefan C1, on the organic two-tumour-merge data. A single 2D FNO
learns

    [ φ₀ (initial two tumours), sep0, sep1, scale, k, t ]  ->  ( u(t), φ(t) )

where u is the nutrient field ((-Δ+k)u=0, u=1 on Γ) and φ the level set. The two tumours drift together
and COALESCE (2 components -> 1), so the operator must produce a topology-changing free boundary.

Outputs -> C1/results/ : model.pt, metrics.json, loss_curve.png, topo_vs_time.png, predictions.png
Run:  python train_c1_tumour.py [epochs]
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
DATA = os.path.join(HERE, "..", "tumour_merge.npy")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
CONN = np.ones((3, 3), int)
DS = 1                                                       # 128^2 full res (promoted 2026-06-21: lower field rel-L2)
MODES = 20                                                   # spectral modes (raised with the resolution)
WIDTH = 32
FS = 22                                                      # font size (> 20)
FIG_DPI = 600
# param indices: [gA, gB, scale, sep0, sep1, rotA, rotB, k]
P_SCALE, P_SEP0, P_SEP1, P_K = 2, 3, 4, 7


def load_split(test_frac=0.2):
    """Time-conditioned (sample,frame) examples; split by SAMPLE, stratified by k quartile."""
    d = np.load(DATA, allow_pickle=True)
    N = len(d); T = d[0]["phi"].shape[0]; H = d[0]["phi"].shape[1] // DS
    rng = np.random.default_rng(0)
    ks = np.array([float(x["params"][P_K]) for x in d])
    q = np.digitize(ks, np.quantile(ks, [0.25, 0.5, 0.75]))   # stratify by reaction-rate k
    tr_s, te_s = [], []
    for c in np.unique(q):
        idx = np.where(q == c)[0]; rng.shuffle(idx)
        kk = max(1, int(round(len(idx) * test_frac)))
        te_s += list(idx[:kk]); tr_s += list(idx[kk:])
    tr_s, te_s = np.array(sorted(tr_s)), np.array(sorted(te_s))

    tnorm = np.linspace(0, 1, T).astype(np.float32)
    X = np.zeros((N * T, H, H, 6), np.float32)               # [phi0, sep0, sep1, scale, k, t]
    Y = np.zeros((N * T, H, H, 2), np.float32)               # [u(t), phi(t)]
    meta = np.zeros((N * T, 3), np.float32)                  # sample, frame, merge_step
    for i, s in enumerate(d):
        phi0 = s["phi"][0][::DS, ::DS]; p = s["params"]
        for t in range(T):
            e = i * T + t
            X[e, ..., 0] = phi0; X[e, ..., 1] = p[P_SEP0]; X[e, ..., 2] = p[P_SEP1]
            X[e, ..., 3] = p[P_SCALE]; X[e, ..., 4] = p[P_K]; X[e, ..., 5] = tnorm[t]
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
            u_p, phi_p = pred[..., 0], pred[..., 1]
            u_g, phi_g = Y[e, ..., 0].numpy(), Y[e, ..., 1].numpy()
            sp, sg = phi_p < 0, phi_g < 0
            rows.append(dict(
                t=int(meta[e, 1]),
                u_l2=float(np.linalg.norm(u_p - u_g) / (np.linalg.norm(u_g) + 1e-8)),
                phi_l2=float(np.linalg.norm(phi_p - phi_g) / (np.linalg.norm(phi_g) + 1e-8)),
                iou=iou(sp, sg), topo=int(n_comp(sp) == n_comp(sg))))
    return rows


def agg(rows, key, frame=None):
    v = [r[key] for r in rows if (frame is None or r["t"] == frame)]
    return float(np.mean(v)) if v else None


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    X, Y, meta, T, tr, te, tr_s, te_s = load_split()
    print(f"train {len(tr_s)} samples ({len(tr)} frames)  test {len(te_s)} samples ({len(te)} frames)")

    xm = X[tr].reshape(-1, 6).mean(0).view(1, 1, 1, 6); xs = X[tr].reshape(-1, 6).std(0).view(1, 1, 1, 6) + 1e-6
    ym = Y[tr].reshape(-1, 2).mean(0).view(1, 1, 1, 2); ys = Y[tr].reshape(-1, 2).std(0).view(1, 1, 1, 2)
    stats = (xm, xs, ym.reshape(2), ys.reshape(2))
    Xn, Yn = (X - xm) / xs, (Y - ym) / ys

    model = FNO2d(modes=MODES, width=WIDTH, in_c=6, out_c=2, n_layers=4)
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
                 "tumour_IoU": agg(rte, "iou"), "topology_acc": agg(rte, "topo"),
                 "topology_acc_per_frame": per_frame_topo},
    }
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    torch.save({"model": model.state_dict(),
                "stats": {"xm": xm, "xs": xs, "ym": ym, "ys": ys}}, os.path.join(OUT, "model.pt"))
    print("\n== TEST =="); print(json.dumps({k: v for k, v in metrics["test"].items()
                                             if k != "topology_acc_per_frame"}, indent=2))

    plt.figure(figsize=(7.5, 5)); plt.semilogy(hist, lw=2)
    plt.xlabel("epoch", fontsize=FS); plt.ylabel("train rel-L2 (u+φ)", fontsize=FS)
    plt.xticks(fontsize=FS - 4); plt.yticks(fontsize=FS - 4)
    plt.title("FROST C1 tumour — training loss", fontsize=FS); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "loss_curve.png"), dpi=FIG_DPI); plt.close()

    plt.figure(figsize=(8.5, 5.8))
    plt.plot(range(T), [per_frame_topo[t] for t in range(T)], "o-", lw=2.5, ms=9)
    plt.xlabel("frame t", fontsize=FS); plt.ylabel("topology accuracy", fontsize=FS); plt.ylim(-0.05, 1.05)
    plt.xticks(fontsize=FS - 4); plt.yticks(fontsize=FS - 4)
    plt.title("topology accuracy through the merge", fontsize=FS); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "topo_vs_time.png"), dpi=FIG_DPI); plt.close()

    plot_predictions(model, X, Y, meta, te_s, T, stats)
    print("saved -> results/{metrics.json, model.pt, loss_curve.png, topo_vs_time.png, predictions.png}")


def plot_predictions(model, X, Y, meta, te_s, T, stats):
    """tumour preview style (YlGnBu nutrient + black/crimson Γ); 6 rows = u/φ GT, pred, |error|."""
    xm, xs2, ym, ys = stats
    d = np.load(DATA, allow_pickle=True)
    i = int(te_s[0])
    ms = int(d[i]["merge_step"])
    times = [0, max(1, ms - 3), ms, min(T - 1, ms + 2), T - 1]
    H = X.shape[1]; gx = np.linspace(-1, 1, H)

    Ug, Up, Pg, Pp = [], [], [], []
    for t in times:
        e = i * T + t
        with torch.no_grad():
            pred = (model(((X[e:e+1] - xm) / xs2))[0] * ys + ym).numpy()
        Ug.append(Y[e, ..., 0].numpy()); Up.append(pred[..., 0])
        Pg.append(Y[e, ..., 1].numpy()); Pp.append(pred[..., 1])
    dU = [np.abs(a - b) for a, b in zip(Ug, Up)]; dP = [np.abs(a - b) for a, b in zip(Pg, Pp)]
    vU, vP = max(x.max() for x in dU), max(x.max() for x in dP)
    Unorm = Normalize(0.0, max(1e-6, max(x.max() for x in Ug)))
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
        field(axs[0, c], Ug[c], Pg[c], "YlGnBu", Unorm); axs[0, c].set_title(f"t={t}  ({n_comp(Pg[c] < 0)}c)", fontsize=FS)
        field(axs[1, c], Up[c], Pp[c], "YlGnBu", Unorm)
        imU = err(axs[2, c], dU[c], vU)
        field(axs[3, c], Pg[c], Pg[c], "coolwarm", Pnorm)
        field(axs[4, c], Pp[c], Pp[c], "coolwarm", Pnorm)
        imP = err(axs[5, c], dP[c], vP)
    for r, lab in enumerate(["u GT", "u pred", "|Δu|", "φ GT", "φ pred", "|Δφ|"]):
        axs[r, 0].set_ylabel(lab, fontsize=FS)
    for im, row in [(imU, 2), (imP, 5)]:
        cb = fig.colorbar(im, ax=list(axs[row, :]), fraction=0.022, pad=0.01)
        cb.ax.tick_params(labelsize=FS - 4)
    fig.savefig(os.path.join(OUT, "predictions.png"), dpi=FIG_DPI); plt.close(fig)


if __name__ == "__main__":
    main()
