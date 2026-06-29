"""
FROST C1 — forward operator on the obstacle benchmark.

Learns the FROST forward operator  chi -> (u, phi):  one FNO predicts BOTH the membrane field u and
the level set phi (signed distance to the contact-set free boundary Γ = ∂{u=χ}). The obstacle problem
is the steady, monotone equilibrium, so it is the clean first test of the FROST forward operator.

Key question this answers: does an explicit level-set output recover the FREE BOUNDARY and its
TOPOLOGY (1 vs 2 contact components) — including on held-out instances — better than reading the
contact set off a predicted field? We report field accuracy, level-set/Γ accuracy, contact IoU, and
topology accuracy, with a head-to-head: phi-head contact  vs  contact thresholded from the u-head.

Outputs -> C1/results/ : model.pt, metrics.json, loss_curve.png, predictions.png
Run:  python train_c1_obstacle.py [epochs]
"""
import os, sys, json, time
import argparse
import numpy as np
import torch
from scipy.ndimage import label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno import FNO2d, LpLoss

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))   # FNO is FFT/BLAS-heavy
torch.manual_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "obstacle.npy")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cpu")
CONTACT_TOL = 2e-3                                       # matches gen_obstacle.py contact definition
CONN = np.ones((3, 3), int)


def load_split(test_frac=0.2):
    d = np.load(DATA, allow_pickle=True)
    chi = np.stack([s["chi"] for s in d]).astype(np.float32)
    u = np.stack([s["u"] for s in d]).astype(np.float32)
    phi = np.stack([s["phi"] for s in d]).astype(np.float32)
    nc = np.array([int(s["n_components"]) for s in d])
    # stratified split by topology so the test set has both 1- and 2-component cases
    rng = np.random.default_rng(0)
    tr, te = [], []
    for c in np.unique(nc):
        idx = np.where(nc == c)[0]; rng.shuffle(idx)
        k = max(1, int(round(len(idx) * test_frac)))
        te += list(idx[:k]); tr += list(idx[k:])
    tr, te = np.array(sorted(tr)), np.array(sorted(te))
    X = torch.from_numpy(chi)[..., None]                # (N,H,W,1)
    Y = torch.from_numpy(np.stack([u, phi], -1))        # (N,H,W,2)
    return X, Y, torch.from_numpy(chi), nc, tr, te


def n_comp(mask):
    return int(label(mask, structure=CONN)[1])


def iou(a, b):
    inter = np.logical_and(a, b).sum(); union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def evaluate(model, X, Y, chi, nc, idx, stats):
    """Denormalized metrics on a set of indices."""
    ym, ysd = stats["ym"].reshape(2), stats["ysd"].reshape(2)
    rows = []
    with torch.no_grad():
        for i in idx:
            xb = ((X[i:i+1] - stats["xm"]) / stats["xsd"]).to(DEVICE)
            pred = model(xb)[0].cpu() * ysd + ym                # (H,W,2) denormalized
            u_p, phi_p = pred[..., 0].numpy(), pred[..., 1].numpy()
            u_g, phi_g = Y[i, ..., 0].numpy(), Y[i, ..., 1].numpy()
            chi_i = chi[i].numpy()
            ct_gt = (u_g - chi_i < CONTACT_TOL) & (chi_i > 0)   # GT contact set
            ct_phi = (phi_p < 0) & (chi_i > 0)                  # FROST level-set head
            ct_u = (u_p - chi_i < CONTACT_TOL) & (chi_i > 0)    # baseline: threshold predicted field
            rows.append(dict(
                i=int(i), nc=int(nc[i]),
                u_l2=float(torch.norm(torch.tensor(u_p - u_g)) / (torch.norm(torch.tensor(u_g)) + 1e-8)),
                phi_l2=float(torch.norm(torch.tensor(phi_p - phi_g)) / (torch.norm(torch.tensor(phi_g)) + 1e-8)),
                iou_phi=iou(ct_phi, ct_gt), iou_u=iou(ct_u, ct_gt),
                topo_phi=int(n_comp(ct_phi) == nc[i]), topo_u=int(n_comp(ct_u) == nc[i])))
    return rows


def agg(rows, key, sub=None):
    v = [r[key] for r in rows if (sub is None or r["nc"] == sub)]
    return float(np.mean(v)) if v else None


BW_SIGMA = 0.05                                         # width (φ units) of the boundary up-weight band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("epochs", nargs="?", type=int, default=300)
    ap.add_argument("--bw", type=float, default=0.0)    # boundary-weight strength (0 = baseline)
    a = ap.parse_args()
    epochs, bw = a.epochs, a.bw
    outdir = OUT if bw == 0 else os.path.join(OUT, f"bw_{bw:g}"); os.makedirs(outdir, exist_ok=True)
    X, Y, chi, nc, tr, te = load_split()
    print(f"train {len(tr)}  test {len(te)}  bw={bw}  (test topo: "
          f"{ {int(c): int((nc[te]==c).sum()) for c in np.unique(nc)} })")

    # standardize input and each target channel using TRAIN statistics
    xm, xsd = X[tr].mean(), X[tr].std()
    ym = Y[tr].reshape(-1, 2).mean(0).view(1, 1, 1, 2)
    ysd = Y[tr].reshape(-1, 2).std(0).view(1, 1, 1, 2)
    stats = dict(xm=xm, xsd=xsd, ym=ym, ysd=ysd)
    Xn, Yn = (X - xm) / xsd, (Y - ym) / ysd
    # boundary weight per cell: ~1 near the free boundary φ=0 (from GT φ, physical units), ~0 away
    Wfull = torch.exp(-(Y[..., 1] / BW_SIGMA) ** 2 / 2.0)              # (N,H,W)

    model = FNO2d(modes=16, width=32, in_c=1, out_c=2, n_layers=4).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lp = LpLoss()
    Xtr, Ytr = Xn[tr].to(DEVICE), Yn[tr].to(DEVICE); Wtr = Wfull[tr].to(DEVICE)

    def boundary_term(pphi, gphi, w):                                  # weighted relative L2 on φ near Γ
        num = (w * (pphi - gphi) ** 2).reshape(len(w), -1).sum(1)
        den = (w * gphi ** 2).reshape(len(w), -1).sum(1) + 1e-8
        return torch.mean(torch.sqrt(num / den))

    bs = 8; hist = []
    print(f"FNO params: {n_par/1e3:.0f}k  epochs {epochs}  -> {outdir}")
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(tr)); tot = 0.0
        for j in range(0, len(tr), bs):
            b = perm[j:j+bs]
            opt.zero_grad()
            pred = model(Xtr[b])
            loss = lp(pred[..., 0], Ytr[b][..., 0]) + lp(pred[..., 1], Ytr[b][..., 1])
            if bw > 0:
                loss = loss + bw * boundary_term(pred[..., 1], Ytr[b][..., 1], Wtr[b])
            loss.backward(); opt.step(); tot += loss.item() * len(b)
        sched.step(); hist.append(tot / len(tr))
        if ep % 25 == 0 or ep == epochs - 1:
            r = evaluate(model, X, Y, chi, nc, te, stats)
            print(f"  ep {ep:4d}  train {hist[-1]:.4f}  test u_l2 {agg(r,'u_l2'):.4f} "
                  f"phi_l2 {agg(r,'phi_l2'):.4f} IoU(phi) {agg(r,'iou_phi'):.3f} "
                  f"topo(phi) {agg(r,'topo_phi'):.2f}", flush=True)

    # ---- final metrics
    rtr = evaluate(model, X, Y, chi, nc, tr, stats)
    rte = evaluate(model, X, Y, chi, nc, te, stats)
    metrics = {
        "n_params": int(n_par), "epochs": epochs, "bw": bw, "train_time_s": round(time.time() - t0, 1),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test": {
            "u_relL2": agg(rte, "u_l2"), "phi_relL2": agg(rte, "phi_l2"),
            "contact_IoU_phi_head": agg(rte, "iou_phi"),
            "contact_IoU_u_threshold_baseline": agg(rte, "iou_u"),
            "topology_acc_phi_head": agg(rte, "topo_phi"),
            "topology_acc_u_threshold_baseline": agg(rte, "topo_u"),
            "by_topology": {f"{c}_comp": {"u_relL2": agg(rte, "u_l2", c), "phi_relL2": agg(rte, "phi_l2", c),
                                          "IoU_phi": agg(rte, "iou_phi", c), "topo_phi": agg(rte, "topo_phi", c)}
                            for c in (1, 2)},
        },
        "train": {"u_relL2": agg(rtr, "u_l2"), "phi_relL2": agg(rtr, "phi_l2"),
                  "contact_IoU_phi_head": agg(rtr, "iou_phi")},
    }
    json.dump(metrics, open(os.path.join(outdir, "metrics.json"), "w"), indent=2)
    torch.save({"model": model.state_dict(), "stats": {k: v for k, v in stats.items()}},
               os.path.join(outdir, "model.pt"))
    print("\n== TEST ==")
    print(json.dumps(metrics["test"], indent=2))

    # ---- loss curve
    plt.figure(figsize=(6, 4)); plt.semilogy(hist); plt.xlabel("epoch"); plt.ylabel("train rel-L2 (u+φ)")
    plt.title("FROST C1 obstacle — training loss"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=600); plt.close()

    # ---- prediction figure: one 1-comp and one 2-comp test sample
    plot_predictions(model, X, Y, chi, nc, te, stats, outdir=outdir)
    print(f"saved -> {outdir}")


def plot_predictions(model, X, Y, chi, nc, te, stats, out_name="predictions.png", outdir=OUT):
    """2D GT-vs-prediction figure: each test case = 2 rows of 4 panels (-> 4x4 for two cases)."""
    xs = np.linspace(-1, 1, X.shape[1])
    TS = 20
    picks = []
    for c in (2, 1):
        cand = [i for i in te if nc[i] == c]
        if cand:
            picks.append(cand[0])
    ym, ysd = stats["ym"].reshape(2), stats["ysd"].reshape(2)
    nrows = 2 * len(picks)
    fig, axs = plt.subplots(nrows, 4, figsize=(15, 3.6 * nrows))
    if nrows == 1:
        axs = axs[None, :]

    def show(ax, img, ttl, cmap, vmin=None, contour=False):
        im = ax.imshow(np.rot90(img), extent=(-1, 1, -1, 1), cmap=cmap, vmin=vmin)
        if contour:
            ax.contour(xs, xs, np.rot90(img), levels=[0], colors="k", linewidths=1.2)
        if cmap == "magma":
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=13)
        ax.set_title(ttl, fontsize=TS); ax.set_xticks([]); ax.set_yticks([])

    # case-grouped: each case = a u-row (rows 2s) then a φ-row (rows 2s+1)
    raw = np.load(DATA, allow_pickle=True)                   # for the per-case obstacle parameters
    for s, i in enumerate(picks):
        with torch.no_grad():
            pred = (model(((X[i:i+1] - stats["xm"]) / stats["xsd"]))[0] * ysd + ym).numpy()
        u_g, phi_g = Y[i, ..., 0].numpy(), Y[i, ..., 1].numpy()
        u_p, phi_p = pred[..., 0], pred[..., 1]
        chi_i = chi[i].numpy()
        ue, pe = np.abs(u_g - u_p), np.abs(phi_g - phi_p)
        ct_gt = (u_g - chi_i < CONTACT_TOL) & (chi_i > 0)
        ct_phi = (phi_p < 0) & (chi_i > 0)
        ur, pr = 2 * s, 2 * s + 1                            # u row, φ row for this case
        # u row: chi, u GT, u pred, |Δu|
        show(axs[ur, 0], chi_i, "χ (input)", "viridis")
        show(axs[ur, 1], u_g, "u GT", "rainbow")
        show(axs[ur, 2], u_p, "u pred", "rainbow")
        show(axs[ur, 3], ue, f"|Δu| (max {ue.max():.3f})", "magma", vmin=0.0)
        # φ row: φ GT, φ pred, contact overlay (col 2), |Δφ| (col 3)
        show(axs[pr, 0], phi_g, "φ GT", "coolwarm", contour=True)
        show(axs[pr, 1], phi_p, "φ pred", "coolwarm", contour=True)
        ax = axs[pr, 2]
        ax.imshow(np.rot90(np.zeros_like(chi_i)), extent=(-1, 1, -1, 1), cmap="gray", vmin=0, vmax=1)
        ax.contour(xs, xs, np.rot90(ct_gt.astype(float)), levels=[0.5], colors="tab:blue", linewidths=2.0)
        ax.contour(xs, xs, np.rot90(ct_phi.astype(float)), levels=[0.5], colors="crimson", linewidths=1.6)
        ax.set_title(f"contact GT(blue)/pred(red)", fontsize=TS); ax.set_xticks([]); ax.set_yticks([])
        show(axs[pr, 3], pe, f"|Δφ| (max {pe.max():.3f})", "magma", vmin=0.0)
        # explicit case difference (obstacle geometry -> contact topology) on the left of each block
        prm = raw[i]["params"]; sep = float(prm[1] - prm[0]); w = float(prm[4])
        kind = "separate" if nc[i] == 2 else "merged"
        axs[ur, 0].set_ylabel(f"CASE {s+1}: {nc[i]} contact region(s) [{kind}]\n"
                              f"bump sep={sep:.2f}, width={w:.2f}", fontsize=15, labelpad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, out_name), dpi=600); plt.close(fig)


if __name__ == "__main__":
    main()
