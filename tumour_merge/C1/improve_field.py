"""
Experiment: can we close the FIELD rel-L2 gap (tumour u ~16% vs obstacle u ~0.9%)?

Diagnosis (see commit msg / README): the tumour nutrient u is a NEAR-BINARY indicator (≈1 inside the
domain Ω={φ<0}, exactly 0 outside) on a small domain (~5% of the grid), with a sharp jump at Γ. A
spectral FNO rings at that discontinuity, and because ‖u‖ is small the relative error is inflated. The
sharp feature spans only ~1–2 cells at the 64² working resolution. So the levers are RESOLUTION (128²)
and more spectral MODES (which must scale up with resolution), plus more epochs.

This script is configurable so we can sweep them without touching the baseline trainer:
  python improve_field.py --ds 1 --modes 24 --width 32 --epochs 200   # 128², near-Nyquist modes
Baseline (documented) is --ds 2 --modes 16 --width 32 --epochs 100.

Outputs -> results/improve_ds{ds}_m{modes}_w{width}/metrics.json (+ model.pt).
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno import FNO2d, LpLoss
from train_c1_tumour import DATA, P_SCALE, P_SEP0, P_SEP1, P_K, evaluate, agg

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
torch.manual_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def load_split(ds, test_frac=0.2):
    d = np.load(DATA, allow_pickle=True)
    N = len(d); T = d[0]["phi"].shape[0]; H = d[0]["phi"].shape[1] // ds
    rng = np.random.default_rng(0)
    ks = np.array([float(x["params"][P_K]) for x in d])
    q = np.digitize(ks, np.quantile(ks, [0.25, 0.5, 0.75]))
    tr_s, te_s = [], []
    for c in np.unique(q):
        idx = np.where(q == c)[0]; rng.shuffle(idx)
        kk = max(1, int(round(len(idx) * test_frac)))
        te_s += list(idx[:kk]); tr_s += list(idx[kk:])
    tr_s, te_s = np.array(sorted(tr_s)), np.array(sorted(te_s))
    tnorm = np.linspace(0, 1, T).astype(np.float32)
    X = np.zeros((N * T, H, H, 6), np.float32); Y = np.zeros((N * T, H, H, 2), np.float32)
    meta = np.zeros((N * T, 3), np.float32)
    for i, s in enumerate(d):
        phi0 = s["phi"][0][::ds, ::ds]; p = s["params"]
        for t in range(T):
            e = i * T + t
            X[e, ..., 0] = phi0; X[e, ..., 1] = p[P_SEP0]; X[e, ..., 2] = p[P_SEP1]
            X[e, ..., 3] = p[P_SCALE]; X[e, ..., 4] = p[P_K]; X[e, ..., 5] = tnorm[t]
            Y[e, ..., 0] = s["u"][t][::ds, ::ds]; Y[e, ..., 1] = s["phi"][t][::ds, ::ds]
            meta[e] = [i, t, s["merge_step"]]
    exp = lambda S: np.concatenate([np.arange(i * T, i * T + T) for i in S])
    return (torch.from_numpy(X), torch.from_numpy(Y), meta, T, exp(tr_s), exp(te_s), tr_s, te_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, default=1)
    ap.add_argument("--modes", type=int, default=24)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--fieldw", type=float, default=1.0)     # weight on the field (u) loss term
    a = ap.parse_args()
    tag = f"improve_ds{a.ds}_m{a.modes}_w{a.width}"
    outdir = os.path.join(OUT, tag); os.makedirs(outdir, exist_ok=True)

    X, Y, meta, T, tr, te, tr_s, te_s = load_split(a.ds)
    H = X.shape[1]
    print(f"[{tag}] grid {H}x{H}  train {len(tr_s)} samples ({len(tr)} frames)  test {len(te_s)}  fieldw {a.fieldw}")
    xm = X[tr].reshape(-1, 6).mean(0).view(1, 1, 1, 6); xs = X[tr].reshape(-1, 6).std(0).view(1, 1, 1, 6) + 1e-6
    ym = Y[tr].reshape(-1, 2).mean(0).view(1, 1, 1, 2); ys = Y[tr].reshape(-1, 2).std(0).view(1, 1, 1, 2)
    stats = (xm, xs, ym.reshape(2), ys.reshape(2))
    Xn, Yn = (X - xm) / xs, (Y - ym) / ys

    model = FNO2d(modes=a.modes, width=a.width, in_c=6, out_c=2, n_layers=4)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lp = LpLoss()
    Xtr, Ytr = Xn[tr], Yn[tr]
    bs = 32; print(f"FNO {n_par/1e3:.0f}k params  modes {a.modes} width {a.width}  epochs {a.epochs}")
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(tr)); tot = 0.0
        for j in range(0, len(tr), bs):
            b = perm[j:j+bs]; opt.zero_grad()
            pred = model(Xtr[b])
            loss = a.fieldw * lp(pred[..., 0], Ytr[b][..., 0]) + lp(pred[..., 1], Ytr[b][..., 1])
            loss.backward(); opt.step(); tot += loss.item() * len(b)
        sched.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            r = evaluate(model, X, Y, meta, te, stats)
            print(f"  ep {ep:4d}  train {tot/len(tr):.4f}  test u_l2 {agg(r,'u_l2'):.4f} "
                  f"phi_l2 {agg(r,'phi_l2'):.4f} IoU {agg(r,'iou'):.3f} topo {agg(r,'topo'):.3f}", flush=True)

    rte = evaluate(model, X, Y, meta, te, stats)
    metrics = {"tag": tag, "grid": H, "ds": a.ds, "modes": a.modes, "width": a.width, "epochs": a.epochs,
               "fieldw": a.fieldw, "n_params": int(n_par), "train_time_s": round(time.time() - t0, 1),
               "baseline_ds2_m16": {"u_relL2": 0.159, "phi_relL2": 0.013, "IoU": 0.961, "topo": 0.996},
               "test": {"u_relL2": agg(rte, "u_l2"), "phi_relL2": agg(rte, "phi_l2"),
                        "tumour_IoU": agg(rte, "iou"), "topology_acc": agg(rte, "topo")}}
    json.dump(metrics, open(os.path.join(outdir, "metrics.json"), "w"), indent=2)
    torch.save({"model": model.state_dict(), "stats": {"xm": xm, "xs": xs, "ym": ym, "ys": ys}},
               os.path.join(outdir, "model.pt"))
    print(f"\n[{tag}] u_relL2 {metrics['test']['u_relL2']:.4f} (baseline 0.159)  "
          f"phi {metrics['test']['phi_relL2']:.4f}  IoU {metrics['test']['tumour_IoU']:.3f}  "
          f"({metrics['train_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
