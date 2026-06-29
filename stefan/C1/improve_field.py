"""
Experiment: close the FIELD rel-L2 gap on Stefan (T ~14.6% vs obstacle u ~0.9%).

Diagnosis: Stefan T is a CONTINUOUS field (liquid exactly 0, solid down to T_cold) but with ~60% of its
spectral energy in high modes — sharp, deep cold wells localized in the ~10% solid region, under-resolved
at the 64² working grid. Levers: RESOLUTION (128²) + more spectral MODES + more epochs. Configurable so we
can sweep without touching the baseline trainer.

  python improve_field.py --ds 1 --modes 20 --width 32 --epochs 100
Baseline (documented): --ds 2 --modes 16 --width 32 --epochs 100  -> T 0.146, φ 0.038.
Outputs -> results/improve_ds{ds}_m{modes}_w{width}/metrics.json (+ model.pt).
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno import FNO2d, LpLoss
from train_c1_stefan import DATA, evaluate, agg

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
torch.manual_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def load_split(ds, test_frac=0.2):
    d = np.load(DATA, allow_pickle=True)
    N = len(d); T = d[0]["phi"].shape[0]; H = d[0]["phi"].shape[1] // ds
    rng = np.random.default_rng(0)
    ns = np.array([int(x["params"][0]) for x in d])
    tr_s, te_s = [], []
    for c in np.unique(ns):
        idx = np.where(ns == c)[0]; rng.shuffle(idx)
        k = max(1, int(round(len(idx) * test_frac)))
        te_s += list(idx[:k]); tr_s += list(idx[k:])
    tr_s, te_s = np.array(sorted(tr_s)), np.array(sorted(te_s))
    tnorm = np.linspace(0, 1, T).astype(np.float32)
    X = np.zeros((N * T, H, H, 4), np.float32); Y = np.zeros((N * T, H, H, 2), np.float32)
    meta = np.zeros((N * T, 3), np.float32)
    for i, s in enumerate(d):
        phi0 = s["phi"][0][::ds, ::ds]; L = float(s["params"][1]); Tc = float(s["params"][2])
        for t in range(T):
            e = i * T + t
            X[e, ..., 0] = phi0; X[e, ..., 1] = L; X[e, ..., 2] = Tc; X[e, ..., 3] = tnorm[t]
            Y[e, ..., 0] = s["u"][t][::ds, ::ds]; Y[e, ..., 1] = s["phi"][t][::ds, ::ds]
            meta[e] = [i, t, s["merge_step"]]
    exp = lambda S: np.concatenate([np.arange(i * T, i * T + T) for i in S])
    return (torch.from_numpy(X), torch.from_numpy(Y), meta, T, exp(tr_s), exp(te_s), tr_s, te_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, default=1)
    ap.add_argument("--modes", type=int, default=20)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--fieldw", type=float, default=1.0)
    a = ap.parse_args()
    tag = f"improve_ds{a.ds}_m{a.modes}_w{a.width}"
    outdir = os.path.join(OUT, tag); os.makedirs(outdir, exist_ok=True)

    X, Y, meta, T, tr, te, tr_s, te_s = load_split(a.ds)
    H = X.shape[1]
    print(f"[stefan {tag}] grid {H}x{H}  train {len(tr_s)} ({len(tr)} frames)  test {len(te_s)}  fieldw {a.fieldw}")
    xm = X[tr].reshape(-1, 4).mean(0).view(1, 1, 1, 4); xs = X[tr].reshape(-1, 4).std(0).view(1, 1, 1, 4) + 1e-6
    ym = Y[tr].reshape(-1, 2).mean(0).view(1, 1, 1, 2); ys = Y[tr].reshape(-1, 2).std(0).view(1, 1, 1, 2)
    stats = (xm, xs, ym.reshape(2), ys.reshape(2))
    Xn, Yn = (X - xm) / xs, (Y - ym) / ys

    model = FNO2d(modes=a.modes, width=a.width, in_c=4, out_c=2, n_layers=4)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lp = LpLoss(); Xtr, Ytr = Xn[tr], Yn[tr]
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
               "baseline_ds2_m16": {"T_relL2": 0.146, "phi_relL2": 0.038, "IoU": 0.909, "topo": 0.907},
               "test": {"T_relL2": agg(rte, "u_l2"), "phi_relL2": agg(rte, "phi_l2"),
                        "solid_IoU": agg(rte, "iou"), "topology_acc": agg(rte, "topo")}}
    json.dump(metrics, open(os.path.join(outdir, "metrics.json"), "w"), indent=2)
    torch.save({"model": model.state_dict(), "stats": {"xm": xm, "xs": xs, "ym": ym, "ys": ys}},
               os.path.join(outdir, "model.pt"))
    print(f"\n[stefan {tag}] T_relL2 {metrics['test']['T_relL2']:.4f} (baseline 0.146)  "
          f"phi {metrics['test']['phi_relL2']:.4f}  IoU {metrics['test']['solid_IoU']:.3f}  "
          f"({metrics['train_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
