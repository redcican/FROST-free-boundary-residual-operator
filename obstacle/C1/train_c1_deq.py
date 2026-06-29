"""
FROST C1 — train the DEQ / implicit-diff forward operator on the obstacle benchmark, and compare it
head-to-head with the feed-forward FNO (train_c1_obstacle.py).

Same data split, standardization, and metrics as the FNO, so the only change is feed-forward -> DEQ
(fixed-point solve + implicit-function-theorem backward). Extra DEQ-specific evidence:
  - forward fixed-point RESIDUAL converges (the equilibrium is actually reached);
  - the obstacle COMPLEMENTARITY residual  ||u - max(chi, mean_nbr(u))||  of the prediction
    (does the learned solution satisfy the physical fixed point?) — reported for DEQ vs FNO.

Outputs -> results/ : deq_model.pt, deq_metrics.json, deq_convergence.png
Run:  python train_c1_deq.py [epochs]
"""
import os, sys, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deq import DEQObstacle
from fno import FNO2d, LpLoss
from train_c1_obstacle import load_split, evaluate, agg, OUT

torch.set_num_threads(8)
torch.manual_seed(0); np.random.seed(0)


def obstacle_residual_np(u, chi):
    """Relative discrete projected-fixed-point residual ||u - max(chi, mean_nbr(u))|| / ||u||."""
    nb = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
    r = u[1:-1, 1:-1] - np.maximum(chi[1:-1, 1:-1], nb)
    return float(np.linalg.norm(r) / (np.linalg.norm(u[1:-1, 1:-1]) + 1e-8))


def mean_residual(model, X, chi, idx, stats):
    ym, ysd = stats["ym"].reshape(2), stats["ysd"].reshape(2)
    vals = []
    with torch.no_grad():
        for i in idx:
            pred = (model(((X[i:i+1] - stats["xm"]) / stats["xsd"]))[0] * ysd + ym).numpy()
            vals.append(obstacle_residual_np(pred[..., 0], chi[i].numpy()))
    return float(np.mean(vals))


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    X, Y, chi, nc, tr, te = load_split()
    xm, xsd = X[tr].mean(), X[tr].std()
    ym = Y[tr].reshape(-1, 2).mean(0).view(1, 1, 1, 2)
    ysd = Y[tr].reshape(-1, 2).std(0).view(1, 1, 1, 2)
    stats = dict(xm=xm, xsd=xsd, ym=ym, ysd=ysd)
    Xn, Yn = (X - xm) / xsd, (Y - ym) / ysd

    model = DEQObstacle(modes=12, width=32, f_iter=30, b_iter=14, tol=1e-3)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lp = LpLoss()
    Xtr, Ytr = Xn[tr], Yn[tr]
    bs = 16; hist = []
    import copy
    best = {"u": 1e9, "state": None, "ep": -1}
    print(f"DEQ params: {n_par/1e3:.0f}k  epochs {epochs}  (f_iter 30, b_iter 20, grad-clip 1.0, keep-best)")
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(tr)); tot = 0.0
        for j in range(0, len(tr), bs):
            b = perm[j:j+bs]
            opt.zero_grad()
            pred = model(Xtr[b])
            loss = lp(pred[..., 0], Ytr[b][..., 0]) + lp(pred[..., 1], Ytr[b][..., 1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stabilize the equilibrium map
            opt.step(); tot += loss.item() * len(b)
        sched.step(); hist.append(tot / len(tr))
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval(); r = evaluate(model, X, Y, chi, nc, te, stats); u_l2 = agg(r, "u_l2")
            if u_l2 < best["u"]:                                       # keep-best (DEQs can drift late)
                best = {"u": u_l2, "state": copy.deepcopy(model.state_dict()), "ep": ep}
            with torch.no_grad():
                _, fres = model(((X[te[0]:te[0]+1] - xm) / xsd), return_res=True)
            print(f"  ep {ep:4d}  train {hist[-1]:.4f}  test u_l2 {u_l2:.4f} "
                  f"phi_l2 {agg(r,'phi_l2'):.4f} IoU {agg(r,'iou_phi'):.3f} topo {agg(r,'topo_phi'):.2f} "
                  f"| fwd-res {fres[-1]:.1e} ({len(fres)} it)", flush=True)

    if best["state"] is not None:                                     # evaluate/save the BEST model
        model.load_state_dict(best["state"])
        print(f"restored best @ep {best['ep']} (test u_l2 {best['u']:.4f})")
    model.eval()
    rte = evaluate(model, X, Y, chi, nc, te, stats)
    deq_resid = mean_residual(model, X, chi, te, stats)

    # load the feed-forward FNO for a head-to-head comparison
    fno_block = None
    fno_path = os.path.join(OUT, "model.pt")
    if os.path.exists(fno_path):
        ck = torch.load(fno_path, map_location="cpu")
        fno_block = FNO2d(modes=16, width=32, in_c=1, out_c=2, n_layers=4)
        fno_block.load_state_dict(ck["model"]); fno_block.eval()
        fno_metrics = json.load(open(os.path.join(OUT, "metrics.json")))["test"]
        fno_resid = mean_residual(fno_block, X, chi, te, {k: v for k, v in ck["stats"].items()})
    else:
        fno_metrics, fno_resid = None, None

    out = {
        "model": "DEQObstacle (fixed-point + IFT implicit backward)", "n_params": int(n_par),
        "epochs": epochs, "best_epoch": best["ep"], "train_time_s": round(time.time() - t0, 1),
        "deq_test": {"u_relL2": agg(rte, "u_l2"), "phi_relL2": agg(rte, "phi_l2"),
                     "contact_IoU_phi_head": agg(rte, "iou_phi"),
                     "topology_acc_phi_head": agg(rte, "topo_phi"),
                     "obstacle_fixedpoint_residual": deq_resid},
        "fno_test_for_comparison": (None if fno_metrics is None else
            {"u_relL2": fno_metrics["u_relL2"], "phi_relL2": fno_metrics["phi_relL2"],
             "contact_IoU_phi_head": fno_metrics["contact_IoU_phi_head"],
             "topology_acc_phi_head": fno_metrics["topology_acc_phi_head"],
             "obstacle_fixedpoint_residual": fno_resid}),
    }
    json.dump(out, open(os.path.join(OUT, "deq_metrics.json"), "w"), indent=2)
    torch.save({"model": model.state_dict(), "stats": {k: v for k, v in stats.items()}},
               os.path.join(OUT, "deq_model.pt"))
    print("\n== DEQ vs FNO ==")
    print(json.dumps(out["deq_test"], indent=2))
    if fno_metrics is not None:
        print("FNO:", json.dumps(out["fno_test_for_comparison"]))

    # convergence figure: forward fixed-point residual vs iteration for a few test samples
    plt.figure(figsize=(7, 5))
    for i in te[:4]:
        with torch.no_grad():
            _, fres = model(((X[i:i+1] - xm) / xsd), return_res=True)
        plt.semilogy(range(1, len(fres) + 1), fres, marker="o", ms=3, label=f"test idx {int(i)}")
    plt.axhline(model.tol, color="k", ls="--", lw=1, label=f"tol {model.tol:g}")
    plt.xlabel("Anderson iteration"); plt.ylabel("relative fixed-point residual")
    plt.title("FROST C1 DEQ — forward equilibrium convergence"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "deq_convergence.png"), dpi=600); plt.close()
    print("saved -> results/{deq_metrics.json, deq_model.pt, deq_convergence.png}")


if __name__ == "__main__":
    main()
