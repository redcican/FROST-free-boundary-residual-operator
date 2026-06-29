"""
Promote the 128²/modes20 tumour C1 model to the default `results/` (it lowers the field rel-L2:
u 0.159 -> 0.121, φ 0.013 -> 0.0107, IoU 0.961 -> 0.974). The weights are already trained
(`results/improve_ds1_m20_w32/`); the network is deterministic, so we REUSE them rather than retrain,
and regenerate the canonical metrics + figures (train_c1_tumour now defaults to DS=1 / MODES=20).

Run:  python promote.py
"""
import os, sys, json, shutil
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno import FNO2d
from train_c1_tumour import (load_split, evaluate, agg, OUT, MODES, WIDTH, FS, FIG_DPI,
                             plot_predictions)

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
torch.manual_seed(0); np.random.seed(0)
SRC = os.path.join(OUT, "improve_ds1_m20_w32", "model.pt")
# training-loss trajectory logged during the 128² run (improve_field_run.log), for the loss curve
LOSS_EP = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
LOSS_VAL = [1.7468, 0.2950, 0.2317, 0.1939, 0.1660, 0.1461, 0.1318, 0.1221, 0.1167, 0.1141, 0.1135]


def main():
    X, Y, meta, T, tr, te, tr_s, te_s = load_split()
    H = X.shape[1]
    ck = torch.load(SRC, map_location="cpu")
    s = ck["stats"]; stats = (s["xm"], s["xs"], s["ym"].reshape(2), s["ys"].reshape(2))
    model = FNO2d(modes=MODES, width=WIDTH, in_c=6, out_c=2, n_layers=4)
    model.load_state_dict(ck["model"]); model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"reusing trained 128² model ({n_par/1e3:.0f}k params), grid {H}x{H}, test {len(te_s)} samples")

    rte = evaluate(model, X, Y, meta, te, stats)
    per_frame_topo = {t: agg(rte, "topo", t) for t in range(T)}
    metrics = {
        "n_params": int(n_par), "epochs": 100, "ds": 1, "modes": MODES, "width": WIDTH, "grid": H,
        "n_train_samples": int(len(tr_s)), "n_test_samples": int(len(te_s)), "frames": T,
        "promoted_from": "improve_ds1_m20_w32 (128² full resolution); previous default was 64²",
        "test": {"u_relL2": agg(rte, "u_l2"), "phi_relL2": agg(rte, "phi_l2"),
                 "tumour_IoU": agg(rte, "iou"), "topology_acc": agg(rte, "topo"),
                 "topology_acc_per_frame": per_frame_topo},
    }
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    shutil.copy(SRC, os.path.join(OUT, "model.pt"))
    print("== TEST ==")
    print(json.dumps({k: v for k, v in metrics["test"].items() if k != "topology_acc_per_frame"}, indent=2))

    # loss curve (from the logged 128² trajectory)
    plt.figure(figsize=(7.5, 5)); plt.semilogy(LOSS_EP, LOSS_VAL, "o-", lw=2)
    plt.xlabel("epoch", fontsize=FS); plt.ylabel("train rel-L2 (u+φ)", fontsize=FS)
    plt.xticks(fontsize=FS - 4); plt.yticks(fontsize=FS - 4)
    plt.title("FROST C1 tumour (128²) — training loss", fontsize=FS); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "loss_curve.png"), dpi=FIG_DPI); plt.close()

    # topology-vs-time
    plt.figure(figsize=(8.5, 5.8))
    plt.plot(range(T), [per_frame_topo[t] for t in range(T)], "o-", lw=2.5, ms=9)
    plt.xlabel("frame t", fontsize=FS); plt.ylabel("topology accuracy", fontsize=FS); plt.ylim(-0.05, 1.05)
    plt.xticks(fontsize=FS - 4); plt.yticks(fontsize=FS - 4)
    plt.title("topology accuracy through the merge", fontsize=FS); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "topo_vs_time.png"), dpi=FIG_DPI); plt.close()

    plot_predictions(model, X, Y, meta, te_s, T, stats)
    print("regenerated -> results/{model.pt, metrics.json, loss_curve.png, topo_vs_time.png, predictions.png}")


if __name__ == "__main__":
    main()
