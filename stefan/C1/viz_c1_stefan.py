"""
Regenerate the Stefan C1 figures from the saved model (no retraining):
  results/predictions.png   (stefan_preview style: Blues_r T + black/crimson Γ; rows T/φ GT/pred/|err|)
  results/topo_vs_time.png  (topology accuracy through the merge)
Both at fontsize > 20 and dpi 600.

Run: python viz_c1_stefan.py
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno import FNO2d
from train_c1_stefan import load_split, plot_predictions, OUT, FS, FIG_DPI


def main():
    X, Y, meta, T, tr, te, tr_s, te_s = load_split()
    ck = torch.load(os.path.join(OUT, "model.pt"), map_location="cpu")
    s = ck["stats"]
    stats = (s["xm"], s["xs"], s["ym"].reshape(2), s["ys"].reshape(2))
    model = FNO2d(modes=16, width=32, in_c=4, out_c=2, n_layers=4)
    model.load_state_dict(ck["model"]); model.eval()
    plot_predictions(model, X, Y, meta, te_s, T, stats)

    m = json.load(open(os.path.join(OUT, "metrics.json")))
    pf = m["test"]["topology_acc_per_frame"]; ts = sorted(int(k) for k in pf)
    plt.figure(figsize=(8.5, 5.8))
    plt.plot(ts, [pf[str(t)] for t in ts], "o-", lw=2.5, ms=9)
    plt.xlabel("frame t", fontsize=FS); plt.ylabel("topology accuracy", fontsize=FS); plt.ylim(-0.05, 1.05)
    plt.xticks(fontsize=FS - 4); plt.yticks(fontsize=FS - 4)
    plt.title("topology accuracy through the merge", fontsize=FS); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "topo_vs_time.png"), dpi=FIG_DPI); plt.close()
    print("saved -> results/{predictions.png, topo_vs_time.png}")


if __name__ == "__main__":
    main()
