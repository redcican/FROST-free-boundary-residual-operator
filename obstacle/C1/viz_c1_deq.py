"""
Visualize the DEQ forward-operator predictions (reuses the FNO viz machinery on the DEQ model).

Produces, from results/deq_model.pt, the same 2D + 3D GT-vs-prediction figures as the FNO:
  results/deq_predictions.png      (2D: χ, u GT/pred, |Δu|, φ GT/pred, |Δφ|, contact GT-vs-pred)
  results/deq_predictions_3d.png   (3D: membrane surface u GT / prediction / |Δu|)

Run:  python viz_c1_deq.py
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deq import DEQObstacle
from train_c1_obstacle import load_split, plot_predictions, OUT
from viz_c1_3d import render_3d


def main():
    X, Y, chi, nc, tr, te = load_split()
    ck = torch.load(os.path.join(OUT, "deq_model.pt"), map_location="cpu")
    stats = ck["stats"]
    model = DEQObstacle(modes=12, width=32, f_iter=30, b_iter=14, tol=1e-3)
    model.load_state_dict(ck["model"]); model.eval()

    plot_predictions(model, X, Y, chi, nc, te, stats, out_name="deq_predictions.png")
    render_3d(model, X, Y, chi, nc, te, stats, os.path.join(OUT, "deq_predictions_3d.png"))
    print("saved -> results/{deq_predictions.png, deq_predictions_3d.png}")


if __name__ == "__main__":
    main()
