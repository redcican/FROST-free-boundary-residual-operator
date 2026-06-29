"""
2D preview of the obstacle-problem benchmark.

The plot uses the same representative samples, orientation, colormap, and global
color scale as viz_obstacle_3d.py. The crimson curve is the free boundary
Gamma = partial{u = chi}.

Run:  python viz_obstacle_2d.py  -> obstacle_preview.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(HERE, "obstacle.npy"), allow_pickle=True)
N = data[0]["u"].shape[0]
xs = np.linspace(-1, 1, N)
NORM = Normalize(0.0, max(float(d["u"].max()) for d in data))
CMAP = "YlGnBu"


def pick_samples():
    two = [x for x in data if x["n_components"] == 2]
    one = [x for x in data if x["n_components"] == 1]
    picks = (two[:4] + one[:2])[:6]
    while len(picks) < 6 and len(picks) < len(data):
        picks.append(data[len(picks)])
    return picks


def main():
    picks = pick_samples()
    fig, axs = plt.subplots(2, 3, figsize=(14.6, 8.2))
    axs = np.asarray(axs).ravel()
    im = None
    for ax, d in zip(axs, picks):
        im = ax.imshow(d["u"].T, origin="lower", extent=(-1, 1, -1, 1),
                       cmap=CMAP, norm=NORM, interpolation="bilinear")
        ax.contour(xs, xs, d["phi"].T, levels=[0], colors="black", linewidths=3.2)
        ax.contour(xs, xs, d["phi"].T, levels=[0], colors="crimson", linewidths=1.7)
        ax.set_title(f"{d['n_components']} contact region(s)", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs[len(picks):]:
        ax.set_axis_off()

    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.05, top=0.88, wspace=0.06, hspace=0.20)
    cax = fig.add_axes([0.91, 0.16, 0.018, 0.66])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("membrane $u(x,y)$", fontsize=10)
    fig.suptitle("Obstacle problem — membrane field $u$ with free boundary Γ = ∂{u=χ} (crimson)",
                 fontsize=12)
    out = os.path.join(HERE, "obstacle_preview.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("contact components of picks:", [int(d["n_components"]) for d in picks])
    print("saved", out)


if __name__ == "__main__":
    main()
