"""
2D preview for the Stefan solidification benchmark.

The figure prefers a clean non-clipped 3/4-seed sample, uses a shared temperature
scale over all five panels, and overlays the solidification front Gamma.

Run: python viz_stefan_2d.py -> stefan_preview.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "stefan.npy")
OUT_PATH = os.path.join(HERE, "stefan_preview.png")
CMAP = "Blues_r"


def touches_box(sample):
    return any(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
               for mask in sample["mask"])


def pick_sample(data):
    candidates = [
        d for d in data
        if int(d["params"][0]) >= 3 and 2 <= int(d["merge_step"]) <= 10 and not touches_box(d)
    ]
    candidates = candidates or [
        d for d in data
        if int(d["params"][0]) >= 3 and 2 <= int(d["merge_step"]) <= 10
    ]
    candidates = candidates or list(data)
    return min(candidates, key=lambda d: abs(int(d["merge_step"]) - 6))


def selected_times(d):
    tmax = d["phi"].shape[0] - 1
    ms = int(d["merge_step"])
    return [0, max(1, ms - 2), ms, min(tmax, ms + 3), tmax]


def main():
    data = np.load(DATA_PATH, allow_pickle=True)
    d = pick_sample(data)
    times = selected_times(d)
    n = d["u"].shape[-1]
    xs = np.linspace(-1, 1, n)
    norm = Normalize(float(d["params"][2]), 0.0)

    fig, axs = plt.subplots(1, len(times), figsize=(17.5, 3.9))
    im = None
    for ax, t in zip(axs, times):
        im = ax.imshow(d["u"][t].T, origin="lower", extent=(-1, 1, -1, 1),
                       cmap=CMAP, norm=norm, interpolation="bilinear")
        ax.contour(xs, xs, d["phi"][t].T, levels=[0], colors="black", linewidths=3.0)
        ax.contour(xs, xs, d["phi"][t].T, levels=[0], colors="crimson", linewidths=1.4)
        ax.set_title(f"t={t}  ({int(d['n_components'][t])} comp.)", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(left=0.02, right=0.91, bottom=0.08, top=0.78, wspace=0.08)
    cax = fig.add_axes([0.935, 0.18, 0.012, 0.54])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("temperature $T$", fontsize=10)
    fig.suptitle("Stefan solidification — temperature $T$ with front Γ = {T=0} (crimson)",
                 fontsize=12)
    fig.savefig(OUT_PATH, dpi=140)
    plt.close(fig)
    print(f"sample seeds={int(d['params'][0])} merge@t={int(d['merge_step'])} panels={times}")
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
