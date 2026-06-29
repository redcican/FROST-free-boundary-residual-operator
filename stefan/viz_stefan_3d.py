"""
Flat 2D view of the Stefan benchmark (replaces the earlier 3D space-time render). The Stefan problem
here is 2D, so this figure stays in the plane and shows the moving free boundary + topology change
directly:
  (left)  free boundary Γ(t) = {T=0} overlaid for every timestep, coloured by time -- the separate
          grain fronts sweep outward and the contours MERGE from N components to 1;
  (right) solidification arrival-time map: the frame index at which each point first froze, so the
          grains appear as nested time bands and the N->1 coalescence shows as the ridge where two
          fronts meet.

Run: python viz_stefan_3d.py  -> stefan_3d.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "stefan.npy")
OUT_PATH = os.path.join(HERE, "stefan_3d.png")
CMAP = "viridis"
N = 128
xs = np.linspace(-1, 1, N)
XX, YY = np.meshgrid(xs, xs, indexing="ij")


def touches_box(d):
    return any(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() for m in d["mask"])


def pick_sample(data):
    """A clean 3/4-seed coalescence that stays off the box edge, merging mid-window."""
    cand = [d for d in data if int(d["params"][0]) >= 3 and 2 <= int(d["merge_step"]) <= 10
            and not touches_box(d)]
    cand = cand or [d for d in data if int(d["params"][0]) >= 3 and 2 <= int(d["merge_step"]) <= 10]
    cand = cand or list(data)
    return min(cand, key=lambda d: abs(int(d["merge_step"]) - 6))


def main():
    data = np.load(DATA_PATH, allow_pickle=True)
    d = pick_sample(data)
    T = d["phi"].shape[0]
    cmap = plt.get_cmap(CMAP)
    norm = Normalize(0, T - 1)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.9))

    # --- left: free-boundary sweep, one contour per timestep coloured by time
    axL.contourf(XX, YY, d["phi"][-1], levels=[-1e9, 0.0], colors=["0.92"])   # final solid footprint
    for t in range(T):
        axL.contour(XX, YY, d["phi"][t], levels=[0.0], colors=[cmap(norm(t))], linewidths=1.6)
    axL.scatter(d["seeds"][:, 0], d["seeds"][:, 1], c="k", s=22, zorder=5, label="seeds")
    axL.set_title(f"Free boundary Γ(t) = {{T=0}} sweep  ({int(d['n_components'][0])} grains → 1)",
                  fontsize=11)
    axL.set_xlim(-1, 1); axL.set_ylim(-1, 1); axL.set_aspect("equal")
    axL.set_xticks([-1, 0, 1]); axL.set_yticks([-1, 0, 1]); axL.legend(loc="upper right", fontsize=8)

    # --- right: solidification arrival-time map (earliest frame each point is solid)
    arrival = np.full((N, N), np.nan)
    for t in range(T - 1, -1, -1):                      # reversed -> earliest frame wins
        arrival[d["mask"][t]] = t
    cmap_b = cmap.copy(); cmap_b.set_bad("white")
    im = axR.imshow(arrival.T, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap_b,
                    vmin=0, vmax=T - 1, interpolation="nearest")
    axR.contour(XX, YY, d["phi"][-1], levels=[0.0], colors="crimson", linewidths=1.6)
    axR.set_title("Solidification arrival time (frame index)\nmerge = ridge where fronts meet",
                  fontsize=11)
    axR.set_xlim(-1, 1); axR.set_ylim(-1, 1); axR.set_aspect("equal")
    axR.set_xticks([-1, 0, 1]); axR.set_yticks([-1, 0, 1])

    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=[axL, axR], fraction=0.045, pad=0.02)
    cbar.set_label("time (frame index)", fontsize=10)
    fig.suptitle(f"Stefan solidification — 2D front evolution  (seeds={int(d['params'][0])}, "
                 f"merge @ t={int(d['merge_step'])})", fontsize=13)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"sample seeds={int(d['params'][0])} merge@t={int(d['merge_step'])}")
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
