"""
3D view of the obstacle-problem benchmark. The obstacle problem is STEADY (no time), so the natural
3D rendering is the membrane field u(x,y) as a height surface (it rises over the obstacle bumps and
is flat where it CONTACTS them). The free boundary Γ = ∂{u=χ} (contact-set boundary) is overlaid as
a black curve lifted onto the surface. A 2×3 grid spans 2-region and 1-region contact sets.

Run:  python viz_obstacle_3d.py  -> obstacle_3d.png
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.ndimage import map_coordinates
from skimage.measure import find_contours

HERE = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(HERE, "obstacle.npy"), allow_pickle=True)
N = data[0]["u"].shape[0]
xs = np.linspace(-1, 1, N); XX, YY = np.meshgrid(xs, xs, indexing="ij")
NORM = Normalize(0.0, max(float(d["u"].max()) for d in data))
CMAP = "YlGnBu"


def lift(u, rc):
    """Map subpixel contour coords to (x,y) in [-1,1] and bilinearly sample u for height."""
    r = np.clip(rc[:, 0], 0, N - 1)
    c = np.clip(rc[:, 1], 0, N - 1)
    x = -1.0 + 2.0 * r / (N - 1)
    y = -1.0 + 2.0 * c / (N - 1)
    z = map_coordinates(u, np.vstack([r, c]), order=1, mode="nearest")
    return x, y, z + 0.055 * (NORM.vmax - NORM.vmin)


def surface(ax, d):
    u = d["u"]
    ax.plot_surface(XX, YY, u, cmap=CMAP, norm=NORM, linewidth=0, antialiased=True, alpha=0.92,
                    rcount=112, ccount=112)
    for rc in find_contours(d["phi"], 0.0):                 # free boundary Γ = ∂{u=χ}
        x, y, z = lift(u, rc)
        ax.plot(x, y, z, color="black", lw=5.0, alpha=0.92)
        ax.plot(x, y, z + 0.006 * (NORM.vmax - NORM.vmin), color="crimson", lw=3.0, alpha=1.0)
    ax.set_title(f"{d['n_components']} contact region(s)", fontsize=10)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_zlim(NORM.vmin, NORM.vmax)
    try:
        ax.set_box_aspect((1, 1, 0.55), zoom=1.55)
    except TypeError:
        ax.set_box_aspect((1, 1, 0.55))
    ax.view_init(elev=32, azim=-60); ax.set_axis_off()


def main():
    two = [x for x in data if x["n_components"] == 2]
    one = [x for x in data if x["n_components"] == 1]
    picks = (two[:4] + one[:2])[:6]
    while len(picks) < 6 and len(picks) < len(data):
        picks.append(data[len(picks)])
    fig = plt.figure(figsize=(19.5, 10.8))
    positions = [
        [0.015, 0.51, 0.285, 0.38],
        [0.315, 0.51, 0.285, 0.38],
        [0.615, 0.51, 0.285, 0.38],
        [0.015, 0.08, 0.285, 0.38],
        [0.315, 0.08, 0.285, 0.38],
        [0.615, 0.08, 0.285, 0.38],
    ]
    for k, d in enumerate(picks):
        surface(fig.add_axes(positions[k], projection="3d"), d)
    sm = cm.ScalarMappable(norm=NORM, cmap=CMAP); sm.set_array([])
    cbar = fig.colorbar(sm, cax=fig.add_axes([0.93, 0.17, 0.018, 0.64]))
    cbar.set_label("membrane $u(x,y)$", fontsize=10)
    fig.suptitle("Obstacle problem — membrane $u(x,y)$ surface with free boundary Γ = ∂{u=χ} (crimson)\n"
                 "(steady free-boundary equilibrium; contact set merges 2→1 as bumps approach)", fontsize=12)
    out = os.path.join(HERE, "obstacle_3d.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("contact components of picks:", [int(d["n_components"]) for d in picks]); print("saved", out)


if __name__ == "__main__":
    main()
