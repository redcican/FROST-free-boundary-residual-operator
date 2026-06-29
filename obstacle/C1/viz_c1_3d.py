"""
3D view of the FROST C1 forward-operator predictions on held-out obstacle cases.

Renders the membrane field u(x,y) as a height surface (z = u, same convention as obstacle_3d.png),
GROUND TRUTH vs PREDICTION side by side, for one 1-component and one 2-component test sample. The free
boundary Γ = ∂{u=χ} (from the level set φ) is lifted onto each surface: crimson = that panel's own Γ;
on the PRED panel the GT Γ is also drawn (blue) so the free-boundary match is visible in 3D.

Run:  python viz_c1_3d.py   ->  results/predictions_3d.png
(requires results/model.pt from train_c1_obstacle.py)
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.measure import find_contours

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno import FNO2d
from train_c1_obstacle import load_split, CONTACT_TOL, plot_predictions

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
N = 128
xs = np.linspace(-1, 1, N)
XX, YY = np.meshgrid(xs, xs, indexing="ij")


def gamma_curves(phi):
    """Free-boundary polylines Γ = {φ=0} as (x, y) index arrays."""
    out = []
    for c in find_contours(phi, 0.0):
        r = np.clip(c[:, 0], 0, N - 1).astype(int)
        cc = np.clip(c[:, 1], 0, N - 1).astype(int)
        out.append((r, cc))
    return out


TS = 20


def _axes(ax, zmax, title):
    ax.set_zlim(0, zmax * 1.05); ax.set_box_aspect((1, 1, 0.6), zoom=1.5)
    ax.view_init(elev=34, azim=-58); ax.set_axis_off()
    ax.text2D(0.5, 0.90, title, transform=ax.transAxes, ha="center", va="top", fontsize=TS)


def draw_surface(ax, u, gammas, zmax, title):
    ax.plot_surface(XX, YY, u, cmap="YlGnBu", rcount=90, ccount=90, linewidth=0,
                    antialiased=True, alpha=0.93, vmin=0.0, vmax=zmax)
    for (r, cc), col in gammas:                            # lift each Γ onto the surface
        ax.plot(xs[r], xs[cc], u[r, cc] + 0.03 * zmax, color=col, lw=2.8)
    _axes(ax, zmax, title)


def draw_error_surface(fig, ax, u_geom, err, zmax, title):
    """Membrane geometry (GT) coloured by absolute field error |u_GT - u_pred|."""
    vmax = max(float(err.max()), 1e-6)
    norm = Normalize(0.0, vmax)
    ax.plot_surface(XX, YY, u_geom, facecolors=cm.magma(norm(err)), shade=False,
                    rcount=N, ccount=N, linewidth=0, antialiased=False)
    _axes(ax, zmax, title)
    sm = cm.ScalarMappable(norm=norm, cmap="magma"); sm.set_array([])
    cax = ax.inset_axes([1.04, 0.2, 0.035, 0.6])           # just OUTSIDE the panel's right edge
    cb = fig.colorbar(sm, cax=cax, ticks=np.linspace(0, vmax, 4))
    cb.ax.tick_params(labelsize=14)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def render_3d(model, X, Y, chi, nc, te, stats, out_path):
    """3D GT / prediction / |Δu| surfaces for a 2-comp and a 1-comp held-out case."""
    ym, ysd = stats["ym"].reshape(2), stats["ysd"].reshape(2)
    picks = []
    for c in (2, 1):
        cand = [i for i in te if nc[i] == c]
        if cand:
            picks.append(int(cand[0]))
    fig = plt.figure(figsize=(15.0, 4.6 * len(picks)))
    for r, i in enumerate(picks):
        with torch.no_grad():
            pred = (model((X[i:i+1] - stats["xm"]) / stats["xsd"])[0] * ysd + ym).numpy()
        u_g, phi_g = Y[i, ..., 0].numpy(), Y[i, ..., 1].numpy()
        u_p, phi_p = pred[..., 0], pred[..., 1]
        zmax = float(u_g.max())
        rel = float(np.linalg.norm(u_p - u_g) / (np.linalg.norm(u_g) + 1e-8))
        g_gt = [(rc, "crimson") for rc in gamma_curves(phi_g)]
        g_pr = [(rc, "crimson") for rc in gamma_curves(phi_p)] + \
               [(rc, "tab:blue") for rc in gamma_curves(phi_g)]      # blue = GT Γ on the pred panel
        axL = fig.add_subplot(len(picks), 3, 3 * r + 1, projection="3d")
        axM = fig.add_subplot(len(picks), 3, 3 * r + 2, projection="3d")
        axR = fig.add_subplot(len(picks), 3, 3 * r + 3, projection="3d")
        draw_surface(axL, u_g, g_gt, zmax, f"GT  —  {nc[i]} contact comp.")
        draw_surface(axM, u_p, g_pr, zmax, f"FROST pred  (u rel-L2 {rel*100:.1f}%)\nΓ: pred=red, GT=blue")
        draw_error_surface(fig, axR, u_g, np.abs(u_g - u_p), zmax,
                           f"|Δu| on surface  (max {np.abs(u_g-u_p).max():.3f})")
    fig.subplots_adjust(left=0.02, right=0.93, bottom=0.0, top=1.0, wspace=0.06, hspace=-0.12)
    fig.savefig(out_path, dpi=600); plt.close(fig)
    print("picks (test idx):", picks, "-> saved", out_path)


def main():
    X, Y, chi, nc, tr, te = load_split()
    ck = torch.load(os.path.join(OUT, "model.pt"), map_location="cpu")
    stats = ck["stats"]
    model = FNO2d(modes=16, width=32, in_c=1, out_c=2, n_layers=4)
    model.load_state_dict(ck["model"]); model.eval()
    plot_predictions(model, X, Y, chi, nc, te, stats)        # 2D (with |Δu|,|Δφ| panels)
    render_3d(model, X, Y, chi, nc, te, stats, os.path.join(OUT, "predictions_3d.png"))


if __name__ == "__main__":
    main()
