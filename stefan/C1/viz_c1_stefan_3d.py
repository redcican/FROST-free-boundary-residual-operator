"""
3D space-time view of the FROST C1 stefan predictions (analogue of the tumour C1 3D figure).
For a held-out test sample we stack the level set φ(t) over time and marching-cubes the φ=0
space-time isosurface (z = time): the N solid grains sweep upward and FUSE into 1 (the N→1 topology
change). Three rows over five time columns:
  row 1 = GT          (from the actual GT φ)
  row 2 = prediction  (the operator's predicted φ, same renderer -> directly comparable)
  row 3 = |Δφ| error  (predicted surface coloured by |φ_GT − φ_pred|)

Run:  python viz_c1_stefan_3d.py   ->  results/predictions_3d.png
(needs results/model.pt from train_c1_stefan.py)
"""
import os, sys
import numpy as np
import torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-frost")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter, label
from skimage.measure import find_contours, marching_cubes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fno import FNO2d
from train_c1_stefan import load_split, OUT, FS, FIG_DPI

Z_MAX, NLAY, CMAP = 100.0, 96, "YlGnBu"
ELEV, AZIM = 25, 45
PANEL_TIMES = [0, 4, 7, 10, 14]                          # around the merge (merge_step ~ 7)
CONN = np.ones((3, 3), int)
FS_TITLE, FS_ROW = 15, 24
T0_VISUAL_END = 2
INTERMEDIATE_RINGS = 42
SURFACE_SIGMA = (0.10, 0.34, 0.34)
ERROR_SIGMA = (0.10, 0.34, 0.34)


def pred_phi_stack(model, X, stats, i, T, H):
    xm, xs, ym, ys = stats
    phi = np.zeros((T, H, H), np.float32)
    with torch.no_grad():
        for t in range(T):
            phi[t] = (model(((X[i*T+t:i*T+t+1] - xm) / xs))[0] * ys + ym).numpy()[..., 1]
    return phi


def gt_phi_stack(Y, i, T, H):
    return np.stack([Y[i*T+t, ..., 1].numpy() for t in range(T)]).astype(np.float32)


def cumulative(vol, t_end):
    if t_end == 0:
        return np.repeat(vol[0][None], 18, 0)           # short extrusion of the t=0 grains
    zt = np.linspace(0, t_end, NLAY); lo = np.floor(zt).astype(int); hi = np.clip(lo + 1, 0, t_end)
    w = (zt - lo)[:, None, None]
    return (1.0 - w) * vol[lo] + w * vol[hi]


def interpolate_frame(vol, t_float):
    lo = int(np.floor(t_float))
    hi = min(lo + 1, vol.shape[0] - 1)
    w = float(t_float - lo)
    return (1.0 - w) * vol[lo] + w * vol[hi]


def boundary_polylines(frame, H):
    lines = []
    for contour in find_contours(frame, 0.0):
        if len(contour) < 3:
            continue
        x = contour[:, 0] / (H - 1) * 2.0 - 1.0
        y = contour[:, 1] / (H - 1) * 2.0 - 1.0
        lines.append(np.column_stack([x, y]))
    return lines


def add_initial_caps(ax, phi_vol, H, z=0.0):
    for line in boundary_polylines(phi_vol[0], H):
        verts = [np.column_stack([line[:, 0], line[:, 1], np.full(len(line), z)])]
        cap = Poly3DCollection(verts, alpha=0.42, facecolor="#ef6a5a",
                               edgecolor="#ef6a5a", linewidths=0.45)
        ax.add_collection3d(cap)


def add_layer_rings(ax, phi_vol, T, H, t_end, color="black", include_dense=True):
    z_end = Z_MAX * (T0_VISUAL_END if t_end == 0 else t_end) / (T - 1)

    if include_dense:
        dense_times = np.linspace(0.0, max(float(t_end), float(T0_VISUAL_END if t_end == 0 else t_end)),
                                  INTERMEDIATE_RINGS)
        for tau in dense_times:
            source_tau = 0.0 if t_end == 0 else min(float(t_end), tau)
            z = z_end * (tau / max(float(t_end), float(T0_VISUAL_END), 1e-6))
            for line in boundary_polylines(interpolate_frame(phi_vol, source_tau), H):
                closed = np.vstack([line, line[0]])
                ax.plot(closed[:, 0], closed[:, 1], np.full(len(closed), z),
                        color=color, linewidth=0.22, alpha=0.26)

    for t in range(t_end + 1):
        z = Z_MAX * t / (T - 1)
        ring_color = "#ef6a5a" if t == 0 else color
        alpha = 0.92 if t == 0 else 0.70
        lw = 0.62 if t == 0 else 0.44
        for line in boundary_polylines(phi_vol[t], H):
            closed = np.vstack([line, line[0]])
            ax.plot(closed[:, 0], closed[:, 1], np.full(len(closed), z),
                    color=ring_color, linewidth=lw, alpha=alpha)


def surface(ax, phi_vol, T, H, t_end, ncg, err_vol=None, vmax=None):
    z_end = Z_MAX * (T0_VISUAL_END if t_end == 0 else t_end) / (T - 1)
    vol = gaussian_filter(cumulative(phi_vol.astype(float), t_end), sigma=SURFACE_SIGMA)
    try:
        verts, faces, _, _ = marching_cubes(vol, level=0.0, step_size=1)
    except (ValueError, RuntimeError):
        ax.set_axis_off(); return None
    xyz = np.column_stack([verts[:, 1] / (H - 1) * 2 - 1, verts[:, 2] / (H - 1) * 2 - 1,
                           verts[:, 0] / (vol.shape[0] - 1) * z_end])
    surf = ax.plot_trisurf(xyz[:, 0], xyz[:, 1], xyz[:, 2], triangles=faces,
                           alpha=0.68, linewidth=0.06, antialiased=True)
    if err_vol is not None:
        ev = gaussian_filter(cumulative(err_vol.astype(float), t_end), sigma=ERROR_SIGMA)
        zi = np.clip(verts[:, 0].round().astype(int), 0, ev.shape[0] - 1)
        yi = np.clip(verts[:, 1].round().astype(int), 0, H - 1); xi = np.clip(verts[:, 2].round().astype(int), 0, H - 1)
        surf.set_array(ev[zi, yi, xi][faces].mean(axis=1)); surf.set_cmap("magma"); surf.set_clim(0, vmax)
    else:
        z_face = xyz[faces].mean(axis=1)[:, 2]
        time_step = Z_MAX / (T - 1)
        surf.set_array(np.round(z_face / time_step) * time_step)
        surf.set_cmap(CMAP); surf.set_clim(0, Z_MAX)
    add_initial_caps(ax, phi_vol, H)
    add_layer_rings(ax, phi_vol, T, H, t_end, color="black", include_dense=True)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, Z_MAX)
    ax.set_proj_type("ortho"); ax.view_init(elev=ELEV, azim=AZIM); ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1.35), zoom=1.27)
    except TypeError:
        pass
    ax.set_title(f"t=0 to {t_end}  ({ncg}c)" if t_end else f"t=0  ({ncg}c)", fontsize=FS_TITLE, pad=2)
    return surf


def main():
    X, Y, meta, T, tr, te, tr_s, te_s = load_split()
    H = X.shape[1]
    ck = torch.load(os.path.join(OUT, "model.pt"), map_location="cpu")
    s = ck["stats"]; stats = (s["xm"], s["xs"], s["ym"].reshape(2), s["ys"].reshape(2))
    model = FNO2d(modes=16, width=32, in_c=4, out_c=2, n_layers=4)
    model.load_state_dict(ck["model"]); model.eval()

    d = np.load(os.path.join(HERE, "..", "stefan.npy"), allow_pickle=True)
    # held-out test sample with >=3 seeds (clearest topology change)
    i = next((int(j) for j in te_s if int(d[j]["params"][0]) >= 3), int(te_s[0]))
    phi_g = gt_phi_stack(Y, i, T, H); phi_p = pred_phi_stack(model, X, stats, i, T, H)
    err = np.abs(phi_g - phi_p); vmax = float(err.max())
    times = [t for t in PANEL_TIMES if t <= T - 1]
    nfn = lambda vol, t: int(label(vol[t] < 0, structure=CONN)[1])

    fig = plt.figure(figsize=(21.5, 14.6))
    xs0, w, hh = 0.045, 0.170, 0.275
    y_rows = [0.69, 0.385, 0.07]
    esurf = None
    for col, t_end in enumerate(times):
        x = xs0 + 0.180 * col
        surface(fig.add_axes([x, y_rows[0], w, hh], projection="3d"), phi_g, T, H, t_end, nfn(phi_g, t_end))
        surface(fig.add_axes([x, y_rows[1], w, hh], projection="3d"), phi_p, T, H, t_end, nfn(phi_p, t_end))
        es = surface(fig.add_axes([x, y_rows[2], w, hh], projection="3d"), phi_p, T, H, t_end,
                     nfn(phi_p, t_end), err_vol=err, vmax=vmax)
        esurf = es if es is not None else esurf

    sm = cm.ScalarMappable(norm=Normalize(0, Z_MAX), cmap=plt.get_cmap(CMAP)); sm.set_array([])
    cb = fig.colorbar(sm, cax=fig.add_axes([0.95, 0.46, 0.011, 0.40])); cb.ax.tick_params(labelsize=FS_TITLE - 3)
    if esurf is not None:
        ecb = fig.colorbar(esurf, cax=fig.add_axes([0.95, 0.10, 0.011, 0.22])); ecb.ax.tick_params(labelsize=FS_TITLE - 3)
    for y, lab in zip(y_rows, ["GT", "prediction", "|Δφ|\nerror"]):
        fig.text(0.012, y + hh / 2, lab, rotation=90, va="center", ha="center", fontsize=FS_ROW, fontweight="bold")
    out = os.path.join(OUT, "predictions_3d.png")
    fig.savefig(out, dpi=FIG_DPI); plt.close(fig)
    print(f"sample {i}  merge@t={int(meta[i*T, 2])}  panels={times}")
    print(f"GT n_comp={[nfn(phi_g, t) for t in times]}  pred n_comp={[nfn(phi_p, t) for t in times]}")
    print("saved", out)


if __name__ == "__main__":
    main()
