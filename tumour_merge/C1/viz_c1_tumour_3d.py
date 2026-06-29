"""
3D space-time view of the FROST C1 tumour predictions, rendered with EXACTLY the tumour_merge_3d.png
machinery (FBNO panel-a style: cumulative space-time boundary surfaces, z = time, red t=0 caps,
YlGnBu radius colouring). Five time columns t = 0, 8, 10, 12, 14 and THREE rows:
  row 1 = GT          (the benchmark `draw_panel`, identical to tumour_merge_3d.png)
  row 2 = FROST pred  (same renderer, fed the operator's predicted level set φ(t))
  row 3 = |Δφ| error  (the predicted space-time surface coloured by |φ_GT − φ_pred|, magma)
so the 2→1 topology change and the prediction error are shown in the benchmark's own style.

Run:  python viz_c1_tumour_3d.py   ->  results/predictions_3d.png
(needs results/model.pt from train_c1_tumour.py)
"""
import os, sys, copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.ndimage import zoom, label, gaussian_filter
from skimage.measure import find_contours, marching_cubes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))   # the benchmark module
import viz_tumour_merge_3d as B                                  # reuse its exact renderers
from fno import FNO2d
from train_c1_tumour import load_split, OUT, FIG_DPI, MODES, WIDTH

N = 128                                                          # benchmark grid (predicted φ is upsampled to this)
CONN = np.ones((3, 3), int)
FS_TITLE, FS_ROW, FS_SUP = 15, 24, 21


def predicted_phi_stack(model, X, stats, i, T, H):
    """Run the operator at every frame -> predicted φ(t) volume, upsampled to the benchmark grid."""
    xm, xs, ym, ys = stats
    phi = np.zeros((T, H, H), np.float32)
    with torch.no_grad():
        for t in range(T):
            e = i * T + t
            phi[t] = (model(((X[e:e+1] - xm) / xs))[0] * ys + ym).numpy()[..., 1]
    return zoom(phi, (1, N / H, N / H), order=1).astype(np.float32)   # (T,128,128)


def boundary_components(phi):
    """Per-frame solid-boundary polylines in [-1,1] (row->x, col->y), matching the benchmark convention."""
    comps = []
    for t in range(phi.shape[0]):
        frame = []
        for c in find_contours(phi[t], 0.0):
            if len(c) >= 3:
                x = c[:, 0] / (N - 1) * 2 - 1; y = c[:, 1] / (N - 1) * 2 - 1
                frame.append(np.column_stack([x, y]))
        comps.append(frame)
    return np.array(comps, dtype=object)


def make_pred_sample(d_gt, phi_pred):
    """A benchmark-shaped sample dict whose φ / boundaries / n_components come from the PREDICTION."""
    d = copy.deepcopy(d_gt)
    d["phi"] = phi_pred
    d["boundary_components"] = boundary_components(phi_pred)
    d["n_components"] = np.array([int(label(phi_pred[t] < 0, structure=CONN)[1]) for t in range(phi_pred.shape[0])])
    return d


PANEL_TIMES = [0, 8, 10, 12, 14]                                 # fixed five time columns


def draw_ls_panel(ax, d, t_end):
    """Cumulative space-time surface from the sample's ACTUAL level set φ (benchmark machinery).
    Used for BOTH GT and prediction so the two rows share one coordinate system (directly comparable;
    the benchmark's own t=0 panel uses an artistic wide separation, not the literal data)."""
    if t_end == 0:
        B.add_t0_complete_bodies(ax, d)
    else:
        B.add_levelset_surface(ax, d, t_end)
    B.apply_fbno_axes(ax, d, t_end)
    title = "t=0" if t_end == 0 else f"t=0 to {t_end}"
    ax.set_title(f"{title}  ({int(d['n_components'][t_end])} comp.)", fontsize=FS_TITLE, pad=2)


def draw_error_panel(ax, d_pred, phi_gt, t_end, vmax):
    """Predicted space-time surface coloured by |φ_GT − φ_pred| (same geometry as the prediction row)."""
    phi_p = d_pred["phi"].astype(float); Tn, n = phi_p.shape[0], phi_p.shape[1]
    if t_end == 0:
        L = 14
        phiv = np.repeat(phi_p[0][None], L, 0)
        errv = np.repeat(np.abs(phi_gt[0] - phi_p[0])[None], L, 0)
        z_end = B.Z_MAX * B.T0_VISUAL_END / (Tn - 1)
    else:
        phiv = B.upsample_to_time(phi_p, t_end)
        errv = B.upsample_to_time(np.abs(phi_gt - phi_p).astype(float), t_end)
        z_end = B.Z_MAX * t_end / (Tn - 1)
    phiv = gaussian_filter(phiv, sigma=(0.45, 0.6, 0.6))
    errv = gaussian_filter(errv, sigma=(0.45, 0.6, 0.6))
    try:
        verts, faces, _, _ = marching_cubes(phiv, level=0.0, step_size=B.MARCHING_CUBES_STEP)
    except (ValueError, RuntimeError):
        B.apply_fbno_axes(ax, d_pred, t_end); return None
    xyz = np.column_stack([verts[:, 1] / (n - 1) * 2 - 1, verts[:, 2] / (n - 1) * 2 - 1,
                           verts[:, 0] / (phiv.shape[0] - 1) * z_end])
    zi = np.clip(verts[:, 0].round().astype(int), 0, phiv.shape[0] - 1)
    yi = np.clip(verts[:, 1].round().astype(int), 0, n - 1); xi = np.clip(verts[:, 2].round().astype(int), 0, n - 1)
    fe = errv[zi, yi, xi][faces].mean(axis=1)
    surf = ax.plot_trisurf(xyz[:, 0], xyz[:, 1], xyz[:, 2], triangles=faces,
                           alpha=0.9, linewidth=0.04, antialiased=True)
    surf.set_array(fe); surf.set_cmap("magma"); surf.set_clim(0, vmax)
    B.add_initial_caps(ax, d_pred); B.apply_fbno_axes(ax, d_pred, t_end)
    title = "t=0" if t_end == 0 else f"t=0 to {t_end}"
    ax.set_title(f"{title}  |Δφ|", fontsize=FS_TITLE, pad=2)
    return surf


def main():
    X, Y, meta, T, tr, te, tr_s, te_s = load_split()
    H = X.shape[1]
    ck = torch.load(os.path.join(OUT, "model.pt"), map_location="cpu")
    s = ck["stats"]; stats = (s["xm"], s["xs"], s["ym"].reshape(2), s["ys"].reshape(2))
    model = FNO2d(modes=MODES, width=WIDTH, in_c=6, out_c=2, n_layers=4)
    model.load_state_dict(ck["model"]); model.eval()

    data = np.load(os.path.join(os.path.dirname(OUT), "..", "tumour_merge.npy"), allow_pickle=True)
    # pick a held-out test sample that is a valid geometry-A merge for the benchmark renderer
    i = next((int(j) for j in te_s
              if int(data[j]["params"][0]) == 0 and int(data[j]["params"][1]) == 0
              and 0 < int(data[j]["merge_step"]) < data[j]["phi"].shape[0] - 1), int(te_s[0]))
    d_gt = data[i]
    phi_gt = d_gt["phi"].astype(np.float32)
    phi_pred = predicted_phi_stack(model, X, stats, i, T, H)
    # render GT and prediction through the SAME machinery on their ACTUAL φ -> directly comparable
    d_gt_r = make_pred_sample(d_gt, phi_gt)
    d_pred = make_pred_sample(d_gt, phi_pred)
    times = [t for t in PANEL_TIMES if t <= T - 1]
    vmax = float(np.abs(phi_gt - phi_pred).max())

    fig = plt.figure(figsize=(21.5, 14.6))
    xs0, w, hh = 0.045, 0.170, 0.275
    y_rows = [0.69, 0.385, 0.07]                                  # GT / prediction / |error|
    err_surf = None
    for col, t_end in enumerate(times):
        x = xs0 + 0.180 * col
        axT = fig.add_axes([x, y_rows[0], w, hh], projection="3d")
        draw_ls_panel(axT, d_gt_r, t_end)                         # GT, from actual φ
        axM = fig.add_axes([x, y_rows[1], w, hh], projection="3d")
        draw_ls_panel(axM, d_pred, t_end)                         # prediction, same machinery
        axB = fig.add_axes([x, y_rows[2], w, hh], projection="3d")
        es = draw_error_panel(axB, d_pred, phi_gt, t_end, vmax)   # |Δφ| error
        err_surf = es if es is not None else err_surf

    sm = cm.ScalarMappable(norm=B.FBNO_NORM, cmap=B.FBNO_CMAP); sm.set_array([])
    cbar = fig.colorbar(sm, cax=fig.add_axes([0.95, 0.46, 0.011, 0.40]), format="%.2f")
    cbar.set_ticks(np.linspace(B.FBNO_GEOM_A_VMIN, B.FBNO_GEOM_A_VMAX, 4)); cbar.ax.tick_params(labelsize=FS_TITLE - 3)
    if err_surf is not None:
        ecb = fig.colorbar(err_surf, cax=fig.add_axes([0.95, 0.10, 0.011, 0.22])); ecb.ax.tick_params(labelsize=FS_TITLE - 3)
    for y, lab in zip(y_rows, ["GT", "prediction", "|Δφ|\nerror"]):
        fig.text(0.012, y + hh / 2, lab, rotation=90, va="center", ha="center",
                 fontsize=FS_ROW, fontweight="bold")
    out = os.path.join(OUT, "predictions_3d.png")
    fig.savefig(out, dpi=FIG_DPI); plt.close(fig)
    print(f"sample {i}  merge@t={int(d_gt['merge_step'])}  panels={times}  vmax|Δφ|={vmax:.3f}")
    print(f"GT n_comp={[int(d_gt_r['n_components'][t]) for t in times]}  "
          f"pred n_comp={[int(d_pred['n_components'][t]) for t in times]}")
    print("saved", out)


if __name__ == "__main__":
    main()
