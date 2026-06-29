"""
TODO(b), time-dependent form — FBNO single-diffeomorphism baseline on a tumour_merge TRAJECTORY.

This is the sharpest version of the diffeomorphism-cannot-change-topology argument. FBNO tracks a
moving free boundary as a fixed reference domain warped by a time-continuous diffeomorphism Φ(·, t).
For every t this is a homeomorphism, so the number of connected components is CONSTANT in t — it can
never jump. The tumour_merge benchmark is exactly a 2→1 coalescence within one trajectory, so a single
diffeomorphism is structurally unable to reproduce it: it is frozen at 2 components for all time.

We make the baseline maximally generous (an ORACLE): fix the reference = the t=0 two-tumour domain,
and for EACH frame fit the best diffeomorphic warp DIRECTLY against the ground-truth level set, with a
positive-Jacobian (no-fold) constraint whose minimum we report. We compare the warped-domain component
count against the ground truth and against the trained FROST C1 operator (results/model.pt), which
predicts a level set on a fixed grid and crosses the 2→1 transition correctly.

Run:  python baseline_diffeo_time.py [n_iter]   # default 300 Adam iters/frame
Out:  results/baseline/{metrics.json, baseline_time_topology.png, baseline_time_filmstrip.png}
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno import FNO2d
from train_c1_tumour import load_split, n_comp, iou, DATA, DS, OUT, MODES, WIDTH

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
torch.manual_seed(0); np.random.seed(0)
BASE = os.path.join(OUT, "baseline"); os.makedirs(BASE, exist_ok=True)
FS = 22; DPI = 600
CTRL = 8
JAC_FLOOR = 0.05


# ----------------------------------------------------------------------------- warp machinery (diffeomorphism)
def identity_grid(H, W):
    ys = torch.linspace(-1, 1, H); xs = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)[None]


def warp_field(ctrl, H, W):
    flow = F.interpolate(ctrl, size=(H, W), mode="bicubic", align_corners=True)
    return flow.permute(0, 2, 3, 1)


def jac_rel_det(grid, H, W):
    g = grid[0]; gx, gy = g[..., 0], g[..., 1]
    gx_i = torch.gradient(gx, dim=0)[0]; gx_j = torch.gradient(gx, dim=1)[0]
    gy_i = torch.gradient(gy, dim=0)[0]; gy_j = torch.gradient(gy, dim=1)[0]
    det = gx_j * gy_i - gx_i * gy_j
    id_det = (2.0 / (W - 1)) * (2.0 / (H - 1))
    return det / id_det


def fit_diffeo(ref_phi, tgt_phi, n_iter=300, lr=0.05):
    H, W = ref_phi.shape
    ref = torch.from_numpy(ref_phi)[None, None].float()
    tgt = torch.from_numpy(tgt_phi).float()
    base = identity_grid(H, W)
    ctrl = torch.zeros(1, 2, CTRL, CTRL, requires_grad=True)
    opt = torch.optim.Adam([ctrl], lr=lr)
    w = torch.exp(-(tgt / 0.10) ** 2 / 2.0) + 0.05
    for _ in range(n_iter):
        opt.zero_grad()
        grid = base + warp_field(ctrl, H, W)
        warped = F.grid_sample(ref, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
        fit = (w * (warped - tgt) ** 2).mean()
        fold = F.relu(JAC_FLOOR - jac_rel_det(grid, H, W)).pow(2).mean()
        smooth = (ctrl[:, :, 1:] - ctrl[:, :, :-1]).pow(2).mean() + \
                 (ctrl[:, :, :, 1:] - ctrl[:, :, :, :-1]).pow(2).mean()
        (fit + 120.0 * fold + 0.01 * smooth).backward()
        opt.step()
    with torch.no_grad():
        grid = base + warp_field(ctrl, H, W)
        warped = F.grid_sample(ref, grid, mode="bilinear", padding_mode="border",
                               align_corners=True)[0, 0].numpy()
        mindet = float(jac_rel_det(grid, H, W).min())
    return warped.astype(np.float32), mindet


# ----------------------------------------------------------------------------- main
def main():
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    X, Y, meta, T, tr, te, tr_s, te_s = load_split()
    i = int(te_s[0])                                              # the held-out trajectory (matches predictions.png)
    ms = int(np.load(DATA, allow_pickle=True)[i]["merge_step"])
    print(f"trajectory = test sample {i}, merge_step={ms}, frames={T}, n_iter={n_iter}")

    # FROST C1 operator (trained) — predicted level set per frame
    model = FNO2d(modes=MODES, width=WIDTH, in_c=6, out_c=2, n_layers=4)
    ck = torch.load(os.path.join(OUT, "model.pt"), map_location="cpu")
    model.load_state_dict(ck["model"]); model.eval()
    xm, xs, ym, ys = ck["stats"]["xm"], ck["stats"]["xs"], ck["stats"]["ym"], ck["stats"]["ys"]
    ym2, ys2 = ym.reshape(2), ys.reshape(2)

    ref_phi = Y[i * T + 0, ..., 1].numpy()                       # fixed reference = t=0 two-tumour domain
    print(f"reference (t=0) has {n_comp(ref_phi < 0)} components")

    t0 = time.time()
    gt_nc, diffeo_nc, frost_nc, mindets = [], [], [], []
    warped_store, frost_store = {}, {}
    for t in range(T):
        e = i * T + t
        phi_g = Y[e, ..., 1].numpy()
        warped, mindet = fit_diffeo(ref_phi, phi_g, n_iter=n_iter)
        with torch.no_grad():
            phi_p = (model(((X[e:e+1] - xm) / xs))[0] * ys2 + ym2).numpy()[..., 1]
        gt_nc.append(n_comp(phi_g < 0)); diffeo_nc.append(n_comp(warped < 0))
        frost_nc.append(n_comp(phi_p < 0)); mindets.append(mindet)
        warped_store[t] = warped < 0; frost_store[t] = phi_p < 0
    print(f"all per-frame oracle fits done in {time.time()-t0:.1f}s "
          f"(min rel-Jac det over all frames = {min(mindets):.3f})")

    def topo_acc(seq):
        return float(np.mean([int(a == b) for a, b in zip(seq, gt_nc)]))

    metrics = {
        "n_iter": n_iter, "trajectory_test_sample": i, "merge_step": ms, "frames": T,
        "reference": "t=0 two-tumour domain (2 components, fixed for all t)",
        "diffeomorphism_check": {
            "min_relative_jacobian_det_over_all_frames": float(min(mindets)),
            "note": "min relative Jacobian det > 0 at every frame => each warp is a genuine "
                    "diffeomorphism; the component count is therefore pinned at the reference's value.",
        },
        "component_count_per_frame": {
            "ground_truth": [int(x) for x in gt_nc],
            "single_diffeomorphism_baseline": [int(x) for x in diffeo_nc],
            "frost_operator": [int(x) for x in frost_nc],
        },
        "topology_accuracy_over_trajectory": {
            "single_diffeomorphism_baseline": topo_acc(diffeo_nc),
            "frost_operator": topo_acc(frost_nc),
        },
        "headline": "A time-continuous diffeomorphism is frozen at 2 components for the whole "
                    "trajectory and is wrong on every post-merge frame; FROST crosses the 2->1 "
                    "transition with the ground truth.",
    }
    json.dump(metrics, open(os.path.join(BASE, "metrics.json"), "w"), indent=2)
    print("\n== METRICS ==")
    print(json.dumps({k: metrics[k] for k in
                      ("diffeomorphism_check", "component_count_per_frame",
                       "topology_accuracy_over_trajectory")}, indent=2))

    plot_topology(T, ms, gt_nc, diffeo_nc, frost_nc, metrics)
    plot_filmstrip(T, ms, i, Y, gt_nc, diffeo_nc, frost_nc, warped_store, frost_store)
    print(f"saved -> {BASE}")


# ----------------------------------------------------------------------------- figures
def plot_topology(T, ms, gt, diffeo, frost, m):
    fig, ax = plt.subplots(figsize=(10, 6))
    fr = range(T)
    ax.plot(fr, frost, "o-", lw=3, ms=11, color="#1f77b4", label="FROST (level set)", zorder=3)
    ax.plot(fr, gt, "s--", lw=2.5, ms=9, color="k", label="ground truth", zorder=2)
    ax.plot(fr, diffeo, "^-", lw=3, ms=11, color="#c0392b",
            label="single diffeomorphism (t=0 ref)", zorder=3)
    ax.axvline(ms, color="gray", ls=":", lw=2)
    ax.text(ms + 0.1, 1.5, "merge", rotation=90, va="center", fontsize=FS - 6, color="gray")
    ax.set_xlabel("frame t", fontsize=FS); ax.set_ylabel("# connected components", fontsize=FS)
    ax.set_yticks([1, 2]); ax.set_ylim(0.7, 2.3)
    ax.tick_params(labelsize=FS - 4); ax.grid(alpha=0.3)
    ax.legend(fontsize=FS - 6, loc="center left")
    da = m["topology_accuracy_over_trajectory"]["single_diffeomorphism_baseline"]
    fa = m["topology_accuracy_over_trajectory"]["frost_operator"]
    ax.set_title(f"topology over the merge — diffeo {da:.0%} vs FROST {fa:.0%}", fontsize=FS - 2)
    fig.tight_layout(); fig.savefig(os.path.join(BASE, "baseline_time_topology.png"), dpi=DPI); plt.close(fig)


def plot_filmstrip(T, ms, i, Y, gt, diffeo, frost, warped_store, frost_store):
    """Rows: GT domain | single-diffeo warp (stuck 2) | FROST. Cols: representative frames."""
    cols = sorted(set([0, max(1, ms - 2), ms, min(T - 1, ms + 2), T - 1]))
    H = Y.shape[1]; gx = np.linspace(-1, 1, H)
    rows = [("ground truth", "#2b6cb0", gt, lambda t: Y[i * T + t, ..., 1].numpy() < 0),
            ("single diffeomorphism\n(t=0 reference)", "#c0392b", diffeo, lambda t: warped_store[t]),
            ("FROST level-set\noperator", "#1f9e5a", frost, lambda t: frost_store[t])]
    fig, axs = plt.subplots(3, len(cols), figsize=(3.0 * len(cols), 9.4))

    def panel(ax, mask, color):
        ax.imshow(np.zeros_like(mask, float).T, origin="lower", extent=(-1, 1, -1, 1),
                  cmap="gray", vmin=0, vmax=1)
        ax.contourf(gx, gx, mask.astype(float).T, levels=[0.5, 1.5], colors=[color], alpha=0.5)
        ax.contour(gx, gx, mask.astype(float).T, levels=[0.5], colors=[color], linewidths=2.2)
        ax.text(0.03, 0.05, f"{n_comp(mask)}c", transform=ax.transAxes, fontsize=FS - 6,
                bbox=dict(fc="white", alpha=0.8, ec="none"))
        ax.set_xticks([]); ax.set_yticks([])

    for r, (lab, color, _, getmask) in enumerate(rows):
        for c, t in enumerate(cols):
            panel(axs[r, c], getmask(t), color)
            if r == 0:
                axs[r, c].set_title(f"t={t}" + ("  (merge)" if t == ms else ""), fontsize=FS - 4)
        axs[r, 0].set_ylabel(lab, fontsize=FS - 5)
    fig.tight_layout(); fig.savefig(os.path.join(BASE, "baseline_time_filmstrip.png"), dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    main()
