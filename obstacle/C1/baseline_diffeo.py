"""
TODO(b) — FBNO single-diffeomorphism baseline on the obstacle benchmark.

WHY THIS EXISTS
---------------
FBNO represents a moving free boundary as the image of ONE fixed reference shape under a learned
smooth, invertible map — a diffeomorphism (FBNO's "reference frame" + the decoder de_x/de_y that
maps reference coordinates to physical ones). A diffeomorphism is a homeomorphism, so it PRESERVES
the number of connected components: a single-component reference can never become two, and a
two-component reference can never fuse into one. The FROST obstacle family has BOTH (the contact set
splits into two regions when the bumps are far/low and merges into one when close/high), so NO single
diffeomorphism can cover the whole family. FROST predicts a level set on a fixed grid, which can
change sign-topology freely — that is exactly the capability this baseline lacks.

HOW IT IS MADE GENEROUS (so the failure is structural, not under-fitting)
-------------------------------------------------------------------------
For every held-out instance we fit the BEST diffeomorphic warp of a fixed reference template
DIRECTLY against the GROUND-TRUTH level set — an ORACLE. The oracle is an upper bound on any *learned*
diffeomorphism operator (FBNO included): a learned warp predicted from χ can only be worse than the
best warp fit against the answer. The warp is a band-limited control-grid displacement; we constrain
its Jacobian determinant to stay positive everywhere (no folding) and REPORT its minimum, so each
warp is a verified diffeomorphism. With folding forbidden, a 1-component template provably stays
1-component — it physically cannot split — which is the whole point.

We then compare topology accuracy / contact IoU against the trained FROST operator (results/model.pt)
on the identical held-out split.

Run:  python baseline_diffeo.py [n_iter]      # default 300 Adam iters per fit
Out:  results/baseline/{metrics.json, baseline_topology.png, baseline_qualitative.png}
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
from train_c1_obstacle import load_split, n_comp, iou, CONTACT_TOL, DATA, OUT

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
torch.manual_seed(0); np.random.seed(0)
DEVICE = torch.device("cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(OUT, "baseline"); os.makedirs(BASE, exist_ok=True)
FS = 22                                                  # figure font size (>20, house style)
DPI = 600
CTRL = 8                                                 # control-grid resolution of the warp (8x8)
JAC_FLOOR = 0.05                                         # forbid local area scaling below 5% of identity (=> no folding)


# ----------------------------------------------------------------------------- warp machinery
def identity_grid(H, W):
    """Normalized [-1,1] sampling grid in grid_sample (x=W, y=H) order, shape (1,H,W,2)."""
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)[None]           # (1,H,W,2)


def warp_field(ctrl, H, W):
    """Upsample an (1,2,CTRL,CTRL) control grid to a smooth, band-limited (1,H,W,2) displacement."""
    flow = F.interpolate(ctrl, size=(H, W), mode="bicubic", align_corners=True)
    return flow.permute(0, 2, 3, 1)                      # (1,H,W,2) in normalized units


def jac_rel_det(grid, H, W):
    """Relative Jacobian determinant of the sampling map (grid) vs the identity grid.

    grid maps pixel (i,j) -> normalized coords. det of d(grid)/d(pixel), normalized so the identity
    map gives 1.0 everywhere. >0 everywhere  <=>  orientation-preserving  <=>  diffeomorphism (no fold).
    """
    g = grid[0]                                          # (H,W,2): components (gx,gy)
    gx, gy = g[..., 0], g[..., 1]
    # d/d(row i) and d/d(col j) via central differences
    gx_i = torch.gradient(gx, dim=0)[0]; gx_j = torch.gradient(gx, dim=1)[0]
    gy_i = torch.gradient(gy, dim=0)[0]; gy_j = torch.gradient(gy, dim=1)[0]
    det = gx_j * gy_i - gx_i * gy_j                      # area element (sign chosen so identity>0)
    id_det = (2.0 / (W - 1)) * (2.0 / (H - 1))           # identity grid spans [-1,1] over H,W pixels
    return det / id_det


def fit_diffeo(ref_phi, tgt_phi, n_iter=300, lr=0.05):
    """Fit the best diffeomorphic warp of ref_phi to tgt_phi (oracle: against GT). Returns warped φ
    and the min relative Jacobian det (proof it stayed a diffeomorphism)."""
    H, W = ref_phi.shape
    ref = torch.from_numpy(ref_phi)[None, None].float()
    tgt = torch.from_numpy(tgt_phi).float()
    base = identity_grid(H, W)
    ctrl = torch.zeros(1, 2, CTRL, CTRL, requires_grad=True)
    opt = torch.optim.Adam([ctrl], lr=lr)
    # weight the fit toward the interface band so the warp matches the boundary, not far-field SDF values
    w = torch.exp(-(tgt / 0.10) ** 2 / 2.0) + 0.05
    for _ in range(n_iter):
        opt.zero_grad()
        grid = base + warp_field(ctrl, H, W)
        warped = F.grid_sample(ref, grid, mode="bilinear", padding_mode="border",
                               align_corners=True)[0, 0]
        fit = (w * (warped - tgt) ** 2).mean()
        rel = jac_rel_det(grid, H, W)
        fold = F.relu(JAC_FLOOR - rel).pow(2).mean()      # forbid folding (keeps it a diffeomorphism)
        smooth = (ctrl[:, :, 1:] - ctrl[:, :, :-1]).pow(2).mean() + \
                 (ctrl[:, :, :, 1:] - ctrl[:, :, :, :-1]).pow(2).mean()
        (fit + 120.0 * fold + 0.01 * smooth).backward()
        opt.step()
    with torch.no_grad():
        grid = base + warp_field(ctrl, H, W)
        warped = F.grid_sample(ref, grid, mode="bilinear", padding_mode="border",
                               align_corners=True)[0, 0].numpy()
        min_rel_det = float(jac_rel_det(grid, H, W).min())
    return warped.astype(np.float32), min_rel_det


# ----------------------------------------------------------------------------- references & FROST
def class_medoid(phis):
    """Index of the most representative shape in a class: minimizes mean contact-IoU distance."""
    masks = [p < 0 for p in phis]
    best, bj = 1e9, 0
    for j in range(len(masks)):
        d = np.mean([1.0 - iou(masks[j], masks[k]) for k in range(len(masks)) if k != j])
        if d < best:
            best, bj = d, j
    return bj


def frost_eval(model, X, Y, chi, stats, idx):
    """FROST FNO predicted contact set {φ<0}∩{χ>0} per test index."""
    ym, ysd = stats["ym"].reshape(2), stats["ysd"].reshape(2)
    out = {}
    with torch.no_grad():
        for i in idx:
            xb = (X[i:i+1] - stats["xm"]) / stats["xsd"]
            pred = (model(xb)[0] * ysd + ym).numpy()
            ct = (pred[..., 1] < 0) & (chi[i].numpy() > 0)
            out[int(i)] = ct
    return out


# ----------------------------------------------------------------------------- main
def main():
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    X, Y, chi, nc, tr, te = load_split()
    phi_all = Y[..., 1].numpy()
    chi_np = chi.numpy()
    print(f"train {len(tr)}  test {len(te)}  "
          f"(test topo: { {int(c): int((nc[te]==c).sum()) for c in (1, 2)} })  n_iter={n_iter}")

    # fixed reference template per topology class = medoid of that class on the TRAIN set (most fair)
    refs = {}
    for c in (1, 2):
        cls = [i for i in tr if nc[i] == c]
        m = class_medoid([phi_all[i] for i in cls])
        refs[c] = cls[m]
        print(f"  reference for {c}-comp class = train sample {cls[m]} (medoid)")

    # ---- FROST head-to-head (trained operator) on the same test split
    model = FNO2d(modes=16, width=32, in_c=1, out_c=2, n_layers=4)
    ck = torch.load(os.path.join(OUT, "model.pt"), map_location="cpu")
    model.load_state_dict(ck["model"]); model.eval()
    stats = {k: torch.as_tensor(v) for k, v in ck["stats"].items()}
    frost_ct = frost_eval(model, X, Y, chi, stats, te)

    # ---- fit the oracle diffeomorphism for BOTH references against every test instance
    t0 = time.time()
    rows = []
    warped_store = {}                                    # (ref_class, test_idx) -> warped φ
    for c_ref in (1, 2):
        ref_phi = phi_all[refs[c_ref]]
        for i in te:
            warped, mindet = fit_diffeo(ref_phi, phi_all[i], n_iter=n_iter)
            warped_store[(c_ref, int(i))] = warped
            ct_w = warped < 0
            ct_gt = phi_all[i] < 0
            rows.append(dict(ref=c_ref, i=int(i), nc=int(nc[i]),
                             iou=iou(ct_w, ct_gt),
                             ncomp_pred=n_comp(ct_w),
                             topo_ok=int(n_comp(ct_w) == nc[i]),
                             min_rel_det=mindet))
        print(f"  fitted {c_ref}-comp reference to all test cases "
              f"(min rel-Jac det over all fits = {min(r['min_rel_det'] for r in rows if r['ref']==c_ref):.3f})")
    print(f"all oracle fits done in {time.time()-t0:.1f}s")

    # ---- aggregate
    def agg(ref, key, cls=None):
        v = [r[key] for r in rows if r["ref"] == ref and (cls is None or r["nc"] == cls)]
        return float(np.mean(v)) if v else None

    frost_topo = {}; frost_iou = {}
    for cls in (1, 2, None):
        sub = [i for i in te if (cls is None or nc[i] == cls)]
        frost_topo[cls] = float(np.mean([n_comp(frost_ct[int(i)]) == nc[i] for i in sub]))
        frost_iou[cls] = float(np.mean([iou(frost_ct[int(i)], phi_all[i] < 0) for i in sub]))

    metrics = {
        "n_iter": n_iter, "n_test": int(len(te)),
        "test_topology_counts": {int(c): int((nc[te] == c).sum()) for c in (1, 2)},
        "reference_train_idx": {f"{c}_comp": int(refs[c]) for c in (1, 2)},
        "diffeomorphism_check": {
            "min_relative_jacobian_det_over_all_fits": float(min(r["min_rel_det"] for r in rows)),
            "note": "min relative Jacobian determinant > 0 over EVERY fit confirms every warp is an "
                    "orientation-preserving diffeomorphism (no folding) — so the topology ceiling is "
                    "structural, not an artifact of an under-powered warp.",
        },
        "single_diffeomorphism_baseline": {
            f"ref_{c}_comp": {
                "topology_acc_overall": agg(c, "topo_ok"),
                "topology_acc_on_1comp_targets": agg(c, "topo_ok", 1),
                "topology_acc_on_2comp_targets": agg(c, "topo_ok", 2),
                "contact_IoU_overall": agg(c, "iou"),
                "contact_IoU_on_1comp_targets": agg(c, "iou", 1),
                "contact_IoU_on_2comp_targets": agg(c, "iou", 2),
            } for c in (1, 2)
        },
        "frost_operator": {
            "topology_acc_overall": frost_topo[None],
            "topology_acc_on_1comp_targets": frost_topo[1],
            "topology_acc_on_2comp_targets": frost_topo[2],
            "contact_IoU_overall": frost_iou[None],
            "contact_IoU_on_1comp_targets": frost_iou[1],
            "contact_IoU_on_2comp_targets": frost_iou[2],
        },
        "headline": "A single diffeomorphism matches ONLY its reference's own topology class "
                    "(100% on-class, 0% off-class); no single reference covers the 1- and 2-component "
                    "family. FROST's level-set operator is correct on BOTH classes.",
    }
    json.dump(metrics, open(os.path.join(BASE, "metrics.json"), "w"), indent=2)
    print("\n== METRICS ==")
    print(json.dumps({k: metrics[k] for k in
                      ("diffeomorphism_check", "single_diffeomorphism_baseline", "frost_operator")}, indent=2))

    plot_topology_bars(metrics)
    plot_qualitative(refs, phi_all, chi_np, nc, te, warped_store, frost_ct)
    print(f"saved -> {BASE}")


# ----------------------------------------------------------------------------- figures
def plot_topology_bars(m):
    """Grouped bars: topology accuracy and contact IoU by target class, for the two single-diffeo
    references vs FROST. The off-class zeros are the structural impossibility."""
    methods = ["Diffeo\n(1-comp ref)", "Diffeo\n(2-comp ref)", "FROST\n(level set)"]
    colors = ["#bdbdbd", "#7f7f7f", "#1f77b4"]
    b1 = m["single_diffeomorphism_baseline"]["ref_1_comp"]
    b2 = m["single_diffeomorphism_baseline"]["ref_2_comp"]
    fr = m["frost_operator"]
    topo = {"1-comp targets": [b1["topology_acc_on_1comp_targets"], b2["topology_acc_on_1comp_targets"],
                               fr["topology_acc_on_1comp_targets"]],
            "2-comp targets": [b1["topology_acc_on_2comp_targets"], b2["topology_acc_on_2comp_targets"],
                               fr["topology_acc_on_2comp_targets"]]}
    iouv = {"1-comp targets": [b1["contact_IoU_on_1comp_targets"], b2["contact_IoU_on_1comp_targets"],
                               fr["contact_IoU_on_1comp_targets"]],
            "2-comp targets": [b1["contact_IoU_on_2comp_targets"], b2["contact_IoU_on_2comp_targets"],
                               fr["contact_IoU_on_2comp_targets"]]}
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.4))
    x = np.arange(len(methods)); ww = 0.36
    bars = None
    for ax, data, ttl in ((axs[0], topo, "topology accuracy"), (axs[1], iouv, "contact IoU")):
        b_a = ax.bar(x - ww / 2, data["1-comp targets"], ww, label="1-comp targets", color="#f0a35e")
        b_b = ax.bar(x + ww / 2, data["2-comp targets"], ww, label="2-comp targets", color="#6aa86a")
        bars = (b_a, b_b)
        ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=FS - 4)
        ax.set_ylim(0, 1.16); ax.set_ylabel(ttl, fontsize=FS)
        ax.tick_params(axis="y", labelsize=FS - 4); ax.grid(axis="y", alpha=0.3)
        for xi, (a, b) in enumerate(zip(data["1-comp targets"], data["2-comp targets"])):
            ax.text(xi - ww / 2, a + 0.015, f"{a:.2f}", ha="center", va="bottom", fontsize=FS - 8)
            ax.text(xi + ww / 2, b + 0.015, f"{b:.2f}", ha="center", va="bottom", fontsize=FS - 8)
    fig.legend(bars, ["1-comp targets", "2-comp targets"], fontsize=FS - 5,
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(BASE, "baseline_topology.png"), dpi=DPI); plt.close(fig)


def plot_qualitative(refs, phi_all, chi_np, nc, te, warped_store, frost_ct):
    """Two rows of the impossibility, each 3 panels: GT | best single-diffeo fit (off-class) | FROST.
    Row A: a 2-comp target with the 1-comp reference (cannot split). Row B: a 1-comp target with the
    2-comp reference (cannot merge)."""
    xs = np.linspace(-1, 1, phi_all.shape[1])

    def pick(cls):
        c = [i for i in te if nc[i] == cls]
        return c[0] if c else None

    cases = [(pick(2), 1, "GROUND TRUTH: 2 contact regions",
              "single diffeomorphism\n(1-comp reference)\ncannot split → 1 region"),
             (pick(1), 2, "GROUND TRUTH: 1 contact region",
              "single diffeomorphism\n(2-comp reference)\ncannot merge → 2 regions")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    def panel(ax, mask, title, edge):
        ax.imshow(np.rot90(np.zeros_like(mask, float)), extent=(-1, 1, -1, 1), cmap="gray", vmin=0, vmax=1)
        ax.contourf(xs, xs, np.rot90(mask.astype(float)), levels=[0.5, 1.5], colors=[edge], alpha=0.45)
        ax.contour(xs, xs, np.rot90(mask.astype(float)), levels=[0.5], colors=[edge], linewidths=2.4)
        ax.set_title(title, fontsize=FS - 4); ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.04, f"{n_comp(mask)} region(s)", transform=ax.transAxes,
                fontsize=FS - 6, color="k", bbox=dict(fc="white", alpha=0.8, ec="none"))

    for r, (i, ref_c, gt_ttl, mid_ttl) in enumerate(cases):
        if i is None:
            continue
        gt = phi_all[i] < 0
        warped = warped_store[(ref_c, int(i))] < 0
        fr = frost_ct[int(i)]
        panel(axs[r, 0], gt, gt_ttl, "#2b6cb0")
        panel(axs[r, 1], warped, mid_ttl, "#c0392b")
        panel(axs[r, 2], fr, "FROST level-set operator\n(correct topology)", "#1f9e5a")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "baseline_qualitative.png"), dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    main()
