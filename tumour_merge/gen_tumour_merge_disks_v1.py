"""
FROST benchmark dataset: TWO-TUMOUR MERGE (topology-change free-boundary problem).

Why this dataset: FBNO represents the moving domain by a single diffeomorphism, which CANNOT change
topology (merge/split). Here two tumours grow and coalesce (components 2 -> 1) -- the case FBNO
cannot represent and FROST (level-set on a fixed grid) can. We cannot run FBNO's MATLAB FEM/FVM
generator, so we build a Python level-set generator: this is also exactly FROST's geometry model.

Model (nutrient-coupled free-boundary growth, in the spirit of the tumour FBP):
  * domain Omega_t = {phi < 0} on a fixed box D; phi is a signed-distance level set.
  * nutrient u solves a steady reaction-diffusion inside Omega_t:  (-Lap + k) u = 0,  u = 1 on Gamma
    (boundary = nutrient source; u high at the rim, low in the centre -- as in FBNO Fig.4).
  * growth: the front advances with normal speed V_n = alpha * mean_Omega(u)  (saturates as the
    tumour starves) -> the two seeds expand and MERGE.  phi <- phi - dt*V_n, then reinitialise to a
    signed distance (scipy EDT, which handles the topology change automatically).

Each sample (varied seeds/params) stores per time step: phi, u (masked), mask  -> the (geometry,
field) ground truth FROST learns.  Output: two_tumour_merge.npz  + a preview figure.

Run:  python gen_two_tumour_merge.py [n_samples]
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
N = 96                      # grid (NxN) on [-1,1]^2
EXT = 1.0
T = 15                      # time steps (matches FBNO tumour)
H = 2 * EXT / (N - 1)
xs = np.linspace(-EXT, EXT, N)
XX, YY = np.meshgrid(xs, xs, indexing="ij")


def signed_distance(mask):
    """Signed distance (negative inside Omega), in physical units."""
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    return (outside - inside) * H


def init_two_disks(c1, r1, c2, r2):
    d1 = np.sqrt((XX - c1[0]) ** 2 + (YY - c1[1]) ** 2) - r1
    d2 = np.sqrt((XX - c2[0]) ** 2 + (YY - c2[1]) ** 2) - r2
    phi0 = np.minimum(d1, d2)                      # union of two disks
    return signed_distance(phi0 < 0)


def solve_nutrient(mask, k):
    """(-Lap + k) u = 0 inside mask, u = 1 on the boundary (cells with an outside neighbour)."""
    u = np.zeros((N, N))
    idx = -np.ones((N, N), dtype=np.int64)
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return u
    for n, (i, j) in enumerate(cells):
        idx[i, j] = n
    M = len(cells)
    rows, cols, data, rhs = [], [], [], np.zeros(M)
    diag = 4.0 / H ** 2 + k
    for n, (i, j) in enumerate(cells):
        rows.append(n); cols.append(n); data.append(diag)
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ii, jj = i + di, j + dj
            if 0 <= ii < N and 0 <= jj < N and mask[ii, jj]:
                rows.append(n); cols.append(idx[ii, jj]); data.append(-1.0 / H ** 2)
            else:
                rhs[n] += (1.0 / H ** 2) * 1.0       # Dirichlet u=1 outside (nutrient source)
    A = sp.csr_matrix((data, (rows, cols)), shape=(M, M))
    sol = spsolve(A, rhs)
    for n, (i, j) in enumerate(cells):
        u[i, j] = sol[n]
    return u


def grow_sample(c1, r1, c2, r2, k, alpha, dt):
    phi = init_two_disks(c1, r1, c2, r2)
    phis, us, masks = [], [], []
    n_components = []
    for _ in range(T):
        mask = phi < 0
        u = solve_nutrient(mask, k)
        phis.append(phi.copy()); us.append(u.copy()); masks.append(mask.copy())
        # connected components (4-connectivity) to detect the merge event
        from scipy.ndimage import label
        n_components.append(int(label(mask)[1]))
        V = alpha * (u[mask].mean() if mask.any() else 0.0)     # saturating, nutrient-coupled
        phi = signed_distance((phi - dt * V) < 0)               # advect + reinitialise
    return (np.array(phis), np.array(us), np.array(masks), np.array(n_components))


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rng = np.random.default_rng(0)
    data = []
    t0 = time.time()
    for s in range(n_samples):
        gap = rng.uniform(0.26, 0.40)                  # half-separation of the two centres
        r1 = rng.uniform(0.14, 0.20); r2 = rng.uniform(0.14, 0.20)
        k = rng.uniform(1.0, 5.0)                       # nutrient consumption
        alpha = rng.uniform(1.0, 1.5)                   # growth rate
        dt = 0.025
        c1, c2 = (-gap, 0.0), (gap, 0.0)
        phis, us, masks, ncomp = grow_sample(c1, r1, c2, r2, k, alpha, dt)
        merge_step = int(np.argmax(ncomp == 1)) if (ncomp == 1).any() else -1
        data.append(dict(phi=phis.astype(np.float32), u=us.astype(np.float32),
                         mask=masks, params=np.array([c1[0], c2[0], r1, r2, k, alpha, dt], np.float32),
                         n_components=ncomp, merge_step=merge_step))
        print(f"  sample {s}: components {ncomp.tolist()}  merge@{merge_step}", flush=True)
    np.save(os.path.join(HERE, "two_tumour_merge.npy"), np.array(data, dtype=object), allow_pickle=True)
    merged = sum(1 for d in data if d["merge_step"] >= 0)
    summary = {"n_samples": n_samples, "grid": N, "T": T,
               "merged_samples": merged,
               "merge_steps": [int(d["merge_step"]) for d in data]}
    with open(os.path.join(HERE, "two_tumour_merge_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{merged}/{n_samples} samples merge. saved two_tumour_merge.npy in {time.time()-t0:.1f}s")

    # preview: one sample that merges, field + zero-level-set over time
    d = next((x for x in data if 0 < x["merge_step"] < T - 1), data[0])
    ms = d["merge_step"]
    show = sorted(set([0, max(0, ms - 2), ms, min(T - 1, ms + 2), T - 1]))
    fig, axs = plt.subplots(1, len(show), figsize=(3 * len(show), 3))
    for ax, ts in zip(axs, show):
        field = np.where(d["mask"][ts], d["u"][ts], np.nan)
        im = ax.imshow(np.rot90(field), extent=(-1, 1, -1, 1), cmap="rainbow", vmin=0, vmax=1)
        ax.contour(XX, YY, d["phi"][ts], levels=[0], colors="k", linewidths=1.2)
        ax.set_title(f"t={ts}  ({d['n_components'][ts]} comp.)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("FROST two-tumour merge — nutrient field + free boundary Γ (black) over time")
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "two_tumour_merge_preview.png"), dpi=100, bbox_inches="tight")
    print(f"preview -> two_tumour_merge_preview.png (sample with merge@{ms})")


if __name__ == "__main__":
    main()
