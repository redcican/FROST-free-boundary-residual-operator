"""
FROST two-tumour-merge dataset -- real FBNO contour topology plus a new merged topology.

The older smooth-disk generator made visually clean "pair of pants" tubes, but it was not a good
tumour benchmark.  This generator starts from the actual FBNO tumour ground-truth boundaries in
domain_test.npy.  Each source contour sequence is scaled once globally, so its original organic
growth/topology is retained instead of normalising every time slice into a constant-size blob.

For a two-tumour sample we place two real contour sequences in the FROST box, then compute their
polygon union at every time step.  Before contact the union has two components; after contact it has
one component with a new exterior boundary.  The masks/level sets are rasterisations of that polygon
union, and the merged boundary components are stored for high-quality visualisation.

Output: tumour_merge.npy with phi/u/mask, source/merged boundary components, params,
n_components and merge_step.  Run: python gen_tumour_merge.py [n_samples]
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.ndimage import distance_transform_edt, label
from matplotlib.path import Path
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
FBNO = os.path.join(HERE, "..", "..", "FBNO", "data", "Tumour_data", "domain_test.npy")
N = 128
EXT = 1.0
T = 15
H = 2 * EXT / (N - 1)
MARGIN = 0.055
BOUNDARY_POINTS = 300
xs = np.linspace(-EXT, EXT, N)
XX, YY = np.meshgrid(xs, xs, indexing="ij")
GRID = np.column_stack([XX.ravel(), YY.ravel()])
CONNECTIVITY = np.ones((3, 3), dtype=int)


def load_real_contours():
    d = np.load(FBNO, allow_pickle=True)
    seen = {}
    for i in range(len(d)):
        k = tuple(np.asarray(d[i]["initial_domain"]).ravel().tolist())
        if k not in seen:
            seen[k] = dict(
                index=i,
                initial_domain=np.asarray(d[i]["initial_domain"]).astype(np.float32),
                contours=np.asarray(d[i]["processed_contours"])[:, :, :2].astype(float),
            )
        if len(seen) == 3:
            break
    return list(seen.values())                       # [A,B,C], each contour array is (15,300,2)


def center_sequence(contours):
    """Recenter each FBNO time slice but preserve its true relative growth."""
    return contours - contours.mean(axis=1, keepdims=True)


def max_abs_extent(seqs):
    return max(float(np.abs(s).max()) for s in seqs)


def rot_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform_contour(contour_t, global_scale, rot, center):
    return contour_t @ rot_matrix(rot).T * global_scale + np.asarray(center)


def valid_polygon(points):
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)
    return poly


def resample_closed(points, n=BOUNDARY_POINTS):
    """Uniformly resample a closed polygon exterior to n points."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-12:
        pts = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = seg > 1e-12
    if not np.all(keep):
        pts = np.vstack([pts[:-1][keep], pts[0]])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    if dist[-1] <= 1e-12:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, dist[-1], n + 1)[:-1]
    return np.column_stack([np.interp(target, dist, pts[:, 0]), np.interp(target, dist, pts[:, 1])])


def polygon_parts(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return list(geom.geoms)


def mask_from_geom(geom):
    mask = np.zeros((N, N), dtype=bool)
    for poly in polygon_parts(geom):
        exterior = np.asarray(poly.exterior.coords)
        inside = Path(exterior).contains_points(GRID).reshape(N, N)
        for interior in poly.interiors:
            hole = Path(np.asarray(interior.coords)).contains_points(GRID).reshape(N, N)
            inside &= ~hole
        mask |= inside
    return mask


def boundary_from_geom(geom):
    parts = []
    for poly in polygon_parts(geom):
        if poly.area > 1e-8:
            parts.append(resample_closed(np.asarray(poly.exterior.coords)[:-1]))
    return parts


def signed_distance(mask):
    return (distance_transform_edt(~mask) - distance_transform_edt(mask)) * H


def solve_nutrient(mask, k):
    """(-Lap + k) u = 0 inside mask, u=1 on the boundary; vectorised sparse assembly."""
    u = np.zeros((N, N))
    cells = np.argwhere(mask); M = len(cells)
    if M == 0:
        return u
    idx = -np.ones((N, N), np.int64); ii, jj = cells[:, 0], cells[:, 1]
    idx[ii, jj] = np.arange(M)
    rows = [np.arange(M)]; cols = [np.arange(M)]; data = [np.full(M, 4 / H ** 2 + k)]
    rhs = np.zeros(M)
    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ni, nj = ii + di, jj + dj
        inb = (ni >= 0) & (ni < N) & (nj >= 0) & (nj < N)
        nb = np.full(M, -1, np.int64); nb[inb] = idx[ni[inb], nj[inb]]
        inside = nb >= 0
        rows.append(np.arange(M)[inside]); cols.append(nb[inside])
        data.append(np.full(inside.sum(), -1 / H ** 2))
        rhs[~inside] += 1.0 / H ** 2                  # Dirichlet u=1 (nutrient source on Gamma)
    A = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(M, M))
    u[ii, jj] = spsolve(A, rhs)
    return u


def build_geometry_sequence(cA, cB, global_scale, sep0, sep1, rotA, rotB):
    """Build the continuous polygon union before rasterising to a fixed FROST grid."""
    geoms, boundaries, masks, ncomp = [], [], [], []
    for t in range(T):
        tau = t / (T - 1)
        smooth = tau * tau * (3.0 - 2.0 * tau)
        sep = sep0 + (sep1 - sep0) * smooth
        p1 = valid_polygon(transform_contour(cA[t], global_scale, rotA, (-sep, 0.0)))
        p2 = valid_polygon(transform_contour(cB[t], global_scale, rotB, (+sep, 0.0)))
        geom = unary_union([p1, p2]).buffer(0)
        geoms.append(geom)
        boundaries.append(boundary_from_geom(geom))
        masks.append(mask_from_geom(geom))
        ncomp.append(len(polygon_parts(geom)))
    return geoms, boundaries, np.array(masks), np.array(ncomp)


def touches_box(mask):
    if not mask.any():
        return True
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def make_sample(cA, cB, global_scale, sep0, sep1, rotA, rotB, k):
    geoms, boundaries, masks, ncomp = build_geometry_sequence(cA, cB, global_scale, sep0, sep1, rotA, rotB)
    phis, us = [], []
    for mask in masks:
        phis.append(signed_distance(mask).astype(np.float32))
        us.append(solve_nutrient(mask, k).astype(np.float32))
    ncomp_grid = np.array([int(label(m, structure=CONNECTIVITY)[1]) for m in masks])
    return np.array(phis), np.array(us), masks, ncomp_grid, boundaries


def choose_candidate(rng, seqs, sample_index):
    # The headline merge example uses exactly the FBNO geometry-A ground-truth topology.
    gA = gB = 0
    target_extent = rng.uniform(0.37, 0.43)
    global_scale = target_extent / max_abs_extent([seqs[gA], seqs[gB]])
    sep0 = rng.uniform(0.36, 0.48)
    sep1 = rng.uniform(0.14, 0.23)
    # Keep the FBNO topology visually recognisable; do not rotate/mirror the canonical
    # A/B/C contour families in the benchmark figure/data.
    rotA = 0.0
    rotB = 0.0
    k = rng.uniform(1.0, 5.0)
    return gA, gB, global_scale, sep0, sep1, rotA, rotB, k


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    sources = load_real_contours()
    seqs = [center_sequence(src["contours"]) for src in sources]
    rng = np.random.default_rng(0)
    data = []; t0 = time.time()
    attempts = 0
    while len(data) < n_samples:
        attempts += 1
        if attempts > 80 * n_samples:
            raise RuntimeError("could not generate enough valid merging samples")
        gA, gB, global_scale, sep0, sep1, rotA, rotB, k = choose_candidate(rng, seqs, len(data))
        geoms, boundaries, masks_preview, ncomp_geom = build_geometry_sequence(
            seqs[gA], seqs[gB], global_scale, sep0, sep1, rotA, rotB
        )
        # Continuous topology and raster topology should agree: separated at t=0, merged later,
        # and never clipped by the FROST computational box.
        ncomp_grid = np.array([int(label(m, structure=CONNECTIVITY)[1]) for m in masks_preview])
        if (not np.array_equal(ncomp_grid, ncomp_geom) or ncomp_geom[0] != 2
                or not (ncomp_geom == 1).any() or (ncomp_grid[0] != 2)
                or not (ncomp_grid == 1).any() or any(touches_box(m) for m in masks_preview)):
            continue
        merge_step = int(np.argmax(ncomp_grid == 1))
        if merge_step <= 2 or merge_step >= T - 1:
            continue
        phis, us, masks, ncomp, boundaries = make_sample(
            seqs[gA], seqs[gB], global_scale, sep0, sep1, rotA, rotB, k
        )
        merge_step = int(np.argmax(ncomp == 1)) if (ncomp == 1).any() else -1
        data.append(dict(phi=phis, u=us, mask=masks,
                         boundary_components=np.array(boundaries, dtype=object),
                         source_indices=np.array([sources[gA]["index"], sources[gB]["index"]], np.int32),
                         params=np.array([gA, gB, global_scale, sep0, sep1, rotA, rotB, k], np.float32),
                         n_components=ncomp, merge_step=merge_step))
        s = len(data) - 1
        if s < 12 or s % 20 == 0:
            print(f"  sample {s}: geoms({chr(65+gA)},{chr(65+gB)}) components {ncomp.tolist()} merge@{merge_step}", flush=True)
    np.save(os.path.join(HERE, "tumour_merge.npy"), np.array(data, dtype=object), allow_pickle=True)
    merged = sum(1 for x in data if x["merge_step"] >= 0)
    json.dump({"n_samples": n_samples, "grid": N, "T": T, "merged_samples": merged,
               "attempts": attempts,
               "merge_steps": [int(x["merge_step"]) for x in data],
              "source": "FBNO geometry-A ground-truth contours; A+A; global contour scale; shapely polygon union"},
              open(os.path.join(HERE, "tumour_merge_summary.json"), "w"), indent=2)
    print(f"\n{merged}/{n_samples} merge. real FBNO contour topology + polygon unions. {time.time()-t0:.1f}s")

    # 2D preview of a merging sample. Use the selected sample's nutrient range;
    # u is close to one everywhere, so a 0..1 scale visually saturates the panels.
    d0 = next((x for x in data if 0 < x["merge_step"] < T - 1), data[0]); ms = d0["merge_step"]
    show = sorted(set([0, max(0, ms - 2), ms, min(T - 1, ms + 2), T - 1]))
    vals = np.concatenate([d0["u"][ts][d0["mask"][ts]] for ts in show])
    norm = Normalize(max(0.0, float(vals.min()) - 0.005), min(1.0, float(vals.max()) + 0.001))
    fig, axs = plt.subplots(1, len(show), figsize=(17.5, 3.9))
    im = None
    for ax, ts in zip(axs, show):
        panel = np.where(d0["mask"][ts], d0["u"][ts], np.nan)
        im = ax.imshow(panel.T, origin="lower", extent=(-1, 1, -1, 1),
                       cmap="YlGnBu", norm=norm, interpolation="bilinear")
        ax.contour(xs, xs, d0["phi"][ts].T, levels=[0], colors="black", linewidths=3.0)
        ax.contour(xs, xs, d0["phi"][ts].T, levels=[0], colors="crimson", linewidths=1.5)
        ax.set_title(f"t={ts} ({d0['n_components'][ts]} comp.)", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(left=0.02, right=0.91, bottom=0.08, top=0.78, wspace=0.08)
    cax = fig.add_axes([0.935, 0.18, 0.012, 0.54])
    cbar = fig.colorbar(im, cax=cax, format="%.3f")
    cbar.set_label("nutrient $u(x,y)$", fontsize=10)
    fig.suptitle("Two-tumour merge from real FBNO contours — nutrient + new union boundary Γ")
    fig.savefig(os.path.join(HERE, "tumour_merge_preview.png"), dpi=140)
    print(f"preview -> tumour_merge_preview.png (merge@{ms})")


if __name__ == "__main__":
    main()
