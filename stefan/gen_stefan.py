"""
FROST benchmark 'stefan' -- 2D two-phase Stefan solidification with grain coalescence.

Adapted from FBNO's stefan_data (../../FBNO/data/stefan_data), which is a 1D melting front s(t):
a single moving point driven by the Stefan flux condition  ds/dt = -(k/rho L) dT/dx|_s  + cooling.
A 1D front cannot change topology, and FBNO's single diffeomorphism cannot represent coalescence.

Here we keep the SAME Stefan physics -- the free boundary is the T = 0 (melting) isotherm and its
velocity is set by the heat-flux jump divided by the latent heat -- but lift it to a 2D FIXED GRID
using the ENTHALPY METHOD (itself FROST's "level set on a fixed background box" idea).  Several solid
seeds grow into a melt and PHYSICALLY MERGE (grain coalescence during solidification is real Stefan
physics).  The number of solid components drops over time = topology change a diffeomorphism cannot do.

Enthalpy method (c = 1, melting temp T = 0):  H = T + L*f_liquid.
    H < 0       -> solid,  T = H,      f_s = 1
    0 <= H <= L -> mushy,  T = 0,      f_s = 1 - H/L     (interface; latent heat buffer)
    H > L       -> liquid, T = H - L,  f_s = 0
Heat equation  dH/dt = D * lap(T).  Cold solid seeds (Dirichlet H = T_cold) are the heat sinks;
the surrounding melt (init T = 0, H = L) loses heat, the T = 0 front advances, grains merge.
Outer box boundary insulated (Neumann).  This is the one-phase Stefan problem on a fixed grid.

Free boundary: the solidification front {f_s = 0.5} == {H = L/2}.  phi = signed distance (neg inside
solid).  Stored per time frame like tumour_merge: phi/u/mask (T,128,128), front polylines, n_components.

Output: stefan.npy (+ summary, preview).  Run single-threaded BLAS:  python gen_stefan.py [n]
"""
import os, sys, json, time
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import find_contours
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
N, EXT = 128, 1.0
H_GRID = 2 * EXT / (N - 1)
T_FRAMES = 15
D = 1.0                                   # dimensionless thermal diffusivity
DT = 0.2 * H_GRID ** 2 / D                # explicit FTCS stability: r = D*dt/h^2 = 0.2 < 0.25
xs = np.linspace(-EXT, EXT, N)
XX, YY = np.meshgrid(xs, xs, indexing="ij")
CONN = np.ones((3, 3), int)               # 8-connectivity for component labelling
MAX_STEPS = 14000
# #1 FBNO physics calibration (from calibrate_from_fbno.py): make 2D fronts advance like FBNO's 1D front
CALIB = json.load(open(os.path.join(HERE, "fbno_calibration.json")))
GAP_RANGE = tuple(CALIB["gen_gap_range"])   # seed inter-edge gap  -> matches FBNO forward excursion
ST_RANGE = tuple(CALIB["gen_St_range"])     # St = |T_cold|/L      -> FBNO active-front regime (p75..p95)


def eff_radius(mask):
    return float(np.sqrt(mask.sum() * H_GRID ** 2 / np.pi))   # area-equivalent front radius


def enthalpy_to_T(Hf, L):
    """Recover temperature from enthalpy (piecewise: solid / mushy=0 / liquid)."""
    T = np.where(Hf < 0.0, Hf, np.where(Hf > L, Hf - L, 0.0))
    return T


def solid_mask_from_H(Hf, L):
    return Hf <= 0.5 * L                   # f_s >= 0.5  <=>  H <= L/2


def laplacian(T):
    lap = np.zeros_like(T)
    lap[1:-1, 1:-1] = (T[:-2, 1:-1] + T[2:, 1:-1] + T[1:-1, :-2] + T[1:-1, 2:]
                       - 4.0 * T[1:-1, 1:-1]) / H_GRID ** 2
    return lap


def signed_distance(mask):
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    return (outside - inside) * H_GRID     # >0 in liquid, <0 in solid, 0 at front


def front_polylines(mask):
    """Solidification-front polylines in [-1,1] coords (for 3D space-time rendering)."""
    polys = []
    padded = np.pad(mask.astype(float), 1)         # pad so seeds touching nothing still close
    for c in find_contours(padded, 0.5):
        c = c - 1.0
        x = np.clip(c[:, 0], 0, N - 1) / (N - 1) * 2 - 1
        y = np.clip(c[:, 1], 0, N - 1) / (N - 1) * 2 - 1
        if len(c) >= 4:
            polys.append(np.column_stack([x, y]))
    return polys


def _step(Hf, seed_cells, L, T_cold):
    T = enthalpy_to_T(Hf, L)
    Hf[1:-1, 1:-1] += DT * laplacian(T)[1:-1, 1:-1]
    Hf[0, :] = Hf[1, :]; Hf[-1, :] = Hf[-2, :]                 # Neumann (insulated) box boundary
    Hf[:, 0] = Hf[:, 1]; Hf[:, -1] = Hf[:, -2]
    Hf[seed_cells] = T_cold                                    # cold nuclei held (Dirichlet sinks)
    return Hf


def simulate(seeds, seed_r, L, T_cold):
    """Enthalpy solidification; St + seed gap are FBNO-calibrated (#1). Window spans the merge event:
    run length = 2 x (steps-to-coalescence) so the N->1 merge lands mid-window."""
    Hf = np.full((N, N), L)                          # whole box = melt at T=0 (H=L)
    seed_cells = np.zeros((N, N), bool)
    for (cx, cy) in seeds:
        seed_cells |= (XX - cx) ** 2 + (YY - cy) ** 2 <= seed_r ** 2
    Hf[seed_cells] = T_cold

    # First pass (scratch copy): run until the grains coalesce (single component).
    scratch = Hf.copy(); merge_it = None
    for it in range(MAX_STEPS):
        _step(scratch, seed_cells, L, T_cold)
        if it % 25 == 0 and label(solid_mask_from_H(scratch, L), structure=CONN)[1] == 1:
            merge_it = it
            break
    nstep = MAX_STEPS if merge_it is None else max(14, min(MAX_STEPS, int(round(2.0 * merge_it))))

    snap_at = np.unique(np.linspace(0, max(nstep, 1), T_FRAMES).astype(int))
    while len(snap_at) < T_FRAMES:                   # guard tiny runs
        snap_at = np.append(snap_at, snap_at[-1])
    snap_at = snap_at[:T_FRAMES]

    frames_u, frames_phi, frames_mask, frames_poly = [], [], [], []
    ncomp, fo, r_eq = [], [], []
    k = 0
    for it in range(snap_at[-1] + 1):
        while k < T_FRAMES and snap_at[k] == it:
            mask = solid_mask_from_H(Hf, L)
            frames_u.append(enthalpy_to_T(Hf, L).astype(np.float32))
            frames_phi.append(signed_distance(mask).astype(np.float32))
            frames_mask.append(mask)
            frames_poly.append(front_polylines(mask))
            ncomp.append(int(label(mask, structure=CONN)[1]))
            fo.append(it * DT)                       # Fourier number (D = Lc = 1)
            r_eq.append(eff_radius(mask))
            k += 1
        _step(Hf, seed_cells, L, T_cold)
    return (np.array(frames_u), np.array(frames_phi), np.array(frames_mask),
            frames_poly, np.array(ncomp), nstep, np.array(fo, np.float32), np.array(r_eq, np.float32))


def sample_seeds(rng, seed_r):
    """2-4 nuclei on a circle, nearest-neighbour EDGE gap drawn from the FBNO-calibrated range (#1):
    each approaching front then travels ~gap/2 to merge, matching FBNO's forward excursion."""
    n_seeds = int(rng.integers(2, 5))
    gap = rng.uniform(*GAP_RANGE)
    R = (gap + 2 * seed_r) / (2 * np.sin(np.pi / n_seeds))     # circle radius for that edge gap
    R = min(R, 0.60)
    base = rng.uniform(0, 2 * np.pi)
    seeds = []
    for j in range(n_seeds):
        ang = base + 2 * np.pi * j / n_seeds + rng.uniform(-0.15, 0.15)
        seeds.append((R * np.cos(ang), R * np.sin(ang)))
    return n_seeds, np.array(seeds), gap


def touches_box(sample):
    return any(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
               for mask in sample["mask"])


def pick_visual_sample(data):
    """Prefer a clean non-clipped 3/4-seed coalescence for publication figures."""
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


def preview_times(sample):
    tmax = sample["phi"].shape[0] - 1
    ms = int(sample["merge_step"])
    return [0, max(1, ms - 2), ms, min(tmax, ms + 3), tmax]


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    rng = np.random.default_rng(0)
    data = []; t0 = time.time(); s = 0; attempts = 0
    while s < n_samples and attempts < n_samples * 30:
        attempts += 1
        seed_r = rng.uniform(0.06, 0.10)
        n_seeds, seeds, gap = sample_seeds(rng, seed_r)
        St = rng.uniform(*ST_RANGE)                   # Stefan number (FBNO active-front regime, #1)
        L = rng.uniform(0.7, 1.3)                     # latent heat
        T_cold = -St * L                              # cold-nucleus temperature so that |T_cold|/L = St
        u, phi, mask, poly, ncomp, nstep, fo, r_eq = simulate(seeds, seed_r, L, T_cold)
        if ncomp[0] < 2 or ncomp[-1] != 1:            # need a real N->1 coalescence
            continue
        merge_step = int(np.argmax(ncomp == 1))
        data.append(dict(
            phi=phi, u=u, mask=mask,
            boundary_components=np.array(poly, dtype=object),
            seeds=seeds.astype(np.float32),
            params=np.array([n_seeds, L, T_cold, seed_r, D, gap], np.float32),
            St=np.float32(St), fo=fo, r_eq=r_eq,
            n_components=ncomp, merge_step=merge_step))
        if s < 12 or s % 20 == 0:
            print(f"  sample {s}: seeds={n_seeds} comps {ncomp[0]}->1 @t={merge_step} "
                  f"St={St:.2f} L={L:.2f} Tc={T_cold:.2f} gap={gap:.2f} ({nstep} steps)", flush=True)
        s += 1
    seed_counts = {int(n): sum(int(x["params"][0]) == n for x in data) for n in (2, 3, 4)}
    sts = np.array([float(x["St"]) for x in data])
    travel = np.array([float(x["params"][5]) / 2 for x in data])   # per-front merge travel ~ gap/2
    np.save(os.path.join(HERE, "stefan.npy"), np.array(data, dtype=object), allow_pickle=True)
    json.dump({"n_samples": len(data), "grid": N, "frames": T_FRAMES, "steady": False,
               "topology": "N->1 grain coalescence", "seed_count_distribution": seed_counts,
               "attempts": attempts,
               "calibration": {"source": CALIB["source"], "St_range": list(ST_RANGE),
                               "gap_range": list(GAP_RANGE),
                               "St_realized": [float(sts.min()), float(sts.max())],
                               "front_travel_realized": [float(travel.min()), float(travel.max())],
                               "fbno_excursion_max": CALIB["front_excursion_fwd"]["max"]}},
              open(os.path.join(HERE, "stefan_summary.json"), "w"), indent=2)
    print(f"\n{len(data)}/{n_samples} solidification samples ({attempts} attempts). "
          f"seeds {seed_counts}. St in [{sts.min():.2f},{sts.max():.2f}], per-front merge travel "
          f"[{travel.min():.2f},{travel.max():.2f}] (FBNO excursion p55-p90 {CALIB['front_excursion_fwd']['p50']:.2f}-{CALIB['front_excursion_fwd']['p90']:.2f}). "
          f"{time.time()-t0:.1f}s")

    # 2D preview: one clean sample's solidification + merge over 5 times.
    d = pick_visual_sample(data)
    times = preview_times(d)
    norm = Normalize(float(d["params"][2]), 0.0)
    fig, axs = plt.subplots(1, 5, figsize=(17.5, 3.9))
    im = None
    for ax, t in zip(axs, times):
        im = ax.imshow(d["u"][t].T, origin="lower", extent=(-1, 1, -1, 1),
                       cmap="Blues_r", norm=norm, interpolation="bilinear")
        ax.contour(xs, xs, d["phi"][t].T, levels=[0], colors="black", linewidths=3.0)
        ax.contour(xs, xs, d["phi"][t].T, levels=[0], colors="crimson", linewidths=1.4)
        ax.set_title(f"t={t}  ({int(d['n_components'][t])} comp.)", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(left=0.02, right=0.91, bottom=0.08, top=0.78, wspace=0.08)
    cax = fig.add_axes([0.935, 0.18, 0.012, 0.54])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("temperature $T$", fontsize=10)
    fig.suptitle("Stefan solidification — temperature $T$ + front Γ = {T=0} (grains grow & coalesce N→1)")
    fig.savefig(os.path.join(HERE, "stefan_preview.png"), dpi=140)
    print("preview -> stefan_preview.png")


if __name__ == "__main__":
    main()
