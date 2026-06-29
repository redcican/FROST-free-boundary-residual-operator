"""
FROST benchmark 'channel' -- turbulent-channel design with a solid-fluid free boundary.

Adapted from the Neural-Topology-Optimization data (../../neural-topology-optimization/data/dataset):
1074 CFD designs of a 2D channel (0.02 x 0.01 m) with a solid baffle, each storing on a 201x101 grid
the fluid fraction gamma in [0,1] (gamma=0 solid, gamma=1 fluid), the concentration C, and pressure.

FROST reframing ("modify the NTO data to our problem"): the SOLID-FLUID INTERFACE {gamma=0.5} is a
FREE BOUNDARY, and we represent it as a level set phi (signed distance, negative inside solid) paired
with the transported field u = C. Crucially the baffle TOPOLOGY varies across designs (1, 2, 3+
disconnected solid pieces) -- the steady, real-CFD analogue of the obstacle benchmark's contact-set
topology, and exactly the design space FROST's topology-optimization loop (C2) explores.

Each sample stores phi/u(=C)/P/gamma/mask (201,101), params[v], n_components.
Output: channel.npy (+ summary, preview).  Run:  python gen_channel.py [n_samples]
"""
import os, sys, json, time, glob, re
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NTO = os.path.join(HERE, "..", "..", "neural-topology-optimization", "data", "dataset")
NX, NY = 201, 101
H = 1e-4                                              # grid spacing (m), square cells
EXTENT = (0.0, 0.02, 0.0, 0.01)                      # physical domain (m)
CONN = np.ones((3, 3), int)
VELS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]


def rasterize(path):
    """CFD node CSV -> regular (201,101) grids for gamma, C, normalized pressure P."""
    df = pd.read_csv(path, skipinitialspace=True); df.columns = [c.strip() for c in df.columns]
    ix = np.round(df["x-coordinate"].values / H).astype(int).clip(0, NX - 1)
    iy = np.round(df["y-coordinate"].values / H).astype(int).clip(0, NY - 1)
    def grid(col, fill):
        a = np.full((NX, NY), fill, np.float32); a[ix, iy] = df[col].values; return a
    gamma = grid("udm-3", 1.0)                        # fluid fraction (1 fluid, 0 solid)
    C = grid("uds-0-scalar", 0.0)                     # concentration
    P = grid("total-pressure", 0.0) / 2000.0 + 0.26   # normalized pressure (paper's convention)
    return gamma, C, P


def signed_distance(solid):
    return (distance_transform_edt(~solid) - distance_transform_edt(solid)) * H   # <0 inside solid


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    by_v = {}
    for f in sorted(glob.glob(os.path.join(NTO, "*.csv"))):
        m = re.search(r"v([\d.]+)N", os.path.basename(f))
        if m and m.group(1) in VELS:
            by_v.setdefault(m.group(1), []).append(f)

    per_v = max(1, n_target // len(VELS) + 1)
    data = []; t0 = time.time(); attempts = 0
    for v in VELS:
        kept = 0
        for f in by_v[v]:
            if kept >= per_v or len(data) >= n_target:
                break
            attempts += 1
            gamma, C, P = rasterize(f)
            solid = gamma < 0.5
            frac = float(solid.mean())
            if not (0.02 < frac < 0.40):              # skip degenerate (empty / all-solid) designs
                continue
            phi = signed_distance(solid).astype(np.float32)
            n = int(label(solid, structure=CONN)[1])
            data.append(dict(phi=phi, u=C.astype(np.float32), P=P.astype(np.float32),
                             gamma=gamma.astype(np.float32), mask=solid,
                             params=np.array([float(v)], np.float32), n_components=n,
                             source=os.path.basename(f)))
            kept += 1
        if len(data) >= n_target:
            break

    from collections import Counter
    comps = dict(sorted(Counter(d["n_components"] for d in data).items()))
    vels = dict(sorted(Counter(float(d["params"][0]) for d in data).items()))
    np.save(os.path.join(HERE, "channel.npy"), np.array(data, dtype=object), allow_pickle=True)
    json.dump({"n_samples": len(data), "grid": [NX, NY], "extent_m": EXTENT, "steady": True,
               "free_boundary": "solid-fluid interface {gamma=0.5}",
               "topology_component_counts": {int(k): int(v) for k, v in comps.items()},
               "velocity_counts": {float(k): int(v) for k, v in vels.items()}, "attempts": attempts},
              open(os.path.join(HERE, "channel_summary.json"), "w"), indent=2)
    print(f"{len(data)}/{n_target} channel designs ({attempts} read). solid-components {comps}. "
          f"velocities {vels}. {time.time()-t0:.1f}s")

    # 2D preview: four DISTINCT designs spanning topology (concentration C + baffle Γ)
    picks, seen = [], set()
    for nc in (1, 2, 3):
        d = next((d for d in data if d["n_components"] == nc and d["source"] not in seen), None)
        if d:
            picks.append(d); seen.add(d["source"])
    for d in sorted(data, key=lambda d: -float(d["params"][0])):       # + a high-velocity design
        if d["source"] not in seen:
            picks.append(d); seen.add(d["source"]); break
    picks = picks[:4]
    xs = np.linspace(EXTENT[0], EXTENT[1], NX); ys = np.linspace(EXTENT[2], EXTENT[3], NY)
    fig, axs = plt.subplots(len(picks), 1, figsize=(7, 2.0 * len(picks)))
    if len(picks) == 1:
        axs = [axs]
    for ax, d in zip(axs, picks):
        ax.imshow(d["u"].T, origin="lower", extent=EXTENT, cmap="jet", aspect="equal", vmin=0, vmax=1)
        ax.contour(xs, ys, d["phi"].T, levels=[0], colors="w", linewidths=1.5)   # free boundary Γ
        ax.contourf(xs, ys, d["mask"].T.astype(float), levels=[0.5, 1.5], colors=["white"])
        ax.set_title(f"v={d['params'][0]:.1f} m/s — {d['n_components']} solid baffle(s)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("FROST channel — concentration $C$ + solid-fluid free boundary Γ = {γ=0.5}")
    plt.tight_layout(); fig.savefig(os.path.join(HERE, "channel_preview.png"), dpi=130, bbox_inches="tight")
    print("preview -> channel_preview.png")


if __name__ == "__main__":
    main()
