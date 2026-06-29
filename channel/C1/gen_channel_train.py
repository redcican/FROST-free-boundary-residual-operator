"""
Stage 1 — expanded channel training set for the FROST operator experiments.

Rasterizes ~K designs per inlet velocity from the NTO CFD CSVs (1074 available) onto a downsampled
grid, storing BOTH representations of the baffle so the φ-vs-γ study (Stage 2) is possible:
  gamma  — fluid fraction in [0,1]   (NTO-style raw density input)
  phi    — signed distance to the solid-fluid interface {γ=0.5}  (FROST level-set input; <0 in solid)
plus the CFD outputs C (concentration), P (normalized pressure), the solid mask, inlet velocity v, and
the number of solid baffles (topology).

Output: channel_train.npy.  Run:  python gen_channel_train.py [per_velocity]
"""
import os, sys, json, glob, re, time
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label

HERE = os.path.dirname(os.path.abspath(__file__))
NTO = os.path.join(HERE, "..", "..", "..", "neural-topology-optimization", "data", "dataset")
NX, NY = 201, 101
DS = 2                                                   # downsample 201x101 -> 100x50 (FFT-friendly)
H = 1e-4 * DS                                            # grid spacing after downsample (m)
CONN = np.ones((3, 3), int)
VELS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]


def rasterize(path):
    df = pd.read_csv(path, skipinitialspace=True); df.columns = [c.strip() for c in df.columns]
    ix = np.round(df["x-coordinate"].values / 1e-4).astype(int).clip(0, NX - 1)
    iy = np.round(df["y-coordinate"].values / 1e-4).astype(int).clip(0, NY - 1)
    def grid(col, fill):
        a = np.full((NX, NY), fill, np.float32); a[ix, iy] = df[col].values; return a
    g = grid("udm-3", 1.0)[: NX - 1 : DS, : NY - 1 : DS]          # (100,50)
    C = grid("uds-0-scalar", 0.0)[: NX - 1 : DS, : NY - 1 : DS]
    P = (grid("total-pressure", 0.0) / 2000.0 + 0.26)[: NX - 1 : DS, : NY - 1 : DS]
    return g, C, P


def signed_distance(solid):
    return ((distance_transform_edt(~solid) - distance_transform_edt(solid)) * H).astype(np.float32)


def main():
    per_v = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    by_v = {}
    for f in sorted(glob.glob(os.path.join(NTO, "*.csv"))):
        m = re.search(r"v([\d.]+)N", os.path.basename(f))
        if m and m.group(1) in VELS:
            by_v.setdefault(m.group(1), []).append(f)

    data = []; t0 = time.time(); attempts = 0
    for v in VELS:
        kept = 0
        for f in by_v[v]:
            if kept >= per_v:
                break
            attempts += 1
            try:
                g, C, P = rasterize(f)
            except Exception:
                continue
            solid = g < 0.5
            if not (0.02 < solid.mean() < 0.40):
                continue
            data.append(dict(gamma=g.astype(np.float32), u=C.astype(np.float32), P=P.astype(np.float32),
                             phi=signed_distance(solid), mask=solid,
                             params=np.array([float(v)], np.float32),
                             n_components=int(label(solid, structure=CONN)[1]),
                             source=os.path.basename(f)))
            kept += 1
        print(f"  v={v}: kept {kept}", flush=True)

    from collections import Counter
    comps = dict(sorted(Counter(d["n_components"] for d in data).items()))
    vels = dict(sorted(Counter(float(d["params"][0]) for d in data).items()))
    H2, W2 = data[0]["u"].shape
    np.save(os.path.join(HERE, "channel_train.npy"), np.array(data, dtype=object), allow_pickle=True)
    json.dump({"n_samples": len(data), "grid": [H2, W2], "downsample": DS,
               "topology_component_counts": {int(k): int(v) for k, v in comps.items()},
               "velocity_counts": {float(k): int(v) for k, v in vels.items()}, "attempts": attempts},
              open(os.path.join(HERE, "channel_train_summary.json"), "w"), indent=2)
    print(f"\n{len(data)} designs ({attempts} read), grid {H2}x{W2}. topology {comps}. {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
