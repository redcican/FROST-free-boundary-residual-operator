"""
FROST benchmark 'obstacle' -- the steady obstacle problem (a well-posed free-boundary equilibrium).

Purpose: the obstacle problem is solved by a CONSTRAINED FIXED POINT  u <- max(chi, mean_neighbours(u))
with u=0 on the boundary -- i.e. exactly the kind of equilibrium FROST's DEQ must solve, but
steady and monotone (guaranteed to converge). The reproductions showed the equilibrium solve is the
hard part of FROST; this benchmark is the clean testbed to develop/debug it before the
time-dependent merge.

Free boundary: the CONTACT SET boundary  partial{u = chi}.  With an obstacle made of two Gaussian
bumps, the contact set has two components when the bumps are far / low and one when they are close /
high -- a steady topology family (the kind a single diffeomorphism cannot track).

Each sample stores the field u, obstacle chi, contact mask, level set phi (signed distance to the
free boundary), params, and the contact-component count. STEADY: one equilibrium per sample (no time).

Output: obstacle.npy (+ summary, preview).  Run (single-threaded BLAS):  python gen_obstacle.py [n]
"""
import os, sys, json, time
import numpy as np
from scipy.ndimage import distance_transform_edt, label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
N, EXT = 128, 1.0
H = 2 * EXT / (N - 1)
xs = np.linspace(-EXT, EXT, N)
XX, YY = np.meshgrid(xs, xs, indexing="ij")
CONN = np.ones((3, 3), int)
II, JJ = np.indices((N, N))
INTERIOR = (II > 0) & (II < N - 1) & (JJ > 0) & (JJ < N - 1)
RED = INTERIOR & ((II + JJ) % 2 == 0)
BLACK = INTERIOR & ((II + JJ) % 2 == 1)


def obstacle(c1, c2, h1, h2, w, base):
    g1 = h1 * np.exp(-((XX - c1[0]) ** 2 + (YY - c1[1]) ** 2) / (2 * w ** 2))
    g2 = h2 * np.exp(-((XX - c2[0]) ** 2 + (YY - c2[1]) ** 2) / (2 * w ** 2))
    return g1 + g2 - base                                  # >0 only near bump cores -> contact there


def solve_obstacle(chi, omega=1.7, sweeps=800, tol=1e-7):
    """Projected red-black SOR: u <- max(chi, (1-w)u + w*mean_nbr), u=0 on the boundary."""
    u = np.maximum(np.zeros((N, N)), chi)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    for it in range(sweeps):
        u_old = u
        for color in (RED, BLACK):
            nbr = np.zeros((N, N))
            nbr[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
            gs = (1 - omega) * u + omega * nbr
            proj = np.maximum(chi, gs)
            u = np.where(color, proj, u)
        if it % 50 == 0 and np.max(np.abs(u - u_old)) < tol:
            break
    return u


def signed_distance(mask):
    return (distance_transform_edt(~mask) - distance_transform_edt(mask)) * H


def make_sample(c1, c2, h1, h2, w, base, contact_tol=2e-3):
    chi = obstacle(c1, c2, h1, h2, w, base)
    u = solve_obstacle(chi)
    contact = (u - chi) < contact_tol                      # free boundary = partial of this set
    contact &= (chi > -base * 0.0 + 1e-6) | (chi > 0)      # only where obstacle is active (near bumps)
    contact = contact & (chi > 0.0)                        # contact only where obstacle pokes above 0
    n = int(label(contact, structure=CONN)[1])
    phi = signed_distance(contact)
    return u.astype(np.float32), chi.astype(np.float32), contact, phi.astype(np.float32), n


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    rng = np.random.default_rng(0)
    data = []; t0 = time.time(); s = 0; attempts = 0
    while s < n_samples and attempts < n_samples * 40:
        attempts += 1
        d = rng.uniform(0.16, 0.46)                        # bump half-separation
        h1, h2 = rng.uniform(0.45, 0.75, 2)                # bump heights
        w = rng.uniform(0.16, 0.24)                        # bump width
        base = rng.uniform(0.12, 0.22)                     # contact threshold
        u, chi, contact, phi, n = make_sample((-d, 0.0), (d, 0.0), h1, h2, w, base)
        if n not in (1, 2) or contact.sum() < 25:          # need a real, well-resolved contact set
            continue
        data.append(dict(u=u, chi=chi, contact=contact, phi=phi,
                         params=np.array([-d, d, h1, h2, w, base], np.float32), n_components=n))
        if s < 12 or s % 20 == 0:
            print(f"  sample {s}: d={d:.2f} h=({h1:.2f},{h2:.2f}) contact_components={n}", flush=True)
        s += 1
    counts = {1: sum(x["n_components"] == 1 for x in data), 2: sum(x["n_components"] == 2 for x in data)}
    np.save(os.path.join(HERE, "obstacle.npy"), np.array(data, dtype=object), allow_pickle=True)
    json.dump({"n_samples": len(data), "grid": N, "steady": True,
               "contact_component_counts": counts, "attempts": attempts},
              open(os.path.join(HERE, "obstacle_summary.json"), "w"), indent=2)
    print(f"\n{len(data)}/{n_samples} obstacle solutions ({attempts} attempts). "
          f"contact components: {counts}. {time.time()-t0:.1f}s")

    # 2D preview: same representative samples and shared color scale as the 3D view.
    two = [x for x in data if x["n_components"] == 2]
    one = [x for x in data if x["n_components"] == 1]
    show = (two[:4] + one[:2])[:6] or data[:6]
    norm = Normalize(0.0, max(float(d0["u"].max()) for d0 in data))
    fig, axs = plt.subplots(2, 3, figsize=(14.6, 8.2))
    axs = np.asarray(axs).ravel()
    im = None
    for ax, d0 in zip(axs, show):
        im = ax.imshow(d0["u"].T, origin="lower", extent=(-1, 1, -1, 1),
                       cmap="YlGnBu", norm=norm, interpolation="bilinear")
        ax.contour(xs, xs, d0["phi"].T, levels=[0], colors="black", linewidths=3.2)
        ax.contour(xs, xs, d0["phi"].T, levels=[0], colors="crimson", linewidths=1.7)
        ax.set_title(f"{d0['n_components']} contact region(s)", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axs[len(show):]:
        ax.set_axis_off()
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.05, top=0.88, wspace=0.06, hspace=0.20)
    cax = fig.add_axes([0.91, 0.16, 0.018, 0.66])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("membrane $u(x,y)$", fontsize=10)
    fig.suptitle("Obstacle problem — membrane field $u$ with free boundary Γ = ∂{u=χ} (crimson)",
                 fontsize=12)
    fig.savefig(os.path.join(HERE, "obstacle_preview.png"), dpi=140)
    print("preview -> obstacle_preview.png")


if __name__ == "__main__":
    main()
