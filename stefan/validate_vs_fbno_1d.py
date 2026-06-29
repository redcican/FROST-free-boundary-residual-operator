"""
#2 Validation: our 2D enthalpy Stefan reduces to FBNO's 1D Stefan in the no-coalescence limit.

A single isolated seed grows radially -- a one-phase Stefan front with no topology change, i.e. the
degenerate slice that should behave like FBNO's 1D front s(t). We run it at a few (FBNO-calibrated)
Stefan numbers and compare its forward radial travel r(tau)-r(0) against the 300 FBNO interface
paths, in the shared dimensionless frame (length = fraction of characteristic length, time = tau in
[0,1] over the run). FBNO's fronts oscillate (cooling) so their band straddles zero; our monotonic
solidifying front should track the UPPER (active) part of that envelope.

Reads fbno_calibration.npz (the 300 resampled curves) so the 2.3 GB file is never reloaded.
Run single-threaded:  python validate_vs_fbno_1d.py  -> stefan_vs_fbno_1d.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gen_stefan as gs                       # reuse the exact 2D enthalpy physics

HERE = os.path.dirname(os.path.abspath(__file__))
T_FRAMES = gs.T_FRAMES
N, DT = gs.N, gs.DT


def run_single_seed(St, travel_target, L=1.0, seed_r=0.08, max_steps=40000):
    """One central seed; sample r_eq over a run that reaches travel_target (no merge, no topology change)."""
    T_cold = -St * L
    Hf = np.full((N, N), L)
    seed = (gs.XX ** 2 + gs.YY ** 2) <= seed_r ** 2
    Hf[seed] = T_cold
    r0 = gs.eff_radius(gs.solid_mask_from_H(Hf, L))

    scratch = Hf.copy(); nstep = max_steps
    for it in range(max_steps):
        gs._step(scratch, seed, L, T_cold)
        if it % 25 == 0 and gs.eff_radius(gs.solid_mask_from_H(scratch, L)) - r0 >= travel_target:
            nstep = it
            break

    snap = np.unique(np.linspace(0, max(nstep, 1), T_FRAMES).astype(int))
    while len(snap) < T_FRAMES:
        snap = np.append(snap, snap[-1])
    snap = snap[:T_FRAMES]
    r_eq, fo = [], []; k = 0
    for it in range(snap[-1] + 1):
        while k < T_FRAMES and snap[k] == it:
            r_eq.append(gs.eff_radius(gs.solid_mask_from_H(Hf, L))); fo.append(it * DT); k += 1
        gs._step(Hf, seed, L, T_cold)
    r_eq = np.array(r_eq)
    return np.array(fo), r_eq - r_eq[0]                  # Fourier number, forward radial travel


def main():
    cal = np.load(os.path.join(HERE, "fbno_calibration.npz"))
    tau, fwd = cal["tau"], cal["forward"]               # (15,), (300,15)
    p5, p50, p95 = cal["env_p5"], cal["env_p50"], cal["env_p95"]
    target = float(np.percentile(fwd.max(axis=1), 95))  # run our seed to FBNO's p95 peak excursion

    st_lo, st_hi = gs.ST_RANGE
    st_list = [st_lo, 0.5 * (st_lo + st_hi), st_hi]
    runs = [(st, *run_single_seed(st, target)) for st in st_list]

    # FBNO normalized-speed envelope (per step) and our mid-St speed
    fbno_spd = np.diff(fwd, axis=1) * (T_FRAMES - 1)
    s_p5, s_p95 = np.percentile(fbno_spd, 5, axis=0), np.percentile(fbno_spd, 95, axis=0)
    mid_tr = runs[1][2]
    mid_spd = np.diff(mid_tr) * (T_FRAMES - 1)
    spd_in = float(np.mean((mid_spd >= s_p5 - 1e-6) & (mid_spd <= s_p95 + 1e-6)))

    # Stefan similarity check: r - r0 should be linear in sqrt(Fo)
    def r2_sqrt_fo(fo, tr):
        x = np.sqrt(fo); A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, tr, rcond=None)
        pred = A @ coef
        ss = 1 - np.sum((tr - pred) ** 2) / max(np.sum((tr - tr.mean()) ** 2), 1e-12)
        return float(ss), float(coef[0])
    r2_mid, _ = r2_sqrt_fo(*runs[1][1:])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.3))
    # Panel A: forward travel vs normalized time, over FBNO band
    axA.fill_between(tau, p5, p95, color="0.82", label="FBNO 1D band (p5–p95)")
    for c in fwd[::12]:
        axA.plot(tau, c, color="0.65", lw=0.4, alpha=0.5)
    axA.plot(tau, p50, color="0.35", lw=2.0, ls="--", label="FBNO median")
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for (st, _fo, tr), col in zip(runs, colors):
        axA.plot(tau, tr, color=col, lw=2.4, marker="o", ms=4, label=f"FROST single seed, St={st:.2f}")
    axA.axhline(0, color="k", lw=0.6, alpha=0.4)
    axA.set_xlabel("normalized time  τ = t / t_run")
    axA.set_ylabel("forward front travel  r(τ) − r(0)")
    axA.set_title(f"Forward excursion — reaches FBNO p95 peak ({target:.3f})\n"
                  "FROST = clean √t similarity → rides the active (upper) envelope", fontsize=10)
    axA.legend(fontsize=8, loc="upper left"); axA.grid(alpha=0.25)
    # Panel B: Stefan similarity — r-r0 vs sqrt(Fourier number)
    for (st, fo, tr), col in zip(runs, colors):
        axB.plot(np.sqrt(fo), tr, color=col, lw=2.2, marker="o", ms=4,
                 label=f"St={st:.2f}  (Fo_end={fo[-1]:.3f})")
    axB.set_xlabel(r"$\sqrt{\mathrm{Fo}}$   (Fo = D·t / L_c²)")
    axB.set_ylabel("front travel  r − r(0)")
    axB.set_title(f"Stefan similarity check: r − r₀ ∝ √Fo  (R² = {r2_mid:.4f})\n"
                  "higher St → smaller Fo to same travel (faster front)", fontsize=10)
    axB.legend(fontsize=8, loc="upper left"); axB.grid(alpha=0.25)
    fig.suptitle("#2  FROST 2D single-seed Stefan front  vs  FBNO 1D interface paths "
                 "(degenerate no-coalescence slice)", fontsize=12)
    out = os.path.join(HERE, "stefan_vs_fbno_1d.png")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"FBNO p95 peak excursion target = {target:.3f}")
    for st, fo, tr in runs:
        print(f"  St={st:.2f}: peak travel {tr.max():.3f}, Fo_end {fo[-1]:.3f}")
    print(f"Stefan similarity R² (r-r0 vs √Fo, mid-St) = {r2_mid:.4f}")
    print(f"mid-St normalized speed inside FBNO [p5,p95] band: {spd_in*100:.0f}% of frames")
    print("saved", out)


if __name__ == "__main__":
    main()
