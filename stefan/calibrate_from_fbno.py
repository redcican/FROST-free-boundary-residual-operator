"""
#1 Calibration: extract FBNO's 1D Stefan physics from stefan_data.npy and turn it into the parameter
ranges for our 2D enthalpy generator, so the 2D fronts advance like FBNO's measured 1D front.

FBNO's data is 1D (temperature_field 101x5001, interface path s(t)).  We CANNOT use the fields as 2D
samples, but we CAN reuse the physics: the front-excursion distribution, the normalized front-speed
envelope, and an effective Stefan number.  We map both problems into a shared dimensionless frame:
    length = fraction of the characteristic domain length   (FBNO: s/L_max in [0,1]; ours: r/1)
    time   = normalized over the run, tau in [0,1]
In that frame the front behaviour is directly comparable; we match the FORWARD excursion (so our
grains travel a FBNO-like distance before they merge) -> the normalized speed then matches by
construction (dr/dtau = excursion over the run).

Outputs (consumed by gen_stefan.py and validate_vs_fbno_1d.py; the 2.3 GB file is never reloaded):
  fbno_calibration.json  - scalar targets + chosen generator ranges
  fbno_calibration.npz   - the 300 interface curves resampled to T_FRAMES (for the 1D validation)

Run single-threaded:  python calibrate_from_fbno.py
"""
import os, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FBNO = os.path.join(HERE, "..", "..", "FBNO", "data", "stefan_data", "stefan_data.npy")
T_FRAMES = 15

# FBNO normalization constants (from FEM_stefan_generator.py / stefan_run.py)
ALPHA_PHYS = 1.15e-6 * 2 * 15        # adjusted thermal diffusivity (m^2/s)
L_MAX = 2.5                          # characteristic length (m)
TIME_MAX = 3600.0                    # characteristic time (s)
ALPHA_NORM = ALPHA_PHYS / L_MAX ** 2 * TIME_MAX   # Fourier number over the full normalized run


def stefan_number_from_lambda(lam):
    """One-phase Stefan similarity  s = 2 lam sqrt(alpha t)  =>  lam e^{lam^2} erf(lam) = St/sqrt(pi)."""
    return math.sqrt(math.pi) * lam * math.exp(lam ** 2) * math.erf(lam)


def main():
    d = np.load(FBNO, allow_pickle=True)
    n = len(d)
    s_all = np.stack([np.asarray(r["interface_position"], np.float32) for r in d])   # (n, 5001)
    del d
    nt = s_all.shape[1]

    # resample each interface path to T_FRAMES normalized times tau in [0,1]
    src = np.linspace(0, 1, nt)
    tau = np.linspace(0, 1, T_FRAMES)
    s15 = np.stack([np.interp(tau, src, s_all[i]) for i in range(n)])                # (n, 15)
    fwd = s15 - s15[:, :1]                                                           # forward excursion
    excursion_fwd = fwd.max(axis=1)                                                  # peak forward travel
    speed_norm = np.diff(s15, axis=1) * (T_FRAMES - 1)                               # dr/dtau over the run

    pe = lambda a, q: float(np.percentile(a, q))

    # map several forward-excursion quantiles to an effective Stefan number via the similarity law
    # s = 2 lam sqrt(alpha t).  FBNO's median run is near-static (St~0.18); the ACTIVE fronts that
    # actually move + would coalesce live in the upper excursion quantiles -> we calibrate to those.
    def st_from_excursion(exc):
        lam = exc / (2.0 * math.sqrt(ALPHA_NORM))
        return lam, stefan_number_from_lambda(lam)

    lam_med, st_median = st_from_excursion(pe(excursion_fwd, 50))
    _, st_p75 = st_from_excursion(pe(excursion_fwd, 75))
    _, st_p95 = st_from_excursion(pe(excursion_fwd, 95))

    # generator ranges
    gap_lo = 2.0 * pe(excursion_fwd, 55)
    gap_hi = 2.0 * pe(excursion_fwd, 90)
    gap_lo, gap_hi = max(0.24, gap_lo), min(0.60, max(gap_hi, gap_lo + 0.10))
    st_lo = max(0.5, round(st_p75, 2))                 # active-front regime (FBNO p75..p95 excursion)
    st_hi = min(1.6, max(st_p95, st_lo + 0.4))

    calib = {
        "source": "FBNO/data/stefan_data/stefan_data.npy (300 x 1D Stefan)",
        "n_fbno": n,
        "alpha_norm_fourier": ALPHA_NORM,
        "front_excursion_fwd": {"p10": pe(excursion_fwd, 10), "p50": pe(excursion_fwd, 50),
                                "p90": pe(excursion_fwd, 90), "max": float(excursion_fwd.max())},
        "front_speed_norm": {"p5": pe(speed_norm, 5), "p50": pe(speed_norm, 50),
                             "p95": pe(speed_norm, 95)},
        "stefan_number": {"median": st_median, "p75": st_p75, "p95": st_p95,
                          "lambda_median": lam_med},
        # --- knobs consumed by gen_stefan.py ---
        "gen_gap_range": [gap_lo, gap_hi],          # seed inter-edge gap (-> FBNO-like front travel)
        "gen_St_range": [st_lo, st_hi],             # St = |T_cold|/L, FBNO active-front regime (p75..p95)
        "note": ("1D fields are NOT used as 2D samples; we match the dimensionless forward-excursion "
                 "and normalized-speed envelope. Absolute Fourier number differs (different time "
                 "normalizations); the per-run dimensionless front behaviour is what is matched."),
    }
    json.dump(calib, open(os.path.join(HERE, "fbno_calibration.json"), "w"), indent=2)
    np.savez(os.path.join(HERE, "fbno_calibration.npz"),
             tau=tau, forward=fwd.astype(np.float32),
             env_p5=np.percentile(fwd, 5, axis=0), env_p50=np.percentile(fwd, 50, axis=0),
             env_p95=np.percentile(fwd, 95, axis=0))

    print(f"FBNO 1D Stefan: {n} samples")
    print(f"  Fourier number over run (alpha_norm) = {ALPHA_NORM:.4f}")
    print(f"  forward excursion  p10/p50/p90/max = "
          f"{calib['front_excursion_fwd']['p10']:.3f}/{calib['front_excursion_fwd']['p50']:.3f}/"
          f"{calib['front_excursion_fwd']['p90']:.3f}/{calib['front_excursion_fwd']['max']:.3f}")
    print(f"  normalized speed   p5/p50/p95     = "
          f"{calib['front_speed_norm']['p5']:.3f}/{calib['front_speed_norm']['p50']:.3f}/"
          f"{calib['front_speed_norm']['p95']:.3f}")
    print(f"  effective Stefan number  median/p75/p95 = {st_median:.3f}/{st_p75:.3f}/{st_p95:.3f}")
    print(f"  => gen gap range  = [{gap_lo:.3f}, {gap_hi:.3f}]")
    print(f"  => gen St range   = [{st_lo:.3f}, {st_hi:.3f}]")
    print("saved fbno_calibration.json + fbno_calibration.npz")


if __name__ == "__main__":
    main()
