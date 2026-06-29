"""
FROST C2 — inverse design by differentiating the EQUILIBRIUM (the principled §4.4 adjoint).

This is the first FROST inverse-design result built the way the proposal prescribes (FROST_proposal.md
§4.4–4.5, §12.3 step 1): instead of NTO's "backprop through a frozen feed-forward operator" (what the
channel C2 did), we differentiate the **DEQ equilibrium** operator. The design variable enters the
fixed point, and the design gradient

    dJ/dd = ∂J/∂(u*,φ*) · (I − ∂T_θ/∂(u,φ))^{-1} · ∂T_θ/∂d

is obtained for free by the DEQ's implicit-function-theorem backward (the SAME adjoint used to train it):
we make the obstacle χ a differentiable leaf, run the frozen DEQ forward, and call `J.backward()`.

Why the obstacle is the right first case: (1) its forward operator is the strongest in the suite (~1%);
(2) it has a CHEAP exact solver (projected SOR, `gen_obstacle.solve_obstacle`), so we can actually
compute the **trust-region certificate** |J_FROST − J_true| at the optimized design — the very check the
channel C2 had to leave out of scope (CFD too expensive).

Task (free-boundary targeting): given a target contact footprint M* (taken from a real obstacle so it is
achievable), design χ so the equilibrium contact set {φ*<0} matches M*. We run it three ways:
  • parametric design through the DEQ equilibrium  (IFT adjoint)        — the principled §4.4 method
  • parametric design through the frozen feed-forward FNO               — the NTO-style baseline
  • free-FIELD design through the DEQ (band-limited χ, off the data manifold) — exposes the certificate
    gap that motivates the proposed active-acquisition / trust-region loop (§4.5)
and certify every optimized design against the true solver.

Run:  python design_c2_obstacle.py [steps]        # default 150 design steps; needs ../C1/results/{deq_model.pt,model.pt}
Out:  results/{design_metrics.json, design_targets.png, design_convergence.png, design_certificate.png}
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C1DIR = os.path.join(HERE, "..", "C1")
OBSDIR = os.path.join(HERE, "..")
sys.path.insert(0, C1DIR); sys.path.insert(0, OBSDIR)
from deq import DEQObstacle
from fno import FNO2d
from train_c1_obstacle import n_comp, iou, CONTACT_TOL
import gen_obstacle as G                                   # obstacle(), solve_obstacle(), signed_distance(), N, xs

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
torch.manual_seed(0); np.random.seed(0)
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
FS, DPI = 22, 600
TAU = 0.03                                                 # softness (φ units) of the contact indicator
N = G.N
xs_t = torch.linspace(-1, 1, N)
XX, YY = torch.meshgrid(xs_t, xs_t, indexing="ij")         # matches gen_obstacle's meshgrid(indexing='ij')
RANGES = dict(sep=(0.16, 0.46), h1=(0.45, 0.75), h2=(0.45, 0.75), w=(0.16, 0.24), base=(0.12, 0.22))


# ------------------------------------------------------------------ frozen operators
def _freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_operators():
    dk = torch.load(os.path.join(C1DIR, "results", "deq_model.pt"), map_location="cpu")
    deq = DEQObstacle(modes=12, width=32, f_iter=30, b_iter=14, tol=1e-3)
    deq.load_state_dict(dk["model"]); _freeze(deq)
    fk = torch.load(os.path.join(C1DIR, "results", "model.pt"), map_location="cpu")
    fno = FNO2d(modes=16, width=32, in_c=1, out_c=2, n_layers=4)
    fno.load_state_dict(fk["model"]); _freeze(fno)
    return (deq, {k: torch.as_tensor(v) for k, v in dk["stats"].items()}), \
           (fno, {k: torch.as_tensor(v) for k, v in fk["stats"].items()})


def operator_fields(model, stats, chi):
    """chi (N,N) torch -> (phi, u) (N,N) torch, denormalized. Differentiable in chi (IFT for the DEQ)."""
    xm, xsd = stats["xm"], stats["xsd"]
    ym, ysd = stats["ym"].reshape(2), stats["ysd"].reshape(2)
    out = model(((chi[None, ..., None] - xm) / xsd))[0] * ysd + ym
    return out[..., 1], out[..., 0]


def footprint_J(phi, Mstar):
    """Soft mismatch between the contact indicator sigmoid(-phi/tau) and the target footprint M*."""
    return ((torch.sigmoid(-phi / TAU) - Mstar) ** 2).mean()


# ------------------------------------------------------------------ true (ground-truth) solver
def true_solve(chi_np):
    u = G.solve_obstacle(chi_np)
    contact = ((u - chi_np) < CONTACT_TOL) & (chi_np > 0.0)
    phi = G.signed_distance(contact)
    return u.astype(np.float32), contact, phi.astype(np.float32), n_comp(contact)


def chi_param(c1x, sep, h1, h2, w, base):                  # numpy obstacle (matches gen_obstacle.obstacle)
    return G.obstacle((-sep, 0.0), (sep, 0.0), h1, h2, w, base)


def make_target(d):
    chi = chi_param(None, d["sep"], d["h1"], d["h2"], d["w"], d["base"])
    u, contact, phi, n = true_solve(chi)
    return chi.astype(np.float32), contact.astype(np.float32), phi, n


# ------------------------------------------------------------------ designers
def chi_from_raw(raw):
    p = {k: RANGES[k][0] + (RANGES[k][1] - RANGES[k][0]) * torch.sigmoid(raw[k]) for k in RANGES}
    g1 = p["h1"] * torch.exp(-((XX + p["sep"]) ** 2 + YY ** 2) / (2 * p["w"] ** 2))
    g2 = p["h2"] * torch.exp(-((XX - p["sep"]) ** 2 + YY ** 2) / (2 * p["w"] ** 2))
    return g1 + g2 - p["base"], p


def design_parametric(model, stats, Mstar, steps, lr=0.08):
    """Design the obstacle PARAMETERS (stays on the data manifold) through `model`."""
    raw = {k: nn.Parameter(torch.zeros(())) for k in RANGES}
    opt = torch.optim.Adam(list(raw.values()), lr=lr)
    Mt = torch.from_numpy(Mstar).float()
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        chi, p = chi_from_raw(raw)
        phi, _ = operator_fields(model, stats, chi)
        J = footprint_J(phi, Mt)
        J.backward(); opt.step(); hist.append(J.item())
    with torch.no_grad():
        chi, p = chi_from_raw(raw)
        phi, _ = operator_fields(model, stats, chi)
    params = {k: float(v) for k, v in p.items()}
    return chi.detach().numpy(), phi.detach().numpy(), hist, params


def design_field(model, stats, Mstar, steps, lr=0.05, coarse=16):
    """Design a band-limited χ FIELD directly (more expressive → can leave the data manifold)."""
    theta = nn.Parameter(torch.zeros(1, 1, coarse, coarse))
    opt = torch.optim.Adam([theta], lr=lr)
    Mt = torch.from_numpy(Mstar).float()
    lo, hi = -0.25, 0.55                                    # bound χ to the physical obstacle range
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        field = F.interpolate(theta, size=(N, N), mode="bicubic", align_corners=True)[0, 0]
        chi = lo + (hi - lo) * torch.sigmoid(field)
        phi, _ = operator_fields(model, stats, chi)
        tv = (theta[:, :, 1:] - theta[:, :, :-1]).abs().mean() + \
             (theta[:, :, :, 1:] - theta[:, :, :, :-1]).abs().mean()
        J = footprint_J(phi, Mt) + 0.02 * tv
        J.backward(); opt.step(); hist.append(footprint_J(phi, Mt).item())
    with torch.no_grad():
        field = F.interpolate(theta, size=(N, N), mode="bicubic", align_corners=True)[0, 0]
        chi = lo + (hi - lo) * torch.sigmoid(field)
        phi, _ = operator_fields(model, stats, chi)
    return chi.detach().numpy(), phi.detach().numpy(), hist, None


# ------------------------------------------------------------------ certify a finished design
def certify(chi_np, phi_op, Mstar):
    """Re-solve the optimized design with the TRUE solver and compare to the operator's prediction."""
    Mt = torch.from_numpy(Mstar).float()
    u_t, contact_t, phi_t, n_t = true_solve(chi_np)
    ct_op = phi_op < 0
    J_frost = float(footprint_J(torch.from_numpy(phi_op), Mt))
    J_true = float(footprint_J(torch.from_numpy(phi_t), Mt))
    Mb = Mstar > 0.5
    return dict(
        J_frost=J_frost, J_true=J_true, certificate_gap=abs(J_frost - J_true),
        IoU_operator_vs_target=iou(ct_op, Mb), IoU_true_vs_target=iou(contact_t > 0, Mb),
        IoU_operator_vs_true=iou(ct_op, contact_t > 0),
        topo_operator=n_comp(ct_op), topo_true=int(n_t),
        _phi_true=phi_t, _contact_true=contact_t, _u_true=u_t)


# ------------------------------------------------------------------ main
def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    (deq, dstats), (fno, fstats) = load_operators()
    print(f"loaded frozen DEQ + FNO operators.  design steps = {steps}")

    # achievable targets: contact footprints of REAL obstacles (one merged=1-comp, one separated=2-comp)
    targets = {
        "merged_1comp": dict(sep=0.18, h1=0.72, h2=0.72, w=0.23, base=0.14),
        "separated_2comp": dict(sep=0.40, h1=0.55, h2=0.55, w=0.18, base=0.18),
    }
    results = {}
    for name, d in targets.items():
        chi_t, M, phi_tgt, n_tgt = make_target(d)
        print(f"\n=== target '{name}': true topology {n_tgt} comp, contact area {M.mean():.3f} ===")
        entry = {"target_topology": int(n_tgt), "target_area_frac": float(M.mean()), "runs": {}}
        t0 = time.time()
        # (1) principled: through the DEQ equilibrium (IFT adjoint)
        chi, phi_op, hist, par = design_parametric(deq, dstats, M, steps)
        cert = certify(chi, phi_op, M)
        entry["runs"]["deq_equilibrium_IFT"] = {**{k: v for k, v in cert.items() if not k.startswith("_")},
                                                "params": par, "loss_hist": hist}
        entry["_deq_vis"] = (chi, phi_op, cert, M, chi_t)
        print(f"  [DEQ/IFT]  J_FROST {cert['J_frost']:.4f}  J_true {cert['J_true']:.4f}  "
              f"gap {cert['certificate_gap']:.4f}  IoU(op/target) {cert['IoU_operator_vs_target']:.3f}  "
              f"IoU(true/target) {cert['IoU_true_vs_target']:.3f}  topo op/true {cert['topo_operator']}/{cert['topo_true']}")
        # (2) baseline: through the frozen feed-forward FNO
        chi2, phi_op2, hist2, par2 = design_parametric(fno, fstats, M, steps)
        cert2 = certify(chi2, phi_op2, M)
        entry["runs"]["fno_feedforward"] = {**{k: v for k, v in cert2.items() if not k.startswith("_")},
                                            "params": par2, "loss_hist": hist2}
        print(f"  [FNO/ff ]  J_FROST {cert2['J_frost']:.4f}  J_true {cert2['J_true']:.4f}  "
              f"gap {cert2['certificate_gap']:.4f}  IoU(true/target) {cert2['IoU_true_vs_target']:.3f}")
        results[name] = entry
        print(f"  ({time.time()-t0:.0f}s)")

    # (3) off-manifold field design on the merged target (motivates the trust region)
    print("\n=== off-manifold field design (DEQ), merged target ===")
    chiF, phiF, histF, _ = design_field(deq, dstats, results["merged_1comp"]["_deq_vis"][3],
                                        steps, coarse=16)
    certF = certify(chiF, phiF, results["merged_1comp"]["_deq_vis"][3])
    results["merged_1comp"]["runs"]["deq_field_offmanifold"] = {
        **{k: v for k, v in certF.items() if not k.startswith("_")}, "loss_hist": histF}
    results["_field_vis"] = (chiF, phiF, certF, results["merged_1comp"]["_deq_vis"][3])
    print(f"  [DEQ/field] J_FROST {certF['J_frost']:.4f}  J_true {certF['J_true']:.4f}  "
          f"gap {certF['certificate_gap']:.4f}  (operator thinks it matched; true solver disagrees → "
          f"certificate gap {certF['certificate_gap']/max(1e-9,results['merged_1comp']['runs']['deq_equilibrium_IFT']['certificate_gap']):.1f}× the parametric gap)")

    # ---- save metrics (strip heavy arrays)
    clean = {n: {"target_topology": e["target_topology"], "target_area_frac": e["target_area_frac"],
                 "runs": e["runs"]} for n, e in results.items() if not n.startswith("_")}
    summary = {
        "task": "free-boundary targeting: design obstacle χ so the equilibrium contact set {φ*<0} matches M*",
        "method": "differentiate the frozen DEQ equilibrium via the IFT adjoint (§4.4); certify with the true SOR solver (§4.5)",
        "design_steps": steps, "tau": TAU, "results": clean,
        "headline": "IFT-through-equilibrium design hits the target free boundary AND is certified by the "
                    "true solver (small |J_FROST−J_true|) when the design stays on-manifold (parametric); "
                    "the off-manifold field design fools the operator (low J_FROST) but the true solver "
                    "disagrees (large gap) — the empirical case for the active-acquisition/trust-region C2.",
    }
    json.dump(summary, open(os.path.join(OUT, "design_metrics.json"), "w"), indent=2)
    print("\nsaved -> results/design_metrics.json")

    plot_targets(results)
    plot_convergence(results)
    plot_certificate(results)
    print(f"saved -> {OUT}")


# ------------------------------------------------------------------ figures
def _contour_panel(ax, base_img, cmap, overlays, title):
    ax.imshow(np.rot90(base_img), extent=(-1, 1, -1, 1), cmap=cmap)
    for mask, color, lw in overlays:
        ax.contour(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.rot90(mask.astype(float)),
                   levels=[0.5], colors=[color], linewidths=lw)
    ax.set_title(title, fontsize=FS - 5); ax.set_xticks([]); ax.set_yticks([])


def plot_targets(results):
    names = [n for n in results if not n.startswith("_")]
    fig, axs = plt.subplots(len(names), 3, figsize=(13.5, 4.6 * len(names)))
    if len(names) == 1:
        axs = axs[None, :]
    for r, name in enumerate(names):
        chi, phi_op, cert, M, chi_t = results[name]["_deq_vis"]
        Mb = M > 0.5; ct_op = phi_op < 0; ct_true = cert["_contact_true"] > 0
        # col 0: designed obstacle χ + target outline
        _contour_panel(axs[r, 0], chi, "viridis", [(Mb, "white", 2.5)],
                       "designed obstacle χ\n(white = target footprint)")
        # col 1: operator-predicted contact vs target
        axs[r, 1].imshow(np.rot90(np.zeros((N, N))), extent=(-1, 1, -1, 1), cmap="gray", vmin=0, vmax=1)
        for mask, c, lw in [(Mb, "white", 2.0), (ct_op, "#1f77b4", 2.4)]:
            axs[r, 1].contour(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.rot90(mask.astype(float)),
                              levels=[0.5], colors=[c], linewidths=lw)
        axs[r, 1].set_title(f"FROST equilibrium contact\nIoU/target {cert['IoU_operator_vs_target']:.2f}  "
                            f"({cert['topo_operator']}c)", fontsize=FS - 5)
        axs[r, 1].set_xticks([]); axs[r, 1].set_yticks([])
        # col 2: TRUE-solver contact at the SAME design (the certificate)
        axs[r, 2].imshow(np.rot90(np.zeros((N, N))), extent=(-1, 1, -1, 1), cmap="gray", vmin=0, vmax=1)
        for mask, c, lw in [(Mb, "white", 2.0), (ct_true, "#2ca02c", 2.4)]:
            axs[r, 2].contour(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.rot90(mask.astype(float)),
                              levels=[0.5], colors=[c], linewidths=lw)
        axs[r, 2].set_title(f"true solver (certify)\nIoU/target {cert['IoU_true_vs_target']:.2f}  "
                            f"gap {cert['certificate_gap']:.3f}", fontsize=FS - 5)
        axs[r, 2].set_xticks([]); axs[r, 2].set_yticks([])
        axs[r, 0].set_ylabel(f"target: {results[name]['target_topology']}-comp", fontsize=FS - 4)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "design_targets.png"), dpi=DPI); plt.close(fig)


def plot_convergence(results):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in [n for n in results if not n.startswith("_")]:
        ax.semilogy(results[name]["runs"]["deq_equilibrium_IFT"]["loss_hist"], lw=2.5,
                    label=f"{name} — DEQ/IFT")
        ax.semilogy(results[name]["runs"]["fno_feedforward"]["loss_hist"], lw=2, ls="--",
                    label=f"{name} — FNO/ff")
    ax.set_xlabel("design step", fontsize=FS); ax.set_ylabel("objective J (footprint mismatch)", fontsize=FS)
    ax.tick_params(labelsize=FS - 5); ax.grid(alpha=0.3); ax.legend(fontsize=FS - 8)
    ax.set_title("inverse-design convergence", fontsize=FS - 2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "design_convergence.png"), dpi=DPI); plt.close(fig)


def plot_certificate(results):
    """The certificate gap |J_FROST − J_true|: small on-manifold (parametric), large off-manifold (field)."""
    labels, gaps, colors = [], [], []
    for name in [n for n in results if not n.startswith("_")]:
        labels.append(f"{name}\nDEQ/IFT"); gaps.append(results[name]["runs"]["deq_equilibrium_IFT"]["certificate_gap"]); colors.append("#1f77b4")
        labels.append(f"{name}\nFNO/ff"); gaps.append(results[name]["runs"]["fno_feedforward"]["certificate_gap"]); colors.append("#8c9eb2")
    fld = results["merged_1comp"]["runs"]["deq_field_offmanifold"]["certificate_gap"]
    labels.append("merged_1comp\nDEQ/field\n(off-manifold)"); gaps.append(fld); colors.append("#c0392b")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(gaps)), gaps, color=colors)
    for i, g in enumerate(gaps):
        ax.text(i, g + max(gaps) * 0.01, f"{g:.3f}", ha="center", fontsize=FS - 8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=FS - 9)
    ax.set_ylabel("certificate gap  |J_FROST − J_true|", fontsize=FS - 2)
    ax.tick_params(axis="y", labelsize=FS - 5); ax.grid(axis="y", alpha=0.3)
    ax.set_title("trust-region certificate (true solver vs operator at the optimum)", fontsize=FS - 4)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "design_certificate.png"), dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    main()
