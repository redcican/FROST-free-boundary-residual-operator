"""
Operator comparison figure for the channel C1 study (A2 φ-vs-γ + A3 topology generalization).
Reads results/{phi,gamma}_{random,topo}/metrics.json -> results/operator_comparison.png

Run: python viz_compare.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
FS, FIG_DPI = 22, 600


def load(cfg):
    p = os.path.join(OUT, cfg, "metrics.json")
    return json.load(open(p))["test"] if os.path.exists(p) else None


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.5))

    # left: A2 random-split, C rel-L2 per topology, φ vs γ
    pr, gr = load("phi_random"), load("gamma_random")
    topos = [1, 2, 3]
    phi_c = [pr["by_topology"][str(t)]["C_relL2"] for t in topos]
    gam_c = [gr["by_topology"][str(t)]["C_relL2"] for t in topos]
    x = np.arange(len(topos)); wbar = 0.38
    a1.bar(x - wbar/2, phi_c, wbar, label="φ-conditioned (FROST)", color="#1f77b4")
    a1.bar(x + wbar/2, gam_c, wbar, label="γ-conditioned (NTO-style)", color="#ff7f0e")
    a1.set_xticks(x); a1.set_xticklabels([f"{t} baffle(s)" for t in topos], fontsize=FS - 4)
    a1.set_ylabel("C rel-L2 (held-out)", fontsize=FS); a1.set_title("A2  representation, random split", fontsize=FS)
    a1.legend(fontsize=FS - 7); a1.tick_params(labelsize=FS - 5); a1.grid(alpha=0.3, axis="y")

    # right: A3 topology generalization — overall C/P rel-L2, random vs topo-extrapolation
    cfgs = ["phi_random", "gamma_random", "phi_topo", "gamma_topo"]
    labels = ["φ\nrandom", "γ\nrandom", "φ\ntopo→3", "γ\ntopo→3"]
    C = [load(c)["C_relL2"] for c in cfgs]; P = [load(c)["P_relL2"] for c in cfgs]
    x = np.arange(len(cfgs))
    a2.bar(x - wbar/2, C, wbar, label="C rel-L2", color="#2ca02c")
    a2.bar(x + wbar/2, P, wbar, label="P rel-L2", color="#9467bd")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=FS - 5)
    a2.set_ylabel("rel-L2 (held-out)", fontsize=FS)
    a2.set_title("A3  topology generalization (topo→3 = train 1–2, test 3)", fontsize=FS - 2)
    a2.legend(fontsize=FS - 7); a2.tick_params(labelsize=FS - 5); a2.grid(alpha=0.3, axis="y")

    fig.tight_layout(); fig.savefig(os.path.join(OUT, "operator_comparison.png"), dpi=FIG_DPI); plt.close(fig)
    print("saved -> results/operator_comparison.png")


if __name__ == "__main__":
    main()
