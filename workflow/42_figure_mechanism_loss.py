#!/usr/bin/env python3
"""Draw figures/step6_mechanism_loss_detail.png from the Step-6.2 artefacts.

Why this replaces an illustration
---------------------------------
The first version of this figure was an AI-drawn schematic, and it twice rendered
phosphoenolpyruvate as an enzyme box and placed pyruvate kinase upstream of its own
substrate. Rather than keep re-prompting, the figure is drawn here from the curated
structures themselves, so every label, count and sparsity pattern is the object the loss
actually uses. Nothing in this figure is hand-entered.
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402

SESSION = REPO_ROOT
WORKFLOW = SESSION / "workflow"
RESULTS = SESSION / "results"
FIGURES = SESSION / "figures"
sys.path.insert(0, str(WORKFLOW))

#: Pathway blocks, in the order the curated reaction list follows. Used only to group
#: the stoichiometry columns visually; membership is the curation's, not invented here.
BLOCKS = [
    ("Glycolysis", ["HEX1", "PGI", "PFK", "FBA", "TPI", "GAPD", "PGK", "PGM", "ENO",
                    "PYK"]),
    ("Fermentation", ["PDC", "ALCD", "ALDD", "ACS"]),
    ("TCA cycle", ["PDH", "CS", "ACONT", "ICDH", "AKGD", "SUCOAS", "SUCD", "FUM",
                   "MDH"]),
    ("Pentose phosphate", ["G6PDH", "PGL", "GND", "RPI", "RPE", "TKT1", "TALA"]),
    ("Ergosterol biosynthesis", ["HMGCOAR", "MEVK", "PMEVK", "DPMVD", "IPDDI", "FRTT",
                                 "SQLS", "SQLE", "LNS", "C14DM", "C14STR", "C24STR",
                                 "C8ISO", "ERGSTt"]),
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    import pandas as pd

    rep = json.loads((RESULTS / "step6_mechanism_loss_report.json").read_text())
    mech = load_module(WORKFLOW / "35_metabolic_mechanism_loss.py", "mech35")

    # Rebuild the structures the loss actually uses, from the same protein roster the
    # Step-6.2 run used, so the figure cannot drift from the objective.
    prot = pd.read_parquet(SESSION / "data" / "step5_protein_stats.parquet")
    proteins = [str(p) for p in prot["protein"].astype(str)]
    struct = mech.build_structures(proteins)
    S = np.asarray(struct["S"])
    rxn_names = list(struct["reactions"])

    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], width_ratios=[1.0, 1.25],
                          hspace=0.34, wspace=0.22)

    # ---- (A) complex coverage -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    cm = rep["complex_membership"]
    names = sorted(cm, key=lambda k: cm[k]["n_measured"])
    cur = [cm[n]["n_curated"] for n in names]
    mea = [cm[n]["n_measured"] for n in names]
    y = np.arange(len(names))
    ax.barh(y, cur, color="#d9e2e8", edgecolor="#6b8394", lw=0.6, label="curated")
    ax.barh(y, mea, color="#2f6d8c", edgecolor="none", label="measured")
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7.5)
    ax.set_xlabel("subunits", fontsize=9)
    ax.set_title(f"A  Curated protein complexes ({len(names)}), and how many of their\n"
                 f"subunits this proteome actually quantifies",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="center right", frameon=False,
              bbox_to_anchor=(1.0, 0.42))
    ax.grid(axis="x", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate(f"{rep['n_complex_edges']:,} within-complex edges,\n"
                r"each weighted $1/(k-1)$",
                xy=(0.97, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=8.2,
                bbox=dict(boxstyle="round,pad=0.34", fc="#f3f7f9", ec="#9fb4c0", lw=0.6))

    # ---- (B) stoichiometry sparsity ------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    if S is not None:
        ax.imshow((S != 0).astype(float), aspect="auto", cmap="Greys",
                  interpolation="nearest", vmin=0, vmax=1.4)
        ax.set_ylabel("balanced metabolite", fontsize=9)
        pos, lab = [], []
        idx = {r: i for i, r in enumerate(rxn_names)}
        for bname, rs in BLOCKS:
            cols = [idx[r] for r in rs if r in idx]
            if not cols:
                continue
            ax.axvline(min(cols) - 0.5, color="#c0392b", lw=0.8, alpha=0.65)
            pos.append(float(np.mean(cols)))
            lab.append(bname)
        ax.set_xticks(pos)
        ax.set_xticklabels(lab, fontsize=8, rotation=18, ha="right")
        ax.set_title(f"B  Stoichiometric matrix $\\mathbf{{S}}$ from iMM904: "
                     f"{S.shape[0]} balanced metabolites $\\times$ {S.shape[1]} "
                     f"reactions\n"
                     r"the flux penalty is $\Vert\mathbf{S}\,v(\hat{\delta})\Vert_2^2$",
                     fontsize=10, loc="left")
    else:
        ax.text(0.5, 0.5, "stoichiometry unavailable", ha="center", va="center")
        ax.set_axis_off()

    # ---- (C) convergence trajectory ------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    traj = rep["convergence_trajectory"]
    ep = [t["epoch"] for t in traj]
    for key, lab, col in (("mse", "masked squared error", "#2f6d8c"),
                          ("pcc_loss", "per-sample correlation", "#7a9a3a"),
                          ("complex", "complex coherence", "#c0392b"),
                          ("flux", "metabolic flux", "#d68910"),
                          ("total", "weighted total", "#333333")):
        ax.plot(ep, [t[key] for t in traj], label=lab, lw=1.9 if key == "total" else 1.3,
                ls="-" if key == "total" else "--", color=col)
    ax.set_yscale("log")
    ax.set_xlabel("optimisation step", fontsize=9)
    ax.set_ylabel("term value (log scale)", fontsize=9)
    ax.set_title("C  All four terms fall together on a fitting trajectory",
                 fontsize=10, loc="left")
    ax.legend(fontsize=7.6, frameon=False, ncol=2)
    ax.grid(lw=0.4, alpha=0.35)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- (D) term isolation + verification -----------------------------------
    ax = fig.add_subplot(gs[1, 1])
    iso = rep["term_isolation"]
    dev = rep["verification"]["per_term_deviation"]
    keys = [("mse", "mse_only", "masked squared\nerror"),
            ("pcc_loss", "pcc_only", "per-sample\ncorrelation"),
            ("complex", "complex_only", "complex\ncoherence"),
            ("flux", "flux_only", "metabolic\nflux")]
    x = np.arange(len(keys))
    starts = [iso[k[1]]["start"] for k in keys]
    ends = [max(iso[k[1]]["end"], 1e-9) for k in keys]
    ax.bar(x - 0.19, starts, 0.36, color="#b8c9d4", edgecolor="#6b8394", lw=0.6,
           label="at initialisation")
    ax.bar(x + 0.19, ends, 0.36, color="#2f6d8c", label="after optimising it alone")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([k[2] for k in keys], fontsize=8.4)
    ax.set_ylabel("isolated term value (log scale)", fontsize=9)
    ax.set_title("D  Each term, switched on alone from a non-degenerate start:\n"
                 "strictly positive, then driven down",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    txt = "  ".join(f"{k[2].replace(chr(10), ' ')}: {dev[k[0]]:.1e}" for k in keys)
    ax.annotate("max |tensor $-$ NumPy| per term\n" + txt,
                xy=(0.5, -0.30), xycoords="axes fraction", ha="center", fontsize=7.6,
                bbox=dict(boxstyle="round,pad=0.34", fc="#f3f7f9", ec="#9fb4c0", lw=0.6))

    fig.savefig(FIGURES / "step6_mechanism_loss_detail.png", dpi=190,
                bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / "step6_mechanism_loss_detail.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote figures/step6_mechanism_loss_detail.png")
    if S is not None:
        print(f"  stoichiometry {S.shape}, {int((S != 0).sum())} non-zero entries")


if __name__ == "__main__":
    main()
