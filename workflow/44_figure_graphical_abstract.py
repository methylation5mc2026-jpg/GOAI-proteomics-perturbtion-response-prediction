#!/usr/bin/env python3
"""Draw figures/step6_graphical_abstract.png from the Step-6 artefacts.

Why this is drawn rather than illustrated
-----------------------------------------
Three successive AI-generated versions of this figure returned a byte-identical cached
image that named a database we never queried, quoted a margin of +13.6% where the measured
value is different, and -- most importantly -- showed only the favourable half of the
result. A graphical abstract is the most-read object in a paper; this one is therefore built
from the same artefacts as the text, so every number in it is the number in the tables, and
the null result is on the figure rather than omitted from it.
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch         # noqa: E402

SESSION = REPO_ROOT
RESULTS = SESSION / "results"
FIGURES = SESSION / "figures"

INK = "#16242c"
TEAL = "#2f8f8a"
NAVY = "#3b4a72"
AMBER = "#d99a2b"
RUST = "#b5502f"
PALE = "#eef3f6"
GREY = "#93a4ae"


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def panel_header(fig, x, y, w, text, n):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, 0.035, boxstyle="round,pad=0.002,rounding_size=0.006",
        transform=fig.transFigure, fc=TEAL, ec="none", zorder=5))
    fig.text(x + 0.008, y + 0.0175, f"{n}   {text}", transform=fig.transFigure,
             fontsize=14.9, weight="bold", color="white", va="center", zorder=6)


def main() -> None:
    aff = load("step6_target_affinity_report.json")
    mval = load("step6_itpv_mechanism_validation.json")
    attn = load("step6_attention_rank_diagnostics.json")
    stack = load("step6_cluster_weights.json")

    head = stack.get("headline_val_total")
    five = stack.get("step5_val_total")
    bench = stack.get("benchmark_total")
    d_five = (head - five) if head and five else None
    pct_bench = 100.0 * (head - bench) / bench if head and bench else None

    fig = plt.figure(figsize=(13.0, 9.4), facecolor="white")
    fig.text(0.5, 0.973, "Mechanism-aware virtual cell modelling of the "
             "$\\it{Saccharomyces\\ cerevisiae}$ perturbation proteome",
             ha="center", va="center", fontsize=15.5, weight="bold", color=INK)

    panel_header(fig, 0.030, 0.905, 0.455, "PERTURBATION PROTEOMICS", "1")
    panel_header(fig, 0.515, 0.905, 0.455, "MECHANISM, IMPORTED AND VALIDATED", "2")
    panel_header(fig, 0.030, 0.408, 0.455, "MECHANISM-AWARE MODEL", "3")
    panel_header(fig, 0.515, 0.408, 0.455, "WHAT IT BOUGHT", "4")

    # ---------------- panel 1 ----------------
    ax = fig.add_axes([0.045, 0.640, 0.175, 0.225])
    rng = np.random.default_rng(42)
    base = rng.normal(0, 0.45, (26, 34))
    base[:, 8:13] += rng.normal(1.15, 0.25, (26, 5))
    base[14:20, :] -= 0.95
    ax.imshow(base, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto",
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("log$_2$ fold change", fontsize=11, color=INK, pad=4)
    ax.set_xlabel("5,243 proteins", fontsize=10.5, color=INK, labelpad=3)
    ax.set_ylabel("samples", fontsize=10.5, color=INK, labelpad=3)
    for sp in ax.spines.values():
        sp.set_color(GREY)

    fig.text(0.238, 0.862,
             "$\\bf{13{,}412}$ samples\n"
             "$\\bf{5{,}243}$ proteins\n"
             "56 chemicals, 4 strains,\ntime course\n\n"
             "OOD regimes: novel\nchemical, novel strain,\nboth novel",
             fontsize=11, color=INK, va="top", linespacing=1.5)

    fig.patches.append(FancyBboxPatch(
        (0.038, 0.478), 0.440, 0.135,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        transform=fig.transFigure, fc=PALE, ec=GREY, lw=1.0, zorder=4))
    fig.text(0.052, 0.545,
             "$\\bf{The\\ obstacle}$:  with 37 compounds in the fitting cohort and\n"
             "5,243 protein columns, the compound-by-protein engagement\n"
             "surface is not identifiable from the abundance data.",
             fontsize=11.2, color=INK, va="center", linespacing=1.55, zorder=5)

    # ---------------- panel 2 ----------------
    ax = fig.add_axes([0.578, 0.640, 0.145, 0.225])
    pos = [p for p in mval.get("positives", []) if p.get("any_hit")]
    lab, val = [], []
    for p in pos[:7]:
        h = max((x for x in p["hits"] if x.get("status") == "annotated"),
                key=lambda x: x["pactivity"], default=None)
        if h:
            _nm = p["compound"].split()[0].rstrip(",")
            _nm = {"(S)-(+)-Camptothecin": "Camptothecin"}.get(_nm, _nm)
            lab.append(f"{_nm} $\\to$ {h['protein']}")
            val.append(h["pactivity"])
    order = np.argsort(val)
    ax.barh(np.arange(len(val)), np.array(val)[order], color=TEAL, height=0.68)
    ax.set_yticks(np.arange(len(val)))
    ax.set_yticklabels([lab[i] for i in order], fontsize=9.2)
    ax.set_xlabel("pActivity  ($-\\log_{10}$ M)", fontsize=10.5, color=INK)
    ax.set_xlim(0, 13); ax.tick_params(axis="x", labelsize=9.5)
    ax.grid(axis="x", lw=0.4, alpha=0.3); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("recovered pharmacology", fontsize=11, color=INK, pad=4)

    fig.text(0.745, 0.868,
             f"$\\bf{{{aff.get('n_activity_records_usable', 0):,}}}$ curated $K_d$, "
             f"$K_i$, IC$_{{50}}$,\nEC$_{{50}}$ records over "
             f"$\\bf{{{aff.get('n_distinct_chembl_targets', 0):,}}}$ ChEMBL\n"
             "targets, mapped onto the\nmeasured yeast proteome\n"
             f"by orthology\n"
             f"($\\bf{{{aff.get('target_to_protein_links', 0):,}}}$ target links)",
             fontsize=11, color=INK, va="top", linespacing=1.5)

    fig.patches.append(FancyBboxPatch(
        (0.524, 0.478), 0.440, 0.135,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        transform=fig.transFigure, fc="#eaf4ee", ec=TEAL, lw=1.2, zorder=4))
    fig.text(0.744, 0.545,
             "$\\bf{Validated\\ against\\ pharmacology\\ fixed\\ in\\ advance}$\n"
             f"{mval.get('n_positive_compounds_recovered', 0)} of "
             f"{mval.get('n_positive_compounds', 0)} positive controls recovered   ·   "
             f"{mval.get('n_negative_pairs_correctly_zero', 0)} of "
             f"{mval.get('n_negative_pairs_tested', 0)} negative controls zero",
             fontsize=11.2, color=INK, ha="center", va="center", linespacing=1.7,
             zorder=5)

    # ---------------- panel 3 ----------------
    fig.text(0.040, 0.378,
             "Chemical context modulates attention over the STRING protein\n"
             "graph; co-complex coherence and metabolite mass balance enter\n"
             "the objective as verified penalties.",
             fontsize=11, color=INK, va="top", linespacing=1.5)

    ax = fig.add_axes([0.052, 0.105, 0.150, 0.180])
    spec = attn.get("normalised_singular_spectrum") or [1]
    cum = np.cumsum(spec)
    ax.plot(np.arange(1, len(cum) + 1), cum, marker="o", ms=5.5, lw=2.1, color=RUST)
    ax.set_ylim(0, 1.08); ax.set_xlim(0.6, len(cum) + 0.4)
    ax.set_xlabel("component", fontsize=10.5)
    ax.set_ylabel("cum. variance", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    ax.set_title("attention modulation", fontsize=11, color=INK, pad=4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.annotate(f"{100 * spec[0]:.0f}% in one\ndirection",
                xy=(1, spec[0]), xytext=(2.4, 0.42), fontsize=10, color=RUST,
                arrowprops=dict(arrowstyle="->", color=RUST, lw=1.3))

    fig.patches.append(FancyBboxPatch(
        (0.225, 0.100), 0.253, 0.192,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        transform=fig.transFigure, fc="#fdf3e3", ec=AMBER, lw=1.2, zorder=4))
    fig.text(0.238, 0.196,
             "$\\bf{But\\ the\\ modulation\\ collapsed.}$\n"
             f"Participation ratio "
             f"{attn.get('participation_ratio', float('nan')):.2f} of a\n"
             f"possible {attn.get('n_modes', 0)} — chemistry gates the\n"
             "attention in $\\it{magnitude}$, not in\n$\\it{pattern}$. We report the "
             "rank statistic,\nnot the heat map it would replace.",
             fontsize=10.8, color=INK, va="center", linespacing=1.6, zorder=5)

    # ---------------- panel 4 ----------------
    ax = fig.add_axes([0.540, 0.135, 0.175, 0.240])
    names = ["control", "benchmark", "Step 5", "Step 6"]
    vals = [0.2515, bench, five, head]
    ax.bar(np.arange(4), vals, color=[GREY, NAVY, TEAL, TEAL], width=0.64)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(names, fontsize=10)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylim(0, 0.63)
    ax.set_ylabel("competition score", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.grid(axis="y", lw=0.4, alpha=0.3); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.014, f"{v:.4f}", ha="center", fontsize=9.6, color=INK)
    ax.annotate("", xy=(3, head + 0.085), xytext=(2, five + 0.085),
                arrowprops=dict(arrowstyle="<->", color=RUST, lw=1.4))
    ax.text(2.5, head + 0.115, f"{d_five:+.4f}", ha="center", fontsize=11,
            color=RUST, weight="bold")

    fig.text(0.735, 0.352,
             f"$\\bf{{vs\\ benchmark}}$:  {head:.4f} against\n"
             f"{bench:.4f} — a margin of $\\bf{{{pct_bench:+.1f}\\%}}$,\n"
             "weights fitted out-of-fold and frozen\nbefore one validation evaluation.\n\n"
             f"$\\bf{{vs\\ previous\\ stage}}$:  ${d_five:+.4f}$.\n"
             "The pre-registered target was not met.",
             fontsize=11, color=INK, va="top", linespacing=1.5)

    fig.patches.append(FancyBboxPatch(
        (0.730, 0.100), 0.234, 0.150,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        transform=fig.transFigure, fc="#fdf3e3", ec=AMBER, lw=1.2, zorder=4))
    fig.text(0.744, 0.175,
             "$\\bf{The\\ central\\ result}$\n"
             "A knowledge import can be verifiably\ncorrect as $\\it{data}$ and inert as "
             "a $\\it{feature}$.\nProvenance and coverage establish\nwhat an import is; "
             "not that a model\ncan use it.",
             fontsize=10.8, color=INK, va="center", linespacing=1.6, zorder=5)

    for x0, x1, y in ((0.489, 0.511, 0.9225), (0.489, 0.511, 0.4255)):
        fig.patches.append(FancyArrowPatch(
            (x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=16, color=NAVY, lw=2.0, zorder=6))

    fig.text(0.5, 0.042,
             "Leave-Chemical-Group-Out 5-fold cross-fitting   ·   every member prediction "
             "from a model that saw neither the row's compound nor its strain   ·   "
             "validation scored exactly once",
             ha="center", fontsize=10.6, color=INK,
             bbox=dict(boxstyle="round,pad=0.55", fc=PALE, ec=GREY, lw=0.9))

    fig.savefig(FIGURES / "step6_graphical_abstract.png", dpi=230,
                bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / "step6_graphical_abstract.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote figures/step6_graphical_abstract.png")
    print(f"  headline {head}  step5 {five}  bench {bench}  "
          f"delta5 {d_five:+.6f}  vs bench {pct_bench:+.2f}%")


if __name__ == "__main__":
    main()
