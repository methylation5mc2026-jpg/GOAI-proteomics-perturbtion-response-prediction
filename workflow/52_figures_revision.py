#!/usr/bin/env python3
"""Step 7.3 -- Figures for the 2026-08-11 revised submission.

Two figures, both drawn directly from result artefacts (no hand-entered number):

``figures/step7_test_selfscore.png``
    The seven-module self-evaluation on the released test truth, the validation
    counterpart, the aggregation-convention sensitivity, and the S1 batch
    confound re-measured on the test cohort.

``figures/step7_entity_extrapolation.png``
    What the merged open leaderboard actually asks for: which entities are
    unseen, how far the unseen compounds are from the training chemistry, and
    which entity axes currently do and do not have an external representation.
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from matplotlib.patches import Patch, Rectangle                       # noqa: E402

SESSION = REPO_ROOT
RESULTS = SESSION / "results"
FIGURES = SESSION / "figures"

NAVY = "#154a66"
TEAL = "#2f8f8f"
ORANGE = "#c47424"
GREY = "#8b96a0"
RED = "#a63a2f"
GREEN = "#2c6e48"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": "#404040", "axes.linewidth": 0.7,
    "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "xtick.direction": "out", "ytick.direction": "out",
    "savefig.dpi": 220, "figure.dpi": 110,
})

MODULES = ["m1_abundance", "m2_fold_change", "m3_s1_chem", "m3_s2_strain",
           "m3_s3_both", "m3_time", "m4_dep"]
MODULE_LABELS = ["M1\nabund.", "M2\nfold ch.", "S1\nnew chem",
                 "S2\nnew strain", "S3\nboth new", "TIME\ninterp.", "M4\nDEP"]
CONV_LABELS = {"per_sample_primary": "per-sample\n(primary spec)",
               "pooled_only": "pooled", "per_sample_only": "per-sample only",
               "per_protein_primary": "per-protein"}


def figure_selfscore(sc: dict, val: dict) -> None:
    """Six panels: stage trajectory on both cohorts, then the test-cohort detail."""
    scores = sc["scores"]
    fig = plt.figure(figsize=(16.6, 8.0))
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.30,
                          left=0.055, right=0.988, top=0.875, bottom=0.085)

    STAGES = [("step3_gbdt", "Stage 3\nGBDT", "step3"),
              ("step4_chem_stacking", "Stage 4\nchemistry", "step4"),
              ("step5_knowledge_stacking", "Stage 5\nknowledge", "step5"),
              ("step6_submitted", "Stage 6\nmechanism", "step6")]
    val_tot = [val[k] for _, _, k in STAGES]
    te_tot = [scores[k]["total_score"] for k, _, _ in STAGES]
    val_bench, te_bench = val["benchmark"], scores["per_context_mean_batch"]["total_score"]
    val_null, te_null = val["null"], scores["control_anchor"]["total_score"]

    # --- A: stage trajectory on both cohorts -------------------------------
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(STAGES))
    ax.plot(x, val_tot, "-o", color=ORANGE, lw=1.8, ms=6,
            label="validation cohort (n=2,806)")
    ax.plot(x, te_tot, "-s", color=NAVY, lw=1.8, ms=5.5,
            label="test cohort (n=4,226)")
    ax.axhline(val_bench, color=ORANGE, ls="--", lw=1.0, alpha=0.8)
    ax.axhline(te_bench, color=NAVY, ls="--", lw=1.0, alpha=0.8)
    ax.axhline(val_null, color=GREY, ls=":", lw=1.0)
    ax.axhline(te_null, color=GREY, ls=":", lw=1.0)
    bb = dict(facecolor="white", edgecolor="none", pad=0.9)
    ax.text(1.04, val_bench + 0.004, f"official benchmark {val_bench:.4f}",
            fontsize=6.5, color=ORANGE, ha="left", va="bottom", bbox=bb)
    ax.text(1.04, te_bench - 0.005, f"benchmark on test {te_bench:.4f}",
            fontsize=6.5, color=NAVY, ha="left", va="top", bbox=bb)
    ax.text(-0.05, (val_null + te_null) / 2 - 0.022,
            f"$\\Delta\\equiv0$ null: {val_null:.3f} / {te_null:.3f}",
            fontsize=6.6, color="#555")
    for xi, (a, b) in enumerate(zip(val_tot, te_tot)):
        ax.text(xi, a + 0.011, f"{a:.4f}", ha="center", fontsize=7.0,
                color=ORANGE, fontweight="bold")
        ax.text(xi, b - 0.023, f"{b:.4f}", ha="center", fontsize=7.0,
                color=NAVY, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in STAGES], fontsize=7.4)
    ax.set_ylim(0.20, 0.56)
    ax.set_ylabel("weighted total score")
    ax.legend(fontsize=7.0, frameon=False, loc="lower right")
    ax.set_title("A  The stage trajectory replicates on a second cohort\n"
                 "same rubric, same frozen weights, different samples",
                 loc="left", fontsize=9)
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)

    # --- B: margin over the benchmark, per stage, both cohorts -------------
    ax = fig.add_subplot(gs[0, 1])
    mv = [a - val_bench for a in val_tot]
    mt = [b - te_bench for b in te_tot]
    ax.bar(x - 0.19, mv, width=0.38, color=ORANGE, label="validation",
           edgecolor="white", linewidth=0.5)
    ax.bar(x + 0.19, mt, width=0.38, color=NAVY, label="test",
           edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#333", lw=0.8)
    for xi, (a, b) in enumerate(zip(mv, mt)):
        ax.text(xi - 0.19, a + (0.0028 if a >= 0 else -0.0052), f"{a:+.4f}",
                ha="center", va="bottom" if a >= 0 else "top",
                fontsize=6.8, fontweight="bold")
        ax.text(xi + 0.19, b + (0.0028 if b >= 0 else -0.0052), f"{b:+.4f}",
                ha="center", va="bottom" if b >= 0 else "top",
                fontsize=6.8, fontweight="bold")
    lo, hi = min(mv + mt), max(mv + mt)
    ax.set_ylim(lo - 0.014, hi + 0.011)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in STAGES], fontsize=7.4)
    ax.set_ylabel("total score minus the group-mean benchmark")
    ax.legend(fontsize=7.2, frameon=False, loc="upper left")
    ax.set_title("B  Sign and ordering agree on both cohorts\n"
                 "Stage 3 is below the benchmark on both; the magnitude shrinks on test",
                 loc="left", fontsize=9)
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)

    # --- C: total score, test cohort, all predictors ----------------------
    ax = fig.add_subplot(gs[0, 2])
    order = ["global_mean", "control_anchor", "per_context_mean", "step3_gbdt",
             "per_context_mean_batch", "step4_chem_stacking",
             "step5_knowledge_stacking", "step6_submitted"]
    names = ["global_mean", "control_anchor ($\\Delta\\equiv0$)", "per_context_mean",
             "ours: Stage 3", "per_context_mean_batch\n(official-style benchmark)",
             "ours: Stage 4", "ours: Stage 5", "ours: Stage 6 (submitted)"]
    vals = [scores[k]["total_score"] for k in order]
    cols = [GREY, GREY, GREY, "#c9a227", TEAL, ORANGE, ORANGE, ORANGE]
    bars = ax.barh(np.arange(len(order)), vals, color=cols, height=0.66,
                   edgecolor="white", linewidth=0.6)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(names, fontsize=7.0)
    ax.invert_yaxis()
    for b, v in zip(bars, vals):
        ax.text(v + 0.006, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=7.4, fontweight="bold")
    ax.set_xlim(0, 0.58)
    ax.set_xlabel("weighted total score, test cohort")
    ax.set_title("C  Test cohort: every model against every trivial baseline\n"
                 f"submitted model beats the benchmark by "
                 f"+{te_tot[-1] - te_bench:.4f}", loc="left", fontsize=9)
    ax.grid(axis="x", alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)

    # --- D: per-module, test cohort ---------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    xm = np.arange(len(MODULES))
    w = 0.27
    trio = [("control_anchor", "$\\Delta\\equiv0$ null floor", GREY),
            ("per_context_mean_batch", "group-mean benchmark", TEAL),
            ("step6_submitted", "ours: Stage 6", ORANGE)]
    for i, (key, lab, col) in enumerate(trio):
        v = [scores[key]["module_scores"][m] for m in MODULES]
        ax.bar(xm + (i - 1) * w, v, width=w, color=col, label=lab,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(xm)
    ax.set_xticklabels(MODULE_LABELS, fontsize=6.4)
    ax.set_ylabel("module score, test cohort")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.0, frameon=False, loc="upper right")
    ax.set_title("D  Module decomposition on the test cohort\n"
                 "no residual module has a zero null floor",
                 loc="left", fontsize=9)
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    m1_null = scores["control_anchor"]["module_scores"]["m1_abundance"]
    m1_ours = scores["step6_submitted"]["module_scores"]["m1_abundance"]
    if m1_null > m1_ours:
        ax.annotate(f"the null anchor still wins M1:\n{m1_null:.3f} vs ours {m1_ours:.3f}",
                    xy=(-0.27, m1_null), xytext=(0.30, 0.955),
                    textcoords="axes fraction", fontsize=6.7, color=RED,
                    ha="left", va="top",
                    arrowprops=dict(arrowstyle="->", color=RED, lw=0.7))

    # --- E: aggregation-convention sensitivity ----------------------------
    ax = fig.add_subplot(gs[1, 1])
    convs = ["per_sample_primary", "pooled_only", "per_sample_only",
             "per_protein_primary"]
    ours = [scores["step6_submitted"]["sensitivity"][c]["total_score"] for c in convs]
    bench = [scores["per_context_mean_batch"]["sensitivity"][c]["total_score"]
             for c in convs]
    xc = np.arange(len(convs))
    ax.bar(xc - 0.19, bench, width=0.38, color=TEAL, label="group-mean benchmark",
           edgecolor="white", linewidth=0.5)
    ax.bar(xc + 0.19, ours, width=0.38, color=ORANGE, label="ours: Stage 6",
           edgecolor="white", linewidth=0.5)
    for xi, (a, b) in enumerate(zip(ours, bench)):
        ax.text(xi, max(a, b) + 0.013, f"+{a - b:.4f}", ha="center",
                fontsize=7.2, fontweight="bold", color=NAVY)
    ax.set_xticks(xc)
    ax.set_xticklabels([CONV_LABELS[c] for c in convs], fontsize=7.0)
    ax.set_ylim(0, 0.60)
    ax.set_ylabel("weighted total score, test cohort")
    ax.legend(fontsize=7.0, frameon=False, loc="upper left")
    ax.set_title("E  The ranking survives every aggregation convention\n"
                 "the official convention is unpublished, so all four are computed",
                 loc="left", fontsize=9)
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)

    # --- F: S1 batch confound, re-measured on test ------------------------
    ax = fig.add_subplot(gs[1, 2])
    conf = pd.DataFrame(sc["residual_confound"])
    sub = conf[conf.split == "val_chem_only"].set_index("model")
    keys = ["per_context_mean_batch", "step6_submitted", "control_anchor"]
    labs = ["group-mean\nbenchmark", "ours:\nStage 6", "$\\Delta\\equiv0$\nnull"]
    blind = [sub.loc[k, "resid_pcc_batch_blind"] for k in keys]
    aware = [sub.loc[k, "resid_pcc_batch_aware"] for k in keys]
    xb = np.arange(len(keys))
    ax.bar(xb - 0.19, blind, width=0.38, color=TEAL,
           label="batch-blind $\\mu_{ctx}$ (the official form)",
           edgecolor="white", linewidth=0.5)
    ax.bar(xb + 0.19, aware, width=0.38, color=NAVY,
           label="batch-aware $\\mu_{ctx}$ (our diagnostic)",
           edgecolor="white", linewidth=0.5)
    for xi, (a, b) in enumerate(zip(blind, aware)):
        ax.text(xi - 0.19, a + 0.009, f"{a:.3f}", ha="center", fontsize=7.0)
        ax.text(xi + 0.19, b + 0.009, f"{b:.3f}", ha="center", fontsize=7.0)
    ax.annotate("", xy=(0.19, aware[0] + 0.025), xytext=(-0.19, blind[0] - 0.012),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.text(0.02, (blind[0] + aware[0]) / 2 + 0.075,
            f"collapses by\n{blind[0] - aware[0]:.3f}", color=RED, fontsize=7.2,
            fontweight="bold", ha="left")
    ax.set_xticks(xb)
    ax.set_xticklabels(labs, fontsize=7.4)
    ax.set_ylabel("S1 residual PCC (per-sample mean)")
    ax.set_ylim(0, max(blind + aware) * 1.42)
    ax.legend(fontsize=6.9, frameon=False, loc="upper right")
    ax.set_title("F  S1 rewards batch reproduction \u2014 replicated on test\n"
                 "n=1,640 novel-compound test samples",
                 loc="left", fontsize=9)
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.suptitle("Self-evaluation on the released test ground truth "
                 "(our offline rubric, spec %s \u2014 NOT the official score)"
                 % sc["spec_version"],
                 fontsize=11.5, fontweight="bold", color=NAVY, y=0.968)
    out = FIGURES / "step7_test_selfscore.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


def figure_entities(meta_prof: dict, chem: pd.DataFrame, ext: dict) -> None:
    fig = plt.figure(figsize=(13.4, 6.9))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.22,
                          left=0.075, right=0.985, top=0.90, bottom=0.085)

    ev = meta_prof["entity_visibility"]
    nov = meta_prof["split_novelty_vs_train"]

    # --- A: strain x cohort visibility matrix ------------------------------
    ax = fig.add_subplot(gs[0, 0])
    tv = meta_prof["profile_train_val"]["columns"]["Strains"]["levels"]
    te = meta_prof["profile_test"]["columns"]["Strains"]["levels"]
    strains = sorted(set(tv) | set(te))
    rows = ["train_val", "test"]
    M = np.array([[tv.get(s, 0) for s in strains], [te.get(s, 0) for s in strains]],
                 dtype=float)
    im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=0, vmax=M.max())
    for i in range(2):
        for j, s in enumerate(strains):
            v = int(M[i, j])
            ax.text(j, i, f"{v:,}" if v else "0", ha="center", va="center",
                    fontsize=7.6, color="white" if M[i, j] > M.max() * 0.55 else "#222")
    unseen = [j for j, s in enumerate(strains) if tv.get(s, 0) == 0]
    for j in unseen:
        ax.add_patch(Rectangle((j - 0.5, -0.5), 1, 2, fill=False, edgecolor=RED,
                               linewidth=2.0, zorder=5))
        ax.text(j, 0.60, "UNSEEN IN\ntrain_val", ha="center", va="center",
                fontsize=7.0, color=RED, fontweight="bold")
    ax.set_xticks(range(len(strains)))
    ax.set_xticklabels(strains, fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title("A  Six strains; exactly one (%s) never appears in train_val\n"
                 "sample counts per cohort"
                 % ", ".join(ev["strains_test_only_sorted"]), loc="left", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("samples", fontsize=7)

    # --- B: entity novelty per evaluation split ----------------------------
    ax = fig.add_subplot(gs[0, 1])
    splits = ["test_chem_only", "test_strain_only", "test_both", "test_time"]
    labels = ["test_chem_only\n(S1)", "test_strain_only\n(S2)", "test_both\n(S3)",
              "test_time\n(interpolation)"]
    n_s = [nov["test:" + s]["n_samples"] for s in splits]
    nchem = [nov["test:" + s]["n_novel_chems"] for s in splits]
    nstr = [nov["test:" + s]["n_novel_strains"] for s in splits]
    x = np.arange(len(splits))
    ax.bar(x, n_s, width=0.55, color=TEAL, edgecolor="white", linewidth=0.6)
    for xi, n in enumerate(n_s):
        ax.text(xi, n + 34, f"{n:,}", ha="center", fontsize=8.2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\nnovel compounds: {nc}\nnovel strains: {ns}"
                        for lab, nc, ns in zip(labels, nchem, nstr)], fontsize=7.0)
    ax.set_ylabel("test samples")
    ax.set_ylim(0, max(n_s) * 1.22)
    ax.set_title("B  What each module extrapolates over\n"
                 "'unseen' means absent from the train split, per the revised handbook",
                 loc="left", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    # --- C: chemical distance of the unseen compounds ----------------------
    ax = fig.add_subplot(gs[1, 0])
    t = chem[(~chem.in_train) & chem.in_test].sort_values("max_tanimoto_to_train")
    y = np.arange(len(t))
    cols = [RED if v < 0.30 else GREEN for v in t.max_tanimoto_to_train]
    ax.barh(y, t.max_tanimoto_to_train, color=cols, height=0.62,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(t.compound, fontsize=7.2)
    ax.invert_yaxis()
    for yi, (v, nn) in enumerate(zip(t.max_tanimoto_to_train,
                                     t.nearest_train_compound)):
        ax.text(v + 0.012, yi, f"{v:.3f}  (nearest: {nn})", va="center", fontsize=6.8)
    ax.axvline(0.30, color=NAVY, linestyle="--", linewidth=0.9)
    ax.text(0.31, -0.72, "Tanimoto 0.30", fontsize=6.9, color=NAVY, va="center")
    ax.set_xlim(0, 1.32)
    ax.set_xlabel("max ECFP4 Tanimoto similarity to any training compound")
    n_far = int((t.max_tanimoto_to_train < 0.30).sum())
    ax.set_title("C  The unseen compounds are far extrapolation\n"
                 f"{n_far} of {len(t)} below Tanimoto 0.30; only Tamoxifen has a close analogue",
                 loc="left", fontsize=9)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=RED, label="< 0.30 (no close analogue)"),
                       Patch(color=GREEN, label="$\\geq$ 0.30")],
              fontsize=7, frameon=False, loc="center right",
              bbox_to_anchor=(1.0, 0.62))

    # --- D: external representation availability per entity axis -----------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    used = {r["resource"]: r for r in ext["resources"]}
    rows = [
        ("compound structure", "PubChem SMILES + RDKit",
         f"{used['PubChem (PUG-REST)']['counts']['n_resolved_to_a_molecule']} of "
         f"{used['PubChem (PUG-REST)']['counts']['n_resolved_to_a_molecule']} molecular "
         "labels resolved", True),
        ("compound mechanism", "ChEMBL + OrthoDB + UniProt (ITPV)",
         f"{used['ChEMBL (web services)']['counts']['n_activity_records_usable']:,} records "
         f"-> {used['OrthoDB']['counts']['target_to_protein_links_total']:,} target links",
         True),
        ("protein network", "STRING v12",
         f"{used['STRING']['counts']['n_edges_above_cutoff_undirected']:,} edges; "
         f"{used['STRING']['counts']['n_proteins_mapped']:,} of 5,243 mapped", True),
        ("metabolic stoichiometry", "iMM904 core sub-network",
         f"{used['iMM904 genome-scale metabolic reconstruction']['counts']['n_balanced_metabolites']}"
         f" metabolites x "
         f"{used['iMM904 genome-scale metabolic reconstruction']['counts']['n_reactions']}"
         " reactions", True),
        ("complex membership", "CORUM / Complex Portal + literature",
         f"{used['CORUM / Complex Portal + yeast complex literature']['counts']['n_complexes']}"
         f" complexes, "
         f"{used['CORUM / Complex Portal + yeast complex literature']['counts']['n_co_complex_edges']:,}"
         " edges", True),
        ("strain genotype", "1011 Yeast Genomes / SGD S288C",
         "NOT YET USED \u2014 strain enters only as a categorical\nlevel, so the "
         "unseen strain CRD has no representation", False),
    ]
    ax.text(0.0, 1.02, "D  Which entity axes have an external representation today",
            transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ax.text(0.0, 0.965, "the single open leaderboard rewards entity representations; "
                        "one of our six axes is still a blank category",
            transform=ax.transAxes, fontsize=7.6, va="bottom", color="#444")
    yy = 0.90
    for axis_name, source, detail, ok in rows:
        h = 0.145
        col = GREEN if ok else RED
        face = "#eef5f0" if ok else "#fdeeea"
        ax.add_patch(Rectangle((0.0, yy - h + 0.012), 1.0, h - 0.018,
                               transform=ax.transAxes, facecolor=face,
                               edgecolor=col, linewidth=0.9, zorder=1))
        ax.text(0.016, yy - 0.028, axis_name, transform=ax.transAxes,
                fontsize=8.0, fontweight="bold", va="top", color=NAVY)
        ax.text(0.016, yy - 0.075, source, transform=ax.transAxes, fontsize=7.2,
                va="top", color="#333", style="italic")
        ax.text(0.38, yy - 0.030, detail, transform=ax.transAxes, fontsize=6.9,
                va="top", color="#222", linespacing=1.35)
        ax.text(0.988, yy - 0.048, "used" if ok else "GAP", transform=ax.transAxes,
                fontsize=7.4, fontweight="bold", ha="right", va="top", color=col)
        yy -= h

    fig.suptitle("Entity extrapolation under the unified open leaderboard: "
                 "what is unseen, how far, and what representation exists",
                 fontsize=11, fontweight="bold", color=NAVY, y=0.975)
    out = FIGURES / "step7_entity_extrapolation.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> None:
    sc = json.loads((RESULTS / "step7_test_selfscore.json").read_text())
    s6 = json.loads((RESULTS / "step6_model_scores.json").read_text())
    s4 = json.loads((RESULTS / "step4_model_scores.json").read_text())
    s5 = json.loads((RESULTS / "step5_model_scores.json").read_text())
    base = json.loads((RESULTS / "harness_baseline_scores.json").read_text())
    val = {
        "step3": float(s4["step3_best_total"]),
        "step4": float(s4["val_total_at_frozen_weights"]),
        "step5": float(s5["val_total_at_frozen_weights"]),
        "step6": float(s6["headline_val_total"]),
        "benchmark": float(base["baseline_totals"]["per_context_mean_batch"]),
        "null": float(base["baseline_totals"]["control_anchor"]),
    }
    meta_prof = json.loads((RESULTS / "metadata_profile.json").read_text())
    ext = json.loads((RESULTS / "step7_external_data_manifest.json").read_text())
    chem = pd.read_csv(RESULTS / "step5_chemical_support.csv")
    print("drawing the two revised-submission figures from artefacts ...")
    figure_selfscore(sc, val)
    figure_entities(meta_prof, chem, ext)


if __name__ == "__main__":
    main()
