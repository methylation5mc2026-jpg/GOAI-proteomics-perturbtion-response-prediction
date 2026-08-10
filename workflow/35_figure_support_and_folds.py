#!/usr/bin/env python3
"""Two real-data figures for the Step-5 manuscript.

figures/step5_chemical_support.png
    Applicability domain of the released split: the distribution of each compound's
    maximum ECFP4 Tanimoto similarity to any TRAINING compound, and the per-compound
    values for the eleven test-only compounds. This is the figure that bounds the
    novel-chemical axis independently of any model, so it is plotted from the measured
    similarities rather than asserted in prose.

figures/step5_lcgo_composition.png
    Realised composition of the five leave-chemical-group-out folds against the official
    test cohort's regime mix -- the check that the calibration cohort resembles the cohort
    the frozen weights are ultimately applied to.
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.REPO_ROOT
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# Official test-split counts, from the released metadata (quoted in 29_lcgo_oof_matrix.py).
TEST_MIX = {"full": 135, "chem_novel": 1640, "strain_novel": 1322, "both_novel": 1129}
REGIMES = ["full", "chem_novel", "strain_novel", "both_novel"]
COLS = {"full": "#4878a8", "chem_novel": "#e08a3c",
        "strain_novel": "#5f9e6e", "both_novel": "#a05195"}
SUPPORT_CUT = 0.30

# =========================================================== chemical support
sup = pd.read_csv(RESULTS / "step5_chemical_support.csv")
sup["max_tanimoto_to_train"] = pd.to_numeric(sup["max_tanimoto_to_train"])
test_only = sup[(~sup["in_train"].astype(bool)) & (sup["in_test"].astype(bool))]
in_train = sup[sup["in_train"].astype(bool)]

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0),
                         gridspec_kw={"width_ratios": [1.0, 1.35]})

ax = axes[0]
bins = np.arange(0, 1.0001, 0.05)
ax.hist(in_train["max_tanimoto_to_train"], bins=bins, color="#b8c6d1",
        edgecolor="#5a6b78", linewidth=0.6, label=f"in training (n={len(in_train)})")
ax.hist(test_only["max_tanimoto_to_train"], bins=bins, color="#c0392b",
        edgecolor="#7d2018", linewidth=0.6, alpha=0.85,
        label=f"test-only (n={len(test_only)})")
ax.axvline(SUPPORT_CUT, color="#333333", ls="--", lw=1.3)
ax.text(SUPPORT_CUT + 0.015, ax.get_ylim()[1] * 0.94,
        f"{SUPPORT_CUT:.2f}\nconvention", fontsize=8.5, va="top", color="#333333")
ax.set_xlabel("maximum ECFP4 Tanimoto similarity to any training compound")
ax.set_ylabel("compounds")
ax.set_title("A  Structural support is low for almost every compound", loc="left",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
t = test_only.sort_values("max_tanimoto_to_train")
y = np.arange(len(t))
colours = ["#c0392b" if v < SUPPORT_CUT else "#3d7a4f"
           for v in t["max_tanimoto_to_train"]]
ax.barh(y, t["max_tanimoto_to_train"], color=colours, edgecolor="#333333",
        linewidth=0.5, height=0.72)
ax.set_yticks(y)
ax.set_yticklabels(t["compound"], fontsize=9)
ax.axvline(SUPPORT_CUT, color="#333333", ls="--", lw=1.3)
for yy, (v, nn) in enumerate(zip(t["max_tanimoto_to_train"],
                                 t["nearest_train_compound"])):
    ax.text(v + 0.012, yy, f"{v:.3f}   nearest: {nn}", va="center", fontsize=8,
            color="#333333")
ax.set_xlim(0, 1.16)
ax.set_xlabel("maximum ECFP4 Tanimoto similarity to any training compound")
n_low = int((t["max_tanimoto_to_train"] < SUPPORT_CUT).sum())
ax.set_title(f"B  The eleven test-only compounds: {n_low} of {len(t)} are unsupported",
             loc="left", fontsize=11, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Applicability domain of the released novel-chemical split",
             fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(FIGS / "step5_chemical_support.png", dpi=155)
fig.savefig(FIGS / "step5_chemical_support.pdf")
plt.close(fig)
print(f"wrote step5_chemical_support.png  ({n_low}/{len(t)} test-only below "
      f"{SUPPORT_CUT})")

# ============================================================ LCGO composition
folds = json.loads((RESULTS / "step5_lcgo_folds.json").read_text(encoding="utf-8"))
comp = folds["composition"]
mix = folds["overall_regime_mix"]

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6),
                         gridspec_kw={"width_ratios": [1.25, 1.0, 0.9]})

# --- per-fold dev composition
ax = axes[0]
names = sorted(comp.keys())
bottom = np.zeros(len(names))
for r in REGIMES:
    vals = np.array([comp[n]["dev_by_regime"].get(r, 0) for n in names], dtype=float)
    ax.bar(names, vals, bottom=bottom, color=COLS[r], edgecolor="white",
           linewidth=0.7, label=r)
    bottom += vals
for i, n in enumerate(names):
    ax.text(i, bottom[i] + 14, f"{int(bottom[i])}", ha="center", fontsize=8.5)
ax.set_ylabel("held-out (dev) samples")
ax.set_title("A  Per-fold dev composition", loc="left", fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)

# --- fit-set size per fold
ax = axes[1]
fitn = [comp[n]["n_fit"] for n in names]
ax.bar(names, fitn, color="#8fa6b8", edgecolor="#4a5a68", linewidth=0.7)
ax.axhline(5078, color="#c0392b", ls="--", lw=1.4)
ax.text(len(names) - 0.45, 5078 + 40,
        "5,078 rows: the fit set of the\nmembers the weights are applied to",
        fontsize=8, ha="right", color="#c0392b")
for i, v in enumerate(fitn):
    ax.text(i, v + 40, f"{v:,}", ha="center", fontsize=8.5)
ax.set_ylim(0, 5600)
ax.set_ylabel("fit rows")
ax.set_title("B  Fit-set size, and the residual mismatch", loc="left", fontsize=11,
             fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

# --- pooled OOF mix vs official test mix
ax = axes[2]
tot_o = sum(mix.values())
tot_t = sum(TEST_MIX.values())
x = np.arange(2)
bottom = np.zeros(2)
for r in REGIMES:
    vals = np.array([100.0 * mix.get(r, 0) / tot_o, 100.0 * TEST_MIX[r] / tot_t])
    ax.bar(x, vals, bottom=bottom, color=COLS[r], edgecolor="white", linewidth=0.7)
    for xi in range(2):
        if vals[xi] > 4.5:
            ax.text(xi, bottom[xi] + vals[xi] / 2, f"{vals[xi]:.1f}%", ha="center",
                    va="center", fontsize=8.5, color="white", fontweight="bold")
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels([f"pooled LCGO\n(n={tot_o:,})", f"official test\n(n={tot_t:,})"],
                   fontsize=9.5)
ax.set_ylabel("share of samples (%)")
ax.set_title("C  Calibration vs deployment mix", loc="left", fontsize=11,
             fontweight="bold")
ax.set_ylim(0, 100)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Realised five-fold leave-chemical-group-out cross-fitting design",
             fontsize=13, fontweight="bold", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.945))
fig.savefig(FIGS / "step5_lcgo_composition.png", dpi=155)
fig.savefig(FIGS / "step5_lcgo_composition.pdf")
plt.close(fig)
print("wrote step5_lcgo_composition.png")
print(f"  pooled both_novel share {100.0 * mix['both_novel'] / tot_o:.1f}% "
      f"vs test {100.0 * TEST_MIX['both_novel'] / tot_t:.1f}%")
