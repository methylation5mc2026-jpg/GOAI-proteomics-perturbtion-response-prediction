"""Independent verification of the Step-2 harness artefacts.

Confirms the exported files are well formed, self-consistent, and internally
reproducible: JSON is strict (no bare NaN/Infinity), the weighted module
contributions re-derive each reported total, the CSV agrees with the JSON, and
the score spec still sums to 1.0.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import harness as H
from common import FIGURES, RESULTS

OK: list[str] = []
BAD: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (OK if cond else BAD).append(name if cond else f"{name}: {detail}")
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""),
          flush=True)


raw = (RESULTS / "harness_baseline_scores.json").read_text(encoding="utf-8")
# json.loads rejects bare NaN/Infinity when parse_constant raises.
def _reject(x):
    raise ValueError(f"non-standard JSON constant present: {x}")


try:
    payload = json.loads(raw, parse_constant=_reject)
    check("harness_baseline_scores.json is strict JSON (no bare NaN/Infinity)", True)
except ValueError as e:
    payload = json.loads(raw)
    check("harness_baseline_scores.json is strict JSON (no bare NaN/Infinity)", False, str(e))

check("score spec weights sum to 1.0",
      abs(sum(payload["score_spec"]["module_weights"].values()) - 1.0) < 1e-12)
check("spec version recorded matches the harness module",
      payload["score_spec"]["spec_version"] == H.SPEC_VERSION,
      f"json={payload['score_spec']['spec_version']} module={H.SPEC_VERSION}")

check("all sanity checks in the payload passed",
      all(payload["sanity_checks"].values()),
      str(payload["sanity_checks"]))

# Every reported total must equal the sum of its weighted module contributions,
# and equal weight-dot-module_scores recomputed from scratch.
for name, b in payload["baselines"].items():
    contrib_sum = sum(b["module_weighted_contributions"].values())
    check(f"{name}: total == sum(weighted contributions)",
          abs(contrib_sum - b["total_score"]) < 1e-9,
          f"{contrib_sum:.9f} vs {b['total_score']:.9f}")
    redot = sum(H.MODULE_WEIGHTS[m] * s for m, s in b["module_scores"].items())
    check(f"{name}: total == weights . module_scores",
          abs(redot - b["total_score"]) < 1e-9, f"{redot:.9f}")
    check(f"{name}: every module score lies in [0, 1]",
          all(0.0 <= v <= 1.0 for v in b["module_scores"].values()),
          str({k: round(v, 4) for k, v in b["module_scores"].items()}))

# CSV must agree with the JSON on every total and module score.
rep = pd.read_csv(RESULTS / "harness_validation_report.csv")
for name, b in payload["baselines"].items():
    tot = rep[(rep["model"] == name) & (rep["module"] == "TOTAL")]["value"]
    check(f"{name}: CSV TOTAL matches JSON",
          len(tot) == 1 and abs(float(tot.iloc[0]) - b["total_score"]) < 1e-9)
    for mod, sc in b["module_scores"].items():
        v = rep[(rep["model"] == name) & (rep["module"] == mod)
                & (rep["metric"] == "MODULE_SCORE")]["value"]
        if len(v) != 1 or abs(float(v.iloc[0]) - sc) < 1e-9:
            continue
        check(f"{name}/{mod}: CSV module score matches JSON", False,
              f"{float(v.iloc[0])} vs {sc}")
check("CSV module scores all match JSON", True)

# The residual-confound story must be reproduced by the exported CSV.
conf = pd.read_csv(RESULTS / "harness_residual_confound_sensitivity.csv")
row = conf[(conf["model"] == "per_context_mean_batch")
           & (conf["split"] == "val_chem_only")].iloc[0]
check("batch confound reproduced in CSV: batch-blind >> batch-aware",
      row["resid_pcc_mu_ctx_batch_blind"] > 0.30
      and row["resid_pcc_mu_ctx_batch_aware"] < 0.05,
      f"blind={row['resid_pcc_mu_ctx_batch_blind']:.4f} "
      f"aware={row['resid_pcc_mu_ctx_batch_aware']:.4f}")

split_df = pd.read_csv(RESULTS / "harness_split_metrics.csv")
check("split metrics cover 4 models x 4 evaluation splits",
      len(split_df) == 16 and split_df["split"].nunique() == 4
      and split_df["model"].nunique() == 4, f"{len(split_df)} rows")

# control_anchor predicts Delta == 0, so its fold-change PCC must be undefined
# (exported as null) rather than a number.
ca_fc = payload["baselines"]["control_anchor"]["modules"]["m2_fold_change"]["pcc_pooled"]
check("control_anchor fold-change PCC exported as null (undefined)", ca_fc is None,
      f"got {ca_fc!r}")

# Leakage report must still assert zero entity overlap on the OOD splits.
lr = payload["split_and_leakage_report"]
check("S1 has zero chemical overlap with train",
      lr["val_chem_only"]["chemicals_overlapping_train"] == 0)
check("S2 has zero strain overlap with train",
      lr["val_strain_only"]["strains_overlapping_train"] == 0)
check("S3 has zero strain and chemical overlap with train",
      lr["val_both"]["strains_overlapping_train"] == 0
      and lr["val_both"]["chemicals_overlapping_train"] == 0)
check("entity-OOD splits are 100% novel at the condition-cell level",
      all(lr[s]["frac_condition_cells_novel"] == 1.0
          for s in ("val_chem_only", "val_strain_only", "val_both")))
check("val_time is characterised as time-grid completion, not extrapolation",
      lr["time_semantics"]["unseen_time_values"] == []
      and lr["time_semantics"]["frac_rows_with_novel_condition_cell"] > 0.5,
      lr["time_semantics"]["generalisation_axis"])

# Baseline ranking must be stable across aggregation conventions (the one
# handbook ambiguity) -- otherwise model selection would depend on our guess.
sens = payload["aggregation_convention_sensitivity"]
convs = list(next(iter(sens.values())).keys())
rankings = {}
for c in convs:
    order = sorted(sens.keys(), key=lambda m: -sens[m][c]["total_score"])
    rankings[c] = order
best = {c: r[0] for c, r in rankings.items()}
check("the top baseline is the same under every aggregation convention",
      len(set(best.values())) == 1, str(best))
for c, r in rankings.items():
    print(f"      {c:22s} {' > '.join(r)}")

for f in ["harness_baseline_scores.json", "harness_validation_report.csv",
          "harness_split_metrics.csv", "harness_residual_confound_sensitivity.csv"]:
    p = RESULTS / f
    check(f"{f} exists and is non-empty", p.exists() and p.stat().st_size > 0,
          f"{p.stat().st_size / 1024:.1f} KB" if p.exists() else "missing")
for f in ["harness_baseline_scores.png", "harness_baseline_scores.pdf"]:
    p = FIGURES / f
    check(f"{f} exists and is non-empty", p.exists() and p.stat().st_size > 0,
          f"{p.stat().st_size / 1024:.1f} KB" if p.exists() else "missing")

print("\n" + "=" * 70)
print(f"Step-2 output verification: {len(OK)} passed, {len(BAD)} failed")
print("=" * 70)
if BAD:
    for b in BAD:
        print(f"  FAILED: {b}")
    sys.exit(1)
print("All Step-2 artefacts verified.")
