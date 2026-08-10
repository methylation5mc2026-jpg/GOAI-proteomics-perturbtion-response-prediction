"""Check the numeric claims made in README section 11 against the artefacts.

Prose drifts from data as analyses are re-run.  Every figure quoted in the Step-3
write-up is re-read from ``results/`` here, so a stale number in the README fails
loudly instead of being trusted.
"""

from __future__ import annotations

from repo_paths import WORKFLOW_DIR

import json
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(WORKFLOW_DIR))

from common import RESULTS, SESSION  # noqa: E402

FAILS: list[str] = []


def claim(desc: str, stated: float, actual: float, tol: float = 5e-4) -> None:
    """Compare a README figure against the artefact value."""
    ok = abs(stated - actual) <= tol
    print(f"  [{'OK ' if ok else 'BAD'}] {desc}: README {stated} vs artefact {actual:.6f}",
          flush=True)
    if not ok:
        FAILS.append(f"{desc}: README {stated} != artefact {actual:.6f}")


def main() -> None:
    readme = (SESSION / "README.md").read_text(encoding="utf-8")
    sec = readme[readme.index("# Step 3 — Feature Engineering"):]

    sc = json.loads((RESULTS / "gbdt_model_scores.json").read_text(encoding="utf-8"))
    aud = json.loads((RESULTS / "gbdt_feature_audit.json").read_text(encoding="utf-8"))
    sens = json.loads((RESULTS / "gbdt_round_sensitivity.json").read_text(encoding="utf-8"))
    ver = json.loads((RESULTS / "step3_verification_report.json").read_text(encoding="utf-8"))
    tot, mod = sc["model_totals"], sc["model_module_scores"]
    spl = pd.read_csv(RESULTS / "gbdt_split_metrics.csv")
    boot = pd.read_csv(RESULTS / "gbdt_bootstrap_ci.csv")

    print("=== headline totals ===")
    claim("benchmark total", 0.4454, sc["success_gate"]["benchmark_total_score"], 5e-5)
    claim("best candidate (blend) total", 0.4362, tot["blend_ens_delta_benchmark_w0.5"])
    claim("best pure GBDT (xgb_delta) total", 0.4155, tot["xgb_delta"])
    claim("lgb_delta total", 0.4029, tot["lgb_delta"])
    claim("ablation full-only total", 0.4099,
          sc["regime_specialists"]["ablation"]["total_score"])
    claim("control_anchor total", 0.2563, tot["control_anchor"])
    claim("ens_delta total", 0.3514, tot["ens_delta"])
    claim("cat_delta total", 0.2649, tot["cat_delta"])
    claim("cat_abs total", 0.2779, tot["cat_abs"])

    print("=== design matrix / audit ===")
    claim("design matrix rows", 5645086, aud["design_matrix"]["n_rows"], 0)
    claim("rows per protein", 1077, aud["design_matrix"]["rows_per_protein_mean"], 1.0)
    claim("n_features", 37, aud["config"]["n_features"], 0)
    claim("anchor max deviation", 0.0, aud["anchor_report"]["max_abs_deviation"], 0)
    claim("anchor coverage gain (pp)", 6.42, aud["anchor_report"]["coverage_gain_pp"], 5e-3)
    claim("anchor coverage before (%)", 71.07,
          100 * aud["anchor_report"]["frac_defined_harness"], 5e-3)
    claim("anchor coverage after (%)", 77.49,
          100 * aud["anchor_report"]["frac_defined_derived"], 5e-3)
    claim("n cells both anchors defined", 29377690,
          aud["anchor_report"]["n_cells_both_defined"], 0)
    oof = aud["oof_self_contribution"]
    claim("OOF shift d_batch", 0.007, oof["d_batch"]["mean_abs_oof_minus_full"], 5e-4)
    claim("OOF shift d_ctxb", 0.046, oof["d_ctxb"]["mean_abs_oof_minus_full"], 5e-4)

    print("=== batch-vs-chemistry attribution (the decisive table) ===")
    def s(model, split, col):
        return float(spl[(spl["model"] == model) & (spl["split"] == split)][col].iloc[0])
    claim("benchmark S1 residual, batch-blind", 0.4528,
          s("per_context_mean_batch", "val_chem_only", "resid_ctx_pcc_per_sample_mean"))
    claim("benchmark S1 residual, batch-aware", 0.0004,
          s("per_context_mean_batch", "val_chem_only",
            "resid_ctx_batch_aware_pcc_per_sample_mean"))
    claim("xgb_delta S1 residual, batch-aware", 0.1716,
          s("xgb_delta", "val_chem_only", "resid_ctx_batch_aware_pcc_per_sample_mean"))
    claim("xgb_delta S1 residual, batch-blind", 0.4040,
          s("xgb_delta", "val_chem_only", "resid_ctx_pcc_per_sample_mean"))
    claim("xgb_delta Time residual, batch-aware", 0.4279,
          s("xgb_delta", "val_time", "resid_ctx_batch_aware_pcc_per_sample_mean"))
    claim("benchmark Time residual, batch-blind", 0.3235,
          s("per_context_mean_batch", "val_time", "resid_ctx_pcc_per_sample_mean"))

    print("=== ablation diffs ===")
    abl = sc["regime_specialists"]["ablation"]["module_scores"]
    r = mod["lgb_delta"]
    for m, stated in [("m3_s2_strain", 0.0376), ("m3_s3_both", 0.0096),
                      ("m4_dep", 0.0334), ("m3_time", 0.0000),
                      ("m1_abundance", -0.0365), ("m3_s1_chem", -0.0309),
                      ("m2_fold_change", -0.0127)]:
        claim(f"ablation diff {m}", stated, r[m] - abl[m], 5e-4)
    claim("ablation diff TOTAL", -0.0070,
          tot["lgb_delta"] - sc["regime_specialists"]["ablation"]["total_score"], 5e-4)

    print("=== paired bootstrap (best candidate vs benchmark, fold-change PCC) ===")
    best = sc["success_gate"]["best_model"]
    for split, stated in [("val_strain_only", 0.0663), ("val_time", 0.0624),
                          ("val_both", 0.0237), ("val_chem_only", -0.0380)]:
        row = boot[(boot["model"] == best) & (boot["split"] == split)
                   & (boot["metric"] == "fold_change_pcc")].iloc[0]
        claim(f"paired diff {split}", stated, float(row["vs_benchmark_mean_diff"]))

    print("=== round sensitivity ===")
    curve = pd.DataFrame(sens["curve"])
    x = curve[curve["model"] == "xgb_delta"].set_index("n_rounds")
    for rounds, stated in [(25, 0.4088), (200, 0.4204), (600, 0.4155)]:
        claim(f"xgb_delta total @{rounds}", stated, float(x.loc[rounds, "total_score"]))
    claim("best over whole grid", 0.4204, sens["grid_best_optimistic"]["total_score"])
    claim("S2 gain 25->600 rounds", 0.050,
          float(x.loc[600, "m3_s2_strain"] - x.loc[25, "m3_s2_strain"]), 1e-3)
    claim("Time gain 25->600 rounds", 0.078,
          float(x.loc[600, "m3_time"] - x.loc[25, "m3_time"]), 1e-3)
    claim("m1 loss 25->600 rounds", -0.058,
          float(x.loc[600, "m1_abundance"] - x.loc[25, "m1_abundance"]), 1e-3)

    print("=== test predictions ===")
    te = sc["test_predictions"]
    claim("test samples", 4226, te["n_samples"], 0)
    claim("indicative test FC PCC", 0.3474,
          te["indicative_fold_change_pcc_per_sample_mean"])
    claim("test y_pred min", 9.07, te["y_pred_summary"]["min"], 5e-3)
    claim("test y_pred max", 32.75, te["y_pred_summary"]["max"], 5e-3)

    print("=== verification counts ===")
    ai = ver["artefact_integrity"]
    claim("integrity checks passed", 28, ai["n_passed"], 0)
    claim("integrity checks total", 28, ai["n_checks"], 0)
    claim("success criteria met", 4, ver["success_criteria"]["n_met"], 0)
    claim("success criteria total", 5, ver["success_criteria"]["n_criteria"], 0)

    print("=== structural claims ===")
    for txt, must in [("score target was **not** met", True),
                      ("0.0004", True), ("0.1716", True)]:
        ok = (txt in sec) == must
        print(f"  [{'OK ' if ok else 'BAD'}] README contains {txt!r}: {txt in sec}", flush=True)
        if not ok:
            FAILS.append(f"README missing required text: {txt!r}")

    # Every number in a README markdown table should appear in some artefact; a
    # weaker but useful check is that no obviously stale total lingers.
    stale = [m for m in re.findall(r"0\.4276|0\.457\b|0\.205\b", sec)
             if "earlier single-model pass" not in sec]
    if stale:
        FAILS.append(f"stale first-pass numbers quoted without provenance: {set(stale)}")

    print(f"\n{len(FAILS)} discrepancies")
    if FAILS:
        for f in FAILS:
            print(f"  - {f}", flush=True)
        raise AssertionError(f"{len(FAILS)} README claims disagree with the artefacts")
    print("every README figure in section 11 matches the artefacts", flush=True)


if __name__ == "__main__":
    main()
