"""Independently verify the Step-3 artefacts, then register them in manifest.json.

Re-reads every declared output from disk and re-checks the claims made about it,
rather than trusting the producing script's own logging.  Any failure raises.
"""

from __future__ import annotations

from repo_paths import WORKFLOW_DIR

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(WORKFLOW_DIR))

import features as F  # noqa: E402
import harness as H  # noqa: E402
from common import RESULTS, SESSION  # noqa: E402

MANIFEST = SESSION / "manifest.json"
STEP = "Step 3 - Feature engineering and GBDT baselines"

NEW = [
    ("workflow/features.py",
     "Long-format feature engineering: 12 group-mean Delta tables, 2 abundance tables, "
     "6 protein statistics, out-of-fold encoding, leakage audits"),
    ("workflow/step3_data.py",
     "Shared Step-3 loading: independent control-anchor re-derivation, inner chemical-holdout split"),
    ("workflow/13b_probe_step3_inputs.py",
     "Bounded probe of metadata cardinalities and finite-cell budgets"),
    ("workflow/13c_smoke_test_features.py",
     "7 mechanical self-tests of features.py on a synthetic cohort"),
    ("workflow/14_feature_engineering.py",
     "Builds and audits the design matrix; writes the leakage-audit and coverage artefacts"),
    ("workflow/15_train_gbdt.py",
     "Inner-holdout hyper-parameter selection then 6 final fits (LightGBM/XGBoost/CatBoost x delta/abs)"),
    ("workflow/16_eval_gbdt.py",
     "Harness scoring, bootstrap CIs, detection sub-cohorts, batch attribution, test predictions"),
    ("data/gbdt_design_train.parquet",
     "Long-format training design matrix: one row per (train sample, protein) finite-Delta cell"),
    ("results/gbdt_feature_audit.json",
     "Anchor identity check, out-of-fold self-contribution, vocabularies, leakage assertions"),
    ("results/gbdt_feature_coverage.csv",
     "Per-split fraction of defined cells for every group-mean feature"),
    ("results/gbdt_tuning_trials.csv",
     "Every (config, n_iterations) candidate and its inner-dev objective"),
    ("results/gbdt_training_report.json",
     "Selected config, fit times, per-model top features, library versions"),
    ("results/gbdt_feature_importance.csv",
     "Gain-based feature importance for all 6 models (normalised to sum to 1 per model)"),
    ("results/gbdt_model_scores.json",
     "Full 5-module harness scores, success gate, bootstrap CIs, sub-cohorts, batch attribution"),
    ("results/gbdt_validation_report.csv",
     "Tidy one-row-per-metric table across all scored predictors"),
    ("results/gbdt_split_metrics.csv",
     "Per-split abundance / fold-change / residual PCC, incl. batch-aware residual"),
    ("results/gbdt_bootstrap_ci.csv",
     "95% bootstrap CIs per split/metric plus paired differences vs the benchmark"),
    ("results/gbdt_detection_subcohorts.csv",
     "Modules 2/3 re-scored on >=50% and >=90% train-detection protein sub-cohorts"),
    ("results/gbdt_test_predictions.csv",
     "Predicted log2 abundance for the 4,226 treated test samples x 5,243 proteins"),
    ("data/gbdt_test_predictions.parquet",
     "Compact mirror of the test abundance predictions"),
    ("data/gbdt_test_delta_predictions.parquet",
     "Predicted Delta for the treated test samples"),
    ("figures/gbdt_performance_comparison.png",
     "Module scores, total scores vs the 0.4454 benchmark, and per-split residual PCC"),
    ("figures/gbdt_feature_importance.png",
     "Top-22 feature importance and gain share by feature family per model"),
    ("figures/gbdt_feature_coverage.png",
     "Heat-map of group-mean feature availability by split"),
    ("workflow/18_probe_feature_availability.py",
     "Measures which features vanish on which OOD split; defines the regime mask groups"),
    ("workflow/19_round_sensitivity.py",
     "Truncates trained boosters at prediction time to isolate the round-count effect"),
    ("workflow/13d_smoke_test_eval.py",
     "6 self-tests of the reconstruction, ensembling and bootstrap functions"),
    ("results/gbdt_feature_availability.json",
     "Empirical per-split feature availability and the derived chem/strain mask groups"),
    ("results/gbdt_round_sensitivity.csv",
     "Total and per-module score for every model at 6 boosting-round truncations"),
    ("results/gbdt_round_sensitivity.json",
     "Round-sensitivity curve with the pre-registered vs best-on-val comparison"),
    ("figures/gbdt_round_sensitivity.png",
     "Score vs model capacity; shows no round count recovers the benchmark"),
    ("workflow/20_regen_figures.py",
     "Re-renders the Step-3 figures from persisted artefacts (restyling without recompute)"),
    ("workflow/21_audit_readme_claims.py",
     "Re-checks all 60 numeric claims in README section 11 against the artefacts"),
]

#: Artefact-integrity checks: a failure here is a *bug* and raises.
CHECKS: list[tuple[str, bool, str]] = []
#: Scientific success criteria: recorded and reported, but a failure is a
#: *result*, not a defect, so it must not be hidden behind an exception.
OUTCOMES: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record and print one artefact-integrity result (failure raises at the end)."""
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""),
          flush=True)


def outcome(name: str, met: bool, detail: str = "") -> None:
    """Record and print one scientific success criterion (never raises)."""
    OUTCOMES.append((name, bool(met), detail))
    print(f"  [{'MET' if met else 'NOT MET'}] {name}" + (f" -- {detail}" if detail else ""),
          flush=True)


def main() -> None:
    print("=== Step 3 output verification ===", flush=True)

    # --- 1. Existence and non-emptiness -----------------------------------
    missing = [rel for rel, _ in NEW if not (SESSION / rel).exists()]
    check("all declared outputs exist", not missing, f"missing: {missing}" if missing else
          f"{len(NEW)} files")
    empty = [rel for rel, _ in NEW if (SESSION / rel).exists()
             and (SESSION / rel).stat().st_size == 0]
    check("no declared output is empty", not empty, f"empty: {empty}" if empty else "")

    # --- 2. Feature audit -------------------------------------------------
    aud = json.loads((RESULTS / "gbdt_feature_audit.json").read_text(encoding="utf-8"))
    check("leakage guard (blanked eval rows) recorded as passing",
          aud.get("leak_guard_blanked_eval") is True)
    check("control anchor identical to Y - Delta where both defined",
          float(aud["anchor_report"]["max_abs_deviation"]) == 0.0,
          f"max|dev| = {aud['anchor_report']['max_abs_deviation']}")
    check("anchor coverage improved over Y - Delta",
          float(aud["anchor_report"]["coverage_gain_pp"]) > 0,
          f"+{aud['anchor_report']['coverage_gain_pp']:.2f} pp")
    oof = aud["oof_self_contribution"]
    engaged = sum(1 for k, v in oof.items()
                  if isinstance(v, dict) and (v.get("mean_abs_oof_minus_full") or 0) > 0)
    check("out-of-fold encoding active for every Delta table",
          engaged == len(F.DELTA_TABLE_KEYS), f"{engaged}/{len(F.DELTA_TABLE_KEYS)} tables")
    unseen = aud["unseen_level_mapping"]
    check("novel chemicals/strains are out-of-vocabulary",
          all(v["frac_mapped_to_UNSEEN"] == 1.0 for v in unseen.values()),
          f"{len(unseen)} split x column combinations")

    # --- 3. Feature coverage semantics ------------------------------------
    cov = pd.read_csv(RESULTS / "gbdt_feature_coverage.csv")
    piv = cov.pivot(index="feature", columns="split", values="frac_cells_defined")
    check("strain-keyed features are unavailable on novel-strain splits",
          float(piv.loc["d_strain", "val_strain_only"]) == 0.0
          and float(piv.loc["d_ctx", "val_strain_only"]) == 0.0)
    check("chemical-keyed features are unavailable on novel-chemical splits",
          float(piv.loc["d_drug", "val_chem_only"]) == 0.0
          and float(piv.loc["d_drug_time", "val_chem_only"]) == 0.0)
    check("batch-keyed features remain available on every split",
          bool((piv.loc["d_batch"] > 0.8).all()) and bool((piv.loc["d_plate"] > 0.7).all()))

    # --- 4. Scores --------------------------------------------------------
    sc = json.loads((RESULTS / "gbdt_model_scores.json").read_text(encoding="utf-8"))
    gate = sc["success_gate"]
    step2 = json.loads((RESULTS / "harness_baseline_scores.json").read_text(encoding="utf-8"))
    ref = float(step2["baseline_totals"]["per_context_mean_batch"])
    check("benchmark predictor reproduces its Step-2 total score",
          abs(float(gate["benchmark_total_score"]) - ref) < 1e-9,
          f"{gate['benchmark_total_score']:.9f} vs {ref:.9f}")
    outcome("SUCCESS CRITERION: best model total score > 0.4454",
            bool(gate["beats_benchmark"]),
            f"{gate['best_model']} = {gate['best_total_score']:.6f} "
            f"vs benchmark {gate['benchmark_total_score']:.6f} ({gate['margin']:+.6f})")
    check("weights in the scored spec still sum to 1",
          abs(sum(sc["score_spec"]["module_weights"].values()) - 1.0) < 1e-12)
    for nm, mods in sc["model_module_scores"].items():
        bad = [k for k, v in mods.items() if v is None or not (0.0 <= float(v) <= 1.0)]
        if bad:
            check(f"module scores in [0,1] for {nm}", False, f"out of range: {bad}")
            break
    else:
        check("every module score is a finite value in [0, 1]", True,
              f"{len(sc['model_module_scores'])} predictors")

    # Recompute the weighted total from the module scores as an arithmetic check.
    w = sc["score_spec"]["module_weights"]
    worst = 0.0
    for nm, mods in sc["model_module_scores"].items():
        recomputed = sum(w[k] * float(v) for k, v in mods.items())
        worst = max(worst, abs(recomputed - float(sc["model_totals"][nm])))
    check("total score equals sum(weight * module score) for every predictor",
          worst < 1e-9, f"max deviation {worst:.2e}")

    # --- 5. Bootstrap -----------------------------------------------------
    b = pd.read_csv(RESULTS / "gbdt_bootstrap_ci.csv")
    fin = b.dropna(subset=["mean", "ci_lo", "ci_hi"])
    check("bootstrap CIs bracket their point estimate",
          bool(((fin["ci_lo"] <= fin["mean"] + 1e-9)
                & (fin["mean"] <= fin["ci_hi"] + 1e-9)).all()),
          f"{len(fin)} intervals")
    best = gate["best_model"]
    pb = b[(b["model"] == best) & (b["metric"] == "fold_change_pcc")].dropna(
        subset=["vs_benchmark_ci_lo"])
    sig = pb[pb["vs_benchmark_ci_lo"] > 0]
    outcome("best model significantly beats the benchmark on >=3 of 4 splits "
            "(paired bootstrap CI excludes 0)",
            len(sig) >= 3,
            f"{len(sig)}/{len(pb)} splits: {sorted(sig['split'])}")

    # Regime routing was introduced to fix the novel-strain deficit; verify it did.
    spl = pd.read_csv(RESULTS / "gbdt_split_metrics.csv")
    abl = sc["regime_specialists"]["ablation"]
    if abl.get("module_scores"):
        s2_routed = float(sc["model_module_scores"][abl["family"]]["m3_s2_strain"])
        s2_full = float(abl["module_scores"]["m3_s2_strain"])
        outcome("regime routing improves the novel-strain (S2) module at matched capacity",
                s2_routed > s2_full,
                f"routed {s2_routed:.4f} vs full-only {s2_full:.4f} "
                f"({s2_routed - s2_full:+.4f})")

    # The batch-vs-chemistry attribution is the scientifically decisive table.
    bench_row = spl[(spl["model"] == "per_context_mean_batch")
                    & (spl["split"] == "val_chem_only")].iloc[0]
    outcome("benchmark's S1 residual score is essentially all batch, not chemistry",
            float(bench_row["resid_ctx_batch_aware_pcc_per_sample_mean"]) < 0.02,
            f"batch-blind {bench_row['resid_ctx_pcc_per_sample_mean']:.4f} -> "
            f"batch-aware {bench_row['resid_ctx_batch_aware_pcc_per_sample_mean']:.4f}")
    gbdt_rows = spl[(spl["model"] == "xgb_delta") & (spl["split"] == "val_chem_only")]
    if len(gbdt_rows):
        gr = gbdt_rows.iloc[0]
        outcome("a GBDT retains genuine non-batch signal on S1 where the benchmark has none",
                float(gr["resid_ctx_batch_aware_pcc_per_sample_mean"]) > 0.10,
                f"xgb_delta batch-aware residual PCC = "
                f"{gr['resid_ctx_batch_aware_pcc_per_sample_mean']:.4f}")

    # --- 6. Test predictions ----------------------------------------------
    req = pd.read_parquet(SESSION / "workflow" / "processed_delta_matrix_test.parquet",
                          columns=["sample_ID"])["sample_ID"].astype(str).tolist()
    pq = pd.read_parquet(SESSION / "data" / "gbdt_test_predictions.parquet")
    check("test predictions cover exactly the required sample_IDs, in order",
          pq["sample_ID"].astype(str).tolist() == req,
          f"{len(req)} samples")
    vals = pq.drop(columns=["sample_ID"]).to_numpy(dtype="float32")
    check("test prediction matrix has the full protein width",
          vals.shape[1] == 5243, f"shape {vals.shape}")
    check("no missing value in the test predictions", bool(np.isfinite(vals).all()),
          f"{int((~np.isfinite(vals)).sum())} non-finite cells")
    check("predicted abundances lie in a plausible log2 range",
          bool(vals.min() > 0 and vals.max() < 40),
          f"[{vals.min():.2f}, {vals.max():.2f}]")

    # The CSV is the declared deliverable; confirm it parses and agrees.
    head = pd.read_csv(RESULTS / "gbdt_test_predictions.csv", nrows=5)
    check("test-prediction CSV parses with the expected header",
          list(head.columns)[:1] == ["sample_ID"] and len(head.columns) == 5244,
          f"{len(head.columns)} columns")
    check("test-prediction CSV agrees with the parquet mirror",
          bool(np.allclose(head.drop(columns=["sample_ID"]).to_numpy(dtype="float32"),
                           vals[:5], atol=1e-4)))

    # --- 7. Importance ----------------------------------------------------
    imp = pd.read_csv(RESULTS / "gbdt_feature_importance.csv")
    # Importance is normalised per fitted booster, and there is one booster per
    # (model family, availability regime).
    key = ["model", "regime"] if "regime" in imp.columns else ["model"]
    sums = imp.groupby(key)["gain_share"].sum()
    check("feature importance sums to 1 per fitted booster",
          bool(np.allclose(sums.to_numpy(), 1.0, atol=1e-6)),
          f"{len(sums)} boosters, max dev {abs(sums - 1).max():.2e}")
    check("importance covers the full declared feature set",
          set(imp["feature"]) == set(F.FEATURE_NAMES),
          f"{imp['feature'].nunique()} of {len(F.FEATURE_NAMES)} features")

    # --- 8. JSON validity -------------------------------------------------
    def _reject_constant(c: str):
        """Raise on a bare NaN / Infinity literal.

        ``json.loads`` accepts those by default, so scanning the raw text used to
        be the workaround -- but that also flags the word "NaN" inside a perfectly
        valid *quoted string* (the score spec documents its NaN policy in prose).
        ``parse_constant`` fires only on the bare literals, which is exactly the
        RFC-8259 violation we care about.
        """
        raise ValueError(f"non-RFC-8259 bare constant: {c}")

    for rel in ["results/gbdt_model_scores.json", "results/gbdt_feature_audit.json",
                "results/gbdt_training_report.json", "results/gbdt_round_sensitivity.json",
                "results/gbdt_feature_availability.json"]:
        txt = (SESSION / rel).read_text(encoding="utf-8")
        try:
            json.loads(txt, parse_constant=_reject_constant)
            ok = True
        except (json.JSONDecodeError, ValueError) as e:
            ok = False
            print(f"    {rel}: {e}", flush=True)
        check(f"{rel} is strict RFC-8259 JSON", ok)

    # --- Summary ----------------------------------------------------------
    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    n_met = sum(1 for _, ok, _ in OUTCOMES if ok)
    print(f"\nartefact integrity : {n_pass}/{len(CHECKS)} checks passed", flush=True)
    print(f"success criteria   : {n_met}/{len(OUTCOMES)} met", flush=True)
    for n, ok, d in OUTCOMES:
        print(f"  [{'MET' if ok else 'NOT MET'}] {n}", flush=True)
    (RESULTS / "step3_verification_report.json").write_text(
        json.dumps({
            "artefact_integrity": {
                "n_checks": len(CHECKS), "n_passed": n_pass,
                "all_passed": n_pass == len(CHECKS),
                "note": "a failure here is a defect and raises",
                "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in CHECKS],
            },
            "success_criteria": {
                "n_criteria": len(OUTCOMES), "n_met": n_met,
                "all_met": n_met == len(OUTCOMES),
                "note": ("a failure here is a scientific result, not a defect; it is reported "
                         "rather than raised so it cannot be mistaken for a broken pipeline"),
                "criteria": [{"name": n, "met": ok, "detail": d} for n, ok, d in OUTCOMES],
            },
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote results/step3_verification_report.json", flush=True)

    update_manifest()

    failed = [n for n, ok, _ in CHECKS if not ok]
    if failed:
        raise AssertionError(f"{len(failed)} artefact-integrity checks FAILED: {failed}")


def update_manifest() -> None:
    """Register the Step-3 artefacts, preserving earlier entries."""
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    outputs = m.get("outputs", [])
    have = {o["relative_path"] for o in outputs}
    added = 0
    for rel, desc in NEW + [("results/step3_verification_report.json",
                             "Independent re-verification of every Step-3 artefact")]:
        p = SESSION / rel
        if not p.exists():
            raise FileNotFoundError(f"declared Step-3 output missing: {rel}")
        entry = {"path": str(p), "relative_path": rel, "description": desc,
                 "exists": True, "size_mb": round(p.stat().st_size / 1e6, 3),
                 "step": STEP}
        if rel in have:
            outputs = [entry if o["relative_path"] == rel else o for o in outputs]
        else:
            outputs.append(entry)
            added += 1

    sc = json.loads((RESULTS / "gbdt_model_scores.json").read_text(encoding="utf-8"))
    m["outputs"] = outputs
    m["step"] = STEP
    m["status"] = "completed"
    m["steps_completed"] = ["Step 1 - Data QC and EDA",
                            "Step 2 - Harness and local OOD cross-validation",
                            STEP]
    m["n_outputs"] = len(outputs)
    m["n_missing"] = sum(1 for o in outputs if not (SESSION / o["relative_path"]).exists())
    m["step3_summary"] = {
        "benchmark": sc["success_gate"]["benchmark_name"],
        "benchmark_total_score": sc["success_gate"]["benchmark_total_score"],
        "best_model": sc["success_gate"]["best_model"],
        "best_total_score": sc["success_gate"]["best_total_score"],
        "margin_over_benchmark": sc["success_gate"]["margin"],
        "target_score_met": bool(sc["success_gate"]["beats_benchmark"]),
        "n_model_families": len(sc["training_report"].get("fits", {})) // len(F.REGIMES)
        if sc.get("training_report", {}).get("fits") else None,
        "n_boosters_trained": len(sc.get("training_report", {}).get("fits", {})),
        "n_regimes": len(F.REGIMES),
        "n_features": len(F.FEATURE_NAMES),
        "score_spec_version": H.SPEC_VERSION,
        "note": ("target score not reached; see README section 11.6 for the "
                 "batch-vs-chemistry attribution that explains the gap"),
    }
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest.json: {added} new entries, {len(outputs)} total, "
          f"{m['n_missing']} missing", flush=True)


if __name__ == "__main__":
    main()
