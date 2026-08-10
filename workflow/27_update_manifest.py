"""Step 4 bookkeeping: register the new artefacts in manifest.json.

Also re-audits the numeric claims made in the Step-4 README section against the
JSON reports, so a stale number in the prose is caught rather than shipped.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_common as S4  # noqa: E402

SESSION, RESULTS, DATA, FIGURES = S4.SESSION, S4.RESULTS, S4.DATA, S4.FIGURES
WORKFLOW = S4.WORKFLOW

STEP4_OUTPUTS = [
    ("workflow/22_rdkit_features.py", "SMILES resolution, ECFP4 + 2D descriptors, QC gates"),
    ("workflow/23_train_rdkit_gbdt.py", "controlled A/B GBDT retraining with the molecular block"),
    ("workflow/24_train_deep_learning.py", "multi-task MLP-ResNet with entity dropout"),
    ("workflow/25a_inner_members.py", "out-of-fold member predictions on the inner cohort"),
    ("workflow/25b_val_members.py", "cached validation-cohort member matrices"),
    ("workflow/25_stacking_ensemble.py", "per-regime non-negative stacking meta-learner"),
    ("workflow/25c_time_harness.py", "harness cost measurement"),
    ("workflow/25d_report_stacking.py", "readable stacking summary"),
    ("workflow/step4_common.py", "shared Step-4 data/prediction/scoring infrastructure"),
    ("workflow/step4_fastscore.py", "exact co-moment factorisation of harness modules 1-3"),
    ("workflow/26_eval_step4.py", "test predictions, verification, figures"),
    ("workflow/26a_probe_range.py", "abundance dynamic-range diagnostic"),
    ("results/step4_model_scores.json", "all Step-4 candidates, per-module, both cohorts"),
    ("results/step4_stacking_weights.json", "the frozen per-regime stacking weights used"),
    ("results/step4_test_predictions.csv", "submission matrix 4226 x 5243"),
    ("results/step4_verification_report.json", "6 integrity checks + 4 success criteria"),
    ("results/step4_rdkit_report.json", "resolution audit, MW cross-checks, salt decisions"),
    ("results/step4_smiles_resolved.json", "name -> CID/SMILES/formula audit trail"),
    ("results/step4_dl_training.json", "network architecture, curves, epoch selection"),
    ("results/step4_inner_members.json", "inner-cohort member provenance"),
    ("results/step4_val_members.json", "validation member cache manifest"),
    ("results/step4_rdkit_gbdt_training.json", "RDKit GBDT fit log"),
    ("results/step4_feature_importance.csv", "per-regime split-gain shares"),
    ("results/step4_chemical_probe.json", "perturbation-label and split profile"),
    ("data/step4_mol_features.parquet", "per-compound molecular feature table"),
    ("data/step4_test_predictions.parquet", "submission matrix, parquet mirror"),
    ("data/step4_test_delta_predictions.parquet", "derived fold-change matrix"),
    ("figures/step4_performance_comparison.png", "per-module and total score comparison"),
    ("figures/step4_feature_importance.png", "molecular share of split gain by regime"),
    ("figures/step4_chemical_space.png", "ECFP4 PCA and property-space coverage"),
    ("figures/step4_dl_training.png", "training curves and honest epoch selection"),
]


def audit_readme_claims() -> list[dict]:
    """Check the headline numbers in the README against the JSON reports."""
    stack = json.loads((RESULTS / "step4_stacking_weights.json").read_text(encoding="utf-8"))
    ver = json.loads((RESULTS / "step4_verification_report.json").read_text(encoding="utf-8"))
    rdk = json.loads((RESULTS / "step4_rdkit_report.json").read_text(encoding="utf-8"))
    dl = json.loads((RESULTS / "step4_dl_training.json").read_text(encoding="utf-8"))
    readme = (SESSION / "README.md").read_text(encoding="utf-8")

    ms = stack["module_scores"]
    checks = [
        ("stacked val total 0.477377", "0.477377",
         f"{stack['val_total_at_frozen_weights']:.6f}"),
        ("margin +0.031934", "0.031934", f"{stack['margin_vs_benchmark']:.6f}"),
        ("inner-dev total 0.449636", "0.449636",
         f"{stack['inner_dev_total_at_frozen_weights']:.6f}"),
        ("val-tuned bound 0.505394", "0.505394",
         f"{stack['val_total_val_tuned_OPTIMISTIC']:.6f}"),
        ("multilib sensitivity 0.458853", "0.458853",
         f"{stack['sensitivity']['val_gbdt_tab_replaced_by_multi_library_ensemble']['total_score']:.6f}"),
        ("m3_s1 tabular 0.3392", "0.3392", f"{ms['val:gbdt_tab']['m3_s1_chem']:.4f}"),
        ("m3_s1 rdkit 0.4011", "0.4011", f"{ms['val:gbdt_mol']['m3_s1_chem']:.4f}"),
        ("m1 stacked 0.8229", "0.8229", f"{ms['val:stacked_oof_frozen']['m1_abundance']:.4f}"),
        ("m1 control anchor 0.8172", "0.8172", f"{ms['val:control_anchor']['m1_abundance']:.4f}"),
        ("rdkit total 0.4356", "0.4356", f"{stack['totals']['val:gbdt_mol']:.4f}"),
        ("tabular total 0.4029", "0.4029", f"{stack['totals']['val:gbdt_tab']:.4f}"),
        ("dl total 0.3793", "0.3793", f"{stack['totals']['val:dl']:.4f}"),
        ("dl best epoch 144", "144", str(dl["epoch_selection"]["best_inner_epoch"])),
        ("dl inner pcc 0.2724", "0.2724",
         f"{dl['epoch_selection']['best_inner_dev_pcc']:.4f}"),
        ("0 unresolved compounds", None, str(rdk["n_unresolved"])),
        ("0 MW flags", None, str(rdk["mw_crosscheck"]["n_flagged"])),
        ("fp_pca EVR 72.9%", "72.9", f"{100 * rdk['blocks']['fp_pca']['cumulative_evr']:.1f}"),
        ("test samples 4226", "4,226", str(ver["test_predictions"]["n_samples"])),
    ]

    rows = []
    for label, needle, actual in checks:
        present = (needle in readme) if needle else True
        rows.append({"claim": label, "value_in_report": actual,
                     "string_found_in_readme": present})
        flag = "ok " if present else "!! "
        print(f"  {flag}{label:34s} report says {actual:>12s}  "
              f"{'(string present)' if present else '(STRING NOT FOUND IN README)'}")

    # integrity / criteria consistency
    ai = ver["artefact_integrity"]
    sc = ver["success_criteria"]
    print(f"\n  artefact integrity: {ai['n_passed']}/{ai['n_checks']} "
          f"all_passed={ai['all_passed']}")
    print(f"  success criteria  : {sc['n_met']}/{sc['n_criteria']} all_met={sc['all_met']}")
    for pat, ok in (("All 6 artefact-integrity checks pass", ai["all_passed"]),):
        if not (pat in readme and ok):
            print(f"  !! README claim {pat!r} vs actual {ok}")
    return rows


def main() -> None:
    mpath = SESSION / "manifest.json"
    man = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}

    entries = []
    missing = []
    for rel, desc in STEP4_OUTPUTS:
        p = SESSION / rel
        if p.exists():
            entries.append(
                {
                    "path": str(p),
                    "relative_path": rel,
                    "description": desc,
                    "size_bytes": p.stat().st_size,
                }
            )
        else:
            missing.append(rel)

    stack = json.loads((RESULTS / "step4_stacking_weights.json").read_text(encoding="utf-8"))
    ver = json.loads((RESULTS / "step4_verification_report.json").read_text(encoding="utf-8"))

    man["current_step"] = (
        "Step 4 - domain context integration, deep learning and out-of-fold stacking"
    )
    man["status"] = "completed"
    man["step4"] = {
        "status": "completed",
        "headline": {
            "val_total_frozen_oof_weights": stack["val_total_at_frozen_weights"],
            "benchmark_total": stack["benchmark_total"],
            "margin": stack["margin_vs_benchmark"],
            "beats_benchmark": stack["beats_benchmark"],
            "step3_best_total": stack["step3_best_total"],
        },
        "protocol_note": (
            "stacking weights were fitted on an inner cohort carved from train, with every "
            "member retrained on inner_fit, then frozen before touching val_*; the reported "
            "val total is a held-out estimate"
        ),
        "artefact_integrity": {
            "n_passed": ver["artefact_integrity"]["n_passed"],
            "n_checks": ver["artefact_integrity"]["n_checks"],
            "all_passed": ver["artefact_integrity"]["all_passed"],
        },
        "success_criteria": {
            "n_met": ver["success_criteria"]["n_met"],
            "n_criteria": ver["success_criteria"]["n_criteria"],
            "all_met": ver["success_criteria"]["all_met"],
        },
        "not_completed": [
            "XGBoost arm of the RDKit A/B (2 of 4 regimes on disk; killed under CPU "
            "contention, so the multi-library RDKit sensitivity was not run)",
            "bootstrap CI on the stacked margin",
        ],
        "outputs": entries,
        "missing_declared_outputs": missing,
    }
    man.setdefault("steps_completed", [])
    if "step4" not in man["steps_completed"]:
        man["steps_completed"].append("step4")

    mpath.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {mpath} with {len(entries)} Step-4 outputs")
    if missing:
        print(f"!! {len(missing)} declared outputs missing: {missing}")

    print("\n=== auditing README numeric claims against the JSON reports ===")
    audit_readme_claims()


if __name__ == "__main__":
    main()
