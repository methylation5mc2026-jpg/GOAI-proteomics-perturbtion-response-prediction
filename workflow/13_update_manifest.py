"""Append the Step-2 artefacts to manifest.json, preserving Step-1 entries."""

from __future__ import annotations

import json

from common import SESSION

MANIFEST = SESSION / "manifest.json"

NEW = [
    ("workflow/harness.py",
     "Official 5-dimensional scoring suite: Modules 1-4 + compute_competition_score"),
    ("workflow/validation_splits.py",
     "Local OOD split extraction, leakage assertions, train-frozen mu_ctx / mu_drug baselines"),
    ("workflow/10_eval_baselines.py",
     "Scores 4 trivial baselines through the harness; writes all Step-2 results"),
    ("workflow/11_test_harness.py",
     "57 self-tests for harness.py (vs scipy/sklearn references and analytic values)"),
    ("workflow/12_verify_step2_outputs.py",
     "36 independent checks that the exported Step-2 artefacts are valid and self-consistent"),
    ("results/harness_baseline_scores.json",
     "Score spec, leakage report, baseline diagnostics, full metric tree, convention sensitivity"),
    ("results/harness_validation_report.csv",
     "Tidy one-row-per-metric table: 4 baselines x 7 modules x all sub-metrics (580 rows)"),
    ("results/harness_split_metrics.csv",
     "Per-split abundance / fold-change / residual PCC for each baseline (16 rows)"),
    ("results/harness_residual_confound_sensitivity.csv",
     "S1/S2/S3/Time residual PCC under batch-blind vs batch-aware mu_ctx (16 rows)"),
    ("figures/harness_baseline_scores.png",
     "Module-score comparison across the 4 trivial baselines"),
    ("figures/harness_baseline_scores.pdf",
     "Vector version of the baseline module-score comparison"),
]


def main() -> None:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    outputs = m.get("outputs", [])
    have = {o["relative_path"] for o in outputs}

    added = 0
    for rel, desc in NEW:
        p = SESSION / rel
        if not p.exists():
            raise FileNotFoundError(f"declared Step-2 output missing: {rel}")
        entry = {
            "path": str(p),
            "relative_path": rel,
            "description": desc,
            "exists": True,
            "size_mb": round(p.stat().st_size / 1e6, 3),
            "step": "Step 2 - Harness and local OOD cross-validation",
        }
        if rel in have:
            outputs = [entry if o["relative_path"] == rel else o for o in outputs]
        else:
            outputs.append(entry)
            added += 1

    m["outputs"] = outputs
    m["step"] = "Step 2 - Harness and local OOD cross-validation"
    m["status"] = "completed"
    m["steps_completed"] = ["Step 1 - Data QC and EDA",
                            "Step 2 - Harness and local OOD cross-validation"]
    m["n_outputs"] = len(outputs)
    m["n_missing"] = sum(1 for o in outputs if not (SESSION / o["relative_path"]).exists())
    m["score_spec_version"] = "1.0.0"

    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest.json: {added} new entries, {len(outputs)} total, "
          f"{m['n_missing']} missing", flush=True)


if __name__ == "__main__":
    main()
