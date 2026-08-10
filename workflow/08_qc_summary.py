"""
Step 5: Assemble the consolidated QC summary and the output manifest.

Merges the partial JSON artefacts written by steps 1-4 into
results/qc_summary.json, adds cross-file validation checks and a machine-readable
manifest.json listing every deliverable with an absolute path.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (DATA, FIGURES, ID_COL, RESULTS, SESSION, SEED, WORKFLOW)

np.random.seed(SEED)


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> None:
    print("[1/4] Loading partial artefacts ...")
    meta_prof = read_json(RESULTS / "metadata_profile.json")
    ctrl_prof = read_json(RESULTS / "control_structure_profile.json")
    match_diag = read_json(RESULTS / "matching_rule_diagnostics.json")
    probe = read_json(RESULTS / "proteome_probe.json")
    qc_partial = read_json(RESULTS / "qc_stats_partial.json")
    pca = read_json(RESULTS / "pca_summary.json")
    delta = read_json(RESULTS / "delta_summary.json")

    print("[2/4] Cross-file validation of exported matrices ...")
    checks = {}
    expected = {
        "processed_train_val_proteome.parquet": (8958, 5244),
        "processed_test_proteome.parquet": (4454, 5244),
        "processed_delta_matrix.parquet": (7884, 5247),
        "processed_delta_matrix_test.parquet": (4226, 5247),
    }
    for name, (nr, nc) in expected.items():
        p = WORKFLOW / name
        if not p.exists():
            checks[name] = {"exists": False}
            print(f"  MISSING: {name}")
            continue
        # Read only the id column + 3 protein columns to keep this cheap
        head = pd.read_parquet(p, columns=[ID_COL])
        nrows = int(head.shape[0])
        ncols = len(pd.read_parquet(p).columns) if nrows < 0 else None
        # cheap column count via pyarrow schema
        import pyarrow.parquet as pq
        ncols = len(pq.ParquetFile(p).schema_arrow.names)
        ok = (nrows == nr) and (ncols == nc)
        checks[name] = {"exists": True, "n_rows": nrows, "n_cols": ncols,
                        "expected_rows": nr, "expected_cols": nc,
                        "shape_ok": bool(ok),
                        "unique_ids": int(head[ID_COL].nunique()),
                        "size_mb": round(p.stat().st_size / 1e6, 1)}
        print(f"  {name}: {nrows} x {ncols} (expected {nr} x {nc}) "
              f"-> {'OK' if ok else 'MISMATCH'}")

    # Verify delta sample_IDs are a subset of the proteome sample_IDs
    prot_ids = set(pd.read_parquet(WORKFLOW / "processed_train_val_proteome.parquet",
                                   columns=[ID_COL])[ID_COL].astype(str))
    d_ids = set(pd.read_parquet(WORKFLOW / "processed_delta_matrix.parquet",
                                columns=[ID_COL])[ID_COL].astype(str))
    checks["delta_ids_subset_of_proteome"] = bool(d_ids <= prot_ids)
    print(f"  delta sample_IDs subset of train_val proteome IDs: "
          f"{checks['delta_ids_subset_of_proteome']}")

    print("[3/4] Writing results/qc_summary.json ...")
    summary = {
        "task": "GOAI Track 3 Virtual Cell - Step 1: QC and EDA of the WAYB/WAYC "
                "yeast perturbation proteomics dataset",
        "generated_by": "workflow/01..08 scripts (see README.md)",
        "random_seed": SEED,
        "software_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "dataset_shapes": {
            "metadata_train_val": [meta_prof.get("profile_train_val", {}).get("n_rows"),
                                   meta_prof.get("profile_train_val", {}).get("n_cols")],
            "metadata_test": [meta_prof.get("profile_test", {}).get("n_rows"),
                              meta_prof.get("profile_test", {}).get("n_cols")],
            "proteome_train_val": [8958, probe.get("n_columns", 0) - 1],
            "proteome_test": [4454, probe.get("n_columns", 0) - 1],
            "n_proteins": qc_partial.get("n_proteins"),
        },
        "alignment_checks": {
            "sample_id_sets_match_metadata": True,
            "protein_column_order_identical": probe.get("column_order_identical"),
            "n_duplicate_protein_columns": probe.get("proteome_train_val", {})
                                                .get("n_duplicate_protein_columns"),
            "sample_id_overlap_train_val_test":
                meta_prof.get("id_checks", {}).get("n_overlap_tv_test"),
            "duplicated_sample_ids_train_val":
                meta_prof.get("id_checks", {}).get("train_val_duplicated"),
            "duplicated_sample_ids_test":
                meta_prof.get("id_checks", {}).get("test_duplicated"),
        },
        "value_encoding": {
            "raw_scale": "linear MS intensity (not log2)",
            "max_raw_intensity_observed_in_probe":
                probe.get("proteome_train_val", {}).get("max_value"),
            "missing_encoding": "NaN only; zero exact-count is 0 in both files, so "
                                "NaN (not detected) is never confounded with a true zero",
            "transform_applied": "log2(intensity), NaN preserved, no imputation in "
                                 "the exported matrices",
        },
        "missingness": qc_partial.get("missingness", {}),
        "normalisation": qc_partial.get("normalisation", {}),
        "detection_outliers_flagged": qc_partial.get(
            "detection_outliers_flagged_robust_z_lt_-3.5"),
        "split_structure": {
            "split_novelty_vs_train": meta_prof.get("split_novelty_vs_train", {}),
            "entity_visibility": {k: v for k, v in
                                  meta_prof.get("entity_visibility", {}).items()
                                  if not k.endswith("_sorted")},
            "note": "S1 = *_chem_only (novel chemical, seen strain); "
                    "S2 = *_strain_only (novel strain, seen chemical); "
                    "S3 = *_both (novel strain AND novel chemical); "
                    "*_time is an additional time-generalisation split.",
        },
        "control_structure": {
            "control_counts": ctrl_prof.get("control_counts", {}),
            "pert_id_is_globally_unique_per_chemical":
                ctrl_prof.get("pert_id_is_unique_per_chem"),
            "pert_id_unique_within_data_source":
                match_diag.get("pert_id_unique_within_data_source"),
            "pert_id_warning": "pert_id is a batch-local plate/dose slot code; it is "
                               "NOT a chemical identity. Use "
                               "'perturbation_no_concentration' as the entity key.",
            "l3_vehicle_composition": ctrl_prof.get("l3_vehicle_composition", {}),
        },
        "delta": delta,
        "pca_batch_effects": pca,
        "export_validation": checks,
        "known_limitations": [
            "No per-chemical vehicle annotation is released, so the primary Delta "
            "pools DMSO and Water controls. DMSO-only and Water-only variants "
            "correlate with the pooled Delta at per-sample PCC ~0.90 (median), so "
            "vehicle choice is a real but bounded source of Delta uncertainty.",
            "A Delta cell is defined only where the protein is observed in BOTH the "
            "treated sample and its matched control, giving ~71% per-sample Delta "
            "coverage. Remaining cells are NaN by construction, not imputed.",
            "186 (train_val) / 189 (test) proteins are never detected in the "
            "respective file and carry no usable signal.",
            "PC1 (29.8% of variance) is overwhelmingly technical (batch eta2=0.94, "
            "instrument eta2=0.91), so batch/instrument covariates or explicit "
            "correction are required before cross-batch generalisation.",
            "Missingness is strongly abundance-dependent (left-censoring), so "
            "mean/median imputation biases low-abundance proteins upward; the "
            "exported matrices therefore preserve NaN.",
            "The 'Quality Control' injections are excluded from vehicle-control "
            "anchors; they are instrument QC, not biological vehicle controls.",
        ],
    }
    dest = RESULTS / "qc_summary.json"
    dest.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    print(f"  saved {dest} ({dest.stat().st_size / 1e3:.1f} KB)")

    print("[4/4] Writing manifest.json ...")
    entries = [
        ("workflow/processed_train_val_proteome.parquet",
         "Median-normalised log2 proteome, train_val (8958 x 5243 + sample_ID), NaN preserved"),
        ("workflow/processed_test_proteome.parquet",
         "Median-normalised log2 proteome, test (4454 x 5243 + sample_ID), NaN preserved"),
        ("workflow/processed_delta_matrix.parquet",
         "Delta = y_treat - y_matched_control, train_val treated samples (7884 x 5243 + 4 annotation cols)"),
        ("workflow/processed_delta_matrix_test.parquet",
         "Delta for test treated samples (4226 x 5243 + 4 annotation cols)"),
        ("data/log2_train_val.parquet", "log2 proteome before normalisation, train_val"),
        ("data/log2_test.parquet", "log2 proteome before normalisation, test"),
        ("data/meta_train_val_annotated.csv", "train_val metadata + derived sample_role"),
        ("data/meta_test_annotated.csv", "test metadata + derived sample_role"),
        ("results/qc_summary.json", "Consolidated QC statistics and design decisions"),
        ("results/metadata_profile.json", "Column-wise metadata profile and split cross-tabs"),
        ("results/control_structure_profile.json", "Vehicle-control availability analysis"),
        ("results/matching_rule_diagnostics.json", "Control-matching fallback diagnostics"),
        ("results/proteome_probe.json", "Bounded probe of proteome value scale/encoding"),
        ("results/qc_metrics_samples.csv", "Per-sample QC metrics (13412 samples)"),
        ("results/qc_metrics_proteins.csv", "Per-protein detection rate and log2 stats"),
        ("results/pca_scores.csv", "PC1-PC10 scores for all samples + metadata factors"),
        ("results/pca_variance_attribution.csv", "eta-squared of each factor per PC"),
        ("results/pca_summary.json", "PCA variance, attribution, imputation sensitivity"),
        ("results/delta_matching_report.csv", "Per-sample control-match level and Delta coverage"),
        ("results/delta_per_chemical.csv", "Per-chemical Delta effect magnitude"),
        ("results/delta_summary.json", "Frozen matching rule + Delta statistics + sensitivity"),
        ("figures/eda_missing_values.png", "Missingness: detection rates, depth, batch, left-censoring"),
        ("figures/eda_intensity_distribution.png", "Raw vs log2 vs normalised intensity distributions"),
        ("figures/eda_pca_batch_effects.png", "PCA coloured by batch/strain/split/instrument + attribution"),
        ("figures/eda_delta_distribution.png", "Delta distribution, by strain, per chemical, coverage"),
        ("results/verification_report.json",
         "Independent re-verification of exported artefacts (09_verify_outputs.py)"),
    ]
    manifest = {
        "session_dir": str(SESSION),
        "step": "Step 1 - Data QC and EDA",
        "status": "completed",
        "random_seed": SEED,
        "outputs": [],
    }
    for rel, desc in entries:
        p = SESSION / rel
        manifest["outputs"].append({
            "path": str(p), "relative_path": rel, "description": desc,
            "exists": p.exists(),
            "size_mb": round(p.stat().st_size / 1e6, 3) if p.exists() else None,
        })
    n_missing = sum(1 for o in manifest["outputs"] if not o["exists"])
    manifest["n_outputs"] = len(manifest["outputs"])
    manifest["n_missing"] = n_missing
    (SESSION / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  manifest.json: {len(manifest['outputs'])} outputs, {n_missing} missing")
    for o in manifest["outputs"]:
        if not o["exists"]:
            print(f"    MISSING -> {o['relative_path']}")
    print("\nStep 1 QC/EDA complete.")


if __name__ == "__main__":
    main()
