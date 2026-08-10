#!/usr/bin/env python
"""Step 6.6 -- refresh manifest.json with the Step-6 artefacts and headline.

Every declared output is stat'ed on disk, so ``exists`` and ``size_mb`` are
measured rather than asserted. A declared output that is missing is recorded as
missing rather than quietly dropped.
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import json
import sys
import time
from pathlib import Path

SESSION = REPO_ROOT
RESULTS = SESSION / "results"

STEP6_OUTPUTS: list[tuple[str, str]] = [
    ("workflow/34_target_affinity_features.py",
     "ChEMBL target affinity (MoA) vector mapping: compound -> ChEMBL -> UniProt -> yeast ortholog"),
    ("workflow/34b_validate_itpv_mechanisms.py",
     "Independent mechanistic validation of the ITPV ortholog mapping (positive + negative controls)"),
    ("workflow/35_metabolic_mechanism_loss.py",
     "iMM904 flux-conservation and protein-complex co-response PyTorch loss (MechanismLoss)"),
    ("workflow/36_dynamic_cross_attention_gnn.py",
     "Dynamic Bio-Chemical Cross-Attention Graph Transformer, cross-fitted over the 5 LCGO folds"),
    ("workflow/37_hierarchical_cluster_stacking.py",
     "Hierarchical nested K=4/8/16/32 non-negative ridge stacking meta-learner"),
    ("workflow/38_eval_step6.py",
     "Compute scaling assessment, final test export and submission integrity verification"),
    ("data/step6_target_features.parquet",
     "Compact per-compound ITPV descriptors (evidence flag, potency summaries, train-only SVD)"),
    ("data/step6_itpv_proteome.parquet",
     "Full Initial Target Perturbation Vector: 57 perturbations x 5243 measured yeast proteins"),
    ("data/step6_chembl_activities.parquet",
     "Curated ChEMBL Kd/Ki/IC50/EC50 records with pActivity, non-censored"),
    ("data/step6_target_protein_map.parquet",
     "ChEMBL target -> yeast protein links with the mapping channel (direct/orthodb/symbol)"),
    ("data/step6_mechanism_structures.npz",
     "Protein-complex co-response edges and the iMM904 metabolite x reaction stoichiometry matrix"),
    ("data/step6_protein_clusters_k32.parquet",
     "Nested protein-cluster hierarchy (k4/k8/k16/k32) over the 5243 proteins"),
    ("data/step6_test_predictions.parquet",
     "Parquet mirror of the Step-6 test abundance matrix"),
    ("data/step6_test_delta_predictions.parquet",
     "Step-6 test fold-change (Delta) matrix"),
    ("results/step6_target_affinity_report.json",
     "ITPV construction report: resolution, coverage, mapping channels, missingness policy"),
    ("results/step6_itpv_mechanism_validation.json",
     "Mechanistic validation of the ortholog mapping against textbook pharmacology"),
    ("results/step6_mechanism_loss_report.json",
     "MechanismLoss term-by-term NumPy cross-check, gradient and convergence evidence"),
    ("results/step6_xattn_training.json",
     "Cross-attention transformer training report with per-epoch trajectories and checkpoints"),
    ("results/step6_model_scores.json",
     "Step-6 stacking scores, selected configuration and frozen weights"),
    ("results/step6_cluster_weights.json",
     "Step-6 cluster stacking weight tensor and hierarchy metadata"),
    ("results/step6_bootstrap_ci.json",
     "Paired bootstrap confidence intervals vs the scalar rung and the Step-5 frozen weights"),
    ("results/step6_test_predictions.csv",
     "Step-6 submission matrix: treated test samples x 5243 proteins"),
    ("results/step6_compute_scaling_report.json",
     "Measured epoch-budget comparison and cloud compute allocation assessment"),
    ("results/step6_verification_report.json",
     "Submission integrity checks and the Step-6 headline summary"),
    ("figures/step6_performance_comparison.png",
     "Members vs ensembles, hierarchical ladder, and per-module decomposition"),
    ("figures/step6_cross_attention_weights.png",
     "Chemistry-driven deviation of attention over STRING PPI edges"),
]


def main() -> None:
    mpath = SESSION / "manifest.json"
    m = json.loads(mpath.read_text())

    outs = []
    missing = []
    for rel, desc in STEP6_OUTPUTS:
        p = SESSION / rel
        ex = p.exists()
        if not ex:
            missing.append(rel)
        outs.append({
            "path": str(p), "relative_path": rel, "description": desc,
            "exists": ex,
            "size_mb": round(p.stat().st_size / 1e6, 3) if ex else None,
        })

    scores = {}
    sp = RESULTS / "step6_model_scores.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        scores = {
            "headline_val_total": s.get("headline_val_total"),
            "step5_val_total": s.get("step5_val_total"),
            "benchmark_total": s.get("benchmark_total"),
            "delta_vs_step5": s.get("delta_vs_step5"),
            "delta_vs_benchmark": s.get("delta_vs_benchmark"),
            "step6_target": s.get("step6_target"),
            "step6_target_met": s.get("step6_target_met"),
            "selected": s.get("selected"),
            "roles": s.get("roles"),
            "n_free_parameters": s.get("n_free_parameters"),
        }

    ta = {}
    tp = RESULTS / "step6_target_affinity_report.json"
    if tp.exists():
        t = json.loads(tp.read_text())
        ta = {k: t.get(k) for k in
              ("itpv_dimension", "n_distinct_chembl_targets", "target_to_protein_links",
               "mapping_channels", "n_perturbations_with_nonzero_itpv",
               "n_proteins_with_nonzero_itpv", "n_activity_records_usable",
               "n_nan_in_feature_matrix")}

    mv = {}
    vp = RESULTS / "step6_itpv_mechanism_validation.json"
    if vp.exists():
        v = json.loads(vp.read_text())
        mv = {k: v.get(k) for k in
              ("n_positive_compounds", "n_positive_compounds_recovered",
               "n_positive_pairs_tested", "n_positive_pairs_annotated",
               "n_negative_pairs_tested", "n_negative_pairs_correctly_zero")}

    ver = {}
    rp = RESULTS / "step6_verification_report.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        tp6 = r.get("test_predictions", {})
        ver = {"all_integrity_checks_passed":
               tp6.get("artefact_integrity", {}).get("all_passed"),
               "n_checks": tp6.get("artefact_integrity", {}).get("n_checks"),
               "n_samples": tp6.get("n_samples"), "n_proteins": tp6.get("n_proteins"),
               "indicative_fold_change_pcc":
                   tp6.get("indicative_fold_change_pcc_per_sample_mean")}

    m["current_step"] = "Step 6 - mechanism-guided neural models and hierarchical cluster stacking"
    m["step6"] = {
        "status": "completed" if not missing else "completed_with_missing_outputs",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "headline": scores,
        "target_affinity": ta,
        "itpv_mechanism_validation": mv,
        "submission_verification": ver,
        "outputs": outs,
        "missing_declared_outputs": missing,
    }
    done = m.get("steps_completed", [])
    label = "Step 5 - knowledge integration, LCGO cross-fitting and protein-cluster stacking"
    if label not in done:
        done.append(label)
    label6 = "Step 6 - mechanism-guided neural models and hierarchical cluster stacking"
    if label6 not in done:
        done.append(label6)
    m["steps_completed"] = done
    mpath.write_text(json.dumps(m, indent=2))

    print(f"manifest.json updated: {len(outs)} Step-6 outputs, {len(missing)} missing")
    for x in missing:
        print(f"  MISSING: {x}")
    if scores:
        print(f"  headline val total: {scores.get('headline_val_total')}")


if __name__ == "__main__":
    sys.exit(main())
