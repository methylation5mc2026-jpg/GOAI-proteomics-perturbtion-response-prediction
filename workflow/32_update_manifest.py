"""Step 5 -- refresh manifest.json with the Step-5 artefacts and headline.

Every declared Step-5 output is checked for existence and size on disk, so a file
that was never written shows up as ``missing`` in the manifest instead of being
listed as though it exists. The headline block is read back out of
``results/step5_verification_report.json`` rather than restated here, so the
manifest cannot drift from the number the verification actually computed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_common as S4  # noqa: E402

SESSION, RESULTS, DATA, FIGURES = S4.SESSION, S4.RESULTS, S4.DATA, S4.FIGURES
log = S4.log

#: relative path -> description. Grouped in pipeline order.
STEP5_OUTPUTS: dict[str, str] = {
    # scripts
    "workflow/27_knowledge_extraction.py":
        "Step 5.1 dual-search prior-knowledge mining from ProteinTalks + STRING, with "
        "run-time-located line provenance and an absence survey",
    "workflow/28_advanced_features.py":
        "Step 5.2 3D conformer chemistry (ETKDGv3/MMFF94, SASA, PMI, Gobbi 3D pharmacophores) "
        "and the STRING PPI spectral embedding + protein clustering",
    "workflow/29_lcgo_oof_matrix.py":
        "Step 5.3 5-fold Leave-Chemical-Group-Out design, per-fold refits, full OOF member "
        "matrices, and the full-train val/test counterparts",
    "workflow/30_gnn_and_cluster_stacking.py":
        "Step 5.4 graph-attention ResNet (cross-fitted) and the protein-cluster non-negative "
        "stacking meta-learner with the ablation ladder and bootstrap CIs",
    "workflow/30a_smoke_clusterscore.py":
        "Step 5.4 exactness smoke test for the cluster co-moment scorer",
    "workflow/31_agentic_loop_runner.py":
        "Step 5.5 agentic self-evolution loop, test export and verification",
    "workflow/step5_clusterscore.py":
        "exact fast harness evaluation under per-(regime, role, protein-cluster) weights",
    "workflow/run_step5_chain.sh":
        "driver that chains the long Step-5 stages so they never contend for the CPU",
    # knowledge
    "results/knowledge_priors.json":
        "structured dual-search prior-knowledge base: 13 priors tagged by search mode, "
        "transfer kind and whether a named script consumes them",
    "results/step5_paper_provenance.json":
        "located quotes and line numbers from the ProteinTalks markdown, plus the absence survey",
    "results/step5_chemical_support.csv":
        "max ECFP4 Tanimoto of every compound to the nearest training compound (prior FD1)",
    # features
    "data/step5_mol3d_features.parquet":
        "per-compound 3D descriptors and train-fitted 3D pharmacophore PCA",
    "results/step5_mol3d_report.json":
        "conformer-generation QC, per-compound audit and leakage note",
    "data/step5_protein_graph.npz":
        "STRING v12 weighted adjacency over the 5,243 measured proteins plus the "
        "giant-component spectral embedding",
    "data/step5_protein_clusters.parquet":
        "protein -> cluster index at K in {1,4,8,12,16}, fitted on train rows only",
    "data/step5_protein_stats.parquet":
        "train-only per-protein abundance / response / degree statistics",
    "results/step5_graph_report.json":
        "STRING mapping and edge QC, component structure, STRING-vs-co-response agreement",
    "figures/step5_knowledge_graph_embedding.png":
        "spectral embedding, degree distribution and what the protein clusters are",
    # LCGO
    "results/step5_lcgo_folds.json":
        "LCGO fold design, realised composition, per-fold fit timings and OOF coverage",
    "results/step5_leakage_audit.json":
        "5 leakage/partition checks re-derived from each fold's realised fit set",
    # stacking
    "results/step5_clusterscore_smoke.json":
        "cluster-scorer exactness: harness agreement, reduction to the Step-4 scorer, "
        "incremental-update exactness, throughput",
    "results/step5_gnn_training.json":
        "graph-attention ResNet training histories, per-fold epoch selection and the "
        "lambda_graph = 0 control",
    "results/step5_dl_training.json":
        "cross-fitted Step-4-architecture deep member training histories",
    "results/step5_model_scores.json":
        "every candidate blend, per module, on both cohorts, with the ablation ladder",
    "results/step5_cluster_weights.json":
        "the frozen protein-cluster weight tensor actually used (same content as above)",
    "results/step5_bootstrap_ci.json":
        "paired non-parametric bootstrap CIs on the per-sample component of the margins",
    "figures/step5_cluster_stacking_weights.png":
        "fitted weight tensor per regime, the per-cluster shrinkage curve and cluster identity",
    "figures/step5_performance_comparison.png":
        "ablation ladder, per-module composition and cluster-count sensitivity",
    # agentic loop and submission
    "results/step5_agentic_loop_log.json":
        "every self-evolution iteration: mutation, motivating prior, score, Pareto status, "
        "and any diagnosis",
    "results/step5_test_predictions.csv":
        "final submission matrix, 4,226 test samples x 5,243 proteins",
    "data/step5_test_predictions.parquet":
        "parquet mirror of the submission matrix",
    "data/step5_test_delta_predictions.parquet":
        "predicted fold-change matrix for the test cohort",
    "results/step5_verification_report.json":
        "artefact integrity checks, success criteria and the headline with its provenance",
}


def main() -> None:
    man_p = SESSION / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))

    existing = {o["relative_path"] for o in man.get("outputs", [])}
    added, missing = [], []
    for rel, desc in STEP5_OUTPUTS.items():
        p = SESSION / rel
        if not p.exists():
            missing.append(rel)
            continue
        if rel in existing:
            continue
        added.append({
            "path": str(p),
            "relative_path": rel,
            "description": desc,
            "size_bytes": int(p.stat().st_size),
            "step": "Step 5",
        })
    man.setdefault("outputs", []).extend(added)
    man["n_outputs"] = len(man["outputs"])
    man["n_missing"] = sum(
        0 if (SESSION / o["relative_path"]).exists() else 1 for o in man["outputs"]
    )

    vr_p = RESULTS / "step5_verification_report.json"
    if vr_p.exists():
        vr = json.loads(vr_p.read_text(encoding="utf-8"))
        man["step5"] = {
            "status": "completed",
            "headline": vr["headline"],
            "artefact_integrity": {
                "n_checks": vr["artefact_integrity"]["n_checks"],
                "n_passed": vr["artefact_integrity"]["n_passed"],
                "all_passed": vr["artefact_integrity"]["all_passed"],
            },
            "success_criteria": {
                "n_criteria": vr["success_criteria"]["n_criteria"],
                "n_met": vr["success_criteria"]["n_met"],
                "all_met": vr["success_criteria"]["all_met"],
                "criteria": [{"name": c["name"], "met": c["met"]}
                             for c in vr["success_criteria"]["criteria"]],
            },
            "ablation_ladder": vr["ablation_ladder"],
            "roles": vr["roles"],
            "n_clusters": vr["n_clusters"],
            "missing_declared_outputs": missing,
        }
        man["step"] = ("Step 5 - dual-search knowledge integration, LCGO cross-fitting and "
                       "protein-cluster agentic stacking")
        man["status"] = "completed"
        man["current_step"] = "Step 5 complete"
        sc = man.setdefault("steps_completed", [])
        label = ("Step 5 - dual-search knowledge integration, LCGO cross-fitting and "
                 "protein-cluster agentic stacking")
        if label not in sc:
            sc.append(label)
    else:
        man["step5"] = {"status": "incomplete",
                        "note": "results/step5_verification_report.json absent",
                        "missing_declared_outputs": missing}

    man_p.write_text(json.dumps(man, indent=2), encoding="utf-8")
    log(f"manifest updated: {len(added)} Step-5 outputs added, "
        f"{man['n_outputs']} total, {man['n_missing']} missing")
    if missing:
        log(f"  declared but absent on disk ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
