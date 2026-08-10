#!/usr/bin/env python
"""Step 7.2 -- External-resource disclosure manifest for the merged open leaderboard.

The 2026-08-11 handbook revision merges the closed-data and open-knowledge
leaderboards into a single open board, and requires that any external public
resource used for entity-feature construction be declared **with its source and
version** (revision table 1 #1, table 2 #14).  The 5% open-source-contribution
dimension additionally requires disclosure of licence, third-party dependencies,
commercial-API use and closed-model use.

This script does not fetch anything.  It reads the artefacts that the pipeline
already wrote and emits one machine-readable disclosure record per external
resource, with the counts taken from those artefacts rather than from prose, so
that the disclosure table in the submission document and the table in the
repository README are generated from the same source.

Outputs
-------
results/step7_external_data_manifest.json
results/step7_external_data_manifest.csv
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import json
from collections import Counter
from pathlib import Path

import pandas as pd

SESSION = REPO_ROOT
RESULTS = SESSION / "results"


def jload(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def main() -> None:
    smiles = jload("step4_smiles_resolved.json")
    rdkit_rep = jload("step4_rdkit_report.json")
    aff = jload("step6_target_affinity_report.json")
    graph = jload("step5_graph_report.json")
    priors = jload("knowledge_priors.json")
    mech = jload("step6_mechanism_loss_report.json")

    src_counts = Counter(v.get("source") for v in smiles.values())
    cids = sorted({int(v["cid"]) for v in smiles.values() if v.get("cid")})

    n_complexes = mech["n_complexes_curated"]
    n_complex_edges = mech["n_complex_edges"]
    stoich = {
        "n_reactions": mech["n_reactions_curated"],
        "n_metabolites": mech["n_balanced_metabolites"],
        "stoichiometry_shape": mech["stoichiometry_shape"],
        "n_reactions_with_measured_enzyme": mech["n_reactions_with_measured_enzyme"],
    }
    _ = priors  # loaded for existence check only

    records = [
        {
            "resource": "PubChem (PUG-REST)",
            "source_urls": ["https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest"],
            "licence_url": "https://pubchem.ncbi.nlm.nih.gov/docs/downloads",
            "role": "compound entity representation: canonical SMILES + CID for every "
                    "chemical perturbation label",
            "recommended_by_organisers": True,
            "version_or_access": "PubChem PUG-REST, accessed 2026-08-05 during this project",
            "licence": "PubChem aggregates contributor records whose licensing can vary; "
                       "the service does not support a blanket public-domain claim. Check "
                       "record-level provenance before redistribution",
            "counts": {
                "n_perturbation_labels_queried": int(rdkit_rep["n_perturbation_labels"]),
                "n_resolved_to_a_molecule": int(rdkit_rep["n_resolved_as_molecule"]),
                "n_non_molecule_labels": int(rdkit_rep["n_non_molecule_labels"]),
                "n_unresolved": int(rdkit_rep["n_unresolved"]),
                "coverage_of_molecular_labels_pct": float(
                    rdkit_rep["coverage_of_molecular_labels_pct"]),
                "resolution_source_breakdown": dict(src_counts),
                "n_distinct_cids": len(cids),
            },
            "runtime_artifact": "results/step4_smiles_resolved.json",
            "public_evidence": ["evidence/step4_smiles_resolved.json"],
            "verification": "molecular weights cross-checked against the PubChem record for "
                            "every resolved compound; salt stripping and parent-molecule "
                            "extraction validated on 55 reference weights",
        },
        {
            "resource": "RDKit",
            "source_urls": ["https://pypi.org/project/rdkit/2026.3.5/"],
            "licence_url": "https://github.com/rdkit/rdkit/blob/master/license.txt",
            "role": "molecular descriptors, ECFP4 fingerprints, ETKDGv3 conformers and "
                    "MMFF94-optimised 3D descriptors from the PubChem SMILES",
            "recommended_by_organisers": True,
            "version_or_access": "RDKit 2026.03.5 (pinned in the environment table)",
            "licence": "BSD-3-Clause",
            "counts": {
                "n_2d_descriptors": 29,
                "n_descriptor_missing_cells": int(rdkit_rep["n_descriptor_missing_cells"]),
            },
            "runtime_artifact": "results/step4_rdkit_report.json, "
                                "data/step4_mol_features.parquet",
            "public_evidence": ["evidence/step4_rdkit_report.json"],
            "verification": "0 missing descriptor cells; independent molecular-weight "
                            "cross-check caught two silent salt-stripping failures",
        },
        {
            "resource": "ChEMBL (web services)",
            "source_urls": ["https://www.ebi.ac.uk/chembl/api/data"],
            "licence_url": "https://chembl.gitbook.io/chembl-interface-documentation/about",
            "role": "target binding potency (Kd/Ki/IC50/EC50) used to build the Inferred "
                    "Target Potency Vector (ITPV)",
            "recommended_by_organisers": False,
            "version_or_access": "ChEMBL web services, accessed 2026-08-07; the exact "
                                 "database release was not retained in the original artefact",
            "licence": "CC BY-SA 3.0; redistribution of regenerated derived tables or "
                       "matrices must preserve the applicable attribution/share-alike terms",
            "counts": {
                "n_perturbations_resolved_to_chembl": int(aff["n_resolved_to_chembl"]),
                "n_with_activity_evidence": int(aff["n_with_activity_evidence"]),
                "n_activity_records_raw": int(aff["n_activity_records_raw"]),
                "n_activity_records_usable": int(aff["n_activity_records_usable"]),
                "n_distinct_targets": int(aff["n_distinct_chembl_targets"]),
                "n_target_uniprot_accessions": int(aff["n_target_uniprot_accessions"]),
            },
            "runtime_artifact": "results/step6_target_affinity_report.json",
            "public_evidence": ["evidence/step6_target_affinity_report.json",
                                "evidence/step6_itpv_mechanism_validation.json"],
            "verification": "12 pre-declared positive pharmacology controls (11 recovered) "
                            "and 10 pre-declared negative controls (10 correctly zero); "
                            "see results/step6_itpv_mechanism_validation.json",
        },
        {
            "resource": "OrthoDB cross-references returned by UniProt REST",
            "source_urls": ["https://www.uniprot.org/help/uniprot_rest_tutorial"],
            "licence_url": None,
            "role": "orthology projection of non-yeast ChEMBL targets onto the measured "
                    "yeast proteome axis; no direct OrthoDB API or bulk file was queried",
            "recommended_by_organisers": False,
            "version_or_access": "UniProt REST response accessed 2026-08-07; the underlying "
                                 "OrthoDB subrelease was not retained",
            "licence": "the upstream OrthoDB terms/version were not captured in the "
                       "original artefact; confirm them before redistributing mapped rows",
            "counts": {
                "target_to_protein_links_total": int(aff["target_to_protein_links"]),
                "via_orthodb": int(aff["mapping_channels"]["orthodb"]),
                "via_gene_symbol": int(aff["mapping_channels"]["symbol"]),
                "direct_yeast_target": int(aff["mapping_channels"]["direct"]),
                "itpv_nonzero_density": float(aff["itpv_nonzero_density"]),
                "n_perturbations_with_nonzero_itpv": int(
                    aff["n_perturbations_with_nonzero_itpv"]),
                "n_proteins_with_nonzero_itpv": int(aff["n_proteins_with_nonzero_itpv"]),
            },
            "runtime_artifact": "results/step6_target_affinity_report.json",
            "public_evidence": ["evidence/step6_target_affinity_report.json"],
            "verification": "rapamycin potency maps to FPR1 (FKBP12 homologue) rather than "
                            "directly to TOR, which is the pharmacologically correct form",
        },
        {
            "resource": "UniProt",
            "source_urls": ["https://rest.uniprot.org/"],
            "licence_url": "https://www.uniprot.org/help/license",
            "role": "accession and gene-symbol normalisation between ChEMBL targets and the "
                    "yeast protein roster",
            "recommended_by_organisers": False,
            "version_or_access": "UniProt REST, accessed 2026-08-07; the response release "
                                 "header was not retained",
            "licence": "CC BY 4.0",
            "counts": {"n_accessions_touched": int(aff["n_target_uniprot_accessions"])},
            "runtime_artifact": "results/step6_target_affinity_report.json",
            "public_evidence": ["evidence/step6_target_affinity_report.json"],
            "verification": "symbol channel contributes only "
                            f"{aff['mapping_channels']['symbol']} of "
                            f"{aff['target_to_protein_links']} links, so a symbol collision "
                            "cannot dominate the mapping",
        },
        {
            "resource": "STRING",
            "source_urls": ["https://string-db.org/cgi/access"],
            "licence_url": "https://string-db.org/cgi/access",
            "role": "protein association network: adjacency for the graph neural member, "
                    "spectral embedding, and the protein functional clusters used by the "
                    "hierarchical-shrinkage meta-learner",
            "recommended_by_organisers": False,
            "version_or_access": f"STRING v{graph['string_version'].split()[0]} "
                                 f"(taxon {graph['species_taxid']}), "
                                 f"confidence >= {graph['required_score']}",
            "licence": "CC BY 4.0",
            "counts": {
                "n_proteins_mapped": int(graph["n_mapped_to_string"]),
                "n_proteins_unmapped": int(graph["n_unmapped"]),
                "n_edges_in_species_file": int(graph["n_edges_in_species_file"]),
                "n_edges_above_cutoff_undirected": int(graph["n_edges_undirected"]),
                "n_isolated_nodes": int(graph["n_isolated_nodes"]),
                "giant_component_size": int(graph["spectral"]["giant_component_size"]),
                "spectral_dims_kept": int(graph["spectral"]["k_kept"]),
                "jaccard_vs_train_coresponse_graph": float(
                    graph["string_vs_coresponse_jaccard"]),
            },
            "runtime_artifact": "results/step5_graph_report.json",
            "public_evidence": ["evidence/step5_graph_report.json"],
            "verification": "Jaccard overlap with the train-only co-response graph is "
                            f"{graph['string_vs_coresponse_jaccard']:.4f}, so the external "
                            "graph carries information complementary to the data itself "
                            "rather than restating it",
        },
        {
            "resource": "iMM904-informed hand-coded metabolic subset",
            "source_urls": ["https://doi.org/10.1186/1752-0509-3-37"],
            "licence_url": None,
            "role": "literature-informed stoichiometric mass-balance penalty (core "
                    "sub-network) in the mechanism loss; no third-party model file is read",
            "recommended_by_organisers": False,
            "version_or_access": "Mo, Palsson & Herrgard (2009) used as the literature "
                                 "reference for a hand-coded 44-reaction subset",
            "licence": "citation-based reimplementation; no upstream model file is "
                       "redistributed and no model-file licence is asserted",
            "counts": {
                "n_reactions": stoich["n_reactions"],
                "n_reactions_with_measured_enzyme":
                    stoich["n_reactions_with_measured_enzyme"],
                "n_balanced_metabolites": stoich["n_metabolites"],
                "stoichiometry_shape": stoich["stoichiometry_shape"],
            },
            "runtime_artifact": "results/step6_mechanism_loss_report.json, "
                                "results/knowledge_priors.json",
            "public_evidence": ["evidence/step6_mechanism_loss_report.json",
                                "evidence/knowledge_priors.json"],
            "verification": "every reaction retained has at least one measured enzyme; the "
                            "flux penalty was validated against an independent NumPy "
                            "reference implementation",
        },
        {
            "resource": "literature-informed hand-coded protein-complex roster",
            "source_urls": ["https://doi.org/10.1093/nar/gkac1015",
                            "https://doi.org/10.1093/nar/gkab991"],
            "licence_url": None,
            "role": "co-complex consistency penalty in the mechanism loss; the code embeds "
                    "a manual member roster rather than a database export",
            "recommended_by_organisers": False,
            "version_or_access": "manual roster documented against CORUM, Complex Portal "
                                 "and yeast-complex literature; per-edge provenance was not "
                                 "retained in the original artefact",
            "licence": "no upstream database export is redistributed; upstream terms must "
                       "be rechecked before replacing the manual roster with source records",
            "counts": {k: v for k, v in {
                "n_complexes": n_complexes,
                "n_co_complex_edges": n_complex_edges,
            }.items() if v is not None},
            "runtime_artifact": "results/step6_mechanism_loss_report.json",
            "public_evidence": ["evidence/step6_mechanism_loss_report.json"],
            "verification": "each complex edge is weighted 1/(k-1) so that a large complex "
                            "contributes in proportion to its subunit count rather than to "
                            "its pair count",
        },
        {
            "resource": "1011 Yeast Genomes Project (1002genomes.u-strasbg.fr/files/)",
            "source_urls": ["http://1002genomes.u-strasbg.fr/files/",
                            "https://doi.org/10.1038/s41586-018-0030-5"],
            "licence_url": None,
            "role": "PLANNED strain entity representation: variant-derived features for the "
                    "unseen strain CRD (S2/S3 extrapolation)",
            "recommended_by_organisers": True,
            "version_or_access": "not yet used in any scored run; declared as the planned "
                                 "source for the next stage",
            "licence": "not verified for redistribution in this project; confirm the "
                       "specific file terms before use",
            "counts": {"n_strain_features_used_in_scored_runs": 0},
            "runtime_artifact": None,
            "public_evidence": [],
            "verification": "NOT USED in any number reported in this submission; the "
                            "submitted model treats strain identity as a categorical "
                            "feature and therefore has no representation for CRD",
        },
        {
            "resource": "SGD S288C reference genome",
            "source_urls": ["https://www.yeastgenome.org/strain/S288C",
                            "https://doi.org/10.1093/genetics/iyab224"],
            "licence_url": None,
            "role": "PLANNED proxy reference for DHY210, per the organisers' recommendation",
            "recommended_by_organisers": True,
            "version_or_access": "not yet used in any scored run",
            "licence": "not verified for redistribution in this project; confirm the "
                       "specific release terms before use",
            "counts": {"n_strain_features_used_in_scored_runs": 0},
            "runtime_artifact": None,
            "public_evidence": [],
            "verification": "NOT USED in any number reported in this submission",
        },
    ]

    payload = {
        "step": "7_2_external_data_manifest",
        "purpose": "source-and-version disclosure for every external public resource, as "
                   "required by the 2026-08-11 handbook revision (merged open leaderboard) "
                   "and by the new 5% open-source-contribution dimension",
        "official_competition_data": {
            "resource": "GOAI Track-3 virtual-cell dataset (4 files: metadata and proteome "
                        "for train_val and test)",
            "role": "training and evaluation",
            "licence": "competition data-use agreement: used only within this competition, "
                       "NOT redistributed; the repository ships readers and structural "
                       "contract checks only",
            "counts": {"train_val_samples": 8958, "test_samples": 4454, "n_proteins": 5243},
            "note": "sample counts and protein width independently reproduced by our QC "
                    "(results/metadata_profile.json, results/qc_summary.json) and they "
                    "agree with the revised handbook",
        },
        "resources": records,
        "commercial_apis_and_closed_models": [
            {
                "item": "large-language-model coding / literature-verification agents",
                "used_for": "script drafting and refactoring, literature search and DOI "
                            "checking, figure generation, typesetting",
                "affects_any_reported_score": False,
                "reason": "no assistant service participates in model training, inference "
                          "or scoring; stochastic model stages pin seed 42, while non-random "
                          "QC and disclosure outputs do not claim a seed field",
            },
            {
                "item": "AI image-generation service",
                "used_for": "a small number of purely conceptual schematics",
                "affects_any_reported_score": False,
                "reason": "every figure that carries a numeric result is drawn by matplotlib "
                          "directly from the result artefacts",
            },
            {
                "item": "online inference services for the model itself",
                "used_for": "none",
                "affects_any_reported_score": False,
                "reason": "all training, inference and scoring run locally",
            },
        ],
    }

    out = RESULTS / "step7_external_data_manifest.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    rows = []
    for r in records:
        rows.append({
            "resource": r["resource"],
            "source_urls": " | ".join(r["source_urls"]),
            "licence_url": r["licence_url"],
            "recommended_by_organisers": r["recommended_by_organisers"],
            "version_or_access": r["version_or_access"],
            "licence": r["licence"],
            "role": r["role"],
            "used_in_scored_runs": r["runtime_artifact"] is not None,
            "key_counts": json.dumps(r["counts"], ensure_ascii=False),
        })
    pd.DataFrame(rows).to_csv(RESULTS / "step7_external_data_manifest.csv", index=False)
    print(f"wrote {out.name} with {len(records)} external resource records")
    for r in records:
        flag = "used" if r["runtime_artifact"] else "PLANNED (not used)"
        print(f"  [{flag:18s}] {r['resource']}  |  {r['licence'][:48]}")


if __name__ == "__main__":
    main()
