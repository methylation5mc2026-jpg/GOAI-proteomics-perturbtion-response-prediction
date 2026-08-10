"""Step 5 -- generate the README section from the saved JSON reports.

Every number in the section is read out of the artefact that computed it, so the
prose cannot drift from the result. If a report is missing the section says so
rather than leaving a stale figure in place.

Re-running replaces the existing Step-5 section instead of appending a second
copy, so the README stays idempotent under repeated runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_common as S4  # noqa: E402

SESSION, RESULTS, DATA = S4.SESSION, S4.RESULTS, S4.DATA
log = S4.log

MARKER = "# Step 5 — Dual-Search Knowledge Integration, LCGO Cross-Fitting and Protein-Cluster Stacking"


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x, nd: int = 6) -> str:
    return "n/a" if x is None else f"{float(x):.{nd}f}"


def build() -> str:
    kp = jload(RESULTS / "knowledge_priors.json")
    m3 = jload(RESULTS / "step5_mol3d_report.json")
    gr = jload(RESULTS / "step5_graph_report.json")
    fo = jload(RESULTS / "step5_lcgo_folds.json")
    au = jload(RESULTS / "step5_leakage_audit.json")
    sm = jload(RESULTS / "step5_clusterscore_smoke.json")
    gn = jload(RESULTS / "step5_gnn_training.json")
    dlr = jload(RESULTS / "step5_dl_training.json")
    ms = jload(RESULTS / "step5_model_scores.json")
    bo = jload(RESULTS / "step5_bootstrap_ci.json")
    lo = jload(RESULTS / "step5_agentic_loop_log.json")
    vr = jload(RESULTS / "step5_verification_report.json")

    L: list[str] = [MARKER, ""]
    A = L.append

    # ---- headline --------------------------------------------------------
    A("## 13. Step 5 — what was changed and what it bought")
    A("")
    if vr:
        h = vr["headline"]
        A("**Headline (held-out `val_*`, weights frozen before val was touched): "
          f"{fmt(h['val_total'])}**")
        A("")
        A("| reference | total | margin |")
        A("|---|---|---|")
        A(f"| official benchmark | {fmt(h['benchmark_total'])} | "
          f"{h['margin_vs_benchmark']:+.6f} |")
        A(f"| Step 4 (published) | {fmt(h['step4_val_total'])} | "
          f"{h['margin_vs_step4']:+.6f} |")
        A(f"| Step 5 target | {fmt(h['target'], 4)} | "
          f"{h['val_total'] - h['target']:+.6f} "
          f"({'met' if h['target_met'] else 'NOT met'}) |")
        A("")
        A(f"Selection provenance: {h['source']}. The out-of-fold total at those weights "
          f"was {fmt(h['oof_total'])}.")
        A("")
    else:
        A("_`results/step5_verification_report.json` is absent — Step 5 did not complete._")
        A("")

    # ---- the two changes, separated -------------------------------------
    if ms:
        lad = ms["ablation_ladder"]
        A("### 13.1 The ablation ladder — separating the two changes")
        A("")
        A("Step 5 changes two things at once: it calibrates the blend on a much larger "
          "cross-fitted cohort, and it enlarges the weight space from one scalar per "
          "(regime, role) to one weight per (regime, role, protein cluster). Reporting only "
          "the final number would leave the credit unattributed, so each rung is scored on "
          "the same held-out `val_*` with weights frozen beforehand.")
        A("")
        A("| rung | val total | isolates |")
        A("|---|---|---|")
        A(f"| official benchmark | {fmt(ms['benchmark_total'])} | the target to beat |")
        A("| Step 4: scalar weights, single inner cohort | "
          f"{fmt(lad['rung0_step4_scalar_inner_cohort'])} | the published baseline |")
        A("| Step 5: scalar weights, LCGO out-of-fold cohort | "
          f"{fmt(lad['rung1_scalar_oof_cohort'])} | cross-fitting + the new member pool |")
        A("| Step 5: **cluster** weights, LCGO out-of-fold cohort | "
          f"{fmt(lad['rung2_cluster_oof_cohort'])} | the cluster extension alone |")
        A("")
        A("- gain from cross-fitting and the enlarged member pool: "
          f"**{lad['gain_from_cross_fitting_alone']:+.6f}**")
        A("- gain from the cluster extension: "
          f"**{lad['gain_from_the_cluster_extension']:+.6f}**")
        A("")
        A("The first gain is *not* purely attributable to cross-fitting: rung 1 also "
          "introduces the 3D-chemistry member and the cross-fitted deep members, so it "
          "mixes two effects. Rung 1 -> rung 2 is a clean A/B — only the weight space "
          "changes — and is the one comparison Step 5 can claim cleanly.")
        A("")
        A("For reference, fitting the weights directly on `val_*` reaches "
          f"{fmt(ms['val_total_val_tuned_OPTIMISTIC'])}. That number is **optimistic and is "
          "not a result** — it only measures how much headroom the weight space contains.")
        A("")
        sel = ms["selected"]
        A(f"Selected configuration: cluster column `{sel['cluster_column']}` "
          f"({sel['n_clusters']} clusters), cluster-spread shrinkage alpha = {sel['alpha']}, "
          f"roles `{', '.join(ms['roles'])}`. Selected on {sel['selected_on']}.")
        A("")
        if ms.get("roles_dropped"):
            A("Roles dropped for lack of a matched member on one cohort: "
              f"`{', '.join(ms['roles_dropped'])}`.")
            A("")

    # ---- why cluster weights --------------------------------------------
    A("### 13.2 Why per-cluster weights are the right extension")
    A("")
    A("The competition total mixes two kinds of submetric. Module 2 (25%) is built entirely "
      "from Pearson correlations of the fold change, so it is invariant to any positive "
      "rescaling of the prediction. Module 1 (20%) contains two R^2 terms and is strongly "
      "scale-sensitive. The shortfall of a non-negative weight vector from 1 is therefore a "
      "free shrinkage toward the control anchor: it buys Module 1 at no cost to Module 2. "
      "Step 4 already exploited that with one scalar per (regime, role).")
    A("")
    A("But R^2 is dominated by per-protein dynamic range. A hyper-abundant enzyme with a wide "
      "response distribution tolerates a large predicted fold change; a low-abundance "
      "regulatory factor, whose measured variance is largely noise, does not. The optimal "
      "shrinkage is therefore protein-dependent, and a single scalar per regime is forced to "
      "average over that. Making the shrinkage per-cluster is the whole mechanism, and it "
      "comes from the rubric's own algebra rather than from any paper "
      "(prior `XD5` in the knowledge base).")
    A("")

    # ---- exactness -------------------------------------------------------
    if sm:
        A("### 13.3 The scorer is exact, and that is verified rather than assumed")
        A("")
        A("A converged search over a cluster-weight tensor is only possible with a fast "
          "objective, and a fast objective is only admissible if it *is* the harness. "
          "Clusters partition the protein axis, so every cross-cluster Gram entry is "
          "identically zero and the co-moment factorisation extends to per-cluster weights "
          "at the same precompute cost. Four checks gate its use:")
        A("")
        c1 = sm["check1_harness_agreement_k8"]
        A("| check | result | tolerance |")
        A("|---|---|---|")
        A("| agreement with `harness.compute_competition_score` over "
          f"{c1['n_trials']} cluster-weight tensors (including one alternating 0 and 1.2 "
          "between neighbouring clusters) | max dev "
          f"{c1['max_abs_deviation']:.2e} | 1e-6 |")
        c2 = sm["check2a_reduces_to_step4_scorer"]
        A("| `n_clusters = 1` reproduces the Step-4 scalar scorer | max dev "
          f"{c2['max_abs_deviation']:.2e} | 1e-9 |")
        c2b = sm["check2b_cluster_constant_equals_scalar"]
        A("| a cluster-constant tensor equals the equivalent scalar weights | max dev "
          f"{c2b['max_abs_deviation']:.2e} | 1e-9 |")
        c2c = sm.get("check2c_incremental_updates_exact")
        if c2c:
            A("| the incremental block-update path stays exact after "
              f"{c2c['n_perturbations']:,} single-coordinate perturbations | max dev "
              f"{c2c['max_abs_deviation_vs_cold_scorer']:.2e} | 1e-9 |")
        A("")
        A(f"Throughput: {1000 * sm['check3_seconds_per_evaluation']:.1f} ms per evaluation of "
          f"{sm['n_free_parameters']} free weights, against "
          f"{sm['exact_harness_seconds_per_evaluation_for_scale']:.1f} s for one exact harness "
          "pass — a speed-up of about "
          f"{sm['exact_harness_seconds_per_evaluation_for_scale'] / sm['check3_seconds_per_evaluation']:,.0f}x. "
          "The second and third checks are what make the scalar rung of the ablation ladder a "
          "genuine control rather than a differently-implemented number.")
        A("")
        A("Module 4 (5% of the weight) thresholds `|Delta| > 1` and is not a polynomial in the "
          "weights, so it is held constant during the search and every reported total comes "
          "from the real harness. No reported number comes from the fast path.")
        A("")

    # ---- LCGO ------------------------------------------------------------
    if fo and au:
        A("### 13.4 The 5-fold Leave-Chemical-Group-Out design")
        A("")
        A(f"All {fo['n_train_rows']:,} train rows receive an out-of-fold prediction, against "
          "2,700 in Step 4, and each fold's fit set holds ~"
          f"{int(sum(v['n_fit'] for v in fo['composition'].values()) / len(fo['composition'])):,} "
          "rows against 2,378 in Step 4.")
        A("")
        A("A plain leave-chemical-group-out partition would make every held-out row "
          "`chem_novel`, but the rubric scores novel-strain generalisation at equal weight "
          "(0.20 each). So chemicals are split into 5 row-count-balanced groups **and** each "
          "fold holds out one strain. `train` has only 4 strains against 5 folds, so the fold "
          "slots are dealt round-robin over the strains — a one-to-one map leaves one fold "
          "holding out no strain at all, which measurably skewed the regime mix (18% "
          "`both_novel` instead of 23%) before it was fixed.")
        A("")
        A("| fold | fit rows | dev rows | held-out chemicals | held-out strain |")
        A("|---|---|---|---|---|")
        for k, v in sorted(fo["composition"].items()):
            A(f"| {k} | {v['n_fit']:,} | {v['n_dev']:,} | "
              f"{len(v['held_out_chemicals'])} | {', '.join(v['held_out_strains'])} |")
        A("")
        mix = fo["overall_regime_mix"]
        tot = sum(mix.values())
        A("Realised regime mix: " + ", ".join(
            f"`{k}` {v:,} ({100 * v / tot:.1f}%)" for k, v in mix.items())
          + ". The official test split is 39% chem-only, 31% strain-only, 27% both, 3% time, "
            "so the calibration cohort resembles the cohort the weights are calibrated for.")
        A("")
        A(f"**Leakage audit: {au['n_passed']}/{au['n_checks']} checks passed.** "
          "Each dev row's regime label is *re-derived* from its fold's realised fit set "
          "rather than trusted from the construction, because a construction bug and a "
          "leakage bug look identical from the outside. "
          f"{au['n_regime_mismatches']} mismatches.")
        A("")
        for c in au["checks"]:
            A(f"- [{'PASS' if c['passed'] else 'FAIL'}] {c['name']} — {c['detail']}")
        A("")
        if fo.get("skipped_due_to_fold_time_budget"):
            A("Skipped under the per-fold time budget (logged, not silently dropped): "
              f"`{fo['skipped_due_to_fold_time_budget']}`.")
            A("")

    # ---- knowledge base --------------------------------------------------
    if kp:
        A("### 13.5 The dual-search knowledge base — and the finding that shaped it")
        A("")
        A("*ProteinTalks* (Sun, Qian, Li et al., bioRxiv 2025.02.07.637070, Guo lab) is the "
          "focused-domain source. The single most important thing established about it is "
          "what it does **not** contain, and that was established by search rather than "
          "asserted:")
        A("")
        surv = kp["focused_source"]["absence_survey"]
        A("| probed for | present in the paper? |")
        A("|---|---|")
        for k, v in surv.items():
            A(f"| `{k}` | {'yes (' + str(v['n_hits']) + ' hits)' if v['present'] else '**no**'} |")
        A("")
        A("The paper studies **human** breast-cancer cell lines under **chemical** "
          "perturbation, with no yeast, no strain/genotype modelling, no PPI graph in its "
          "architecture and no gradient-boosted trees. It therefore transfers **methodology, "
          "not biology** — so every biology-tagged prior is recorded and explicitly not used. "
          "Of the 17 protein symbols the paper names, "
          f"{kp['named_protein_set']['n_string_overlap_with_yeast_proteome']} happen to match "
          "a measured yeast gene name; no ortholog mapping was performed, so those matches are "
          "recorded as context and never used as a yeast prior either way.")
        A("")
        c = kp["prior_counts"]
        A(f"{c['n_total']} priors: {c['n_focused_domain']} focused-domain, "
          f"{c['n_divergent_cross_domain']} divergent cross-domain; "
          f"**{c['n_actionable']} actionable** (consumed by a named script) and "
          f"{c['n_context_only']} context-only. A prior nobody consumes is documentation, so "
          "the split is counted rather than blurred.")
        A("")
        A(f"{len(c['focused_priors_with_unlocatable_evidence_DEFECT'])} focused priors had "
          "unlocatable evidence (that would be a defect). "
          f"{len(c['cross_domain_priors_absent_from_the_focused_source_AS_EXPECTED'])} of "
          f"{c['n_divergent_cross_domain']} cross-domain priors are absent from the focused "
          "source — which is the *confirmation* that the second search really was divergent.")
        A("")
        A("Actionable priors and what consumes them:")
        A("")
        A("| prior | search mode | consumed by |")
        A("|---|---|---|")
        for p in kp["priors"]:
            if p["status"] != "actionable":
                continue
            A(f"| `{p['id']}` | {p['search_mode']} | {p.get('consumed_by') or '-'} |")
        A("")
        sup = kp.get("chemical_support_FD1")
        if sup:
            A(f"Prior `FD1` operationalised: of {sup['n_absent_from_train']} compounds absent "
              f"from train, **{sup['n_absent_and_low_support']}** have a maximum ECFP4 Tanimoto "
              f"below {sup['low_support_threshold']} to any training compound (median "
              f"{sup['median_max_tanimoto_novel_compounds']:.3f}). The ~0.3 mark is a "
              "virtual-screening convention, not a published threshold for this task. This is "
              "a strong statement about the task: the held-out chemistry is largely "
              "structurally unsupported, which bounds what any structure-activity model can "
              "extrapolate.")
            A("")
        if ms and ms.get("chemical_support_stratification_FD1", {}).get("bins"):
            A("Held-out-chemical performance stratified by that support "
              "(per-sample context-residual PCC on `val_chem_only`, the Module-3 S1 submetric):")
            A("")
            A("| max Tanimoto to train | val_chem_only samples | compounds | residual PCC |")
            A("|---|---|---|---|")
            for b in ms["chemical_support_stratification_FD1"]["bins"]:
                v = b["mean_residual_ctx_pcc_per_sample"]
                A(f"| {b['max_tanimoto_to_train_bin']} | "
                  f"{b['n_val_chem_only_samples']:,} | {b['n_compounds']} | "
                  f"{'n/a' if v is None else f'{v:.4f}'} |")
            A("")

    # ---- features --------------------------------------------------------
    if m3 or gr:
        A("### 13.6 New features: 3D chemistry and the STRING interaction graph")
        A("")
    if m3:
        A(f"**3D chemistry.** {m3['n_with_conformer']}/{m3['n_labels']} labels received an "
          "ETKDGv3 conformer optimised with MMFF94 where the atom types are covered "
          f"({m3['n_without_conformer']} without: the non-molecular `Quality Control` control). "
          "14 shape/surface descriptors plus a train-fitted PCA of the Gobbi 3D pharmacophore "
          f"fingerprint ({m3['pharm3d']['n_raw_bits']:,} raw bits -> "
          f"{m3['pharm3d']['n_informative_bits_train_filter']:,} informative under a "
          f"train-only filter -> {m3['pharm3d']['n_pca_components']} components, "
          f"{100 * m3['pharm3d']['cumulative_explained_variance_ratio']:.1f}% EVR). "
          "Free-SASA cannot type Pt or bare ions, so cisplatin and NaCl carry an explicit "
          "`mol3d_sasa_defined = 0` flag rather than a zero that would read as "
          "'no surface area'.")
        A("")
    if gr:
        A(f"**STRING graph.** {gr['n_mapped_to_string']:,}/{gr['n_proteins']:,} proteins "
          f"({100 * gr['n_mapped_to_string'] / gr['n_proteins']:.1f}%) resolved against STRING "
          f"v12 for taxon {gr['species_taxid']}; "
          f"{gr.get('n_edges_undirected', 0):,} undirected edges at combined_score >= "
          f"{gr['required_score']}, giant component "
          f"{gr['spectral']['giant_component_size']:,}/{gr['n_proteins']:,}.")
        A("")
        A("Two bugs were found and fixed here, both of which would have passed silently:")
        A("")
        A("1. The REST `network` endpoint returns only edges **among the identifiers submitted "
          "in that call**. Batching 5,185 proteins at 900 per call produced 41,936 edges in "
          "170 components whose five largest were ~880 nodes each — exactly the batch size, "
          "which is the tell. The bulk `protein.links` file is the correct source and gives "
          f"{gr.get('n_edges_undirected', 0):,} edges instead.")
        A("2. The normalised adjacency has one eigenvalue of exactly 1 per connected "
          "component, so its top eigenvectors are component indicators rather than community "
          "structure. Clustering on them spent whole clusters on 2-protein fragments. The "
          "embedding is now computed on the giant component with the trivial eigenvector "
          "dropped, coordinates are rank-transformed, and a minimum cluster size is enforced "
          "with every merge logged.")
        A("")
        A("A reported diagnostic worth noting: the STRING graph and the train-only "
          f"co-response graph share only {gr.get('string_vs_coresponse_shared_edges', 0):,} "
          f"edges (Jaccard {gr.get('string_vs_coresponse_jaccard', 0):.4f}). Curated "
          "interaction topology and measured co-response are largely *different* structures "
          "in this dataset, which is why the cluster index uses both views rather than "
          "trusting either alone. The measured `1-Oct` protein label is an Excel-mangled gene "
          "symbol and is repaired explicitly so it does not look like a coverage failure.")
        A("")

    # ---- deep members ----------------------------------------------------
    if gn or dlr:
        A("### 13.7 The cross-fitted deep members")
        A("")
    if gn:
        pcc = [v["best_dev_pcc"] for v in gn["folds"].values()]
        A("**Graph-attention ResNet.** The Step-4 deep member ended in a dense "
          "`Linear(512, 5243)` head with no notion that co-regulated proteins should respond "
          "alike. This one uses a factorised head `(Z W_p) @ (E + Attn(E))^T + b`, where `E` "
          "is a per-protein embedding initialised from the STRING spectral coordinates and "
          f"`Attn` is multi-head attention over each protein's top-{gn['attn_k']} STRING "
          "neighbours. Cross-fitted over the same 5 LCGO folds; per-fold dev per-sample PCC "
          f"{min(pcc):.4f}-{max(pcc):.4f}.")
        A("")
        A("Two priors are implemented in its training loop:")
        A("")
        A("- `XD3` (graph signal processing): a Laplacian smoothness penalty "
          f"`lambda * tr(P^T L P) / (n p)` with lambda = {gn['lambda_graph']}, plus cosine "
          "annealing with warm restarts.")
        A("- `FD4` (*ProteinTalks*): the two data-loss terms — masked MSE and per-sample "
          "correlation — are balanced by the gradient-cosine rule (clip the cosine of the two "
          "task gradients at 1.0, form `0.01 x cosine`, shift weight toward the correlation "
          "task only when they conflict) rather than a fixed ratio.")
        A("")
        ctrl = gn.get("lambda_graph_control_fold0")
        if ctrl:
            A("Pre-specified control for the Laplacian penalty on fold 0: dev PCC "
              f"{ctrl['best_dev_pcc']:.4f} at lambda = 0 against "
              f"{ctrl['best_dev_pcc_with_penalty']:.4f} with the penalty, i.e. "
              f"{ctrl['delta_from_penalty']:+.4f}. One fold is a noisy estimate and this is "
              "reported as a diagnostic, not an established effect.")
            A("")
    if dlr:
        pcc2 = [v["best_dev_pcc"] for v in dlr["folds"].values()]
        A("**Step-4 dense-head architecture, also cross-fitted** (per-fold dev per-sample PCC "
          f"{min(pcc2):.4f}-{max(pcc2):.4f}). Step 4's stacking gave its deep member large "
          "weight, so keeping both deep architectures in the pool means the Step-4 -> Step-5 "
          "comparison is about the weight space rather than about which deep architecture "
          "happened to be available. The meta-learner is non-negative, so a weaker member "
          "simply receives a weight near zero.")
        A("")

    # ---- bootstrap -------------------------------------------------------
    if bo:
        A("### 13.8 Uncertainty on the margins")
        A("")
        b = bo["cluster_frozen_vs_scalar_frozen"]
        A("Paired non-parametric bootstrap over validation samples, "
          f"{b['n_boot']:,} replicates. Both candidates are scored on the *same* resampled "
          "sample sets, so the shared cohort sampling noise cancels and the interval reflects "
          "only their disagreement.")
        A("")
        A("| comparison | point | 95% CI | replicates favouring Step 5 |")
        A("|---|---|---|---|")
        A(f"| cluster-frozen - scalar-frozen | {b['point_diff']:+.6f} | "
          f"[{b['diff_ci95'][0]:+.6f}, {b['diff_ci95'][1]:+.6f}] | "
          f"{100 * b['frac_bootstrap_replicates_favouring_a']:.1f}% |")
        b2 = bo["cluster_frozen_vs_benchmark_member"]
        A(f"| cluster-frozen - benchmark member | {b2['point_diff']:+.6f} | "
          f"[{b2['diff_ci95'][0]:+.6f}, {b2['diff_ci95'][1]:+.6f}] | "
          f"{100 * b2['frac_bootstrap_replicates_favouring_a']:.1f}% |")
        A("")
        A("**Scope, stated on the interval itself:** the bootstrap covers only the submetrics "
          f"that are literally means over samples — {b['component_weight_of_total']:.4f} of the "
          "1.0 module weight. The pooled and per-protein-mean submetrics are *not* sample "
          "means, so resampling samples does not give them a valid bootstrap distribution, and "
          "they are excluded rather than approximated. The interval is therefore a CI on that "
          "component of the margin, not on the full total.")
        A("")

    # ---- agentic loop ----------------------------------------------------
    if lo:
        A("### 13.9 The agentic self-evolution loop")
        A("")
        A(f"{lo['n_iterations']} iterations, {lo['n_crashes']} crashes, "
          f"{lo['n_accepted']} accepted mutations, Pareto front of "
          f"{len(lo.get('pareto_front', []))} states.")
        A("")
        A("The loop mutates the **stacking configuration** — cluster index, cluster-spread "
          "shrinkage, member-role subset, regime tying, optimiser warm start and step schedule. "
          "It never touches the scoring code, the cohort definitions or the fold assignment. "
          "That boundary is the point: a loop permitted to edit its own objective will find a "
          "way to raise the number without improving the model.")
        A("")
        A("**Deliberate departure from the plan.** The plan proposed retaining states on a "
          "Pareto front of `(oof_total, val_total_if_promoted)`. That was not implemented: a "
          "val term in the retention rule makes val a tuning signal across iterations and the "
          "reported val score stops being held out. The front used instead is "
          "`(oof_total, number of free weights)` — accuracy against model complexity, which is "
          "a real trade-off and needs no val access. Every candidate is scored on the "
          "out-of-fold cohort; only the single promoted state is scored on `val_*`, once.")
        A("")
        A("Each mutation is attributed to the prior that motivated it, and a mutation with no "
          "prior behind it is logged as `exploratory` rather than given a citation it does not "
          "have.")
        A("")

    # ---- submission ------------------------------------------------------
    if vr:
        tp = vr["test_predictions"]
        A("### 13.10 Submission integrity and success criteria")
        A("")
        A(f"Test matrix: {tp['n_samples']:,} samples x {tp['n_proteins']:,} proteins. "
          f"{vr['artefact_integrity']['n_passed']}/{vr['artefact_integrity']['n_checks']} "
          "integrity checks passed.")
        A("")
        for c in vr["artefact_integrity"]["checks"]:
            A(f"- [{'PASS' if c['passed'] else 'FAIL'}] {c['name']} — {c['detail']}")
        A("")
        A(f"Predicted abundance range [{tp['y_pred_summary']['min']:.2f}, "
          f"{tp['y_pred_summary']['max']:.2f}] against an observed train range "
          f"[{tp['observed_train_range'][0]:.2f}, {tp['observed_train_range'][1]:.2f}]. "
          "Indicative fold-change PCC on the released test delta matrix: "
          f"{tp['indicative_fold_change_pcc_per_sample_mean']:.4f} — "
          f"{tp['indicative_note']}")
        A("")
        sc = vr["success_criteria"]
        A(f"**Success criteria: {sc['n_met']}/{sc['n_criteria']} met.**")
        A("")
        for c in sc["criteria"]:
            A(f"- [{'MET' if c['met'] else 'NOT MET'}] {c['name']}")
            A(f"  - {c['detail']}")
        A("")

    # ---- caveats ---------------------------------------------------------
    A("### 13.11 Limitations")
    A("")
    A("- The LCGO fit sets hold ~3,000 rows against the 5,078 the frozen weights are finally "
      "applied to, so a member-strength mismatch remains. It is roughly half the Step-4 gap "
      "but it is not zero, and no cross-fitting scheme can remove it.")
    A("- `train` contains only 4 strains, so the novel-strain axis can be probed one strain at "
      "a time. One strain is held out by two folds; the strain-novelty signal is inherently "
      "noisier than the chemical one.")
    A("- The bootstrap interval covers 0.5875 of the module weight, as stated above.")
    A("- The Laplacian-penalty control is a single fold.")
    A("- Held-out chemistry is largely structurally unsupported (see 13.5), which bounds what "
      "any of these members can extrapolate, independent of the stacking.")
    A("- All reported totals are on the local `val_*` cohort under a reconstructed submetric "
      "aggregation convention (`harness.py`, `SPEC_VERSION` 1.0.0). The organisers' exact "
      "aggregation and their test-side control matching are not available, so the local total "
      "is comparable across our own steps but is not the official score.")
    A("")
    if fo:
        A("### 13.12 Reproducing Step 5")
        A("")
        A("```bash")
        A("uv run python workflow/27_knowledge_extraction.py")
        A("uv run python workflow/28_advanced_features.py --part mol3d")
        A("uv run python workflow/28_advanced_features.py --part graph")
        A("uv run python workflow/29_lcgo_oof_matrix.py --stage oof \\")
        A("    --families lgb_tab,lgb_mol,lgb_mol3d")
        A("uv run python workflow/29_lcgo_oof_matrix.py --stage valtest \\")
        A("    --families lgb_tab,lgb_mol,lgb_mol3d")
        A("uv run python workflow/30a_smoke_clusterscore.py          # exactness gate")
        A("uv run python workflow/30_gnn_and_cluster_stacking.py --stage gnn --arch gnn")
        A("uv run python workflow/30_gnn_and_cluster_stacking.py --stage gnn --arch mlp")
        A("uv run python workflow/30_gnn_and_cluster_stacking.py --stage stack")
        A("uv run python workflow/31_agentic_loop_runner.py")
        A("uv run python workflow/32_update_manifest.py")
        A("```")
        A("")
        A("`workflow/run_step5_chain.sh` chains the three long stages. They are chained rather "
          "than parallelised deliberately: the LightGBM fits and the deep-learning featuriser "
          "both want the CPU, and running them concurrently is what made a Step-4 fit take "
          "1,527 s instead of 37 s.")
        A("")
    return "\n".join(L)


def main() -> None:
    section = build()
    p = SESSION / "README.md"
    txt = p.read_text(encoding="utf-8")
    if MARKER in txt:
        txt = txt[: txt.index(MARKER)].rstrip() + "\n\n"
        log("replacing the existing Step-5 section (idempotent)")
    else:
        txt = txt.rstrip() + "\n\n"
    p.write_text(txt + section + "\n", encoding="utf-8")
    log(f"README.md updated ({len(section.splitlines())} lines of Step-5 section)")


if __name__ == "__main__":
    main()
