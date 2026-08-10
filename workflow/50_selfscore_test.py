#!/usr/bin/env python
"""Step 7.1 -- Self-evaluation on the released test proteome ground truth.

Why this script exists
----------------------
The 2026-08-11 revision of the Track-3 handbook (item 2.2 #7) states that the
**test-set proteome ground truth is now released together with the data package**
for self-evaluation, while the final ranking is decided on a separate internal
evaluation set held by the organisers.  Before that revision, the test truth was
assumed to be withheld, so Step 6 could only report an *indicative* fold-change
PCC on the test cohort.

This script uses the released truth to run the **full seven-module offline
scoring rubric** on the test cohort -- the same harness, the same frozen
``mu_ctx`` / ``mu_drug`` construction (train rows only), the same aggregation
spec -- for

* the Step-6 submitted prediction (hierarchical cluster stacking), and
* the four trivial baselines from Step 2 (``global_mean``, ``per_context_mean``,
  ``per_context_mean_batch``, ``control_anchor``),

so that the test-cohort null floors and the official-style group-mean benchmark
are available on exactly the cohort that was submitted.

Discipline preserved
--------------------
* ``mu_ctx`` / ``mu_drug`` and every baseline group mean are fitted on
  ``split_final == 'train'`` rows of ``train_val`` **only**.  Test rows never
  enter any fit; a leakage guard re-fits with all non-train rows blanked and
  requires bit-identical baselines.
* The test truth is used for **scoring only**.  No model, weight, cluster
  assignment or hyper-parameter in the submitted pipeline was selected using it
  -- the Step-6 predictions read here were frozen on 2026-08-07, before this
  script existed.
* This is *not* the official score.  The organisers' scorer may differ in the
  aggregation convention and in test-side control matching, and the final
  ranking uses an independent internal evaluation set.

Outputs
-------
results/step7_test_selfscore.json
results/step7_test_selfscore_modules.csv
"""
from __future__ import annotations

from repo_paths import REPO_ROOT

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SESSION = REPO_ROOT
WORKFLOW = SESSION / "workflow"
DATA = SESSION / "data"
RESULTS = SESSION / "results"
SEED = 42
sys.path.insert(0, str(WORKFLOW))

import harness as H  # noqa: E402
import validation_splits as VS  # noqa: E402

np.random.seed(SEED)
T0 = time.time()

#: test split name -> the val split name the harness spec keys on.
SPLIT_ALIAS = {
    "test_chem_only": "val_chem_only",
    "test_strain_only": "val_strain_only",
    "test_both": "val_both",
    "test_time": "val_time",
}
KEY_COLS = ["data_source", "Strains", "Medium", "Temperature", "pert_time",
            "perturbation_no_concentration", "split_final"]


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def load_cohort(delta_path: Path, prot_path: Path, meta_path: Path,
                tag: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Row-align (meta, Y, D) for the treated rows of one cohort."""
    dl = pd.read_parquet(delta_path)
    drop = ["sample_ID", "match_level", "split_final", "perturbation_no_concentration"]
    proteins = [c for c in dl.columns if c not in drop]
    pr = pd.read_parquet(prot_path)
    if [c for c in pr.columns if c != "sample_ID"] != proteins:
        raise ValueError(f"{tag}: protein order differs between delta and proteome matrices")
    meta_all = pd.read_csv(meta_path, dtype=str)

    ids = dl["sample_ID"].astype(str).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError(f"{tag}: duplicate sample_ID in the delta matrix")
    pos = pd.Series(np.arange(len(pr)), index=pr["sample_ID"].astype(str).to_numpy())
    order = pos.reindex(ids).to_numpy()
    if np.isnan(order).any():
        raise ValueError(f"{tag}: delta sample_IDs absent from the proteome matrix")
    Y = pr[proteins].to_numpy(dtype="float32")[order.astype(int)]
    D = dl[proteins].to_numpy(dtype="float32")

    mpos = pd.Series(np.arange(len(meta_all)),
                     index=meta_all["sample_ID"].astype(str).to_numpy())
    morder = mpos.reindex(ids).to_numpy()
    if np.isnan(morder).any():
        raise ValueError(f"{tag}: delta sample_IDs absent from the annotated metadata")
    meta = meta_all.iloc[morder.astype(int)].reset_index(drop=True)
    if (meta["split_final"].to_numpy() != dl["split_final"].to_numpy()).any():
        raise ValueError(f"{tag}: split_final disagrees between metadata and delta matrix")
    if (meta["sample_role"] != "treatment").any():
        raise ValueError(f"{tag}: delta matrix contains non-treatment rows")
    log(f"  {tag}: {Y.shape[0]} treated rows x {len(proteins)} proteins; "
        f"Delta defined on {100 * float(np.isfinite(D).mean()):.3f}% of cells")
    return meta, Y, D, proteins


def main() -> None:
    log("=== Step 7.1: seven-module self-evaluation on the released test truth ===")
    log(f"numpy {np.__version__} | pandas {pd.__version__} | "
        f"python {platform.python_version()} | score spec {H.SPEC_VERSION} | seed {SEED}")

    meta_tv, Y_tv, D_tv, proteins = load_cohort(
        WORKFLOW / "processed_delta_matrix.parquet",
        WORKFLOW / "processed_train_val_proteome.parquet",
        DATA / "meta_train_val_annotated.csv", "train_val")
    meta_te, Y_te, D_te, proteins_te = load_cohort(
        WORKFLOW / "processed_delta_matrix_test.parquet",
        WORKFLOW / "processed_test_proteome.parquet",
        DATA / "meta_test_annotated.csv", "test")
    if proteins_te != proteins:
        raise ValueError("protein column order differs between the two cohorts")

    n_tv, n_te = len(meta_tv), len(meta_te)
    meta_all = pd.concat([meta_tv[KEY_COLS], meta_te[KEY_COLS]], ignore_index=True)
    Y_all = np.concatenate([Y_tv, Y_te], axis=0)
    D_all = np.concatenate([D_tv, D_te], axis=0)
    del Y_tv, D_tv

    fit_mask = np.zeros(len(meta_all), dtype=bool)
    fit_mask[:n_tv] = (meta_tv["split_final"].to_numpy() == "train")
    log(f"fit cohort (train rows of train_val only): n={int(fit_mask.sum())}; "
        f"test rows to score: n={n_te}")

    # --- train-frozen Module-3 residual baselines, evaluated on the test rows -
    log("building train-frozen mu_ctx / mu_drug over the pooled frame ...")
    mu_ctx, lvl_ctx, info_ctx = VS.frozen_delta_baseline(
        meta_all, D_all, fit_mask, VS.CTX_LEVELS, "mu_ctx_test")
    mu_drug, lvl_drug, info_drug = VS.frozen_delta_baseline(
        meta_all, D_all, fit_mask, VS.DRUG_LEVELS, "mu_drug_test")
    log("building the batch-aware mu_ctx used only for the confound sensitivity ...")
    mu_ctx_b, _, info_ctx_b = VS.frozen_delta_baseline(
        meta_all, D_all, fit_mask, VS.CTX_LEVELS_BATCH, "mu_ctx_batch_test")

    # Leakage guard: blank every non-fit row and require bit-identical baselines.
    D_guard = D_all.copy()
    D_guard[~fit_mask] = np.nan
    mu_ctx_guard, _, _ = VS.frozen_delta_baseline(
        meta_all, D_guard, fit_mask, VS.CTX_LEVELS, "mu_ctx_leakguard")
    identical = np.array_equal(np.nan_to_num(mu_ctx, nan=-9e9),
                               np.nan_to_num(mu_ctx_guard, nan=-9e9))
    if not identical:
        raise AssertionError("mu_ctx changed when non-train rows were blanked -> leakage")
    log("leakage guard PASSED (baselines depend on train rows only, bit-identical)")
    del D_guard, mu_ctx_guard

    # --- baseline predictors, fitted on train rows only ----------------------
    log("fitting the four trivial baselines on train rows only ...")
    with np.errstate(all="ignore"):
        global_vec = np.nanmean(Y_all[fit_mask], axis=0).astype("float32")
    Y_global = np.broadcast_to(global_vec, Y_all.shape)
    Y_ctx, _, info_abund_ctx = VS.frozen_delta_baseline(
        meta_all, Y_all, fit_mask, VS.CTX_LEVELS, "abund_ctx_test")
    Y_ctxb, _, info_abund_ctxb = VS.frozen_delta_baseline(
        meta_all, Y_all, fit_mask, VS.CTX_LEVELS_BATCH, "abund_ctx_batch_test")

    # --- per-stage frozen test predictions -----------------------------------
    #  Every stage exported its own test matrix at the time it was frozen, so
    #  scoring all four gives the validation-to-test generalisation profile of the
    #  whole trajectory, not just of the final model.
    stage_files = {
        "step3_gbdt": "gbdt_test_predictions.parquet",
        "step4_chem_stacking": "step4_test_predictions.parquet",
        "step5_knowledge_stacking": "step5_test_predictions.parquet",
        "step6_submitted": "step6_test_predictions.parquet",
    }
    ids_te = meta_te["sample_ID"].astype(str).to_numpy()
    stage_pred: dict[str, np.ndarray] = {}
    for tag, fname in stage_files.items():
        log(f"loading the frozen {tag} test predictions ({fname}) ...")
        pred = pd.read_parquet(DATA / fname)
        if [c for c in pred.columns if c != "sample_ID"] != proteins:
            raise ValueError(f"{tag}: prediction protein order differs from the truth")
        ppos = pd.Series(np.arange(len(pred)),
                         index=pred["sample_ID"].astype(str).to_numpy())
        porder = ppos.reindex(ids_te).to_numpy()
        if np.isnan(porder).any():
            raise ValueError(f"{tag}: predictions do not cover every treated test sample")
        stage_pred[tag] = pred[proteins].to_numpy(dtype="float32")[porder.astype(int)]
        del pred

    te = slice(n_tv, n_tv + n_te)
    C_te = Y_te - D_te                      # frozen matched-control anchor
    predictors: dict[str, np.ndarray] = {
        **stage_pred,
        "global_mean": np.ascontiguousarray(np.asarray(Y_global)[te]),
        "per_context_mean": np.ascontiguousarray(Y_ctx[te]),
        "per_context_mean_batch": np.ascontiguousarray(Y_ctxb[te]),
        "control_anchor": C_te,
    }
    mu_ctx_te = np.ascontiguousarray(mu_ctx[te])
    mu_drug_te = np.ascontiguousarray(mu_drug[te])
    mu_ctx_b_te = np.ascontiguousarray(mu_ctx_b[te])
    del mu_ctx, mu_drug, mu_ctx_b, Y_ctx, Y_ctxb, Y_all, D_all

    meta_score = meta_te.copy()
    meta_score["split_final"] = meta_score["split_final"].map(SPLIT_ALIAS)
    if meta_score["split_final"].isna().any():
        raise ValueError("unmapped test split name")
    log("test-cohort split sizes: " +
        ", ".join(f"{k}={v}" for k, v in
                  meta_score["split_final"].value_counts().sort_index().items()))

    results: dict[str, dict] = {}
    rows: list[dict] = []
    confound: list[dict] = []
    for name, Yhat in predictors.items():
        log(f"--- scoring '{name}' on the test cohort ---")
        Yhat = np.ascontiguousarray(Yhat, dtype="float32")
        Dhat = Yhat - C_te
        res = H.compute_competition_score(
            Y_te, Yhat, D_te, Dhat, meta_score,
            mu_ctx=mu_ctx_te, mu_drug=mu_drug_te, verbose=False)
        results[name] = {
            "total_score": res["total_score"],
            "module_scores": res["module_scores"],
            "module_weights": res["module_weights"],
            "module_weighted_contributions": res["module_weighted_contributions"],
            "primary_submetrics": res["primary_submetrics"],
            "warnings": res["warnings"],
            "sensitivity": H.score_sensitivity(res),
        }
        log(f"    total_score = {res['total_score']:.6f}")
        for mod, sc in res["module_scores"].items():
            log(f"      {mod:16s} {sc:.6f}  (weight {res['module_weights'][mod]:.2f})")
            rows.append({"model": name, "module": mod,
                         "score": sc, "weight": res["module_weights"][mod],
                         "weighted": res["module_weights"][mod] * sc})
        rows.append({"model": name, "module": "TOTAL", "score": res["total_score"],
                     "weight": 1.0, "weighted": res["total_score"]})

        # batch-blind vs batch-aware mu_ctx on each split (the S1 confound)
        for split in ("val_chem_only", "val_strain_only", "val_both", "val_time"):
            sm = (meta_score["split_final"] == split).to_numpy()
            if not sm.any():
                continue
            blind = H.module3_residual(D_te[sm], Dhat[sm], mu_ctx_te[sm], prefix="ctx")
            aware = H.module3_residual(D_te[sm], Dhat[sm], mu_ctx_b_te[sm], prefix="ctxb")
            confound.append({
                "model": name, "split": split, "n_samples": int(sm.sum()),
                "resid_pcc_batch_blind": blind["ctx_pcc_per_sample_mean"],
                "resid_pcc_batch_aware": aware["ctxb_pcc_per_sample_mean"],
                "delta_attributable_to_batch":
                    blind["ctx_pcc_per_sample_mean"] - aware["ctxb_pcc_per_sample_mean"],
            })

    payload = {
        "step": "7_1_test_selfscore",
        "seed": SEED,
        "spec_version": H.SPEC_VERSION,
        "provenance": {
            "why": "the 2026-08-11 handbook revision releases the test proteome truth "
                   "for self-evaluation (item 2.2 #7); the final ranking uses a "
                   "separate internal evaluation set held by the organisers",
            "predictions_scored": "results/step6_test_predictions.csv (frozen 2026-08-07)",
            "truth": "workflow/processed_test_proteome.parquet and "
                     "workflow/processed_delta_matrix_test.parquet",
            "baselines_fitted_on": "split_final == 'train' rows of train_val only",
            "not_the_official_score": True,
            "caveat": "the organisers' aggregation convention and test-side control "
                      "matching are not published; this is our offline rubric applied "
                      "to the released truth, and the test truth was used for scoring "
                      "only -- never for model, weight or hyper-parameter selection",
        },
        "cohort": {
            "n_test_treated_rows": int(n_te),
            "n_proteins": len(proteins),
            "n_fit_rows": int(fit_mask.sum()),
            "split_sizes": {k: int(v) for k, v in
                            meta_te["split_final"].value_counts().sort_index().items()},
            "delta_defined_fraction": round(float(np.isfinite(D_te).mean()), 6),
            "match_levels": {"L1_full_ctx": int(n_te)},
        },
        "baseline_diagnostics": {
            "mu_ctx": info_ctx, "mu_drug": info_drug,
            "mu_ctx_batch": info_ctx_b,
            "abund_ctx": info_abund_ctx, "abund_ctx_batch": info_abund_ctxb,
            "leakage_guard_bit_identical": bool(identical),
        },
        "scores": results,
        "residual_confound": confound,
        "runtime_seconds": round(time.time() - T0, 1),
    }

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return None
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    out = RESULTS / "step7_test_selfscore.json"
    out.write_text(json.dumps(_clean(payload), indent=2, ensure_ascii=False))
    pd.DataFrame(rows).to_csv(RESULTS / "step7_test_selfscore_modules.csv", index=False)
    log(f"wrote {out.name} and step7_test_selfscore_modules.csv")

    log("=== summary (test cohort, offline rubric, NOT the official score) ===")
    for name in predictors:
        log(f"  {name:24s} total = {results[name]['total_score']:.6f}")


if __name__ == "__main__":
    main()
